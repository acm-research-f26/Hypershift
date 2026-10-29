"""Replay fixed style-only, validation and original targets under matched policies."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import POLICIES
from rishi_mechanism_batch1 import stationary_indices,paired_contrast


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch61_validation_style_accounts';large=Path('results/RISHI_VALIDATION_STYLE_ACCOUNTS_20261029')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    d=study.load_data();market=engine.market_inputs(d);names=np.asarray(d['panel'].tickers);ids=np.concatenate([study.fold(d,y)[2] for y in (2022,2023,2024,2025)])
    common=doc/'execution_batch58_validation_style_partial/COMMON_STOCKS.csv';f=pd.read_csv(common);overlaps=pd.read_csv(doc/'execution_batch60_validation_style_overlap/ORIGINS.csv').set_index(['group','origin','control','allocation']);assert overlaps.index.is_unique
    roots={'issuer5':Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029'),'equal10':Path('results/RISHI_VALIDATION_TOP10_20261029')}
    targets={k:np.load(p/'TARGETS.npz',allow_pickle=False) for k,p in roots.items()};prior={k:np.load(p/'PATHS.npz',allow_pickle=False) for k,p in roots.items()}
    dates=prior['issuer5']['dates'];np.testing.assert_array_equal(dates,prior['equal10']['dates'])
    for z in targets.values():np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['security_ids'],names)
    paths={};all_targets={};metrics=[];calendar=[];parity=[];check=0
    for group in ('original_five','new_five','ten_seed'):
        controls={c:{k:[] for k in roots} for c in ('beta','liquidity','equal_blend')}
        cursor=0
        for year in (2022,2023,2024,2025):
            _,_,test=study.fold(d,year);sub=f[(f.group==group)&(f.year==year)].set_index('security_id');support=np.isin(names,sub.index);idx={v:i for i,v in enumerate(names)};scores={}
            for label,col in [('beta','beta252'),('liquidity','log10adv20')]:
                a=np.full(len(names),np.nan)
                for name,value in sub[col].rank(method='average',pct=True).items():a[idx[name]]=value
                scores[label]=a
            scores['equal_blend']=(scores['beta']+scores['liquidity'])/2
            for label,score in scores.items():
                score=score.copy();score[~support]=np.median(score[support])
                for kind in roots:
                    w=engine.allocations(d,test,np.broadcast_to(score,(len(test),len(names))),kind);ref=targets[kind][f'{group}__fixed'][cursor:cursor+len(test)]
                    for j,i in enumerate(test):
                        row=overlaps.loc[(group,int(i),label,kind)];assert row.shared_names==np.count_nonzero((w[j]>0)&(ref[j]>0));np.testing.assert_allclose(row.weight_overlap,np.minimum(w[j],ref[j]).sum(),atol=1e-12,rtol=0);check+=1
                    controls[label][kind].append(w)
            cursor+=len(test)
        for kind in roots:
            for arm in ('original','fixed','beta','liquidity','equal_blend'):
                w=targets[kind][f'{group}__{arm}'] if arm in ('original','fixed') else np.concatenate(controls[arm][kind]);all_targets[f'{group}__{kind}__{arm}']=w
                policy='vol20' if kind=='issuer5' else 'full';risk=engine.risk_inputs(d,engine.daily_allocation(d,ids,w));exposure=engine.exposure_series(POLICIES[policy],market,risk)
                m,net,ledger=engine.simulate(d,ids,w,exposure,'weekly',multiplier=1.,detailed=True);assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                if arm in ('original','fixed'):
                    oldkey=f'{group}__{arm}__vol20' if kind=='issuer5' else f'{group}__{arm}__c1';err=float(np.max(abs(net-prior[kind][oldkey])));old=pd.read_csv(roots[kind]/f'{oldkey}_ledger.csv');nav=float(np.max(abs(ledger.nav-old.nav))/1e6);assert err<=1e-10 and nav<=1e-8;parity.append(dict(group=group,kind=kind,arm=arm,daily_error=err,nav_error=nav))
                key=f'{group}__{kind}__{arm}';paths[key]=net;ledger.to_csv(large/f'{key}_ledger.csv',index=False);metrics.append(dict(group=group,kind=kind,arm=arm,**m,mean_exposure=float(ledger.equity_exposure.mean())))
                for year in (2022,2023,2024,2025):
                    mask=pd.DatetimeIndex(dates).year==year;calendar.append(dict(group=group,kind=kind,arm=arm,year=year,calendar_return=float(np.prod(1+net[mask])-1)))
                print(key,flush=True)
    assert check==3618 and len(metrics)==30 and len(parity)==12
    contrasts=[]
    for block in (5,21,63):
        draw=stationary_indices(len(dates),10000,block,29102026)
        for group in ('original_five','new_five','ten_seed'):
            for kind in roots:
                for arm in ('original','fixed'):
                    for control in ('beta','liquidity','equal_blend'):
                        contrasts.append(dict(group=group,kind=kind,arm=arm,control=control,block=block,**paired_contrast(paths[f'{group}__{kind}__{arm}'],paths[f'{group}__{kind}__{control}'],draw,252)))
    assert len(contrasts)==108
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'REPLAY.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths);np.savez_compressed(large/'TARGETS.npz',ids=ids,security_ids=names,**all_targets)
    inputs=[common,doc/'execution_batch60_validation_style_overlap/ORIGINS.csv',doc/'VALIDATION_SIMPLE_STYLE_ACCOUNTS_PROTOCOL_20261029.md',Path(__file__)]+[p/x for p in roots.values() for x in ('TARGETS.npz','PATHS.npz')]
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(pd.DataFrame(contrasts).query("group=='ten_seed' and block==21")[['kind','arm','control','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
