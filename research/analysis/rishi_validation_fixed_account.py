"""Replay fixed validation preference against saved issuer-five model accounts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import POLICIES
from rishi_mechanism_batch1 import stationary_indices, paired_contrast
from rishi_validation_stable_rank import ranks


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch49_validation_account')
    large=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029')
    for p in (out,large):
        if p.exists(): raise FileExistsError(p)
        p.mkdir(parents=True)
    data=study.load_data();ids=np.concatenate([study.fold(data,y)[2] for y in (2022,2023,2024,2025)])
    saved=np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz',allow_pickle=False)
    prior=np.load('results/RISHI_MECHANISM_RISK_20261027/PATHS_PORTABLE.npz',allow_pickle=False)
    np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['security_ids'],data['panel'].tickers)
    dates=prior['dates']; market=engine.market_inputs(data); original=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch48_validation_fixed/ORIGINS.csv').set_index(['group','fill_date'])
    roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')]
    inputs={};paths={};targets={};metrics=[];replays=[];calendar=[];overlap_count=0
    for group,seeds in study.GROUPS.items():
        fixed_parts=[]
        for year in (2022,2023,2024,2025):
            _,val,test=study.fold(data,year); assert data['exit'][val].max()<data['origin'][test].min()
            families=[]
            for root in roots:
                ranks_list=[]
                for seed in seeds:
                    path=root/str(year)/f'{seed}_predictions.npz';inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as z:ranks_list.append(ranks(z['validation'],data['mask'][val]))
                families.append(np.mean(ranks_list,axis=0))
            blend=(families[0]+families[1])/2;n=np.isfinite(blend).sum(0)
            mean=np.divide(np.nansum(blend,axis=0),n,out=np.full(len(n),np.nan),where=n>0)
            supported=(n>=5)&np.isfinite(mean);assert supported.any();mean[~supported]=np.median(mean[supported])
            fixed_parts.append(engine.allocations(data,test,np.broadcast_to(mean,(len(test),len(mean))),'issuer5'))
        weights_fixed=np.concatenate(fixed_parts);weights_original=saved[f'after_hours_premarket__{group}__issuer5']
        for j,i in enumerate(ids):
            key=(group,str(data['panel'].dates[data['entry'][i]].date()));row=original.loc[key]
            assert row.shared_names==np.count_nonzero((weights_fixed[j]>0)&(weights_original[j]>0))
            np.testing.assert_allclose(row.target_weight_overlap,np.minimum(weights_fixed[j],weights_original[j]).sum(),atol=1e-12,rtol=0);overlap_count+=1
        for arm,weights in [('original',weights_original),('fixed',weights_fixed)]:
            targets[f'{group}__{arm}']=weights;risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,weights))
            for policy in ('full','vol20'):
                exposure=engine.exposure_series(POLICIES[policy],market,risk)
                m,net,ledger=engine.simulate(data,ids,weights,exposure,'weekly',multiplier=1.,detailed=True)
                assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                if arm=='original':
                    key=f'E/{group}/issuer5/{policy}/modeled1'; ref=prior['returns'][:,np.flatnonzero(prior['ids']==key)[0]]
                    old=pd.read_csv('results/RISHI_MECHANISM_RISK_20261027/'+key.replace('/','__')+'.csv')
                    err=float(np.max(abs(net-ref)));nav=float(np.max(abs(ledger.nav-old.nav))/1e6)
                    assert err<1e-10 and nav<1e-8;replays.append(dict(group=group,policy=policy,daily_error=err,nav_error=nav))
                key=f'{group}__{arm}__{policy}';paths[key]=net;ledger.to_csv(large/f'{key}_ledger.csv',index=False)
                metrics.append(dict(group=group,arm=arm,policy=policy,**m,mean_exposure=float(ledger.equity_exposure.mean()),min_cash=float(ledger.cash.min())))
                for year in (2022,2023,2024,2025):
                    mask=pd.DatetimeIndex(dates).year==year;calendar.append(dict(group=group,arm=arm,policy=policy,year=year,calendar_return=float(np.prod(1+net[mask])-1)))
                print(key,flush=True)
    assert overlap_count==603 and len(metrics)==12 and len(replays)==6
    contrasts=[]
    for block in (5,21,63):
        draws=stationary_indices(len(dates),10000,block,27102029)
        for group in study.GROUPS:
            for policy in ('full','vol20'):
                contrasts.append(dict(group=group,policy=policy,block=block,**paired_contrast(paths[f'{group}__original__{policy}'],paths[f'{group}__fixed__{policy}'],draws,252)))
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(replays).to_csv(out/'REPLAY.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths);np.savez_compressed(large/'TARGETS.npz',ids=ids,security_ids=np.asarray(data['panel'].tickers),**targets)
    (out/'INPUT_HASHES.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    print(pd.DataFrame(contrasts).query("group=='ten_seed' and block==21").to_string(index=False))


if __name__=='__main__':main()
