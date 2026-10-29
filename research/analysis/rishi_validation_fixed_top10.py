"""Compare validation-fixed and model top-ten full-exposure accounts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
import run_ndx_drawdown_research as study
from audit_ndx_weak_market_short import value_replay
from rishi_mechanism_batch1 import stationary_indices,paired_contrast
from rishi_validation_stable_rank import ranks


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch54_validation_top10');large=Path('results/RISHI_VALIDATION_TOP10_20261029')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    data=study.load_data();ids=np.concatenate([study.fold(data,y)[2] for y in (2022,2023,2024,2025)])
    source=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028');scores=np.load(source/'SCORES.npz',allow_pickle=False);prior=np.load(source/'PATHS.npz',allow_pickle=False);dates=prior['dates']
    np.testing.assert_array_equal(ids,scores['ids']);roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')]
    paths={};targets={};metrics=[];calendar=[];checks=[];hashes={}
    for group,seeds in study.GROUPS.items():
        parts=[]
        for year in (2022,2023,2024,2025):
            _,val,test=study.fold(data,year);assert data['exit'][val].max()<data['origin'][test].min();families=[]
            for root in roots:
                family=[]
                for seed in seeds:
                    path=root/str(year)/f'{seed}_predictions.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as z:family.append(ranks(z['validation'],data['mask'][val]))
                families.append(np.mean(family,axis=0))
            blend=(families[0]+families[1])/2;n=np.isfinite(blend).sum(0);mean=np.divide(np.nansum(blend,axis=0),n,out=np.full(len(n),np.nan),where=n>0)
            support=(n>=5)&np.isfinite(mean);assert support.any();mean[~support]=np.median(mean[support]);parts.append(engine.allocations(data,test,np.broadcast_to(mean,(len(test),len(mean))),'equal10'))
        fixed=np.concatenate(parts);original=engine.allocations(data,ids,scores[f'after_hours_premarket__{group}'],'equal10')
        for arm,w in [('original',original),('fixed',fixed)]:
            targets[f'{group}__{arm}']=w
            for cost in (1,2):
                m,net,ledger=engine.simulate(data,ids,w,np.ones(len(data['panel'].dates)),'weekly',multiplier=cost,detailed=True)
                assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                if arm=='original' and cost==1:
                    key=f'after_hours_premarket__{group}__k10';old=pd.read_csv(source/f'{key}_ledger.csv');err=float(np.max(abs(net-prior[key])));nav=float(np.max(abs(ledger.nav-old.nav))/1e6)
                    assert err<=1e-10 and nav<=1e-8;checks.append(dict(group=group,kind='saved_1x',daily_error=err,nav_error=nav))
                if arm=='original' and cost==2:
                    ref=value_replay(data,ids,w,0.,2.);err=float(np.max(abs(net-ref)));assert err<=1e-10;checks.append(dict(group=group,kind='independent_2x',daily_error=err,nav_error=np.nan))
                key=f'{group}__{arm}__c{cost}';paths[key]=net;ledger.to_csv(large/f'{key}_ledger.csv',index=False);metrics.append(dict(group=group,arm=arm,cost=cost,**m,mean_exposure=float(ledger.equity_exposure.mean())))
                for year in (2022,2023,2024,2025):
                    mask=pd.DatetimeIndex(dates).year==year;calendar.append(dict(group=group,arm=arm,cost=cost,year=year,calendar_return=float(np.prod(1+net[mask])-1)))
                print(key,flush=True)
    contrasts=[]
    for block in (5,21,63):
        draw=stationary_indices(len(dates),10000,block,27102032)
        for group in study.GROUPS:
            for cost in (1,2):
                contrasts.append(dict(group=group,cost=cost,block=block,**paired_contrast(paths[f'{group}__original__c{cost}'],paths[f'{group}__fixed__c{cost}'],draw,252)))
    assert len(metrics)==12 and len(checks)==6 and len(contrasts)==18
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(checks).to_csv(out/'CHECKS.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths);np.savez_compressed(large/'TARGETS.npz',ids=ids,security_ids=np.asarray(data['panel'].tickers),**targets)
    for p in (source/'SCORES.npz',source/'PATHS.npz',Path(__file__),Path('docs/rishi_mechanism_20261027/VALIDATION_FIXED_TOP10_PROTOCOL_20261029.md')):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(pd.DataFrame(contrasts).query("group=='ten_seed' and block==21").to_string(index=False))


if __name__=='__main__':main()
