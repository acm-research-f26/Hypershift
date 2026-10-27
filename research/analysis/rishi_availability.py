"""Frozen post-score common-availability screen and membership-cohort attribution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold,continuous_backtest
from rishi_mechanism_selection import CANDIDATES
from rishi_mechanism_batch1 import stationary_indices,paired_contrast
from rishi_mechanism_risk import write_paths


def common_screen(data,ids,training_counts):
    return data['mask'][ids]&np.isfinite(data['momentum'][ids])&np.isfinite(data['reversal'][ids])&(training_counts>=30)


def membership_age(members):
    age=np.zeros(np.shape(members),dtype=np.int32);previous=np.zeros(np.shape(members)[1],dtype=np.int32)
    for day,current in enumerate(members):previous=np.where(current,previous+1,0);age[day]=previous
    return age


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch11_availability'
    large=Path('results/RISHI_MECHANISM_AVAILABILITY_20261027')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');scores={};ids=None
    for name in CANDIDATES:
        with np.load(source/f'scores_{name}_ten_seed.npz') as saved:
            if ids is None:ids=saved['ids'].copy()
            else:np.testing.assert_array_equal(ids,saved['ids'])
            scores[name]=saved['scores'].copy()
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy();counts=np.zeros_like(scores['E'],dtype=int)
    for year in [2022,2023,2024,2025]:
        train,_,_=fold(data,year);counts[years==year]=data['labelmask'][train].sum(0)
    common=common_screen(data,ids,counts)
    if np.any(common.sum(1)<2):raise ValueError('Common screen insufficient; do not tune')
    selected=data.copy();selected['mask']=data['mask'].copy();selected['mask'][ids]=common
    diagnostics=[]
    for row,i in enumerate(ids):
        mask=data['mask'][i]
        diagnostics.append(dict(origin_id=int(i),year=int(years[row]),original_eligible=int(mask.sum()),common_eligible=int(common[row].sum()),
            unavailable_momentum=int((mask&~np.isfinite(data['momentum'][i])).sum()),
            unavailable_reversal=int((mask&~np.isfinite(data['reversal'][i])).sum()),
            insufficient_training_labels=int((mask&(counts[row]<30)).sum())))
    pd.DataFrame(diagnostics).to_csv(out/'MASK_DIAGNOSTICS.csv',index=False)
    metrics=[];paths={}
    for name in CANDIDATES+['EW']:
        value=np.zeros_like(scores['E']) if name=='EW' else scores[name]
        for k in ([0] if name=='EW' else [5,10]):
            m,curve,ledger=continuous_backtest(selected,ids,value,10 if name=='EW' else k)
            key=f'{name}/{k}';metrics.append(dict(candidate=name,k=k,**m));paths[key]=curve.net.to_numpy()
            curve.to_csv(large/(key.replace('/','_')+'.csv'),index=False)
    pd.DataFrame(metrics).to_csv(out/'ACCOUNT_METRICS.csv',index=False)
    dates=data['panel'].dates[int(data['entry'][ids[0]]):]
    write_paths(large/'PATHS.npz',paths,dates)
    contrasts=[]
    for block in [21,5,63]:
        draws=stationary_indices(len(dates),10000,block,27102036)
        for right in ['B','E_Hlabel']:
            value=paired_contrast(paths['E/10'],paths[right+'/10'],draws,252);value.pop('mean_p_one_sided')
            contrasts.append(dict(right=right,block=block,**value))
    pd.DataFrame(contrasts).to_csv(out/'PAIRED_SENSITIVITY.csv',index=False)
    age=membership_age(data['membership']);cohorts=[];old_e=None;account_count=0
    def add(key,arrays,axis_dates,security_ids):
        nonlocal account_count
        assert security_ids==data['panel'].tickers
        positions=data['panel'].dates.get_indexer(pd.DatetimeIndex(axis_dates));assert (positions>=0).all()
        current=data['membership'][positions];young=current&(age[positions]<=126)
        masks={'outside_snapshot':~current,'recent_member126':young,'established_member':current&~young}
        total=0.
        for label,mask in masks.items():
            contribution=float(np.where(mask,arrays['net_contribution'],0).sum()/len(positions)*252);total+=contribution
            cohorts.append(dict(id=key,cohort=label,annual_arithmetic_contribution=contribution,
                gross_dollars=float(np.where(mask,arrays['gross_dollars'],0).sum()),fees=float(np.where(mask,arrays['fees'],0).sum()),
                held_security_days=int((mask&(arrays['units']>1e-12)).sum())))
        np.testing.assert_allclose(total,arrays['net_contribution'].sum()/len(positions)*252,atol=1e-12)
        account_count+=1
    old=Path('results/RISHI_MECHANISM_TRACE_20261027');axes=json.loads((old/'AXES.json').read_text())
    for name in ['B','D','E','Hlabel','E_Hlabel']:
        for k in (5,10):
            with np.load(old/f'{name}_ten_seed_k{k}.npz',allow_pickle=False) as saved:arrays={f:saved[f].copy() for f in ['units','gross_dollars','fees','net_contribution']}
            add(f'{name}/ten_seed/{k}/full',arrays,axes['dates'],axes['security_ids'])
            if name=='E' and k==5:old_e=arrays['net_contribution'].copy()
    leaders=pd.read_csv(doc/'execution_batch9_leader_trace/RECONCILIATION.csv')
    for key in leaders.id:
        if key in ['SPY','QQQ']:continue
        path=Path('results/RISHI_MECHANISM_LEADER_TRACE_20261027')/(key.replace('/','__')+'.npz')
        with np.load(path,allow_pickle=False) as saved:
            arrays={f:saved[f].copy() for f in ['units','gross_dollars','fees','net_contribution']}
            if key=='equal5/full/ten_seed':np.testing.assert_allclose(old_e,arrays['net_contribution'],atol=1e-12);continue
            add(key,arrays,saved['dates'].tolist(),saved['security_ids'].tolist())
    assert account_count==19 and len(cohorts)==57
    pd.DataFrame(cohorts).to_csv(out/'MEMBERSHIP_COHORTS.csv',index=False)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),doc/'AVAILABILITY_ENTRY_PROTOCOL.md']}
    (out/'RESULTS.json').write_text(json.dumps(dict(common_accounts=len(metrics),contrasts=len(contrasts),
        common_eligible_min=int(common.sum(1).min()),common_eligible_max=int(common.sum(1).max()),
        cohort_accounts=account_count,cohort_rows=len(cohorts),source_hashes=hashes,
        scope='Post-score ex-ante common screen and descriptive source-membership cohorts; original primary masks unchanged'),indent=2)+'\n')
    print(json.dumps(dict(common_accounts=len(metrics),cohort_accounts=account_count,common_eligible_min=int(common.sum(1).min()))))


if __name__=='__main__':main()
