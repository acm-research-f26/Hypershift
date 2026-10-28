"""Independent training-only selection, target and path audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold
from research_engine import target_weights,performance
from ndx_drawdown_engine import ISSUERS
from rishi_mechanism_batch1 import stationary_indices


def main():
    doc=Path('docs/rishi_mechanism_20261027/execution_batch39_training_exclusion');large=Path('results/RISHI_TRAINING_EXCLUSION_20261028')
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');data=load_data();tickers=np.asarray(data['panel'].tickers)
    issuer=np.array([ISSUERS.get(t,t) for t in tickers]);selection=pd.read_csv(doc/'TRAINING_SELECTION.csv');paths=np.load(large/'PATHS.npz',allow_pickle=False)
    metrics=pd.read_csv(doc/'METRICS.csv');contrasts=pd.read_csv(doc/'CONTRASTS.csv');dates=paths['dates'];ids=None;scores={}
    for name in ('E','E_Hlabel','Hlabel'):
        z=np.load(source/f'scores_{name}_ten_seed.npz',allow_pickle=False)
        if ids is None:ids=z['ids'].copy()
        else:np.testing.assert_array_equal(ids,z['ids'])
        scores[name]=z['scores'].copy()
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy();chosen={}
    for year in (2022,2023,2024,2025):
        train,_,_=fold(data,year);j=int(np.flatnonzero(years==year)[0]);mask=data['mask'][ids[j]]
        labels=data['y'][train];valid=data['labelmask'][train]&np.isfinite(labels);counts=valid.sum(0)
        means=np.divide(np.where(valid,labels,0).sum(0),counts,out=np.zeros(len(tickers),float),where=counts>0)
        representatives={}
        for idx in np.flatnonzero(mask&(counts>=30)):
            name=issuer[idx];value=float(means[idx])
            if name not in representatives or value>representatives[name][0] or (value==representatives[name][0] and tickers[idx]<representatives[name][1]):
                representatives[name]=(value,tickers[idx])
        order=sorted(representatives,key=lambda s:(-representatives[s][0],s))[:3]
        expected=selection[selection.year==year].sort_values('order')
        if order!=expected.issuer.tolist() or [representatives[x][1] for x in order]!=expected.representative.tolist():raise ValueError('Training-only selection mismatch')
        chosen[year]=order
    count=0;max_nav=0.;max_metric=0.;max_boot=0.
    for n in (0,1,3):
        for model in ('E','E_Hlabel'):
            for k in (5,10):
                key=f'{model}_k{k}_exclude{n}';saved=np.load(large/f'{key}_weights.npz',allow_pickle=False)
                np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['security_ids'],tickers)
                for i,origin in enumerate(ids):
                    eligible=data['mask'][origin]&~np.isin(issuer,chosen[years[i]][:n])
                    expected=target_weights(scores[model][i],eligible,k,'equal')
                    np.testing.assert_array_equal(expected,saved['weights'][i]);count+=1
                ledger=pd.read_csv(large/f'{key}_ledger.csv');r=paths[key]
                if ledger.date.tolist()!=dates.tolist():raise ValueError('Dates mismatch')
                max_nav=max(max_nav,float(np.max(abs(ledger.nav.to_numpy()-1e6*np.cumprod(1+r)))/1e6))
                m=performance(r);row=metrics[(metrics.model==model)&(metrics.k==k)&(metrics['exclude']==n)].iloc[0]
                max_metric=max(max_metric,*[abs(m[c]-row[c]) for c in ('sharpe','annualized_return','maximum_drawdown')])
    for block in (5,21,63):
        draws=stationary_indices(len(dates),10000,block,27102028)
        for row in contrasts[contrasts.block==block].itertuples():
            difference=paths[row.left]-paths[row.right];samples=difference[draws].mean(1)*252
            values=np.r_[difference.mean()*252,np.quantile(samples,[.025,.975])]
            max_boot=max(max_boot,float(np.max(abs(values-[row.mean_difference,row.mean_low,row.mean_high]))))
    if count!=201*12 or len(contrasts)!=36 or max_nav>1e-8 or max_metric>1e-10 or max_boot>1e-12:raise ValueError('Audit failed')
    (doc/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(targets=count,accounts=12,intervals=36,max_nav_error=max_nav,max_metric_error=max_metric,max_boot_error=max_boot,
        paths_sha256=hashlib.sha256((large/'PATHS.npz').read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Verified training-label issuer choices,',count,'target allocations, 12 accounts and 36 mean intervals',max_nav,max_metric,max_boot)


if __name__=='__main__':main()
