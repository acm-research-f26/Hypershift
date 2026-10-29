"""Independently rebuild saved parent blend ranks, weights, paths and intervals."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pit_research_engine import load_data,fold
from research_engine import target_weights,performance
from rishi_mechanism_batch1 import stationary_indices

RUN=Path('runs/ndx_extended_hours_20261024');SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
LARGE=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028');OUT=Path('docs/rishi_mechanism_20261027/execution_batch42_extended_parent')
SEEDS=[7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941]
GROUPS={'original_five':SEEDS[:5],'new_five':SEEDS[5:],'ten_seed':SEEDS}
VIEWS=('daily','after_hours','after_hours_premarket')


def main():
    data=load_data();ids=np.concatenate([fold(data,y)[2] for y in (2022,2023,2024,2025)]);mask=data['mask'][ids]
    published=np.load(RUN/'scores.npz',allow_pickle=False);saved_scores=np.load(LARGE/'SCORES.npz',allow_pickle=False)
    targets=np.load(LARGE/'TARGETS.npz',allow_pickle=False);paths=np.load(LARGE/'PATHS.npz',allow_pickle=False)
    for z in (published,saved_scores,targets):np.testing.assert_array_equal(z['ids'],ids)
    np.testing.assert_array_equal(targets['security_ids'],data['panel'].tickers)
    lock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())['hashes'];checked=0;max_score=0.;target_count=0;max_nav=0.;max_metric=0.
    metrics=pd.read_csv(OUT/'METRICS.csv');contrasts=pd.read_csv(OUT/'CONTRASTS.csv')
    for view in VIEWS:
        raw={}
        for seed in SEEDS:
            pieces=[]
            for year in (2022,2023,2024,2025):
                path=RUN/view/str(year)/f'{seed}_predictions.npz';key=f'{view}\\{year}\\{seed}_predictions.npz'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=lock[key]:raise ValueError('Parent hash')
                with np.load(path,allow_pickle=False) as z:pieces.append(z['test'].copy())
                checked+=1
            raw[seed]=np.concatenate(pieces)
        for group,seeds in GROUPS.items():
            ranked=[]
            for seed in seeds:
                one=np.zeros(mask.shape,dtype=float)
                for i,eligible in enumerate(mask):one[i,eligible]=rankdata(raw[seed][i,eligible],method='average')/eligible.sum()
                ranked.append(one)
            reconstructed=np.mean(ranked,axis=0)
            max_score=max(max_score,float(np.max(abs(reconstructed-published[view+'__'+group]))))
            d=np.load(SOURCE/f'scores_D_{group}.npz',allow_pickle=False);np.testing.assert_array_equal(d['ids'],ids)
            combined=.5*reconstructed+.5*d['scores'];max_score=max(max_score,float(np.max(abs(combined-saved_scores[view+'__'+group]))))
            for k in (5,10):
                key=f'{view}__{group}__k{k}';weights=targets[key]
                for i in range(len(ids)):
                    np.testing.assert_allclose(target_weights(combined[i],mask[i],k,'equal'),weights[i],rtol=0,atol=1e-12);target_count+=1
                net=paths[key];ledger=pd.read_csv(LARGE/f'{key}_ledger.csv')
                if ledger.date.tolist()!=paths['dates'].tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Ledger feasibility')
                max_nav=max(max_nav,float(np.max(abs(ledger.nav-1e6*np.cumprod(1+net)))/1e6))
                m=performance(net);row=metrics[(metrics.view==view)&(metrics['group']==group)&(metrics.k==k)].iloc[0]
                max_metric=max(max_metric,*[abs(m[c]-row[c]) for c in ('sharpe','annualized_return','maximum_drawdown')])
    max_interval=0.
    for block in (5,21,63):
        draw=stationary_indices(len(paths['dates']),10000,block,27102028)
        for row in contrasts[contrasts.block==block].itertuples():
            a=paths[f'after_hours_premarket__{row.group}__k{row.k}'];b=paths[f'{row.comparator}__{row.group}__k{row.k}']
            diff=a-b;expected=np.r_[252*diff.mean(),np.quantile(252*diff[draw].mean(1),[.025,.975])]
            max_interval=max(max_interval,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    if checked!=120 or target_count!=3618 or len(contrasts)!=36 or max_score>1e-12 or max_nav>1e-8 or max_metric>1e-10 or max_interval>1e-12:raise ValueError('Audit mismatch')
    receipt=dict(parent_files=checked,target_allocations=target_count,accounts=len(metrics),mean_intervals=len(contrasts),
        max_score_error=max_score,max_nav_error=max_nav,max_metric_error=max_metric,max_interval_error=max_interval,
        paths_sha256=hashlib.sha256((LARGE/'PATHS.npz').read_bytes()).hexdigest(),targets_sha256=hashlib.sha256((LARGE/'TARGETS.npz').read_bytes()).hexdigest())
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
