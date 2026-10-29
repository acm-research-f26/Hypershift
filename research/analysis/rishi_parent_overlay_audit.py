"""Independently rebuild target groups, timing, metrics and mean intervals."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import target_weights,performance
from ndx_drawdown_engine import ISSUERS
from rishi_mechanism_batch1 import stationary_indices

OUT=Path('docs/rishi_mechanism_20261027/execution_batch44_parent_overlay');ROOT=Path('results/RISHI_PARENT_OVERLAY_20261028')
SCORES=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028/SCORES.npz')


def independent_weights(score,mask,tickers,allocation):
    if allocation=='equal5':return target_weights(score,mask,5,'equal')
    groups={}
    for j in np.flatnonzero(mask):groups.setdefault(ISSUERS.get(tickers[j],tickers[j]),[]).append(j)
    names=sorted(groups);top=np.array([np.max(score[groups[g]]) for g in names])
    budget=target_weights(top,np.ones(len(names),dtype=bool),5,'equal');w=np.zeros(len(score))
    for name,best,value in zip(names,top,budget):
        equal=[j for j in groups[name] if score[j]==best];w[equal]=value/len(equal)
    return w


def main():
    data=load_data();ticks=data['panel'].tickers;scores=np.load(SCORES,allow_pickle=False);targets=np.load(ROOT/'TARGETS.npz',allow_pickle=False)
    paths=np.load(ROOT/'PATHS.npz',allow_pickle=False);ids=scores['ids'];np.testing.assert_array_equal(ids,targets['ids']);np.testing.assert_array_equal(ticks,targets['security_ids'])
    metrics=pd.read_csv(OUT/'METRICS.csv');contrasts=pd.read_csv(OUT/'CONTRASTS.csv');target_count=0;max_nav=max_metric=max_exp=max_interval=0.
    q=pd.read_csv('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv',index_col=0,parse_dates=True)['QQQ'];ma=q.rolling(200,min_periods=200).mean().to_numpy();q=q.to_numpy()
    origins=data['origin'][ids];dates=data['panel'].dates;dates_idx=dates.get_indexer(pd.DatetimeIndex(paths['dates']));returns=data['returns']
    for view in ('after_hours_premarket','daily','after_hours'):
        for group in ('original_five','new_five','ten_seed'):
            score=scores[f'{view}__{group}']
            for allocation in ('equal5','issuer5'):
                w=targets[f'{view}__{group}__{allocation}']
                for i,origin in enumerate(ids):
                    np.testing.assert_allclose(independent_weights(score[i],data['mask'][origin],ticks,allocation),w[i],rtol=0,atol=1e-12);target_count+=1
                for policy in ('vol20','vol25trend'):
                    key=f'{view}__{group}__{allocation}__{policy}';r=paths[key];ledger=pd.read_csv(ROOT/f'{key}_ledger.csv')
                    if ledger.date.tolist()!=paths['dates'].tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Ledger feasibility')
                    max_nav=max(max_nav,float(np.max(abs(ledger.nav-1e6*np.cumprod(1+r)))/1e6))
                    m=performance(r);row=metrics[(metrics.view==view)&(metrics['group']==group)&(metrics.allocation==allocation)&(metrics.policy==policy)].iloc[0]
                    max_metric=max(max_metric,*[abs(m[c]-row[c]) for c in ('sharpe','annualized_return','maximum_drawdown')])
                    calculated=[]
                    for day in dates_idx:
                        i=int(np.searchsorted(origins,day-1,side='right')-1)
                        if i<0:raise ValueError('Missing prior signal')
                        origin=int(origins[i]);active=w[i]>0;hist=np.nan_to_num(returns[origin-19:origin+1,active],nan=0.)@w[i,active]
                        risk=max(float(hist.std(ddof=1)*np.sqrt(252)),.05)
                        exposure=min(1.,(.2 if policy=='vol20' else .25)/risk)
                        if policy=='vol25trend' and q[origin]<ma[origin]:exposure=min(exposure,.5)
                        calculated.append(exposure)
                    max_exp=max(max_exp,float(np.max(abs(np.asarray(calculated)-ledger.target_exposure.to_numpy()))))
    for block in (5,21,63):
        draw=stationary_indices(len(paths['dates']),10000,block,27102028)
        for row in contrasts[contrasts.block==block].itertuples():
            a=paths[f'after_hours_premarket__{row.group}__{row.allocation}__{row.policy}'];b=paths[f'{row.comparator}__{row.group}__{row.allocation}__{row.policy}']
            d=a-b;expected=np.r_[d.mean()*252,np.quantile(d[draw].mean(1)*252,[.025,.975])]
            max_interval=max(max_interval,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    if target_count!=3618 or len(metrics)!=36 or len(contrasts)!=72 or max_nav>1e-8 or max_metric>1e-10 or max_exp>1e-10 or max_interval>1e-12:raise ValueError('Audit failed')
    receipt=dict(targets=target_count,accounts=len(metrics),mean_intervals=len(contrasts),max_nav_error=max_nav,max_metric_error=max_metric,
        max_target_exposure_error=max_exp,max_interval_error=max_interval,paths_sha256=hashlib.sha256((ROOT/'PATHS.npz').read_bytes()).hexdigest(),
        targets_sha256=hashlib.sha256((ROOT/'TARGETS.npz').read_bytes()).hexdigest())
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
