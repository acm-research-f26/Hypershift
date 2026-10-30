"""Verify cost replay ledgers, NAV, metrics, exposure and paired mean intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    root=Path('results/RISHI_VALIDATION_FIXED_COST_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch51_validation_cost')
    paths=np.load(root/'PATHS.npz',allow_pickle=False);targets=np.load('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029/TARGETS.npz',allow_pickle=False)
    metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv');d=load_data();origins=d['origin'][targets['ids']]
    days=d['panel'].dates.get_indexer(pd.DatetimeIndex(paths['dates']));maximum=0.;count=0
    for row in metrics.itertuples():
        key=f'{row.group}__{row.arm}__{row.policy}__c{row.cost}';r=paths[key];ledger=pd.read_csv(root/f'{key}_ledger.csv');w=targets[f'{row.group}__{row.arm}']
        assert ledger.date.tolist()==paths['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
        maximum=max(maximum,float(np.max(abs(ledger.nav/1e6-np.cumprod(1+r)))))
        m=performance(r);maximum=max(maximum,*[abs(m[k]-getattr(row,k)) for k in ('sharpe','annualized_return','maximum_drawdown')])
        e=[]
        for day in days:
            j=int(np.searchsorted(origins,day-1,side='right')-1);assert j>=0
            if row.policy=='full':e.append(1.);continue
            origin=origins[j];active=w[j]>0;x=np.nan_to_num(d['returns'][origin-19:origin+1,active],nan=0.)@w[j,active]
            e.append(min(1.,.2/max(float(x.std(ddof=1)*np.sqrt(252)),.05)))
        maximum=max(maximum,float(np.max(abs(np.asarray(e)-ledger.target_exposure))));count+=1
    for block in (5,21,63):
        draw=stationary_indices(len(days),10000,block,27102031)
        for row in contrasts[contrasts.block==block].itertuples():
            delta=paths[f'{row.group}__original__{row.policy}__c{row.cost}']-paths[f'{row.group}__fixed__{row.policy}__c{row.cost}']
            expected=np.r_[252*delta.mean(),np.quantile(252*delta[draw].mean(1),[.025,.975])]
            maximum=max(maximum,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    assert count==48 and len(contrasts)==54 and maximum<1e-10
    print(f'{count} accounts, exposure schedules and 54 mean intervals verified; maximum error {maximum:.3g}')


if __name__=='__main__':main()
