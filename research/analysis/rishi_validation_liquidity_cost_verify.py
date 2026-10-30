"""Independently check replayed account paths, exposures and paired means."""
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    root=Path('results/RISHI_VALIDATION_LIQUIDITY_COST_20261029')
    out=Path('docs/rishi_mechanism_20261027/execution_batch63_liquidity_cost')
    p=np.load(root/'PATHS.npz',allow_pickle=False)
    t=np.load('results/RISHI_VALIDATION_STYLE_ACCOUNTS_20261029/TARGETS.npz',allow_pickle=False)
    d=load_data(); ids=t['ids']; np.testing.assert_array_equal(t['security_ids'],d['panel'].tickers)
    origins=d['origin'][ids]; days=d['panel'].dates.get_indexer(pd.DatetimeIndex(p['dates']))
    metrics=pd.read_csv(out/'METRICS.csv'); contrasts=pd.read_csv(out/'CONTRASTS.csv'); maximum=0.
    for row in metrics.itertuples():
        key=f'{row.kind}__{row.arm}__c{row.cost}'; r=p[key]; ledger=pd.read_csv(root/f'{key}_ledger.csv'); w=t[f'ten_seed__{row.kind}__{row.arm}']
        assert ledger.date.tolist()==p['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
        maximum=max(maximum,float(np.max(abs(ledger.nav/1e6-np.cumprod(1+r)))))
        m=performance(r); maximum=max(maximum,*[abs(m[k]-getattr(row,k)) for k in ('sharpe','annualized_return','maximum_drawdown')])
        exposure=[]
        for day in days:
            j=int(np.searchsorted(origins,day-1,side='right')-1); assert j>=0
            if row.kind=='equal10': exposure.append(1.); continue
            origin=origins[j]; active=w[j]>0; hist=np.nan_to_num(d['returns'][origin-19:origin+1,active],nan=0.)@w[j,active]
            exposure.append(min(1.,.2/max(float(hist.std(ddof=1)*np.sqrt(252)),.05)))
        maximum=max(maximum,float(np.max(abs(np.asarray(exposure)-ledger.target_exposure))))
    for block in (5,21,63):
        draw=stationary_indices(len(days),10000,block,29102027)
        for row in contrasts[contrasts.block==block].itertuples():
            delta=p[f'{row.kind}__fixed__c{row.cost}']-p[f'{row.kind}__liquidity__c{row.cost}']
            expected=np.r_[252*delta.mean(),np.quantile(252*delta[draw].mean(1),[.025,.975])]
            maximum=max(maximum,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    assert len(metrics)==16 and len(contrasts)==18 and maximum<1e-10
    print(f'16 accounts and exposures, 18 paired mean intervals verified; max error {maximum:.3g}')


if __name__=='__main__': main()
