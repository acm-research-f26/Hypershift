"""Independently rebuild fee components and synchronized block intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('results/RISHI_INDUSTRY_COVERAGE_20261028')
out=Path('docs/rishi_mechanism_20261027/execution_batch35_coverage_fee')
saved=np.load(root/'PATHS.npz',allow_pickle=False)
daily=pd.read_csv(out/'DAILY.csv',float_precision='round_trip')
summary=pd.read_csv(out/'ACCOUNTS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv')
max_error=0.
for model in ('E','E_Hlabel'):
    a={}
    for arm in ('original','coverage6'):
        ledger=pd.read_csv(root/f'{model}_{arm}_ledger.csv',float_precision='round_trip')
        prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]]
        net=saved[f'{model}_{arm}'];fee=ledger.fee.to_numpy()/prev
        a[arm]={'net':net,'fee':fee,'gross':net+fee,'turnover':ledger.traded.to_numpy()/prev}
        part=daily[daily.model==model]
        assert part.date.tolist()==ledger.date.tolist()
        for key,value in a[arm].items():
            max_error=max(max_error,float(np.max(abs(value-part[f'{arm}_{key}']))))
            expected=summary[(summary.model==model)&(summary.arm==arm)].iloc[0][key+'_annual']
            max_error=max(max_error,abs(value.mean()*252-expected))
    for block in (5,21,63):
        rng=np.random.default_rng(27102028);n=len(saved['dates']);draw=np.empty((10000,n),dtype=np.int32)
        draw[:,0]=rng.integers(n,size=10000)
        for j in range(1,n):
            reset=rng.random(10000)<1/block;starts=rng.integers(n,size=10000)
            draw[:,j]=np.where(reset,starts,(draw[:,j-1]+1)%n)
        row=contrasts[(contrasts.model==model)&(contrasts.block==block)].iloc[0]
        for key in ('net','fee','gross'):
            delta=a['original'][key]-a['coverage6'][key];samples=252*delta[draw].mean(axis=1)
            values=[delta.mean()*252,*np.quantile(samples,[.025,.975])]
            max_error=max(max_error,float(np.max(np.abs(np.asarray(values)-row[[key+'_difference',key+'_low',key+'_high']].to_numpy(float)))))
assert max_error<1e-12
print('Four ledgers and 18 paired component intervals independently reconstructed; max error',max_error)
