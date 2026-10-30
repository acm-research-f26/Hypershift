"""Rebuild traced daily sums and issuer/stock paired accounting."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('results/RISHI_INDUSTRY_COVERAGE_TRACE_20261028');source=Path('results/RISHI_INDUSTRY_COVERAGE_20261028')
out=Path('docs/rishi_mechanism_20261027/execution_batch37_coverage_trace')
paths=np.load(source/'PATHS.npz',allow_pickle=False);stocks=pd.read_csv(out/'STOCKS.csv')
paired=pd.read_csv(out/'PAIRED_STOCKS.csv');issuers=pd.read_csv(out/'PAIRED_ISSUERS.csv')
maxerr=0.
for model in ('E','E_Hlabel'):
    allnet={}
    for arm in ('original','coverage6'):
        z=np.load(root/f'{model}_{arm}.npz',allow_pickle=False);ledger=pd.read_csv(source/f'{model}_{arm}_ledger.csv')
        assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
        net=z['net'];gross=z['gross'];fees=z['fees'];allnet[arm]=net
        for value in (np.max(abs(net.sum(1)-paths[f'{model}_{arm}'])),np.max(abs(gross-fees-net))):maxerr=max(maxerr,float(value))
        prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]]
        maxerr=max(maxerr,float(np.max(abs(fees.sum(1)-ledger.fee.to_numpy()/prev))))
        table=stocks[(stocks.model==model)&(stocks.arm==arm)].set_index('security_id').loc[z['security_ids'].tolist()]
        for column,arr in [('annual_net',net),('annual_gross',gross),('annual_fee',fees),('mean_exposure',z['exposure'])]:
            expected=arr.mean(0)*(1 if column=='mean_exposure' else 252)
            maxerr=max(maxerr,float(np.max(abs(table[column].to_numpy()-expected))))
    diff=(allnet['original']-allnet['coverage6']).mean(0)*252
    table=paired[paired.model==model].set_index('security_id').loc[z['security_ids'].tolist()]
    maxerr=max(maxerr,float(np.max(abs(table.annual_difference.to_numpy()-diff))))
    expected=table.groupby('issuer').annual_difference.sum();actual=issuers[issuers.model==model].set_index('issuer').annual_difference
    maxerr=max(maxerr,float(np.max(abs(expected.sort_index()-actual.sort_index()))))
assert maxerr<1e-10
print('Four daily paths and all stock/issuer summaries reconstructed; max error',maxerr)
