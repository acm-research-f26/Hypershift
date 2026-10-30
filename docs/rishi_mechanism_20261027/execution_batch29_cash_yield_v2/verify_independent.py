"""Independent closed-form check of accumulated interest sidecars."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch29_cash_yield_v2')
daily=pd.read_csv(root/'DAILY_LEDGER.csv')
summary=pd.read_csv(root/'SUMMARY.csv').set_index('id')
assert len(daily)==24*1003 and daily.id.nunique()==24 and len(summary)==24
max_balance=0.;max_terminal=0.;max_wealth=0.
for key,group in daily.groupby('id',sort=False):
    rate=group.rf.to_numpy().copy();rate[0]=0.
    growth=np.cumprod(1+rate)
    prior_cash=np.r_[0.,group.source_cash.to_numpy()[:-1]]
    reconstructed=growth*np.cumsum(rate*prior_cash/growth)
    max_balance=max(max_balance,float(np.max(np.abs(reconstructed-group.sidecar.to_numpy()))))
    max_terminal=max(max_terminal,abs(float(reconstructed[-1]-summary.loc[key,'interest_total'])))
    max_wealth=max(max_wealth,float(np.max(np.abs(group.augmented_wealth-group.source_nav-group.sidecar))))
assert max_balance<1e-7 and max_terminal<1e-7 and max_wealth<1e-8
print('24 accounts, 24072 rows; max sidecar error',max_balance,'terminal error',max_terminal,'wealth identity error',max_wealth)
