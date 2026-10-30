"""Verify all partial rank correlations with a QR projection."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch58_validation_style_partial');data=pd.read_csv(root/'COMMON_STOCKS.csv');rows=pd.read_csv(root/'CORRELATIONS.csv')
features=['beta252','momentum126','volatility20','log10adv20'];maximum=0.
for row in rows.itertuples():
    f=data[(data.group==row.group)&(data.year==row.year)];assert len(f)==row.pairs and not f.security_id.duplicated().any()
    r=f[['preference']+features].rank(method='average');x=r[row.feature].to_numpy();y=r.preference.to_numpy();controls=r[[c for c in features if c!=row.feature]].to_numpy()
    q,_=np.linalg.qr(np.column_stack([np.ones(len(f)),controls]));rx=x-q@(q.T@x);ry=y-q@(q.T@y)
    maximum=max(maximum,abs(row.raw-np.corrcoef(x,y)[0,1]),abs(row.partial-np.corrcoef(rx,ry)[0,1]))
assert len(rows)==48 and len(data)==1140 and maximum<1e-12
print(f'48 common-support correlations and 1140 stock rows verified; max error {maximum:.3g}')
