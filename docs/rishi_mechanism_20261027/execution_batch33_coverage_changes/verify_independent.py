"""Recompute pair correlations from saved changes with QR projections."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

root=Path('docs/rishi_mechanism_20261027/execution_batch33_coverage_changes')
details=pd.read_csv(root/'PAIRED_STOCKS.csv',float_precision='round_trip')
cells=pd.read_csv(root/'PER_PAIR.csv').set_index(['year','model','feature','old_origin','origin'])
keys=['year','model','feature','old_origin','origin']
assert len(cells)==1576 and not cells.index.has_duplicates
max_error=0.;groups=0
for key,g in details.groupby(keys,sort=False):
    row=cells.loc[key];assert len(g)==row.support and not g.security_id.duplicated().any()
    x,y,z=[rankdata(g[c].to_numpy()) for c in ('score_change','coverage_change','log_adv_change')]
    x=x-x.mean();y=y-y.mean();z=z-z.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    if len(g)<20 or den<1e-12:
        assert pd.isna(row.raw) and pd.isna(row.adjusted)
    else:
        raw=float(x@y/den);max_error=max(max_error,abs(raw-row.raw))
        design=np.column_stack([np.ones(len(g)),z]);q,_=np.linalg.qr(design,mode='reduced')
        rx=x-q@(q.T@x);ry=y-q@(q.T@y);adjusted=float(rx@ry/(np.linalg.norm(rx)*np.linalg.norm(ry)))
        max_error=max(max_error,abs(adjusted-row.adjusted))
    groups+=1
assert groups==1576 and max_error<1e-10
print('1576 pair cells;',len(details),'paired stock rows; max correlation error',max_error)
