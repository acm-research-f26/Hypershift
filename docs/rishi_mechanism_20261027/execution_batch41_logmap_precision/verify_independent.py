"""Rebuild scalar numerical summaries from saved endpoint arrays."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch41_logmap_precision')
z=np.load(root/'ENDPOINTS.npz',allow_pickle=False);cells=pd.read_csv(root/'CELLS.csv',float_precision='round_trip')
assert len(cells)==int(z['eligible'].sum()) and not cells.duplicated(['origin','security_id']).any()
ids={int(v):i for i,v in enumerate(z['ids'])};names={v:i for i,v in enumerate(z['security_ids'])}
maxerr=0.
for row in cells.itertuples():
    i=ids[row.origin];j=names[row.security_id];assert z['eligible'][i,j]
    cpu=float(z['cpu_endpoint'][i,j]);gpu=float(z['cuda_endpoint'][i,j]);r=max(abs(cpu),abs(gpu))
    def logmap(x):
        y=np.clip(x,-.99999,.99999)
        return .5*(np.log1p(y)-np.log1p(-y))
    values=[cpu,gpu,r,abs(cpu-gpu),abs(logmap(cpu)-logmap(gpu)),abs(z['cpu_base'][i,j]-z['cuda_base'][i,j])]
    actual=[row.cpu_endpoint,row.cuda_endpoint,row.radius,row.raw_delta,row.mapped_delta,row.base_delta]
    maxerr=max(maxerr,float(np.max(abs(np.array(values)-np.array(actual)))))
    assert bool(row.clipped)==bool(r>=.99999)
    if r<.99999:maxerr=max(maxerr,abs(row.unclipped_derivative-1/(1-r*r)))
bands=pd.read_csv(root/'BANDS.csv').set_index('band')
group=cells.groupby('band')
assert bands.cells.sum()==len(cells)
for name,part in group:
    b=bands.loc[name];assert b.cells==len(part)
    for col,field in [('raw_delta','raw_max'),('mapped_delta','mapped_max'),('base_delta','base_max')]:
        maxerr=max(maxerr,abs(part[col].max()-b[field]))
assert maxerr<1e-10
print(len(cells),'eligible cells and all bands independently reconstructed; max error',maxerr)
