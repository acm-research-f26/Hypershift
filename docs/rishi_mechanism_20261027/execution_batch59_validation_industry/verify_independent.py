"""Verify classified-support rank adjustment with QR projection."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch59_validation_industry');support=pd.read_csv(root/'SUPPORT.csv');rows=pd.read_csv(root/'CORRELATIONS.csv')
features=['beta252','momentum126','volatility20','log10adv20'];maximum=0.
for row in rows.itertuples():
    allrows=support[(support.group==row.group)&(support.year==row.year)];f=allrows.dropna(subset=['industry']).sort_values('security_id')
    assert len(f)==row.classified and len(allrows)-len(f)==row.unclassified and not f.security_id.duplicated().any()
    ranks=f[['preference']+features].rank();cats=sorted(f.industry.unique());assert len(cats)==row.industries
    inds=np.column_stack([(f.industry.to_numpy()==c).astype(float) for c in cats[1:]]) if len(cats)>1 else np.empty((len(f),0))
    x=ranks[row.feature].to_numpy();y=ranks.preference.to_numpy();other=ranks[[c for c in features if c!=row.feature]].to_numpy()
    q,_=np.linalg.qr(np.column_stack([np.ones(len(f)),other,inds]));rx=x-q@(q.T@x);ry=y-q@(q.T@y)
    maximum=max(maximum,abs(row.raw-np.corrcoef(x,y)[0,1]),abs(row.adjusted-np.corrcoef(rx,ry)[0,1]))
assert len(rows)==48 and maximum<1e-12
print(f'48 correlations and classified/unclassified supports verified; max error {maximum:.3g}')
