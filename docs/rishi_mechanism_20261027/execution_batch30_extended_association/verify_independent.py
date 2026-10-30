"""Rebuild all origin-level correlations directly from published rank rows."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch30_extended_association')
r=pd.read_csv(root/'RANKS.csv');cells=pd.read_csv(root/'PER_ORIGIN.csv')
keys=['year','origin','model','feature'];security=['year','model','feature','security_id']
assert len(cells)==3216 and not cells.duplicated(keys).any()
assert not r.duplicated(keys+['security_id']).any()
for c in ('score_rank','feature_rank'):
    expected=r[c]-r.groupby(security)[c].transform('mean')
    assert np.allclose(expected,r[c+'_centered'],atol=1e-12,rtol=0)

def corr(a,b):
    if len(a)<10 or a.nunique()<2 or b.nunique()<2:return np.nan
    return a.corr(b)

rows=[]
for key,g in r.groupby(keys,sort=False):
    rows.append((*key,len(g),corr(g.score_rank,g.feature_rank),corr(g.score_rank_centered,g.feature_rank_centered)))
x=pd.DataFrame(rows,columns=keys+['count','raw','centered']).merge(cells,on=keys,validate='one_to_one')
assert len(x)==len(cells) and (x['count']==x.support).all()
raw=np.nanmax(np.abs(x.raw-x.raw_correlation));center=np.nanmax(np.abs(x.centered-x.centered_correlation))
assert np.array_equal(x.raw.isna(),x.raw_correlation.isna())
assert np.array_equal(x.centered.isna(),x.centered_correlation.isna())
assert raw<1e-12 and center<1e-12
print('3216 origin cells;',len(r),'rank rows; maximum raw/centered error:',raw,center)
