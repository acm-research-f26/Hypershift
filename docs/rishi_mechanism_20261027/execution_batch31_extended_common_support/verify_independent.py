"""Reconstruct common-support ranks using pandas' independent Spearman path."""
from pathlib import Path
import numpy as np
import pandas as pd

doc=Path('docs/rishi_mechanism_20261027')
original=pd.read_csv(doc/'execution_batch30_extended_association/RANKS.csv')
result=pd.read_csv(doc/'execution_batch31_extended_common_support/PER_ORIGIN.csv')
keys=['year','origin','model','security_id']
assert len(result)==1608 and not result.duplicated(['year','origin','model','segment']).any()
max_error=0.;matched=0
for segment in ('AH','PM'):
    c=original[original.feature==segment+'_coverage'][keys+['score_rank','feature_rank']]
    r=original[original.feature==segment+'_return'][keys+['score_rank','feature_rank']]
    data=c.merge(r,on=keys,suffixes=('_c','_r'),validate='one_to_one')
    for (year,origin,model),g in data.groupby(keys[:-1],sort=False):
        row=result[(result.year==year)&(result.origin==origin)&(result.model==model)&(result.segment==segment)]
        assert len(row)==1 and len(g)==int(row.common_support.iloc[0])
        cval=g.score_rank_c.corr(g.feature_rank_c,method='spearman') if len(g)>=10 else np.nan
        rval=g.score_rank_r.corr(g.feature_rank_r,method='spearman') if len(g)>=10 else np.nan
        for actual,expected in [(cval,row.coverage_correlation.iloc[0]),(rval,row.return_correlation.iloc[0])]:
            assert bool(pd.isna(actual))==bool(pd.isna(expected))
            if pd.notna(actual):max_error=max(max_error,abs(float(actual-expected)))
        matched+=1
assert matched==1608 and max_error<1e-12
print('1608 cells independently reconstructed; max correlation difference',max_error)
