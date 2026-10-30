"""Rebuild partial correlations from raw ranks, liquidity CSV and SIC rows."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

doc=Path('docs/rishi_mechanism_20261027')
rank=pd.read_csv(doc/'execution_batch30_extended_association/RANKS.csv')
rank=rank[rank.feature.isin(['AH_coverage','PM_coverage'])]
meta=pd.read_csv(doc/'execution_batch21_saved_industry/DETAIL.csv')
meta=meta[meta.model=='B'][['year','security_id','industry']]
rank=rank.merge(meta,on=['year','security_id'],how='left',validate='many_to_one')
timing=pd.read_csv('runs/ndx_extended_hours_20261024/TIMING.csv')
dv=pd.read_csv('data/nasdaq_monthly_pit/assembled/dollar_volume.csv',index_col=0,parse_dates=True).rolling(20,min_periods=20).mean()
date=pd.to_datetime(timing.prior_regular_close.iloc[rank.origin.to_numpy()].to_numpy())
positions=dv.index.get_indexer(date);stocks=dv.columns.get_indexer(rank.security_id)
assert (positions>=0).all() and (stocks>=0).all()
rank['adv']=dv.to_numpy()[positions,stocks]
actual=pd.read_csv(doc/'execution_batch32_coverage_style/PER_ORIGIN.csv').set_index(['year','origin','model','feature'])
maximum=0.;cases=0
for key,g in rank.groupby(['year','origin','model','feature'],sort=False):
    g=g[g.industry.notna() & np.isfinite(g.adv) & (g.adv>0)]
    row=actual.loc[key];assert len(g)==row.support
    x=rankdata(g.score_rank.to_numpy());y=rankdata(g.feature_rank.to_numpy());a=rankdata(np.log(g.adv.to_numpy()))
    categories=np.unique(g.industry);n=len(g)
    for view in ('raw','adv','industry','both'):
        cols=[np.ones(n)]
        if view in ('adv','both'):cols.append(a)
        if view in ('industry','both'):cols.extend((g.industry.to_numpy()==c).astype(float) for c in categories[1:])
        design=np.column_stack(cols);q,_=np.linalg.qr(design,mode='reduced')
        xr=x-q@(q.T@x);yr=y-q@(q.T@y);computed=float(xr@yr/(np.linalg.norm(xr)*np.linalg.norm(yr)))
        maximum=max(maximum,abs(computed-row[view]));cases+=1
assert cases==1608*4 and maximum<1e-10
print('1608 source/date/ID joins; 6432 QR residual correlations; max error',maximum)
