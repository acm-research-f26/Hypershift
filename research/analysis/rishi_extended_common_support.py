"""Fixed common-support comparison of saved rank/feature associations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_extended_association import correlation

DOC=Path('docs/rishi_mechanism_20261027')
SOURCE=DOC/'execution_batch30_extended_association'


def common_correlations(coverage,returns):
    merged=coverage.merge(returns,on='security_id',suffixes=('_coverage','_return'),validate='one_to_one')
    def rank(col):return merged[col].rank(method='average').to_numpy()
    sc=rank('score_rank_coverage');sr=rank('score_rank_return')
    if not np.array_equal(sc,sr):raise ValueError('Saved score ordering disagrees on common support')
    return len(merged),correlation(sc,rank('feature_rank_coverage')),correlation(sr,rank('feature_rank_return'))


def main():
    out=DOC/'execution_batch31_extended_common_support'
    out.mkdir(parents=True,exist_ok=False)
    rows=pd.read_csv(SOURCE/'RANKS.csv');prior=pd.read_csv(SOURCE/'PER_ORIGIN.csv')
    keys=['year','origin','model','feature','security_id']
    if rows.duplicated(keys).any() or prior.duplicated(keys[:-1]).any():raise ValueError('Duplicate source rows')
    prior=prior.set_index(keys[:-1]);groups=rows.groupby(keys[:-1],sort=False);results=[]
    for year,origin,model in rows[['year','origin','model']].drop_duplicates().itertuples(index=False,name=None):
        for segment in ('AH','PM'):
            cov=groups.get_group((year,origin,model,segment+'_coverage'))
            ret=groups.get_group((year,origin,model,segment+'_return'))
            n,c,r=common_correlations(cov,ret)
            old=prior.loc[(year,origin,model,segment+'_coverage')]
            results.append(dict(year=year,origin=origin,model=model,segment=segment,common_support=n,
                coverage_support=len(cov),return_support=len(ret),coverage_correlation=c,return_correlation=r,
                correlation_difference=c-r,full_coverage_correlation=old.raw_correlation))
    frame=pd.DataFrame(results)
    if len(frame)!=201*4*2 or frame.duplicated(['year','origin','model','segment']).any():raise ValueError('Incomplete fixed cells')
    frame.to_csv(out/'PER_ORIGIN.csv',index=False)
    summary=frame.groupby(['year','model','segment']).agg(origins=('origin','size'),mean_common=('common_support','mean'),
        coverage_defined=('coverage_correlation','count'),return_defined=('return_correlation','count'),
        coverage_mean=('coverage_correlation','mean'),return_mean=('return_correlation','mean'),
        difference_defined=('correlation_difference','count'),difference_mean=('correlation_difference','mean'),
        difference_median=('correlation_difference','median'),full_coverage_mean=('full_coverage_correlation','mean')).reset_index()
    summary.to_csv(out/'ANNUAL.csv',index=False)
    paths=[SOURCE/'RANKS.csv',SOURCE/'PER_ORIGIN.csv',SOURCE/'RESULTS.json',Path(__file__),
           Path('research/analysis/rishi_extended_association.py'),DOC/'EXTENDED_COMMON_SUPPORT_PROTOCOL.md']
    (out/'RESULTS.json').write_text(json.dumps(dict(cells=len(frame),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitation='Exploratory same-support correlations, no intervention or predictive outcome'),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
