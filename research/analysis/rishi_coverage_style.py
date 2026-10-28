"""Exploratory conditional associations of coverage and saved ranks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data

DOC=Path('docs/rishi_mechanism_20261027')
RANKS=DOC/'execution_batch30_extended_association/RANKS.csv'
INDUSTRY=DOC/'execution_batch21_saved_industry/DETAIL.csv'
VIEWS=('raw','adv','industry','both')


def conditional_correlation(score,coverage,adv,industry,view):
    vectors=[np.asarray(v) for v in (score,coverage,adv,industry)]
    if any(v.ndim!=1 or len(v)!=len(vectors[0]) for v in vectors):raise ValueError('Mismatched vectors')
    if view not in VIEWS:raise ValueError('Unknown view')
    score,coverage,adv,industry=vectors;n=len(score)
    if n<20:return np.nan,'too_few'
    if not all(np.isfinite(a).all() for a in vectors) or (adv<=0).any():raise ValueError('Nonfinite or nonpositive input')
    rank=lambda v:pd.Series(v).rank(method='average').to_numpy()
    x=rank(score);y=rank(coverage)
    cols=[np.ones(n)]
    if view in ('adv','both'):cols.append(rank(np.log(adv)))
    groups=np.unique(industry)
    if view in ('industry','both'):
        if len(groups)<2:return np.nan,'one_group'
        cols.extend((industry==g).astype(float) for g in groups[1:])
    design=np.column_stack(cols)
    if n<=design.shape[1] or np.linalg.matrix_rank(design)!=design.shape[1]:return np.nan,'singular'
    x=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    y=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    norm=float(np.linalg.norm(x)*np.linalg.norm(y))
    if norm<1e-12:return np.nan,'constant_residual'
    return float(x@y/norm),'ok'


def main():
    out=DOC/'execution_batch32_coverage_style';out.mkdir(parents=True,exist_ok=False)
    rank=pd.read_csv(RANKS);rank=rank[rank.feature.isin(['AH_coverage','PM_coverage'])]
    meta=pd.read_csv(INDUSTRY);meta=meta[meta.model=='B'][['year','security_id','industry']]
    if meta.duplicated(['year','security_id']).any():raise ValueError('Duplicate dated industry')
    data=load_data();ticker=np.asarray(data['panel'].tickers);ticker_index={str(t):j for j,t in enumerate(ticker)}
    if len(ticker_index)!=len(ticker):raise ValueError('Duplicate security ID')
    merged=rank.merge(meta,on=['year','security_id'],how='left',validate='many_to_one')
    if not merged.security_id.isin(ticker_index).all():raise ValueError('Unknown rank ID')
    index=merged.security_id.map(ticker_index).to_numpy();origin=merged.origin.to_numpy(dtype=int)
    if ((origin<0)|(origin>=len(data['origin']))).any():raise ValueError('Origin axis')
    merged['adv']=data['adv'][data['origin'][origin],index]
    results=[];keys=['year','origin','model','feature']
    for key,group in merged.groupby(keys,sort=False):
        valid_adv=np.isfinite(group.adv) & (group.adv>0)
        good=group.industry.notna() & valid_adv
        g=group.loc[good];row=dict(zip(keys,key));row.update(source_support=len(group),support=len(g),
            excluded_unknown_industry=int(group.industry.isna().sum()),
            excluded_invalid_adv=int((group.industry.notna() & ~valid_adv).sum()),groups=int(g.industry.nunique()))
        for view in VIEWS:
            row[view],row[view+'_status']=conditional_correlation(g.score_rank,g.feature_rank,g.adv,g.industry,view)
        results.append(row)
    frame=pd.DataFrame(results)
    if len(frame)!=201*4*2 or frame.duplicated(keys).any():raise ValueError('Incomplete fixed cells')
    frame.to_csv(out/'PER_ORIGIN.csv',index=False)
    aggregates={'origins':('origin','size'),'mean_support':('support','mean'),'mean_groups':('groups','mean')}
    for view in VIEWS:
        aggregates.update({view+'_mean':(view,'mean'),view+'_median':(view,'median'),view+'_defined':(view,'count')})
    summary=frame.groupby(['year','model','feature']).agg(**aggregates).reset_index()
    summary.to_csv(out/'ANNUAL.csv',index=False)
    base=Path('data/nasdaq_monthly_pit/assembled')
    paths=[RANKS,INDUSTRY,Path(__file__),DOC/'COVERAGE_STYLE_PROTOCOL.md',Path('research/core/pit_research_engine.py'),
           Path('research/core/research_engine.py'),*(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))]
    (out/'RESULTS.json').write_text(json.dumps(dict(cells=len(frame),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limits='Selected-history same-support rank residuals; noncausal and not financial outcomes'),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
