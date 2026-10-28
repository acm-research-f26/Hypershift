"""Consecutive saved score and extended-hours coverage rank changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data

DOC=Path('docs/rishi_mechanism_20261027')
RANKS=DOC/'execution_batch30_extended_association/RANKS.csv'


def change_correlations(score,coverage,liquidity):
    arrays=[np.asarray(v,dtype=float) for v in (score,coverage,liquidity)]
    if any(v.ndim!=1 or len(v)!=len(arrays[0]) for v in arrays):raise ValueError('Mismatched vectors')
    if not all(np.isfinite(v).all() for v in arrays):raise ValueError('Nonfinite changes')
    if len(arrays[0])<20:return np.nan,np.nan,'too_few'
    x,y,z=[pd.Series(v).rank(method='average').to_numpy() for v in arrays]
    x=x-x.mean();y=y-y.mean();z=z-z.mean()
    xy=np.linalg.norm(x)*np.linalg.norm(y)
    if xy<1e-12:return np.nan,np.nan,'constant_score_or_coverage'
    raw=float(x@y/xy);zz=float(z@z)
    if zz<1e-12:return raw,np.nan,'constant_liquidity'
    rx=x-z*(x@z/zz);ry=y-z*(y@z/zz);denom=np.linalg.norm(rx)*np.linalg.norm(ry)
    if denom<1e-12:return raw,np.nan,'constant_residual'
    return raw,float(rx@ry/denom),'ok'


def main():
    out=DOC/'execution_batch33_coverage_changes';out.mkdir(parents=True,exist_ok=False)
    data=load_data();ticker=np.asarray(data['panel'].tickers);lookup={str(t):j for j,t in enumerate(ticker)}
    if len(lookup)!=len(ticker):raise ValueError('Duplicate tickers')
    rank=pd.read_csv(RANKS,float_precision='round_trip')
    rank=rank[rank.feature.isin(['AH_coverage','PM_coverage'])].copy()
    keys=['year','origin','model','feature','security_id']
    if rank.duplicated(keys).any() or not rank.security_id.isin(lookup).all():raise ValueError('Invalid source axes')
    idx=rank.security_id.map(lookup).to_numpy();origins=rank.origin.to_numpy(dtype=int)
    if ((origins<0)|(origins>=len(data['origin']))).any():raise ValueError('Invalid origin')
    rank['adv']=data['adv'][data['origin'][origins],idx]
    cells=[];pairs=[]
    for (year,model,feature),group in rank.groupby(['year','model','feature'],sort=False):
        ids=sorted(group.origin.unique())
        for old,new in zip(ids[:-1],ids[1:]):
            left=group[group.origin==old];right=group[group.origin==new]
            joined=left.merge(right,on='security_id',suffixes=('_old','_new'),validate='one_to_one')
            valid=np.isfinite(joined.adv_old)&np.isfinite(joined.adv_new)&(joined.adv_old>0)&(joined.adv_new>0)
            j=joined.loc[valid].copy()
            ds=j.score_rank_new.to_numpy()-j.score_rank_old.to_numpy()
            dc=j.feature_rank_new.to_numpy()-j.feature_rank_old.to_numpy()
            da=np.log(j.adv_new.to_numpy())-np.log(j.adv_old.to_numpy())
            raw,partial,status=change_correlations(ds,dc,da)
            cells.append(dict(year=year,model=model,feature=feature,old_origin=old,origin=new,
                lag_sessions=int(data['origin'][new]-data['origin'][old]),old_support=len(left),new_support=len(right),
                common_ids=len(joined),invalid_adv=int((~valid).sum()),support=len(j),raw=raw,adjusted=partial,status=status))
            pairs.append(pd.DataFrame(dict(year=year,model=model,feature=feature,old_origin=old,origin=new,
                security_id=j.security_id,score_change=ds,coverage_change=dc,log_adv_change=da)))
    frame=pd.DataFrame(cells);detail=pd.concat(pairs,ignore_index=True)
    if len(frame)!=197*4*2 or frame.duplicated(['year','model','feature','origin']).any():raise ValueError('Incomplete fixed cells')
    frame.to_csv(out/'PER_PAIR.csv',index=False);detail.to_csv(out/'PAIRED_STOCKS.csv',index=False)
    summary=frame.groupby(['year','model','feature']).agg(pairs=('origin','size'),mean_support=('support','mean'),
        raw_mean=('raw','mean'),raw_median=('raw','median'),raw_defined=('raw','count'),
        adjusted_mean=('adjusted','mean'),adjusted_median=('adjusted','median'),adjusted_defined=('adjusted','count')).reset_index()
    summary.to_csv(out/'ANNUAL.csv',index=False)
    base=Path('data/nasdaq_monthly_pit/assembled')
    paths=[RANKS,Path(__file__),DOC/'COVERAGE_RANK_CHANGE_PROTOCOL.md',Path('research/core/pit_research_engine.py'),
           Path('research/core/research_engine.py'),*(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))]
    (out/'RESULTS.json').write_text(json.dumps(dict(cells=len(frame),paired_stock_rows=len(detail),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limits='Descriptive adjacent-origin rank changes; no financial outcomes or trained intervention'),indent=2)+'\n')
    print(summary.to_string(index=False));print('statuses:',frame.status.value_counts().to_dict())


if __name__=='__main__':main()
