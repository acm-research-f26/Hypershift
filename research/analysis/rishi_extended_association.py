"""Descriptive associations between saved ranks and raw extended-hours inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold

DOC=Path('docs/rishi_mechanism_20261027')
CUBE=Path('data/nasdaq_monthly_pit/extended_hours_features_20261024/features.npz')
SCORES=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
TIMING=Path('runs/ndx_extended_hours_20261024/TIMING.csv')
FEATURES=[('AH_return','after_hours','return'),('AH_coverage','after_hours','coverage'),
          ('PM_return','premarket','return'),('PM_coverage','premarket','coverage')]


def rank_pair(score,feature,eligible):
    score=np.asarray(score,float);feature=np.asarray(feature,float);eligible=np.asarray(eligible,bool)
    if score.shape!=feature.shape or score.shape!=eligible.shape or score.ndim!=1:raise ValueError('Mismatched axes')
    support=eligible & np.isfinite(score) & np.isfinite(feature)
    x=pd.Series(score[support]).rank(method='average',pct=True).to_numpy()
    y=pd.Series(feature[support]).rank(method='average',pct=True).to_numpy()
    return support,x,y


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)!=len(y):raise ValueError('Mismatched vectors')
    if len(x)<10 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    return float(np.corrcoef(x,y)[0,1])


def main():
    out=DOC/'execution_batch30_extended_association'
    out.mkdir(parents=True,exist_ok=False)
    data=load_data();panel=data['panel'];timing=pd.read_csv(TIMING)
    np.testing.assert_array_equal(timing.origin,data['origin']);np.testing.assert_array_equal(timing.entry,data['entry'])
    files={name:SCORES/f'scores_{name}_ten_seed.npz' for name in ('B','D','E','Hlabel')}
    exports={name:np.load(path,allow_pickle=False) for name,path in files.items()}
    cells=[];members=[]
    with np.load(CUBE,allow_pickle=False) as cube:
        np.testing.assert_array_equal(cube['dates'],panel.dates.strftime('%Y-%m-%d'))
        np.testing.assert_array_equal(cube['tickers'],panel.tickers)
        names=cube['feature_names'].tolist()
        arrays={segment:cube[segment] for segment in ('after_hours','premarket')}
        for year in (2022,2023,2024,2025):
            _,_,origins=fold(data,year)
            for origin in origins:
                eligible=data['mask'][origin].astype(bool)
                for name,segment,field in FEATURES:
                    day=data['origin'][origin] if segment=='after_hours' else data['entry'][origin]
                    features=arrays[segment][day,:,names.index(field)]
                    for model,z in exports.items():
                        matches=np.flatnonzero(z['ids']==origin)
                        if len(matches)!=1 or z['scores'].shape[1]!=len(panel.tickers):raise ValueError('Score axis mismatch')
                        score=z['scores'][matches[0]];support,x,y=rank_pair(score,features,eligible)
                        key=dict(year=year,origin=int(origin),model=model,feature=name)
                        cells.append(dict(**key,eligible=int(eligible.sum()),support=int(support.sum()),
                            unique_score=int(np.unique(x).size),unique_feature=int(np.unique(y).size),raw_correlation=correlation(x,y)))
                        for stock,a,b in zip(np.flatnonzero(support),x,y):
                            members.append(dict(**key,security_id=str(panel.tickers[stock]),score_rank=a,feature_rank=b))
    for z in exports.values():z.close()
    member=pd.DataFrame(members);groups=['year','model','feature','security_id']
    for col in ('score_rank','feature_rank'):
        member[col+'_centered']=member[col]-member.groupby(groups)[col].transform('mean')
    origin_keys=['year','origin','model','feature']
    centered=member.groupby(origin_keys,sort=False).apply(lambda x:correlation(x.score_rank_centered,x.feature_rank_centered),include_groups=False).rename('centered_correlation').reset_index()
    frame=pd.DataFrame(cells).merge(centered,on=origin_keys,how='left',validate='one_to_one')
    if len(frame)!=201*4*4 or frame.duplicated(origin_keys).any():raise ValueError('Incomplete fixed cells')
    frame.to_csv(out/'PER_ORIGIN.csv',index=False);member.to_csv(out/'RANKS.csv',index=False)
    summary=frame.groupby(['year','model','feature']).agg(origins=('origin','size'),mean_support=('support','mean'),
        raw_defined=('raw_correlation','count'),raw_mean=('raw_correlation','mean'),raw_median=('raw_correlation','median'),
        centered_defined=('centered_correlation','count'),centered_mean=('centered_correlation','mean'),centered_median=('centered_correlation','median')).reset_index()
    summary.to_csv(out/'ANNUAL.csv',index=False)
    wide=frame.pivot(index=['year','origin','feature'],columns='model',values='raw_correlation')
    wide['D_minus_B']=wide.D-wide.B
    paired=wide.groupby(['year','feature']).D_minus_B.agg(['count','mean','median']).reset_index()
    paired.to_csv(out/'PAIRED.csv',index=False)
    data_root=Path('data/nasdaq_monthly_pit/assembled')
    paths=[CUBE,CUBE.parent/'REVIEW.json',TIMING,*files.values(),Path(__file__),DOC/'EXTENDED_FEATURE_ASSOCIATION_PROTOCOL.md',
           *(data_root/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv')),
           Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    (out/'RESULTS.json').write_text(json.dumps(dict(cells=len(frame),rank_rows=len(member),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Descriptive selected-history associations; no outcome or causal claim'),indent=2)+'\n')
    print(summary.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':main()
