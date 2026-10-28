"""Describe saved model rank and target exposure by dated SIC at fixed origins."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold
from research_engine import target_weights
from rishi_dated_sic import between_fraction


def percentile(score,eligible):
    score=np.asarray(score,float);eligible=np.asarray(eligible,bool)
    if score.shape!=eligible.shape or not eligible.any() or not np.isfinite(score[eligible]).all():raise ValueError('Finite eligible scores required')
    out=np.full(score.shape,np.nan);out[eligible]=pd.Series(score[eligible]).rank(method='average',pct=True).to_numpy();return out


def industry_masses(weights,industry):
    weights=np.asarray(weights,float);industry=np.asarray(industry,float)
    if weights.shape!=industry.shape or not np.isfinite(weights).all():raise ValueError('Aligned finite weights required')
    result={'unclassified':float(weights[~np.isfinite(industry)].sum())}
    for value in np.unique(industry[np.isfinite(industry)]):result[str(int(value))]=float(weights[industry==value].sum())
    return result


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch21_saved_industry'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    early=doc/'execution_batch15_dated_sic/ELIGIBLE_JOIN.csv';late=doc/'execution_batch20_sec_combined/JOINED.csv'
    left=pd.read_csv(early);right=pd.read_csv(late)
    left=left[left.year.isin([2022,2023])][['year','signal_date','security_id','industry']]
    right=right[['year','signal_date','security_id','combined_industry']].rename(columns={'combined_industry':'industry'})
    classification=pd.concat([left,right],ignore_index=True)
    if classification.duplicated(['year','signal_date','security_id']).any():raise ValueError('Duplicate classification keys')
    data=load_data();tickers=np.asarray(data['panel'].tickers);base=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    score_files={name:base/f'scores_{name}_ten_seed.npz' for name in ['B','D','E','Hlabel']}
    scores={name:np.load(path,allow_pickle=False) for name,path in score_files.items()}
    summaries=[];groups=[];mass=[];details=[]
    for year in (2022,2023,2024,2025):
        _,_,test=fold(data,year);origin=int(test[0]);date=str(data['panel'].dates[data['origin'][origin]].date());eligible=data['mask'][origin]
        cls=classification[(classification.year==year)&(classification.signal_date==date)].set_index('security_id')
        if set(cls.index)!=set(tickers[eligible]):raise ValueError('Classification universe/date mismatch')
        industry=cls.industry.reindex(tickers).to_numpy(float);covered=eligible&np.isfinite(industry)
        for name,z in scores.items():
            row=np.flatnonzero(z['ids']==origin)
            if len(row)!=1 or z['scores'].shape[1]!=len(tickers):raise ValueError('Saved score axes mismatch')
            raw=z['scores'][row[0]];rank=percentile(raw,eligible)
            fraction,between,within,total=between_fraction(rank[covered],industry[covered])
            summaries.append(dict(year=year,signal_date=date,model=name,eligible=int(eligible.sum()),covered=int(covered.sum()),
                between_fraction=fraction,between_ss=between,within_ss=within,total_ss=total,covered_mean=float(rank[covered].mean()),
                unclassified_mean=float(rank[eligible&~covered].mean()) if (eligible&~covered).any() else np.nan))
            for val in np.unique(industry[covered]):
                r=rank[covered&(industry==val)];groups.append(dict(year=year,model=name,industry=int(val),ids=len(r),mean_rank=float(r.mean()),sd_rank=float(r.std())))
            for k in (5,10):
                weights=target_weights(raw,eligible,k,'equal')
                for group,value in industry_masses(weights,industry).items():mass.append(dict(year=year,model=name,k=k,industry=group,target_weight=value))
            for j in np.flatnonzero(eligible):details.append(dict(year=year,model=name,security_id=tickers[j],industry=industry[j],score=float(raw[j]),rank=float(rank[j])))
    for z in scores.values():z.close()
    pd.DataFrame(summaries).to_csv(out/'SUMMARY.csv',index=False);pd.DataFrame(groups).to_csv(out/'INDUSTRY_RANKS.csv',index=False)
    pd.DataFrame(mass).to_csv(out/'TARGET_INDUSTRY_MASS.csv',index=False);pd.DataFrame(details).to_csv(out/'DETAIL.csv',index=False)
    paths=[early,late,Path(__file__),doc/'SAVED_RANK_INDUSTRY_PROTOCOL.md',*score_files.values(),Path('research/core/pit_research_engine.py')]
    (out/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limits':'One-origin exploratory saved-score description, not causal or checkpoint training-prediction analysis'},indent=2)+'\n')
    print(pd.DataFrame(summaries)[['year','model','covered','between_fraction']].to_string(index=False))


if __name__=='__main__':main()
