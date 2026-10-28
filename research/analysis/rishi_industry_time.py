"""Frozen first-origin SIC categories across all saved forecast origins."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold
from research_engine import target_weights
from rishi_dated_sic import between_fraction
from rishi_saved_industry import percentile,industry_masses


def frozen_labels(tickers,known):
    return np.array([known.get(str(t),np.nan) for t in tickers],float)


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch22_industry_time'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    detail=doc/'execution_batch21_saved_industry/DETAIL.csv';first=doc/'execution_batch21_saved_industry/SUMMARY.csv'
    f=pd.read_csv(detail,float_precision='round_trip');f=f[f.model=='E'];prior=pd.read_csv(first)
    if f.duplicated(['year','security_id']).any():raise ValueError('Duplicate first-origin classification')
    data=load_data();tickers=np.asarray(data['panel'].tickers);base=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    files={name:base/f'scores_{name}_ten_seed.npz' for name in ('B','D','E','Hlabel')}
    exports={name:np.load(path,allow_pickle=False) for name,path in files.items()};rows=[];masses=[]
    for year in (2022,2023,2024,2025):
        _,_,origins=fold(data,year);known=f[f.year==year].set_index('security_id').industry.to_dict();industry=frozen_labels(tickers,known)
        for t,origin in enumerate(origins):
            eligible=data['mask'][origin];covered=eligible&np.isfinite(industry);date=str(data['panel'].dates[data['origin'][origin]].date())
            for model,z in exports.items():
                found=np.flatnonzero(z['ids']==origin)
                if len(found)!=1 or z['scores'].shape[1]!=len(tickers):raise ValueError('Score axes mismatch')
                raw=z['scores'][found[0]];rank=percentile(raw,eligible)
                fraction,_,_,_=between_fraction(rank[covered],industry[covered])
                if t==0:
                    reference=prior[(prior.year==year)&(prior.model==model)]
                    if len(reference)!=1 or not np.isclose(fraction,reference.between_fraction.iloc[0],atol=1e-12):raise ValueError('First-origin prior mismatch')
                rows.append(dict(year=year,origin=int(origin),signal_date=date,model=model,eligible=int(eligible.sum()),covered=int(covered.sum()),between_fraction=fraction))
                for k in (5,10):
                    w=target_weights(raw,eligible,k,'equal');partition=industry_masses(w,industry)
                    if not np.isclose(sum(partition.values()),1,atol=1e-12):raise ValueError('Target mass mismatch')
                    for industry_id,value in partition.items():masses.append(dict(year=year,origin=int(origin),model=model,k=k,industry=industry_id,target_weight=value))
    for z in exports.values():z.close()
    frame=pd.DataFrame(rows);target=pd.DataFrame(masses)
    if len(frame)!=804 or frame.duplicated(['year','origin','model']).any():raise ValueError('Incomplete fixed cells')
    frame.to_csv(out/'PER_ORIGIN.csv',index=False);target.to_csv(out/'TARGET_INDUSTRY_MASS.csv',index=False)
    annual=frame.groupby(['year','model']).agg(origins=('origin','size'),mean_between=('between_fraction','mean'),median_between=('between_fraction','median'),mean_covered=('covered','mean')).reset_index()
    annual.to_csv(out/'ANNUAL_RANKS.csv',index=False)
    wide=frame.pivot(index=['year','origin'],columns='model',values='between_fraction');wide['difference']=wide.D-wide.B
    paired=wide.groupby(level='year').agg(mean_difference=('difference','mean'),fraction_dynamic_greater=('difference',lambda x:float((x>0).mean())),origins=('difference','size')).reset_index()
    paired.to_csv(out/'PAIRED_DYNAMIC_PARENT.csv',index=False)
    target.groupby(['year','model','k','industry']).target_weight.mean().reset_index().to_csv(out/'ANNUAL_TARGET_MASS.csv',index=False)
    paths=[detail,first,Path(__file__),doc/'SAVED_INDUSTRY_TIME_PROTOCOL.md',*files.values(),Path('research/analysis/rishi_saved_industry.py')]
    (out/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'rows':len(frame),
        'limits':'First-origin SIC frozen for the year, new IDs unclassified; saved-score description only'},indent=2)+'\n')
    print(annual.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':main()
