"""Retrospective SEC SIC feasibility and training-derived preference description."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold

_spec=importlib.util.spec_from_file_location('dated_sic_edgar',Path('src/hypershift/data/edgar.py'))
_edgar=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_edgar)
ff12=_edgar.ff12


def match_sic(ticker,day,mapping,sic):
    segments=mapping.get(ticker,{}).get('segments',[])
    covering=[int(c) for start,end,c in segments if start<=day<=end]
    if not covering:return ('no_covering_segment',None,None,None)
    if len(covering)!=1:return ('ambiguous_segments',None,None,None)
    cik=covering[0];record=sic.get(str(cik))
    if not record or not record[0]:return ('missing_sic',cik,None,None)
    code,date=record
    if date>=day:return ('not_yet_filed',cik,int(code),date)
    return ('matched',cik,int(code),date)


def between_fraction(values,groups):
    values=np.asarray(values,float);groups=np.asarray(groups)
    if not len(values) or len(values)!=len(groups) or not np.isfinite(values).all():raise ValueError('Finite aligned values required')
    mean=values.mean();total=float(np.sum((values-mean)**2));between=0.;within=0.
    for group in np.unique(groups):
        v=values[groups==group];between+=len(v)*(v.mean()-mean)**2;within+=np.sum((v-v.mean())**2)
    return (between/total if total>0 else np.nan,float(between),float(within),total)


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch15_dated_sic'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    mapping_path=Path('data/edgar/cik_map.json');sic_path=Path('data/edgar/sic_hist.json')
    score_path=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_Hlabel_ten_seed.npz')
    mapping=json.loads(mapping_path.read_text());sic=json.loads(sic_path.read_text());data=load_data()
    z=np.load(score_path,allow_pickle=False);score_ids=z['ids'];scores=z['scores']
    rows=[];summary=[];group_rows=[]
    for year in (2022,2023,2024,2025):
        _,_,test=fold(data,year);origin=int(test[0]);day=str(data['panel'].dates[data['origin'][origin]].date())
        found=np.flatnonzero(score_ids==origin)
        if len(found)!=1:raise ValueError('Expected unique saved origin')
        for j,ticker in enumerate(data['panel'].tickers):
            if not data['mask'][origin,j]:continue
            status,cik,code,filed=match_sic(str(ticker),day,mapping,sic)
            score=float(scores[found[0],j])
            if not np.isfinite(score):raise ValueError('Missing eligible saved Hlabel score')
            rows.append(dict(year=year,signal_date=day,security_id=ticker,status=status,cik=cik,sic=code,filing_date=filed,
                industry=ff12(code) if status=='matched' else None,hlabel_percentile=score))
        frame=pd.DataFrame([r for r in rows if r['year']==year]);covered=frame[frame.status=='matched']
        summary.append(dict(year=year,signal_date=day,eligible=len(frame),matched=len(covered),
            no_covering_segment=int((frame.status=='no_covering_segment').sum()),
            ambiguous_segments=int((frame.status=='ambiguous_segments').sum()),missing_sic=int((frame.status=='missing_sic').sum()),
            not_yet_filed=int((frame.status=='not_yet_filed').sum()),
            between_fraction=between_fraction(covered.hlabel_percentile.to_numpy(),covered.industry.to_numpy())[0] if len(covered) else np.nan))
        for industry,g in covered.groupby('industry'):
            group_rows.append(dict(year=year,industry=int(industry),ids=len(g),mean_hlabel_percentile=g.hlabel_percentile.mean(),
                sd_hlabel_percentile=g.hlabel_percentile.std(ddof=0)))
    pd.DataFrame(rows).to_csv(out/'ELIGIBLE_JOIN.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'COVERAGE.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'INDUSTRY_PREFERENCE.csv',index=False)
    paths=[mapping_path,sic_path,score_path,Path(__file__),doc/'DATED_SIC_PROTOCOL.md',Path('src/hypershift/data/edgar.py')]
    (out/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limits':'Retrospective S&P-universe entity segments, not independent point-in-time NDX identity; SIC can become stale'},indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
