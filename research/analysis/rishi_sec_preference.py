"""Descriptive Hlabel ranks by pre-audited dated SEC cover/header SIC."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_dated_sic import ff12,between_fraction

KEYS=['year','signal_date','security_id']


def checked_join(scores,receipts):
    if scores.duplicated(KEYS).any() or receipts.duplicated(KEYS).any():raise ValueError('Duplicate universe keys')
    out=scores.merge(receipts,on=KEYS,how='outer',indicator=True,validate='one_to_one')
    if not out['_merge'].eq('both').all():raise ValueError('Universe keys differ')
    return out.drop(columns='_merge')


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch17_sec_preference'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=doc/'execution_batch15_dated_sic/ELIGIBLE_JOIN.csv';receipt=doc/'execution_batch16_sec_cover/RECEIPTS.csv'
    audit_path=doc/'execution_batch16_sec_cover/AUDIT.json';support=doc/'execution_batch2_exports/TRAINING_SUPPORT.csv'
    audit=json.loads(audit_path.read_text())
    if audit['receipt_count']!=202 or audit['matched_validated']!=160 or audit['issues']:raise ValueError('SEC receipt audit not passed')
    for path,digest in audit['input_sha256'].items():
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('SEC audit input changed: '+str(path))
    scores=pd.read_csv(source);scores=scores[scores.year.isin([2024,2025])][KEYS+['hlabel_percentile']]
    receipts=pd.read_csv(receipt)[KEYS+['status','cik','accession','filing_date','sic']]
    joined=checked_join(scores,receipts)
    counts=pd.read_csv(support);counts=counts[counts.family=='Hlabel'][['year','ticker','count']]
    joined=joined.merge(counts,left_on=['year','security_id'],right_on=['year','ticker'],how='left',validate='one_to_one')
    if joined['count'].isna().any() or not np.isfinite(joined.hlabel_percentile).all():raise ValueError('Missing original preference or support')
    joined['covered']=joined.status.eq('matched')
    if joined.loc[joined.covered,'sic'].isna().any():raise ValueError('Matched without SIC')
    joined['industry']=joined.sic.where(joined.covered).map(lambda x:ff12(int(x)) if pd.notna(x) else np.nan)
    summary=[];groups=[];coverage=[]
    for year,part in joined.groupby('year'):
        covered=part[part.covered]
        frac=between_fraction(covered.hlabel_percentile.to_numpy(),covered.industry.to_numpy())[0] if len(covered) else np.nan
        summary.append(dict(year=year,eligible=len(part),covered=len(covered),between_industry_fraction=frac))
        for kind,g in part.groupby('covered'):
            coverage.append(dict(year=year,covered=bool(kind),ids=len(g),mean_rank=g.hlabel_percentile.mean(),
                median_rank=g.hlabel_percentile.median(),under30=int((g['count']<30).sum())))
        for industry,g in covered.groupby('industry'):
            groups.append(dict(year=year,industry=int(industry),ids=len(g),mean_rank=g.hlabel_percentile.mean(),sd_rank=g.hlabel_percentile.std(ddof=0)))
    joined.to_csv(out/'JOINED.csv',index=False);pd.DataFrame(summary).to_csv(out/'SUMMARY.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'COVERAGE_PREFERENCE.csv',index=False);pd.DataFrame(groups).to_csv(out/'INDUSTRY_PREFERENCE.csv',index=False)
    paths=[source,receipt,audit_path,support,Path(__file__),doc/'SEC_PREFERENCE_PROTOCOL.md',Path('research/analysis/rishi_dated_sic.py'),Path('src/hypershift/data/edgar.py')]
    (out/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope':'Descriptive Hlabel ranks only; no neural causation, financial scoring or point-in-time NDX identity certification'},indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(coverage).to_string(index=False))


if __name__=='__main__':main()
