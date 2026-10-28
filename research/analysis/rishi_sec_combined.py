"""Frozen coverage-sensitivity description; no financial scoring."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_dated_sic import ff12,between_fraction

KEYS=['year','signal_date','security_id']


def combine(base,extra):
    if base.duplicated(KEYS).any() or extra.duplicated(KEYS).any():raise ValueError('Duplicate keys')
    expected=set(map(tuple,base.loc[base.status!='matched',KEYS].to_numpy()))
    received=set(map(tuple,extra[KEYS].to_numpy()))
    if expected!=received:raise ValueError('Supplement must match all and only originally unmatched keys')
    z=base.merge(extra[KEYS+['status','sic']],on=KEYS,how='left',suffixes=('_original','_supplement'),validate='one_to_one')
    original=z.status_original.eq('matched');supplement=z.status_supplement.eq('matched')&~original
    z['source']=np.where(original,'original',np.where(supplement,'supplement','unresolved'))
    z['combined_sic']=z.sic_original.where(original,z.sic_supplement.where(supplement))
    if z.loc[z.source!='unresolved','combined_sic'].isna().any():raise ValueError('Matched without SIC')
    return z


def main():
    study=Path('docs/rishi_mechanism_20261027');out=study/'execution_batch20_sec_combined'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    base_path=study/'execution_batch17_sec_preference/JOINED.csv';extra_path=study/'execution_batch19_sec_supplement/RECEIPTS.csv'
    audit_path=study/'execution_batch19_sec_supplement/AUDIT.json';old_path=study/'execution_batch17_sec_preference/SUMMARY.csv'
    a=json.loads(audit_path.read_text())
    if a['receipts']!=42 or a['matches_verified']!=38:raise ValueError('Supplement audit incomplete')
    base=pd.read_csv(base_path);extra=pd.read_csv(extra_path);z=combine(base,extra)
    z['combined_industry']=z.combined_sic.map(lambda x:ff12(int(x)) if pd.notna(x) else np.nan)
    old=pd.read_csv(old_path);summary=[];sources=[];industries=[]
    for year,frame in z.groupby('year'):
        initial=frame[frame.source=='original'];covered=frame[frame.source!='unresolved']
        initial_fraction=between_fraction(initial.hlabel_percentile.to_numpy(),initial.industry.to_numpy())[0]
        expected=old.loc[old.year==year,'between_industry_fraction']
        if len(expected)!=1 or not np.isclose(initial_fraction,expected.iloc[0],rtol=1e-12,atol=1e-12):raise ValueError('Earlier result does not reproduce')
        fraction,between,within,total=between_fraction(covered.hlabel_percentile.to_numpy(),covered.combined_industry.to_numpy())
        summary.append(dict(year=year,eligible=len(frame),original=len(initial),supplement=int((frame.source=='supplement').sum()),
            unresolved=int((frame.source=='unresolved').sum()),original_fraction=initial_fraction,combined_fraction=fraction,between_ss=between,within_ss=within,total_ss=total))
        for name,g in frame.groupby('source'):
            sources.append(dict(year=year,source=name,ids=len(g),mean_rank=g.hlabel_percentile.mean(),median_rank=g.hlabel_percentile.median(),under30=int((g['count']<30).sum())))
        for industry,g in covered.groupby('combined_industry'):
            industries.append(dict(year=year,industry=int(industry),ids=len(g),mean_rank=g.hlabel_percentile.mean(),sd_rank=g.hlabel_percentile.std(ddof=0)))
    z.to_csv(out/'JOINED.csv',index=False);pd.DataFrame(summary).to_csv(out/'SUMMARY.csv',index=False)
    pd.DataFrame(sources).to_csv(out/'SOURCE_GROUPS.csv',index=False);pd.DataFrame(industries).to_csv(out/'INDUSTRY_GROUPS.csv',index=False)
    paths=[base_path,extra_path,audit_path,old_path,Path(__file__),study/'SEC_COMBINED_PREFERENCE_PROTOCOL.md',Path('research/analysis/rishi_dated_sic.py')]
    (out/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope':'Hlabel ranks, not neural predictions or financial returns; no independent NDX identity certification'},indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(sources).to_string(index=False))


if __name__=='__main__':main()
