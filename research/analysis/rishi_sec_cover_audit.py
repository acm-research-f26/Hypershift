"""Verify saved SEC identity receipts against their raw response bytes."""
import hashlib,json,re
from collections import Counter
from datetime import date
from pathlib import Path
import pandas as pd
from lxml import html


def main():
    root=Path('results/RISHI_SEC_COVER_20261028');out=Path('docs/rishi_mechanism_20261027/execution_batch16_sec_cover')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    eligible=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch15_dated_sic/ELIGIBLE_JOIN.csv')
    eligible=eligible[eligible.year.isin([2024,2025])]
    rows=[json.loads(p.read_text()) for p in root.glob('*.json')]
    expected={(int(r.year),r.security_id,r.signal_date) for r in eligible.itertuples()}
    actual={(r['year'],r['security_id'],r['signal_date']) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:raise ValueError('Receipt universe mismatch')
    issues=[];checked=0
    for r in rows:
        if r['status']!='matched':continue
        stem=f"{r['year']}_{r['security_id']}";head=(root/(stem+'_header.bin')).read_bytes();doc=(root/(stem+'_filing.bin')).read_bytes()
        text=head.decode('latin-1');filing=re.search(r'FILED AS OF DATE:\s*(\d{8})',text,re.I)
        accession=re.search(r'ACCESSION NUMBER:\s*(\S+)',text,re.I);cik=re.search(r'CENTRAL INDEX KEY:\s*(\d+)',text,re.I)
        codes=re.findall(r'STANDARD INDUSTRIAL CLASSIFICATION:[^\[]*\[(\d{4})\]',text,re.I)
        symbols={''.join(e.itertext()).strip().upper() for e in html.fromstring(doc).iter() if (e.get('name') or '').lower()=='dei:tradingsymbol'}
        dt=date.fromisoformat(r['filing_date']);age=(date.fromisoformat(r['signal_date'])-dt).days
        good=(filing and filing.group(1)==dt.strftime('%Y%m%d') and accession and accession.group(1)==r['accession'] and
            cik and int(cik.group(1))==r['cik'] and set(codes)=={str(r['sic'])} and r['security_id'] in symbols and
            sorted(symbols)==r['symbols'] and 0<age<=400 and hashlib.sha256(head).hexdigest()==r['header_sha256'] and
            hashlib.sha256(doc).hexdigest()==r['filing_sha256'])
        checked+=1
        if not good:issues.append(stem)
    if issues:raise ValueError('Receipt validation failed: '+str(issues))
    summary=[]
    for year in (2024,2025):
        counts=Counter(r['status'] for r in rows if r['year']==year)
        summary.append(dict(year=year,eligible=sum(counts.values()),**counts))
    pd.DataFrame(summary).fillna(0).to_csv(out/'COVERAGE.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'RECEIPTS.csv',index=False)
    paths=[Path('docs/rishi_mechanism_20261027/SEC_COVER_IDENTITY_PROTOCOL.md'),Path('research/analysis/rishi_sec_cover.py'),
        Path(__file__),Path('data/edgar/cik_map.json'),Path('docs/rishi_mechanism_20261027/execution_batch15_dated_sic/ELIGIBLE_JOIN.csv')]
    (out/'AUDIT.json').write_text(json.dumps({'receipt_count':len(rows),'matched_validated':checked,'issues':issues,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limits':'Filing date evidence; no intervening change, NDX identity or index-vintage certification'},indent=2)+'\n')
    print('validated',checked,'matches from',len(rows),'receipts; no discrepancies')


if __name__=='__main__':main()
