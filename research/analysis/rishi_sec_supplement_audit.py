"""Check SEC missing-issuer supplement receipts against raw captured evidence."""
import gzip,hashlib,json,re
from collections import Counter
from datetime import date
from pathlib import Path
import pandas as pd
from lxml import html
from rishi_sec_cover import select_filing
from rishi_sec_supplement import choose_candidate


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    root=Path('results/RISHI_SEC_COVER_SUPPLEMENT_20261028');prior=Path('results/RISHI_SEC_COVER_20261028')
    out=Path('docs/rishi_mechanism_20261027/execution_batch19_sec_supplement')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    rows=[json.loads(p.read_text()) for p in root.glob('*.json')];original=[json.loads(p.read_text()) for p in prior.glob('*.json')]
    expected={(r['year'],r['security_id']) for r in original if r['status']!='matched'}
    if len(rows)!=42 or {(r['year'],r['security_id']) for r in rows}!=expected:raise ValueError('Incomplete supplement')
    source='https://www.sec.gov/files/company_tickers.json';cache=Path('data/edgar/json')/(sha(source.encode())[:24]+'.json.gz')
    cached=json.loads(gzip.decompress(cache.read_bytes()));current={}
    for v in cached['body'].values():current.setdefault(v['ticker'].upper(),[]).append(int(v['cik_str']))
    old={(r['year'],r['security_id']):r for r in original};verified=0
    for r in rows:
        if r['status']!='matched':continue
        previous=old[r['year'],r['security_id']];status,cik=choose_candidate(r['security_id'],current,previous.get('candidate_ciks',[]))
        if status!='candidate' or cik!=r['cik']:raise ValueError('Candidate mismatch')
        h=(root/(sha(r['header_url'].encode())+'.bin')).read_bytes();d=(root/(sha(r['filing_url'].encode())+'.bin')).read_bytes();t=h.decode('latin-1')
        acc=re.search(r'ACCESSION NUMBER:\s*(\S+)',t,re.I);ck=re.search(r'CENTRAL INDEX KEY:\s*(\d+)',t,re.I)
        filed=re.search(r'FILED AS OF DATE:\s*(\d{8})',t,re.I);sic=re.findall(r'STANDARD INDUSTRIAL CLASSIFICATION:[^\[]*\[(\d{4})\]',t,re.I)
        sym=sorted({''.join(e.itertext()).strip().upper() for e in html.fromstring(d).iter() if (e.get('name') or '').lower()=='dei:tradingsymbol'})
        day=date.fromisoformat(r['filing_date']);age=(date.fromisoformat(r['signal_date'])-day).days
        if not (sha(h)==r['header_sha256'] and sha(d)==r['filing_sha256'] and acc and acc.group(1)==r['accession'] and ck and int(ck.group(1))==cik and filed and filed.group(1)==day.strftime('%Y%m%d') and set(sic)=={str(r['sic'])} and sym==r['symbols'] and r['security_id'] in sym and 0<age<=400):raise ValueError('Filing evidence mismatch '+r['security_id'])
        blocks=[]
        for s in r['submissions_sources']:
            if 'cache_file' in s:
                raw=Path(s['cache_file']).read_bytes();record=json.loads(gzip.decompress(raw));block=record['body']
                if record['url']!=s['url']:raise ValueError('Cache provenance mismatch')
            else:raw=Path(s['download_file']).read_bytes();block=json.loads(raw)
            if sha(raw)!=s['sha256']:raise ValueError('Submission hash mismatch')
            blocks.append(block['filings']['recent'] if 'filings' in block else block)
        options=[{k:b[k][i] for k in ('form','filingDate','accessionNumber','primaryDocument')} for b in blocks for i in range(len(b['form']))]
        selected=select_filing(options,r['signal_date'])
        if not selected or selected['accessionNumber']!=r['accession']:raise ValueError('Selection mismatch')
        verified+=1
    summary=[]
    for year in (2024,2025):
        c=Counter(r['status'] for r in rows if r['year']==year);base=sum(r['status']=='matched' for r in original if r['year']==year)
        summary.append(dict(year=year,baseline_matches=base,new_matches=c['matched'],combined_matches=base+c['matched'],unresolved=sum(c.values())-c['matched'],supplement_statuses=dict(c)))
    pd.DataFrame(rows).to_csv(out/'RECEIPTS.csv',index=False);pd.DataFrame(summary).to_csv(out/'COVERAGE.csv',index=False)
    (out/'AUDIT.json').write_text(json.dumps({'receipts':len(rows),'matches_verified':verified,'current_list_sha256':sha(cache.read_bytes()),'coverage':summary},indent=2)+'\n')
    print('verified',verified,'supplement matches;',summary)


if __name__=='__main__':main()
