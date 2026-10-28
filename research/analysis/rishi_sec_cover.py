"""Bounded SEC filing-cover feasibility audit; no strategy scoring."""
import gzip,hashlib,json,os,re,time
from datetime import date
from pathlib import Path
import pandas as pd
import requests
from lxml import html


def select_filing(rows,day):
    target=date.fromisoformat(day)
    good=[r for r in rows if r.get('form') in ('10-K','10-Q','20-F','40-F') and r.get('filingDate') and r.get('primaryDocument') and 0<(target-date.fromisoformat(r['filingDate'])).days<=400]
    return max(good,key=lambda r:(r['filingDate'],r['accessionNumber'])) if good else None


def parse_filing(header,document,cik,accession,ticker):
    text=header.decode('latin-1');a=re.search(r'ACCESSION NUMBER:\s*(\S+)',text,re.I);c=re.search(r'CENTRAL INDEX KEY:\s*(\d+)',text,re.I)
    if not a or not c or a.group(1)!=accession or int(c.group(1))!=int(cik):return ('header_identity_mismatch',None,[])
    sic=re.findall(r'STANDARD INDUSTRIAL CLASSIFICATION:[^\[]*\[(\d{4})\]',text,re.I)
    if len(set(sic))!=1:return ('missing_or_ambiguous_sic',None,[])
    root=html.fromstring(document)
    symbols=sorted({''.join(e.itertext()).strip().upper() for e in root.iter() if (e.get('name') or '').lower()=='dei:tradingsymbol'})
    return ('matched' if ticker.upper() in symbols else 'ticker_not_tagged',int(sic[0]),symbols)


def cached_json(url):
    p=Path('data/edgar/json')/(hashlib.sha256(url.encode()).hexdigest()[:24]+'.json.gz')
    if not p.exists():return None
    record=json.loads(gzip.decompress(p.read_bytes()))
    if record.get('url')!=url:raise ValueError('Cache URL collision')
    return record['body']


def submission_rows(cik,day):
    url=f'https://data.sec.gov/submissions/CIK{cik:010d}.json';body=cached_json(url)
    if body is None:return None
    blocks=[body['filings']['recent']]
    if min(blocks[0]['filingDate'],default='9999')>=day:
        for f in body['filings'].get('files',[]):
            if f.get('filingFrom','9999')<day:
                page=cached_json('https://data.sec.gov/submissions/'+f['name'])
                if page is None:return None
                blocks.append(page)
    rows=[]
    for b in blocks:
        for i in range(len(b['form'])):
            rows.append({k:b[k][i] for k in ('form','filingDate','accessionNumber','primaryDocument')})
    return rows


def main():
    root=Path('results/RISHI_SEC_COVER_20261028');root.mkdir(parents=True,exist_ok=True)
    source=Path('docs/rishi_mechanism_20261027/execution_batch15_dated_sic/ELIGIBLE_JOIN.csv')
    f=pd.read_csv(source);f=f[f.year.isin([2024,2025])];mapping=json.loads(Path('data/edgar/cik_map.json').read_text())
    session=requests.Session();session.headers.update({'User-Agent':os.environ['SEC_USER_AGENT'],'Accept-Encoding':'gzip, deflate'})
    last=0.
    def fetch(url,path):
        nonlocal last
        if path.exists():return path.read_bytes()
        delay=.5-(time.monotonic()-last)
        if delay>0:time.sleep(delay)
        last=time.monotonic();r=session.get(url,timeout=20);r.raise_for_status();path.write_bytes(r.content);return r.content
    for r in f.itertuples():
        receipt=root/f'{r.year}_{r.security_id}.json'
        if receipt.exists():continue
        row={'year':int(r.year),'signal_date':r.signal_date,'security_id':r.security_id}
        ciks=sorted({int(c) for a,b,c in mapping.get(r.security_id,{}).get('segments',[]) if mapping[r.security_id].get('checked',{}).get(str(c),{}).get('ok')})
        row['candidate_ciks']=ciks
        try:
            if len(ciks)!=1:row['status']='no_candidate' if not ciks else 'ambiguous_candidates'
            else:
                cik=ciks[0];rows=submission_rows(cik,r.signal_date)
                if rows is None:row['status']='missing_cached_submission_history'
                else:
                    item=select_filing(rows,r.signal_date)
                    if not item:row['status']='no_recent_filing'
                    else:
                        acc=item['accessionNumber']
                        base=f'https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace("-", "")}'
                        hurl=base+'/'+acc+'-index-headers.html';durl=base+'/'+item['primaryDocument']
                        head=fetch(hurl,root/f'{r.year}_{r.security_id}_header.bin');doc=fetch(durl,root/f'{r.year}_{r.security_id}_filing.bin')
                        status,sic,symbols=parse_filing(head,doc,cik,acc,r.security_id)
                        row.update(status=status,cik=cik,accession=acc,filing_date=item['filingDate'],form=item['form'],sic=sic,symbols=symbols,
                            header_url=hurl,filing_url=durl,header_sha256=hashlib.sha256(head).hexdigest(),filing_sha256=hashlib.sha256(doc).hexdigest())
        except Exception as e:row.update(status='error',error=f'{type(e).__name__}: {e}')
        receipt.write_text(json.dumps(row,indent=2)+'\n');print(r.year,r.security_id,row['status'],flush=True)
    print('Completed',len(f),'fixed identity receipts')


if __name__=='__main__':main()
