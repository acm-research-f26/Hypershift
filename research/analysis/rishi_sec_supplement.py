"""Resolve missing issuer candidates using dated SEC covers, never current SIC."""
import gzip,hashlib,json,os,time
from pathlib import Path
import requests
from rishi_sec_cover import parse_filing,select_filing


def choose_candidate(ticker,current,historical):
    ciks=sorted(set(current.get(ticker,[])))
    if not ciks:return ('no_current_candidate',None)
    if len(ciks)!=1:return ('ambiguous_current',None)
    if len(historical)>1 and ciks[0] not in historical:return ('historical_conflict',None)
    return ('candidate',ciks[0])


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    root=Path('results/RISHI_SEC_COVER_SUPPLEMENT_20261028');root.mkdir(parents=True,exist_ok=True)
    prior=Path('results/RISHI_SEC_COVER_20261028');records=[json.loads(p.read_text()) for p in prior.glob('*.json')]
    missing=sorted([r for r in records if r['status']!='matched'],key=lambda x:(x['year'],x['security_id']))
    if len(records)!=202 or len(missing)!=42:raise ValueError('Unexpected original universe or missing count')
    source='https://www.sec.gov/files/company_tickers.json';cache=Path('data/edgar/json')/(hashlib.sha256(source.encode()).hexdigest()[:24]+'.json.gz')
    cached=json.loads(gzip.decompress(cache.read_bytes()))
    if cached.get('url')!=source:raise ValueError('Incorrect cached source')
    current={}
    for v in cached['body'].values():current.setdefault(v['ticker'].upper(),[]).append(int(v['cik_str']))
    session=requests.Session();session.headers.update({'User-Agent':os.environ['SEC_USER_AGENT'],'Accept-Encoding':'gzip, deflate'});last=0.
    def fetch(url):
        nonlocal last
        name=sha(url.encode());path=root/(name+'.bin')
        if path.exists():return path.read_bytes()
        delay=.5-(time.monotonic()-last)
        if delay>0:time.sleep(delay)
        last=time.monotonic();r=session.get(url,timeout=20);r.raise_for_status();path.write_bytes(r.content);return r.content
    def json_source(url):
        cached_path=Path('data/edgar/json')/(sha(url.encode())[:24]+'.json.gz')
        if cached_path.exists():
            raw=cached_path.read_bytes();item=json.loads(gzip.decompress(raw))
            if item['url']!=url:raise ValueError('Cache URL collision')
            return item['body'],{'url':url,'cache_file':str(cached_path),'sha256':sha(raw)}
        raw=fetch(url)
        return json.loads(raw),{'url':url,'download_file':str(root/(sha(url.encode())+'.bin')),'sha256':sha(raw)}
    for priorrow in missing:
        ticker=priorrow['security_id'];year=priorrow['year'];receipt=root/f'{year}_{ticker}.json'
        if receipt.exists():continue
        row={k:priorrow[k] for k in ('year','signal_date','security_id')};row['prior_status']=priorrow['status']
        status,cik=choose_candidate(ticker,current,priorrow.get('candidate_ciks',[]));row.update(status=status,cik=cik)
        try:
            if status=='candidate':
                url=f'https://data.sec.gov/submissions/CIK{cik:010d}.json';body,provenance=json_source(url);sources=[provenance]
                blocks=[body['filings']['recent']]
                if min(blocks[0]['filingDate'],default='9999')>=row['signal_date']:
                    for page in body['filings'].get('files',[]):
                        if page.get('filingFrom','9999')<row['signal_date']:
                            b,prov=json_source('https://data.sec.gov/submissions/'+page['name']);blocks.append(b);sources.append(prov)
                options=[]
                for b in blocks:
                    for i in range(len(b['form'])):options.append({k:b[k][i] for k in ('form','filingDate','accessionNumber','primaryDocument')})
                chosen=select_filing(options,row['signal_date']);row['submissions_sources']=sources
                if not chosen:row['status']='no_recent_filing'
                else:
                    acc=chosen['accessionNumber'];base=f'https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace("-", "")}'
                    hurl=base+'/'+acc+'-index-headers.html';durl=base+'/'+chosen['primaryDocument'];head=fetch(hurl);doc=fetch(durl)
                    status,sic,symbols=parse_filing(head,doc,cik,acc,ticker)
                    row.update(status=status,accession=acc,filing_date=chosen['filingDate'],form=chosen['form'],sic=sic,symbols=symbols,
                        header_url=hurl,filing_url=durl,header_sha256=sha(head),filing_sha256=sha(doc))
        except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        receipt.write_text(json.dumps(row,indent=2)+'\n');print(year,ticker,row['status'],flush=True)
    print('Completed',len(missing),'fixed missing-name receipts',flush=True)


if __name__=='__main__':main()
