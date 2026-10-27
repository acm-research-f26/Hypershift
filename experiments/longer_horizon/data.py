from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import numpy as np
from experiments.strategy_search.storage import read,sha,atomic
from experiments.closing_followup.data import stock_market,empty_quotes
from experiments.strategy_validation.quotes import credentials,request,fetch_window,iso
from .config import SCOPES


def download_auctions(reference,prior,output,env_file,workers=4):
    reference,prior,output=map(Path,(reference,prior,output));rp=read(reference/'protocol.json')
    protected=set(read(prior/'protected_sessions.json')['reserved_session_dates'])
    manifest=output/'auctions'/'manifest.json'
    if manifest.exists():
        for name,digest in read(manifest)['files'].items():
            if sha(manifest.parent/name)!=digest:raise ValueError('Changed auction cache')
    symbols=read(Path(rp['market_root'])/'metadata.json')['symbols'];headers=credentials(env_file);tasks=[]
    for scope in SCOPES:
        cache=Path(rp['study_root'])/'cache'/scope;meta=read(cache/'market.json')
        sessions=np.load(cache/'sessions.npy',mmap_mode='r');ends=np.load(cache/'completed.npy',mmap_mode='r')
        allowed={d for p in rp['partitions'][scope].values() for d in p['dates']}
        for sid in meta['session_ids']:
            ix=np.flatnonzero(sessions==sid);start=int(ends[ix[0]])-60_000_000_000;end=int(ends[ix[-1]])+600_000_000_000
            date=str(np.datetime64(start,'ns').astype('datetime64[D]'))
            if date not in allowed or date in protected:raise ValueError('Auction request violates session guard')
            tasks.append((date,start,end))
    def one(item):
        date,start,end=item;p=output/'auctions'/(date+'.json')
        if p.exists():return date
        params=dict(symbols=','.join(symbols),start=iso(start),end=iso(end),feed='sip',limit=10000,sort='asc')
        data={};pages=0
        while True:
            body=request('/v2/stocks/auctions',params,headers);pages+=1
            for s,records in body.get('auctions',{}).items():data.setdefault(s,[]).extend(records)
            token=body.get('next_page_token')
            if not token:break
            if pages>=50:raise ValueError('Incomplete auction pagination')
            params['page_token']=token
        atomic(p,dict(date=date,symbols=symbols,auctions=data,pages=pages,feed='sip',
            retrieved_UTC=datetime.now(timezone.utc).isoformat(),use='Terminal price outcome only; no imbalance signal'))
        return date
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k,date in enumerate(pool.map(one,tasks),1):
            if k%40==0:print(f'AUCTION_CACHE {k}/{len(tasks)}',flush=True)
    files={p.name:sha(p) for p in sorted((output/'auctions').glob('*.json')) if p.name!='manifest.json'}
    atomic(output/'auctions'/'manifest.json',dict(files=files,sessions=len(tasks),read_only_market_data=True))


class QuoteCache:
    def __init__(self,m,output,prior,env_file):
        self.m=m;self.root=Path(output)/'quotes'/m['scope'];self.root.mkdir(parents=True,exist_ok=True)
        self.values=empty_quotes(m['open'].shape);self.known=np.zeros(m['open'].shape,bool)
        self.records={};self.headers=None;self.env_file=env_file
        self.old=Path(prior).parent/'quotes'/m['scope'];self.old_manifest=read(self.old/'manifest.json')['files']
        self.old_docs={}
        index={str(int(t)):i for i,t in enumerate(m['starts'])};symbols={s:j for j,s in enumerate(m['symbols'])}
        for p in self.root.glob('*.json'):
            if p.name=='manifest.json':continue
            d=read(p);i=index[str(d['arrival_ns'])];j=symbols[d['symbol']]
            self.known[i,j]=True
            if d['quote'] is not None:self.values[i,j]=tuple(d['quote'])
            self.records[p.name]=sha(p)
        manifest=self.root/'manifest.json'
        if manifest.exists():
            for name,digest in read(manifest)['files'].items():
                if self.records.get(name)!=digest:raise ValueError('Changed execution quote cache')

    def missing(self,orders):
        if not len(orders):return []
        pairs=np.unique(orders[:,:2].astype(int),axis=0)
        return [(int(i),int(j)) for i,j in pairs if not self.known[i,j]]

    def fetch(self,pairs,workers=4):
        by_time={}
        for i,j in pairs:
            if not self.known[i,j]:by_time.setdefault(i,set()).add(j)
        if not by_time:return 0
        if self.headers is None:self.headers=credentials(self.env_file)
        def one(item):
            i,js=item;stamp=int(self.m['starts'][i]);old_name=str(stamp)+'.json';values={};references={}
            if old_name in self.old_manifest:
                path=self.old/old_name
                if sha(path)!=self.old_manifest[old_name]:raise ValueError('Changed original quote snapshot')
                doc=read(path)
                for j in js:
                    s=self.m['symbols'][j]
                    if s in doc['quotes']:values[s]=tuple(doc['quotes'][s]);references[s]=str(path)
            wanted=[self.m['symbols'][j] for j in js if self.m['symbols'][j] not in values]
            if wanted:
                fresh,pages=fetch_window(stamp,wanted,self.headers);values.update(fresh)
                absent=[s for s in wanted if s not in values]
                if absent:
                    older,count=fetch_window(stamp,absent,self.headers,60_000_000_000);values.update(older)
            records=[]
            for j in sorted(js):
                s=self.m['symbols'][j];p=self.root/(str(stamp)+'_'+s+'.json');q=values.get(s)
                atomic(p,dict(arrival_ns=stamp,symbol=s,quote=q,feed='sip',lookback_seconds=60,
                    reused_from=references.get(s),retrieved_UTC=datetime.now(timezone.utc).isoformat(),
                    size_units='shares; normalized for pre-2025-11-03 source round lots'))
                records.append((i,j,q,p.name,sha(p)))
            return records
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for k,records in enumerate(pool.map(one,sorted(by_time.items())),1):
                for i,j,q,name,digest in records:
                    self.known[i,j]=True
                    if q is not None:self.values[i,j]=tuple(q)
                    self.records[name]=digest
                if k%100==0:print(f'QUOTES {self.m["scope"]} {k}/{len(by_time)} windows',flush=True)
        self.save();return len(by_time)

    def save(self):
        atomic(self.root/'manifest.json',dict(files=dict(sorted(self.records.items())),symbols_queried=int(self.known.sum()),
            primary_max_age_seconds=5,sensitivity_max_age_seconds=60,feed='sip'))
