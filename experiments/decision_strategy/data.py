"""Date-guarded private input adapters and resumable SIP quote requests."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from experiments.strategy_search.storage import read,sha,atomic,utc
from experiments.strategy_search.features import returns,clocks,stock_volatility,window_sum,peers
from experiments.strategy_validation.quotes import credentials,fetch_window
from experiments.closing_followup.data import load_etfs
from .config import STUDY,REFERENCE,PRIOR,ETF_ROOT,SCOPES

ETF_MAP={'Materials':'XLB','Communication Services':'XLC','Energy':'XLE','Financials':'XLF',
    'Industrials':'XLI','Information Technology':'XLK','Consumer Staples':'XLP','Real Estate':'XLRE',
    'Utilities':'XLU','Health Care':'XLV','Consumer Discretionary':'XLY'}


def guard_dates(dates,scope,protocol):
    b=protocol['binding'];reserved=set(b['protected']['reserved_session_dates'])
    allowed={d for p in b['partitions'][scope].values() for d in p['dates']}
    if reserved.intersection(dates) or not set(dates).issubset(allowed):
        raise ValueError('Unauthorized or reserved date; outcomes must stay sealed')


def load_market(scope,protocol,output):
    """Validate metadata dates BEFORE opening price, mask or forecast arrays."""
    output=Path(output)
    if scope==SCOPES[2] and not (output/'selection.json').exists():
        raise RuntimeError('Freeze first-wave shortlist before later data access')
    root=STUDY/'cache'/scope;meta=read(root/'market.json');guard_dates(meta['dates'],scope,protocol)
    classification=read(output/'causal_classifications.json')
    if classification['status']!='verified' or len(classification['rows'])!=250:
        raise ValueError('All250 pre-2025 classification bindings are required')
    if any(r['published']>meta['dates'][0] for r in classification['rows']):
        raise ValueError('Classification publication follows evaluated history')
    names=('open','close','sessions','completed','signal_index','eligible','actions')
    files={name+'.npy':sha(root/(name+'.npy')) for name in names}
    if any(files[n]!=meta['files'][n] for n in files):raise ValueError('Changed safe cohort cache')
    m={k:np.load(root/(k+'.npy'),mmap_mode='r') for k in names}
    starts=np.asarray(m['completed'],np.int64)-60_000_000_000
    actual=np.datetime_as_string(starts.astype('datetime64[ns]').astype('datetime64[D]'))
    guard_dates(set(actual),scope,protocol)
    market_root=Path(protocol['binding']['market_root']);symbols=read(market_root/'metadata.json')['symbols']
    classification={r['symbol']:r for r in classification['rows']}
    groups=sorted({r['industry_group'] for r in classification.values()});sectors=sorted(ETF_MAP)
    industry=np.array([groups.index(classification[s]['industry_group']) for s in symbols])
    sector=np.array([sectors.index(classification[s]['sector_name']) for s in symbols])
    raw,valid=returns(m['close'],m['sessions']);vol,vol_ok=stock_volatility(raw,valid,m['sessions'])
    # Full archive is memory mapped but ONLY the authorized cached cohort slice
    # is touched; never hash/read the entire raw history containing sealed data.
    archive=np.load(market_root/'1m'/'bars.npy',mmap_mode='r')
    bars=archive[meta['begin']:meta['end']]
    np.testing.assert_array_equal(bars[:,:,0],m['open']);np.testing.assert_array_equal(bars[:,:,3],m['close'])
    volume=np.asarray(bars[:,:,4],float)
    elig=np.zeros_like(valid);rows=np.flatnonzero(m['signal_index']>=0);elig[rows]=m['eligible']
    elig &= valid & vol_ok & ~m['actions']
    m.update(starts=starts,scope=scope,meta=meta,symbols=list(symbols),industry=industry,sector=sector,
        sector_names=sectors,raw=raw,valid=valid,vol=vol,volume=volume,eligible_full=elig,
        evaluate_ids=meta['session_ids'][60:] if scope==SCOPES[0] else meta['session_ids'],clock=clocks(m['sessions']))
    etf=load_etfs(m,ETF_ROOT)
    chosen=[etf['symbols'].index(ETF_MAP[s]) for s in sectors]
    for k in ('open','close','volume','actions','eligible_full','raw','valid','vol'):
        if k in etf:m[k]=np.concatenate((m[k],etf[k][:,chosen]),axis=1)
    # ETF valid/raw are computed independently because the prior adapter does
    # not expose them. Their causal masks never use endpoint/label availability.
    m['raw'],m['valid']=returns(m['close'],m['sessions'])
    m['vol'],_=stock_volatility(m['raw'],m['valid'],m['sessions'])
    m['symbols'] += [ETF_MAP[s] for s in sectors]
    m['sector']=np.r_[sector,np.arange(len(sectors))]
    m['industry']=np.r_[industry,np.arange(len(sectors))+len(groups)]
    m['stock_etf']=np.array([250+int(s) for s in sector])
    m['groups']=groups
    atomic(output/'data_audits'/(scope+'.json'),dict(scope=scope,dates=meta['dates'],files=files,universe=250,
        extended_etf_universe=m['symbols'][250:],classification_sha256=sha(output/'causal_classifications.json'),
        raw_archive_accessed_slice=[meta['begin'],meta['end']],no_full_archive_hash=True,
        future_outcomes_accessed=False,target_mask_used=False,training_excluded=60 if scope==SCOPES[0] else 0))
    return m


class Quotes:
    """Sparse exact-nanosecond snapshots, including explicit negative responses."""
    def __init__(self,m,output,protocol,env_file='.env'):
        self.m=m;self.protocol=protocol;self.root=Path(output)/'quotes'/m['scope'];self.env=env_file
        self.records={};self.values={};self.headers=None;self.old={}
        old=PRIOR.parent/'quotes'/m['scope'];manifest=old/'manifest.json'
        self.old_root=old;self.old_manifest=read(manifest)['files'] if manifest.exists() else {}
        self.root.mkdir(parents=True,exist_ok=True)
        # Reuse the verified training snapshots from the interrupted v1 run.
        # Hard links preserve exact bytes without duplicating the licensed
        # cache; its source manifest remains read-only and hash-bound.
        previous=STUDY/'decision_strategy_execution_2026-10-27'/'quotes'/m['scope']
        pm=previous/'manifest.json'
        if previous!=self.root and pm.exists():
            import os,shutil
            proof=read(pm)
            for name,h in proof['files'].items():
                src=previous/name;target=self.root/name
                if sha(src)!=h:raise ValueError('Changed reusable v1 snapshot')
                d=read(src);date=str(np.datetime64(int(d['arrival_ns']),'ns').astype('datetime64[D]'))
                guard_dates({date},m['scope'],protocol)
                if not target.exists():
                    try:os.link(src,target)
                    except OSError:shutil.copyfile(src,target)
                if sha(target)!=h:raise ValueError('Changed linked snapshot')
                self.records[name]=h;self.values[(d['arrival_ns'],d['symbol'])]=d['quote']
            atomic(self.root/'reuse_v1.json',dict(source=str(previous),manifest_sha256=sha(pm),files=len(proof['files']),immutable=True))
        manifest=self.root/'manifest.json'
        if manifest.exists():
            for n,h in read(manifest)['files'].items():
                p=self.root/n
                if sha(p)!=h:raise ValueError('Changed quote snapshot')
                d=read(p);self.records[n]=h;self.values[(d['arrival_ns'],d['symbol'])]=d['quote']

    def get(self,i,js):
        stamp=int(self.m['starts'][i])
        return {int(j):self.values[(stamp,self.m['symbols'][j])] for j in js if (stamp,self.m['symbols'][j]) in self.values}

    def fetch(self,requests):
        grouped={}
        for i,j in requests:
            stamp=int(self.m['starts'][i]);symbol=self.m['symbols'][j]
            date=str(np.datetime64(stamp,'ns').astype('datetime64[D]'));guard_dates({date},self.m['scope'],self.protocol)
            if (stamp,symbol) not in self.values:grouped.setdefault(stamp,set()).add(symbol)
        if not grouped:return 0
        if self.headers is None:self.headers=credentials(self.env)
        def one(item):
            stamp,symbols=item;name=str(stamp)+'.json';values={};references={};pages=0
            if name in self.old_manifest:
                p=self.old_root/name
                if sha(p)!=self.old_manifest[name]:raise ValueError('Changed reused SIP snapshot')
                d=read(p)
                if d.get('feed')!='sip' or d['arrival_ns']!=stamp:raise ValueError('Bad SIP provenance')
                for symbol in symbols:
                    if symbol in d['quotes']:values[symbol]=d['quotes'][symbol];references[symbol]=str(p)
            wanted=sorted(symbols-set(values))
            if wanted:
                result,pages=fetch_window(stamp,wanted,self.headers);values.update(result)
            for symbol in sorted(symbols):
                q=values.get(symbol)
                if q is not None:q=[*map(float,q[:4]),int(q[4])]
                p=self.root/(str(stamp)+'_'+symbol+'.json')
                atomic(p,dict(arrival_ns=stamp,symbol=symbol,quote=q,feed='sip',lookback_seconds=5,
                    pages=pages,reused_from=references.get(symbol),retrieved_UTC=utc(),size_units='shares',
                    negative_response=q is None,complete=True))
            return stamp,sorted(symbols)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for k,(stamp,symbols) in enumerate(pool.map(one,sorted(grouped.items())),1):
                for symbol in symbols:
                    p=self.root/(str(stamp)+'_'+symbol+'.json');d=read(p)
                    self.records[p.name]=sha(p);self.values[(stamp,symbol)]=d['quote']
                if k%100==0:print('QUOTE_WINDOWS',self.m['scope'],k,'/',len(grouped),flush=True)
                self.save()
        return len(grouped)

    def save(self):
        atomic(self.root/'manifest.json',dict(files=self.records,feed='sip',timestamp_dtype='int64 nanoseconds',
            requests=len(self.values),negative_responses=sum(v is None for v in self.values.values()),primary_age_seconds=5))
