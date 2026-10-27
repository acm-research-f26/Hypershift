"""Reuse immutable stock snapshots; cache additional ETF bars and quotes locally."""
import concurrent.futures as cf
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from experiments.strategy_search.storage import read, sha, atomic, arrays
from experiments.strategy_search.features import stock_volatility
from experiments.strategy_validation.market import load
from experiments.strategy_validation.quotes import credentials, request, fetch_window, iso, QUOTE_DTYPE
from .config import SCOPES, ETFS


def check_dates(m, protocol):
    dates = np.datetime_as_string(m['starts'].astype('datetime64[ns]').astype('datetime64[D]'))
    if set(dates) & set(protocol['binding']['reserved_session_dates']):
        raise ValueError('A protected future session was requested')
    allowed = {d for block in protocol['binding']['partitions'][m['scope']].values() for d in block['dates']}
    if not set(dates).issubset(allowed):
        raise ValueError('Only the previously exposed cohort dates are authorized in this experiment')


def stock_market(reference, scope, protocol):
    m, base, diagnostic = load(reference, scope)
    check_dates(m, protocol)
    return m, base


def empty_quotes(shape):
    a = np.zeros(shape, dtype=QUOTE_DTYPE)
    for f in ('bid', 'ask', 'bid_size', 'ask_size'): a[f] = np.nan
    a['timestamp'] = -1
    return a


def stock_quotes(m, prior, output):
    root = Path(prior).parent / 'quotes' / m['scope']
    manifest = read(root / 'manifest.json')['files']
    values = empty_quotes(m['open'].shape)
    lookup = {str(int(t)): i for i, t in enumerate(m['starts'])}
    symbols = {s: j for j, s in enumerate(m['symbols'])}
    reused = {}
    for name, digest in manifest.items():
        stem = Path(name).stem
        if stem not in lookup: continue
        p = root / name
        if sha(p) != digest: raise ValueError('Changed original quote cache')
        d = read(p); i = lookup[stem]
        if d['arrival_ns'] != int(m['starts'][i]) or d['feed'] != 'sip': raise ValueError('Invalid cached quote provenance')
        for s, q in d['quotes'].items():
            if s in symbols: values[i, symbols[s]] = tuple(q)
        reused[name] = digest
    atomic(Path(output) / 'reuse' / (m['scope'] + '.json'), dict(root=str(root), files=reused))
    return values


def prepare_etfs(m, output, env_file, workers=4):
    """Fetch exactly the same session dates, never the gaps or sealed periods."""
    root = Path(output) / 'etf_cache'
    headers = credentials(env_file)
    groups = [(sid, np.flatnonzero(m['sessions'] == sid)) for sid in m['meta']['session_ids']]
    def one(item):
        sid, ix = item
        date = str(np.datetime64(int(m['starts'][ix[0]]), 'ns').astype('datetime64[D]'))
        p = root / 'bars' / (date + '.json')
        if p.exists():
            d = read(p)
            if d['symbols'] != list(ETFS): raise ValueError('Changed ETF universe')
        else:
            params = dict(symbols=','.join(ETFS), timeframe='1Min', start=iso(m['starts'][ix[0]]),
                end=iso(int(m['starts'][ix[-1]])+60_000_000_000), feed='sip', adjustment='raw', limit=10000, sort='asc')
            records = {s: [] for s in ETFS}; pages = 0
            while True:
                body = request('/v2/stocks/bars', params, headers); pages += 1
                for s, bars in body.get('bars', {}).items(): records[s].extend(bars)
                token = body.get('next_page_token')
                if not token: break
                if pages >= 50: raise ValueError('Incomplete ETF bars')
                params['page_token'] = token
            d = dict(symbols=list(ETFS), date=date, feed='sip', adjustment='raw', bars=records,
                     retrieved_UTC=datetime.now(timezone.utc).isoformat())
            atomic(p, d)
        return date
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for k, date in enumerate(pool.map(one, groups), 1):
            if k % 40 == 0: print(f'ETF bars {m["scope"]}: {k}/{len(groups)}', flush=True)
    return load_etfs(m, output)


def load_etfs(stock, output):
    shape = (len(stock['starts']), len(ETFS))
    m = {k: stock[k] for k in ('sessions', 'starts', 'meta', 'evaluate_ids', 'scope', 'completed')}
    m.update(open=np.full(shape, np.nan), close=np.full(shape, np.nan), volume=np.zeros(shape),
             actions=np.zeros(shape, bool), sector=np.arange(len(ETFS)), symbols=list(ETFS))
    # SPY and QQQ are a single hedge group. Sector ETFs each have a separate group.
    m['sector'][1] = 0
    lookup = {int(t): i for i, t in enumerate(m['starts'])}
    hashes = {}
    for sid in m['meta']['session_ids']:
        ix = np.flatnonzero(m['sessions'] == sid)
        date = str(np.datetime64(int(m['starts'][ix[0]]), 'ns').astype('datetime64[D]'))
        p = Path(output) / 'etf_cache' / 'bars' / (date + '.json'); d = read(p); hashes[p.name] = sha(p)
        for j, s in enumerate(ETFS):
            for bar in d['bars'].get(s, []):
                stamp = int(np.datetime64(bar['t'].rstrip('Z'), 'ns').astype(np.int64))
                if stamp in lookup:
                    i = lookup[stamp]; m['open'][i,j] = bar['o']; m['close'][i,j] = bar['c']; m['volume'][i,j] = bar['v']
    valid = np.isfinite(m['close']) & (m['close'] > 0)
    raw = np.zeros(shape); ok = np.zeros(shape, bool)
    ok[1:] = valid[1:] & valid[:-1] & (m['sessions'][1:] == m['sessions'][:-1])[:,None]
    np.log(np.divide(m['close'][1:], m['close'][:-1], out=np.ones_like(raw[1:]), where=ok[1:]),out=raw[1:])
    m['vol'], vol_ok = stock_volatility(raw, ok, m['sessions'])
    m['eligible_full'] = valid & vol_ok
    manifest=Path(output)/'etf_cache'/(m['scope']+'.bars_manifest.json')
    if manifest.exists() and read(manifest)!=hashes:
        raise ValueError('Changed cached ETF bars')
    atomic(manifest, hashes)
    # Raw ETF split discontinuities cannot be traded through; quarantine entire sessions.
    for sid in m['meta']['session_ids']:
        ix = np.flatnonzero(m['sessions'] == sid)
        if ix[0] and (m['sessions'][ix[0]-1] == sid-1):
            ratio = m['open'][ix[0]] / m['close'][ix[0]-1]
            suspicious = np.isfinite(ratio) & ((ratio < .65) | (ratio > 1.5))
            if suspicious.any(): m['eligible_full'][np.ix_(ix, np.flatnonzero(suspicious))] = False
    return m


def etf_quote_windows(m):
    windows=[]
    for sid in m['meta']['session_ids']:
        ix = np.flatnonzero(m['sessions'] == sid)
        # Decision quote as of completed signal bar; three arrivals, five exits.
        for local in [len(ix)-29, len(ix)-28, len(ix)-27, len(ix)-26, *range(len(ix)-5,len(ix))]:
            windows.append(int(ix[local]))
    return windows


def prepare_etf_quotes(m, output, env_file, workers=4):
    headers = credentials(env_file); root = Path(output)/'etf_cache'/'quotes'
    windows = etf_quote_windows(m)
    def one(i):
        stamp=int(m['starts'][i]); p=root/(str(stamp)+'.json')
        if not p.exists():
            quotes,pages=fetch_window(stamp,list(ETFS),headers)
            atomic(p,dict(feed='sip',arrival_ns=stamp,symbols=list(ETFS),quotes=quotes,pages=pages,
                          retrieved_UTC=datetime.now(timezone.utc).isoformat()))
        return i
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for k,i in enumerate(pool.map(one,windows),1):
            if k%100==0: print(f'ETF quotes {m["scope"]}: {k}/{len(windows)}',flush=True)
    return etf_quotes(m,output)


def etf_quotes(m, output):
    values=empty_quotes(m['open'].shape); files={};root=Path(output)/'etf_cache'/'quotes'
    for i in etf_quote_windows(m):
        p=root/(str(int(m['starts'][i]))+'.json')
        if not p.exists(): raise ValueError('Missing ETF quote window: '+str(p))
        d=read(p)
        if d['arrival_ns']!=int(m['starts'][i]) or d['symbols']!=list(ETFS): raise ValueError('Changed ETF quote provenance')
        for j,s in enumerate(ETFS):
            if s in d['quotes']: values[i,j]=tuple(d['quotes'][s])
        files[p.name]=sha(p)
    dest=Path(output)/'etf_cache'/(m['scope']+'.quotes_manifest.json')
    if dest.exists() and read(dest)!=files: raise ValueError('Changed ETF quote cache')
    atomic(dest,files)
    return values
