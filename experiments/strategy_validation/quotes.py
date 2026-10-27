"""Read-only SIP downloader. No trading endpoint or broker-order capability."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np

from experiments.strategy_search.storage import atomic, arrays, read, sha

_lock = threading.Lock()
_next_request = 0.
SIZE_CHANGE = np.datetime64('2025-11-03', 'ns').astype(np.int64)
QUOTE_DTYPE = np.dtype([('bid', 'f8'), ('ask', 'f8'), ('bid_size', 'f8'), ('ask_size', 'f8'), ('timestamp', 'i8')])


def credentials(env_file):
    values = dict(os.environ)
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values.setdefault(key.strip(), value.strip().strip('\"\''))
    key = values.get('ALPACA_API_KEY', values.get('APCA_API_KEY_ID'))
    secret = values.get('ALPACA_API_KEY_SECRET', values.get('APCA_API_SECRET_KEY'))
    if not key or not secret:
        raise ValueError('Missing Alpaca market-data credentials; never put keys in command arguments')
    return {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}


def request(path, params, headers):
    global _next_request
    if not path.startswith('/v2/stocks/') and path != '/v1/corporate-actions':
        raise ValueError('Only stock market-data endpoints are supported')
    for attempt in range(5):
        with _lock:
            now = time.monotonic()
            wait = max(0., _next_request - now)
            _next_request = max(now, _next_request) + .34
        if wait:
            time.sleep(wait)
        req = urllib.request.Request('https://data.alpaca.markets' + path + '?' + urllib.parse.urlencode(params), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(10., 2. ** attempt))
                continue
            raise RuntimeError(f'Alpaca data request failed: HTTP {error.code}') from None
        except urllib.error.URLError:
            if attempt == 4:
                raise RuntimeError('Alpaca market-data network unavailable') from None
            time.sleep(2. ** attempt)


def iso(ns):
    return str(np.datetime64(int(ns), 'ns')) + 'Z'


def normalized_quote(quote, arrival):
    timestamp = int(np.datetime64(quote['t'].rstrip('Z'), 'ns').astype(np.int64))
    multiplier = 100 if timestamp < SIZE_CHANGE else 1
    return (float(quote['bp']), float(quote['ap']),
            float(quote['bs']) * multiplier, float(quote['as']) * multiplier, timestamp)


def fetch_window(timestamp, symbols, headers, lookback_ns=5_000_000_000):
    params = dict(symbols=','.join(symbols), start=iso(timestamp - lookback_ns),
                  end=iso(timestamp + 1), feed='sip', sort='desc', limit=10000)
    latest = {}
    pages = 0
    while True:
        body = request('/v2/stocks/quotes', params, headers)
        pages += 1
        for symbol, records in body.get('quotes', {}).items():
            for q in records:
                value = normalized_quote(q, timestamp)
                if value[4] <= timestamp and (symbol not in latest or value[4] > latest[symbol][4]):
                    latest[symbol] = value
        token = body.get('next_page_token')
        if not token:
            break
        if pages >= 100:
            raise RuntimeError('Quote pagination exceeded 100 pages; incomplete window not accepted')
        params['page_token'] = token
    missing = [s for s in symbols if s not in latest]
    if missing and lookback_ns < 5_000_000_000:
        extra, count = fetch_window(timestamp, missing, headers, 5_000_000_000)
        latest.update(extra)
        pages += count
    return latest, pages


def download(m, windows, output, env_file, workers=4):
    """Resume independently hashed per-minute snapshots; retain as-of provenance."""
    output = Path(output) / 'quotes' / m['scope']
    output.mkdir(parents=True, exist_ok=True)
    headers = credentials(env_file)
    def one(item):
        i, indices = item
        timestamp = int(m['starts'][i])
        symbols = [m['symbols'][j] for j in indices]
        p = output / f'{timestamp}.json'
        if p.exists():
            old = read(p)
            if old['symbols'] != symbols or old['arrival_ns'] != timestamp or old['feed'] != 'sip':
                raise ValueError('Changed cached quote request')
            return i, 0
        values, pages = fetch_window(timestamp, symbols, headers)
        atomic(p, dict(feed='sip', arrival_ns=timestamp, symbols=symbols, quotes=values, pages=pages,
                       retrieved_UTC=datetime.now(timezone.utc).isoformat(),
                       sizes='shares; source round lots converted for dates before 2025-11-03'))
        return i, pages
    completed = 0
    pages = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, count in pool.map(one, windows):
            completed += 1
            pages += count
            if completed % 100 == 0:
                print(json.dumps(dict(scope=m['scope'], quote_windows=completed, total=len(windows), downloaded_pages=pages)), flush=True)
    return assemble(m, windows, output)


def assemble(m, windows, root):
    values = np.zeros(m['open'].shape, dtype=QUOTE_DTYPE)
    for field in ('bid', 'ask', 'bid_size', 'ask_size'):
        values[field] = np.nan
    values['timestamp'] = -1
    hashes = {}
    previous = read(Path(root) / 'manifest.json')['files'] if (Path(root) / 'manifest.json').exists() else {}
    for i, indices in windows:
        p = Path(root) / f'{int(m["starts"][i])}.json'
        if not p.exists():
            continue
        doc = read(p)
        if doc['feed'] != 'sip' or doc['arrival_ns'] != int(m['starts'][i]):
            raise ValueError('Quote provenance mismatch')
        hashes[p.name] = sha(p)
        if p.name in previous and hashes[p.name] != previous[p.name]:
            raise ValueError('Changed cached bid/ask snapshot')
        for j in indices:
            q = doc['quotes'].get(m['symbols'][j])
            if q is not None:
                values[i, j] = tuple(q)
    receipt = Path(root) / 'manifest.json'
    atomic(receipt, dict(feed='sip', files=hashes, windows=len(windows), cached=len(hashes),
                         columns=['bid', 'ask', 'bid_size_shares', 'ask_size_shares', 'timestamp_ns'],
                         complete=len(hashes) == len(windows)))
    return values
