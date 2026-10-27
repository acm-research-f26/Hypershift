"""Download quarantined 15-minute Alpaca stock bars; never places orders.

Default symbols are the historical monthly NASDAQ-100 ticker union plus SPY/QQQ.
No new package is required beyond requirements-data.txt and requirements.txt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
ENDPOINT = 'https://data.alpaca.markets/v2/stocks/bars'
MEMBERS = ROOT / 'data/nasdaq_monthly_pit/assembled/monthly_members.csv'
TIMEZONE = 'America/New_York'
COLUMNS = ['symbol', 'timestamp_utc', 'bar_end_utc', 'open', 'high', 'low',
           'close', 'volume', 'trade_count', 'vwap']
ENV_KEYS = {'APCA_API_KEY_ID', 'APCA_API_SECRET_KEY', 'APCA_API_BASE_URL',
            'ALPACA_DATA_FEED'}


class AlpacaError(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_env(path):
    """Literal KEY=value settings only; no shell expansion or execution.

    Supports blank/comment lines, optional export, and matching outer quotes.
    Process environment overrides the local file. Errors never include values.
    """
    values = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            key, sep, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if key not in ENV_KEYS:
                continue
            if not sep:
                raise ValueError('Invalid Alpaca setting in .env; use KEY=value.')
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError('Unmatched quote in an Alpaca .env setting.')
                value = value[1:-1]
            values[key] = value
    for key in ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def auth_headers(settings):
    keys = ['APCA_API_KEY_ID', 'APCA_API_SECRET_KEY']
    if any(not settings.get(k, '').strip() or
           settings[k].upper().startswith('PASTE_') for k in keys):
        raise AlpacaError('Fill APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env first. '
                          'No request sent. Keep credentials in the local file.')
    if any(any(ch.isspace() for ch in settings[k]) for k in keys):
        raise AlpacaError('Alpaca credentials must not contain whitespace. No request sent.')
    return {'APCA-API-KEY-ID': settings[keys[0]],
            'APCA-API-SECRET-KEY': settings[keys[1]]}


def date_bound(value):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Dates must use YYYY-MM-DD.')
    return pd.Timestamp(value, tz=TIMEZONE)


def chunks(start, end):
    boundaries = [start]
    for year in range(start.year + 1, end.year + 1):
        boundary = pd.Timestamp(f'{year}-01-01', tz=TIMEZONE)
        if boundary < end:
            boundaries.append(boundary)
    return list(zip(boundaries + [end], (boundaries + [end])[1:]))


def symbols_for(explicit, members=MEMBERS):
    if explicit:
        symbols = sorted(set(s.upper() for s in explicit))
        source = None
    else:
        frame = pd.read_csv(members, dtype=str)
        symbols = sorted(set(frame.ticker) | {'SPY', 'QQQ'})
        source = {'path': str(members), 'sha256': digest(members),
                  'rule': 'All historical ticker spellings, plus SPY/QQQ; no current-survivor filter.'}
    if not symbols or any(not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,19}', s) for s in symbols):
        raise ValueError('Invalid or empty stock symbol list.')
    return symbols, source


class AlpacaClient:
    def __init__(self, headers, rpm=150, session=None, sleep=time.sleep,
                 clock=time.monotonic, retries=5):
        self.headers = headers
        self.session = session if session is not None else requests.Session()
        self.sleep, self.clock = sleep, clock
        self.interval, self.next_request, self.retries = 60 / rpm, 0., retries

    def page(self, params):
        for attempt in range(self.retries):
            self.sleep(max(0., self.next_request - self.clock()))
            self.next_request = self.clock() + self.interval
            response = None
            try:
                response = self.session.get(ENDPOINT, headers=self.headers, params=params,
                                            timeout=(10, 60), allow_redirects=False)
            except requests.RequestException:
                if attempt + 1 == self.retries:
                    raise AlpacaError('Market-data request failed after retries; details suppressed.') from None
            if response is not None:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise AlpacaError('Market-data endpoint returned invalid JSON.') from None
                    if not isinstance(payload, dict) or 'bars' not in payload or 'next_page_token' not in payload:
                        raise AlpacaError('Unrecognized market-data response schema.')
                    return payload
                if response.status_code in (401, 403):
                    raise AlpacaError(f'Alpaca HTTP {response.status_code}: verify the key pair and '
                                      'historical feed entitlement. No feed was substituted.')
                if response.status_code != 429 and response.status_code not in (500, 502, 503, 504):
                    raise AlpacaError(f'Alpaca HTTP {response.status_code}; response details suppressed.')
            if attempt + 1 == self.retries:
                raise AlpacaError('Alpaca rate limit or service failure persisted after retries.')
            retry_after = response.headers.get('Retry-After', '') if response is not None else ''
            try:
                delay = float(retry_after)
                if not math.isfinite(delay):
                    raise ValueError
            except ValueError:
                delay = 2 ** attempt
            self.sleep(min(30., max(1., delay)))

    def bars(self, symbol, start, end, feed, adjustment):
        params = dict(symbols=symbol, timeframe='15Min', feed=feed,
                      adjustment=adjustment, asof='-', sort='asc', limit=10000,
                      start=start.tz_convert('UTC').isoformat(),
                      end=(end.tz_convert('UTC') - pd.Timedelta(microseconds=1)).isoformat())
        bars, tokens, pages = [], set(), 0
        while True:
            payload = self.page(params)
            mapping = payload['bars']
            if mapping is None:
                mapping = {}
            if not isinstance(mapping, dict) or set(mapping) - {symbol}:
                raise AlpacaError('Unexpected symbols in Alpaca response; no identity remapping accepted.')
            values = mapping.get(symbol, [])
            if not isinstance(values, list):
                raise AlpacaError('Unrecognized bars schema.')
            bars.extend(values)
            pages += 1
            token = payload['next_page_token']
            if not token:
                break
            if not isinstance(token, str) or token in tokens:
                raise AlpacaError('Invalid or repeated pagination token; incomplete chunk rejected.')
            tokens.add(token)
            params['page_token'] = token
        return normalize(bars, symbol, start, end), pages


def normalize(bars, symbol, start, end):
    if not bars:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(bars)
    fields = dict(t='timestamp_utc', o='open', h='high', l='low', c='close',
                  v='volume', n='trade_count', vw='vwap')
    if not set(fields).issubset(frame.columns):
        raise AlpacaError('Historical bars have missing required fields.')
    frame = frame.rename(columns=fields)
    timestamps = pd.to_datetime(frame.timestamp_utc, utc=True, errors='raise')
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise AlpacaError('Missing, duplicate, or unsorted bar timestamps; chunk rejected.')
    if (timestamps.dt.minute % 15 != 0).any() or (timestamps.dt.second != 0).any() or (timestamps.dt.microsecond != 0).any():
        raise AlpacaError('Bars do not align to 15-minute boundaries.')
    end_times = timestamps + pd.Timedelta(minutes=15)
    if ((timestamps < start) | (end_times > end)).any():
        raise AlpacaError('Bar outside the requested interval or incomplete at its end.')
    numeric = list(fields.values())[1:]
    values = frame[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise AlpacaError('Nonfinite bar values; chunk rejected.')
    prices = values[['open', 'high', 'low', 'close', 'vwap']]
    if ((prices <= 0).any().any() or (values[['volume', 'trade_count']] < 0).any().any() or
            (values.high < values[['open', 'low', 'close']].max(axis=1)).any() or
            (values.low > values[['open', 'high', 'close']].min(axis=1)).any() or
            (values.trade_count % 1 != 0).any()):
        raise AlpacaError('Invalid OHLC, volume, or trade count; chunk rejected.')
    frame[numeric] = values
    frame['symbol'] = symbol
    frame['timestamp_utc'] = timestamps.map(lambda t: t.isoformat())
    frame['bar_end_utc'] = end_times.map(lambda t: t.isoformat())
    return frame[COLUMNS]


def save_chunk(client, out, symbol, start, end, feed, adjustment):
    stem = f'{start.date()}_{end.date()}'
    folder = out / 'bars' / symbol
    path, record_path = folder / (stem + '.csv.gz'), folder / (stem + '.json')
    query = dict(symbol=symbol, start=str(start), end_exclusive=str(end),
                 feed=feed, adjustment=adjustment, timeframe='15Min', asof='-')
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['query'] != query or not path.exists() or digest(path) != record['sha256']:
            raise AlpacaError('Completed cache chunk changed; choose a fresh output directory.')
        return record
    frame, pages = client.bars(symbol, start, end, feed, adjustment)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    frame.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(path)
    record = dict(query=query, rows=len(frame), pages=pages,
                  status='downloaded_unreviewed' if len(frame) else 'empty_requires_review',
                  first_bar=None if frame.empty else frame.timestamp_utc.iloc[0],
                  last_bar=None if frame.empty else frame.timestamp_utc.iloc[-1],
                  file=str(path.relative_to(out)), sha256=digest(path),
                  retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--symbols', nargs='+', help='Default: every historical membership ticker plus SPY/QQQ.')
    parser.add_argument('--start', default='2018-01-01', help='Inclusive New York date.')
    parser.add_argument('--end', default='2026-01-01', help='Exclusive New York date.')
    parser.add_argument('--feed', choices=['sip', 'iex'], help='Default: ALPACA_DATA_FEED or sip; never auto-switches.')
    parser.add_argument('--adjustment', choices=['raw', 'split', 'all'], default='raw')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--requests-per-minute', type=int, default=150)
    parser.add_argument('--dry-run', action='store_true', help='Show non-secret plan; no network or dataset writes.')
    args = parser.parse_args(argv)
    out, plan, records, started = None, None, [], False
    try:
        settings = read_env(args.env_file)
        feed = args.feed or settings.get('ALPACA_DATA_FEED', 'sip')
        if feed not in ('sip', 'iex'):
            raise ValueError('ALPACA_DATA_FEED must be sip or iex.')
        start, end = date_bound(args.start), date_bound(args.end)
        if start >= end or end.tz_convert('UTC') > pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=16):
            raise ValueError('Use start < end and a completed historical end date at least 16 minutes ago.')
        if not 2 <= args.requests_per_minute <= 1000:
            raise ValueError('Requests per minute must be between 2 and 1000.')
        symbols, membership = symbols_for(args.symbols)
        intervals = chunks(start, end)
        out = (args.out or ROOT / 'data/nasdaq_monthly_pit/alpaca_15min' /
               f'{feed}_{args.adjustment}_{start:%Y%m%d}_{end:%Y%m%d}').resolve()
        plan = dict(provider='Alpaca', endpoint=ENDPOINT, timeframe='15Min',
                    start=str(start), end_exclusive=str(end), date_timezone=TIMEZONE,
                    feed=feed, adjustment=args.adjustment, asof='-', symbols=symbols,
                    sessions='All returned sessions retained; no regular-session filtering or forward fill.',
                    membership_source=membership, downloader_sha256=digest(Path(__file__)),
                    chunk_count=len(symbols)*len(intervals),
                    status='price_only_unreviewed_not_training_inputs')
        if args.dry_run:
            print(json.dumps(dict(plan=plan, output_directory=str(out)), indent=2))
            return 0
        headers = auth_headers(settings)  # Validate before creating outputs or sending requests.
        lock = out / 'REQUEST.json'
        if lock.exists():
            if json.loads(lock.read_text(encoding='utf-8')) != plan:
                raise AlpacaError('Download settings changed; choose a fresh output directory.')
        elif out.exists() and any(out.iterdir()):
            raise AlpacaError('Output directory is nonempty and has no matching request lock.')
        out.mkdir(parents=True, exist_ok=True)
        write_json(lock, plan)
        write_json(out / 'manifest.json', dict(**plan, download_status='in_progress'))
        started = True
        client = AlpacaClient(headers, rpm=args.requests_per_minute)
        for symbol in symbols:
            for left, right in intervals:
                record = save_chunk(client, out, symbol, left, right, feed, args.adjustment)
                records.append(record)
                print(f'{symbol} {left.date()}..{right.date()} {record["rows"]} bars ({record["status"]})', flush=True)
        coverage = pd.DataFrame([dict(symbol=r['query']['symbol'], start=r['query']['start'],
                          end_exclusive=r['query']['end_exclusive'], rows=r['rows'],
                          status=r['status'], first_bar=r['first_bar'], last_bar=r['last_bar'],
                          file=r['file'], sha256=r['sha256']) for r in records])
        coverage.to_csv(out / 'coverage.csv', index=False)
        write_json(out / 'manifest.json', dict(**plan, download_status='completed_with_review_required',
            completed_chunks=len(records), rows=int(coverage.rows.sum()),
            empty_chunks=int((coverage.rows == 0).sum()), chunks=records,
            limitations=['No security identity, corporate action, or historical member coverage certification.',
                         'Empty chunks are disclosed; IPOs, delistings, missing data and ticker reuse require review.',
                         'Raw bars are not total returns. Adjusted bars are retrospective vendor adjustments.',
                         'Bar timestamp marks the interval start; features become available at bar end.',
                         'Extended hours retained. Apply an exchange calendar including early closes before fitting.',
                         '15-minute bars do not guarantee better forecasts or independent market observations.']))
        print(f'Download complete: {len(records)} chunks. Coverage requires review: {out}', flush=True)
        return 0
    except (AlpacaError, ValueError, OSError, KeyError):
        # Never expose request/response bodies, header values, or arbitrary exception text.
        import sys
        error = sys.exc_info()[1]
        message = str(error) if isinstance(error, AlpacaError) else 'Invalid configuration, filesystem, or bar data; details suppressed.'
        # Only update a manifest that this invocation successfully locked.
        if started:
            try:
                if json.loads((out / 'REQUEST.json').read_text()) == plan:
                    write_json(out / 'manifest.json', dict(**plan, download_status='interrupted_or_failed',
                               completed_chunks=len(records), message=message))
            except (OSError, ValueError):
                pass
        parser.exit(1, message + '\n')


if __name__ == '__main__':
    main()
