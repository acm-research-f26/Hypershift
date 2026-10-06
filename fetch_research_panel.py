"""Immutable OHLCV snapshot for the controlled architecture experiment."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('data/research_panel'))
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a fresh output directory.')
    tickers = json.loads(Path('config/modern_source_snapshot.json').read_text())['tickers']
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    raw = yf.download(tickers + ['SPY'], start='2018-01-01', end='2026-10-01',
                      auto_adjust=False, actions=True, progress=False, threads=False,
                      timeout=30, multi_level_index=True)
    if raw is None or raw.empty:
        raise RuntimeError('Provider returned no data; no substitute data used.')
    a.out.mkdir(parents=True)
    hashes = {}
    for field in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']:
        frame = raw[field].reindex(columns=tickers + ['SPY'])
        path = a.out / (field.lower().replace(' ', '_') + '.csv')
        frame.to_csv(path, index_label='Date')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = dict(provider='Yahoo Finance via yfinance', version=yf.__version__,
                downloaded_utc=datetime.now(timezone.utc).isoformat(),
                start='2018-01-01', end_exclusive='2026-10-01', tickers=tickers,
                benchmark='SPY total-return ETF proxy for S&P 500', hashes=hashes,
                limitations=['Fixed 12-stock survivor universe selected in prior work.',
                             'Revised historical data, not a vintage point-in-time database.',
                             'No delisting, historical membership or securities-lending feed.'])
    (a.out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(rows=len(raw), first=str(raw.index[0]), last=str(raw.index[-1]),
                          missing=int(raw['Close'].isna().sum().sum()))), flush=True)


if __name__ == '__main__':
    main()
