"""Download a timestamped, immutable hourly OHLCV research snapshot.

Hourly source timestamps label bar STARTS. A completed hour's features become
available at bar start + one hour. Never use a still-forming bar as a closed one.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


def full_hour_mask(index, now):
    local = index.tz_convert('America/New_York')
    # Keep ordinary full-hour bars, omit the regular 15:30-16:00 half-hour.
    # The provider's early-close half-bars need a calendar for precise handling;
    # sample construction below additionally requires a following full hour.
    return ((local.minute == 30) & (local.hour >= 9) & (local.hour <= 14)
            & (index + pd.Timedelta(hours=1) + pd.Timedelta(minutes=20) <= now))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', default='1y')
    parser.add_argument('--out', type=Path, default=Path('data/hourly'))
    parser.add_argument('--tickers', nargs='+', default=['AAPL','MSFT','GOOGL','NVDA','JPM','BAC','GS','MS','XOM','CVX','COP','SLB'])
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new output directory to preserve previous snapshots.')
    args.out.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    now = pd.Timestamp(datetime.now(timezone.utc))
    raw = yf.download(args.tickers, period=args.period, interval='1h', auto_adjust=False,
                      prepost=False, threads=False, progress=False, timeout=30,
                      group_by='column', multi_level_index=True)
    if raw is None or raw.empty:
        raise RuntimeError('Provider returned no hourly observations.')
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()
    raw.to_csv(args.out / 'provider_ohlcv.csv', index_label='bar_start_utc')
    fields = ['Open','High','Low','Close','Volume']
    close = raw['Close'].reindex(columns=args.tickers)
    complete = raw.loc[:, raw.columns.get_level_values(0).isin(fields)].notna().all(axis=1)
    values_valid = (close > 0).all(axis=1) & np.isfinite(close).all(axis=1)
    mask = complete & values_valid & full_hour_mask(raw.index, now)
    usable = raw.loc[mask]
    close = usable['Close'].reindex(columns=args.tickers)
    # Require an additional timestamp exactly one hour later on the same date.
    # This excludes overnight gaps and the last available bar on early-close days.
    raw_local = raw.index.tz_convert('America/New_York')
    day_last_start = pd.Series(raw.index, index=raw_local.date).groupby(level=0).max()
    usable_local = close.index.tz_convert('America/New_York')
    conservative = np.array([t < day_last_start.loc[d] for t, d in zip(close.index, usable_local.date)])
    close = close.loc[conservative]
    close.to_csv(args.out / 'full_hour_closes.csv', index_label='bar_start_utc')
    rows = []
    for i in range(len(close) - 1):
        start, next_start = close.index[i], close.index[i + 1]
        if next_start - start != pd.Timedelta(hours=1):
            continue
        for ticker in args.tickers:
            current, future = float(close.iloc[i][ticker]), float(close.iloc[i + 1][ticker])
            rows.append({'ticker': ticker, 'feature_bar_start_utc': start.isoformat(),
                         'signal_available_utc': (start + pd.Timedelta(hours=1)).isoformat(),
                         'target_close_utc': (next_start + pd.Timedelta(hours=1)).isoformat(),
                         'current_close': current, 'next_hour_close': future,
                         'next_hour_return_pp': 100 * (future / current - 1), 'actual_up': future > current})
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError('No complete consecutive-hour pairs were found.')
    labels.to_csv(args.out / 'next_hour_outcomes.csv', index=False)
    metadata = {'provider': 'Yahoo Finance via yfinance', 'yfinance_version': yf.__version__,
                'downloaded_at_utc': now.isoformat(), 'requested_period': args.period,
                'interval': '1h', 'auto_adjust': False, 'regular_session_only': True,
                'provider_timestamp_semantics': 'bar start', 'raw_rows': len(raw),
                'completion_buffer_minutes': 20,
                'buffer_meaning': 'Conservative downloader choice, not a measurement or guarantee of provider latency.',
                'incomplete_or_invalid_rows': int((~(complete & values_valid)).sum()),
                'full_hour_rows': len(close), 'hourly_forecast_origins': int(labels.signal_available_utc.nunique()),
                'stock_hour_outcomes': len(labels), 'first_bar_start_utc': raw.index[0].isoformat(),
                'last_bar_start_utc': raw.index[-1].isoformat(), 'tickers': args.tickers,
                'limitations': 'Research snapshot, not guaranteed real-time feed. Last session bar omitted to avoid early-close partial bars. No forward filling. Outcomes are labels, not trained predictions.',
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.csv')}}
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
