"""Modern prices for the pre-existing 2017 research cohorts, with coverage audit.

The ticker list is frozen before downloads. Unavailable symbols are retained as
missing columns, never replaced by current winners. Public-vendor availability
can nevertheless introduce survivorship bias; this is not a PIT stock database.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FIELDS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/recent_cohorts'))
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise SystemExit('Completed immutable snapshot exists. Choose a fresh directory.')
    selection = {}
    for market in ['NYSE', 'NASDAQ']:
        names = Path(f'config/{market}_source_cohort.txt').read_text().splitlines()
        selection[market + '_recent'] = sorted(sorted(names, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:100])
    selection_path = root / 'selection.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection
    else:
        selection_path.write_text(json.dumps(selection, indent=2))
    symbols = sorted(set(sum(selection.values(), []) + ['SPY', 'QQQ']))
    # Only provider punctuation conversion, not successor-company substitution.
    aliases = {name: name.replace('.', '-') for name in symbols}
    (root / 'aliases.json').write_text(json.dumps(aliases, indent=2))
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    cache = root / 'raw'
    cache.mkdir(exist_ok=True)
    failures = []
    for i in range(0, len(symbols), 10):
        batch = symbols[i:i+10]
        needed = [s for s in batch if not (cache / f'{s}.csv').exists()
                  or pd.read_csv(cache / f'{s}.csv')['Adj Close'].notna().sum() == 0]
        if needed:
            raw = yf.download([aliases[s] for s in needed], start='2018-01-01', end='2026-01-01',
                              auto_adjust=False, actions=True, progress=False, threads=4,
                              timeout=25, multi_level_index=True)
            for name in needed:
                frame = pd.DataFrame(index=raw.index)
                for field in FIELDS:
                    key = (field, aliases[name])
                    frame[field] = raw[key] if key in raw.columns else np.nan
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.to_csv(cache / f'{name}.csv', index_label='Date')
        print(f'Downloaded/cached {min(i+10, len(symbols))}/{len(symbols)} symbols', flush=True)
    spy = pd.read_csv(cache / 'SPY.csv', index_col=0, parse_dates=True)
    dates = spy.loc[spy['Adj Close'].notna()].index
    if len(dates) < 1900 or dates.max() < pd.Timestamp('2025-12-30'):
        raise RuntimeError('Incomplete SPY calendar; cannot evaluate recent years.')
    all_fields = {field: {} for field in FIELDS}
    coverage = []
    for name in symbols:
        frame = pd.read_csv(cache / f'{name}.csv', index_col=0, parse_dates=True).reindex(dates)
        valid = frame['Adj Close'].notna() & (frame['Adj Close'] > 0)
        if not valid.any():
            failures.append(name)
        row = dict(ticker=name, provider_symbol=aliases[name],
                   first=str(dates[valid][0].date()) if valid.any() else None,
                   last=str(dates[valid][-1].date()) if valid.any() else None)
        for year in range(2018, 2026):
            row[f'observed_{year}'] = int(valid[dates.year == year].sum())
        coverage.append(row)
        for field in FIELDS:
            all_fields[field][name] = frame[field]
    for field, columns in all_fields.items():
        pd.DataFrame(columns, index=dates).to_csv(root / (field.lower().replace(' ', '_') + '.csv'), index_label='Date')
    pd.DataFrame(coverage).to_csv(root / 'coverage.csv', index=False)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob('*.csv')}
    manifest = dict(provider='Yahoo Finance via yfinance', version=yf.__version__,
                    downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    start='2018-01-01', end_exclusive='2026-01-01', selection=selection,
                    benchmark='SPY adjusted close for BOTH cohorts; QQQ retained as secondary reference',
                    no_history=failures, hashes=hashes,
                    limitations=['Frozen 2017 research cohort; no replacement based on recent returns.',
                                 'Public-provider missing histories can cause survivorship/availability bias.',
                                 'Cohorts include funds/preferred shares; historical exchange labels are not current membership.',
                                 'Revised adjusted prices, not point-in-time corporate-action vintages.',
                                 'No certified delisting-return or historical membership feed.'])
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(sessions=len(dates), missing_entire_histories=failures)), flush=True)


if __name__ == '__main__':
    main()
