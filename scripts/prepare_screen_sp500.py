"""Assemble local S&P 500 yfinance daily panel for the v3 exploratory h5 screen; no download.

Source: data/fresh/sp500_daily (scripts/fetch_fresh.py, yfinance auto-adjusted closes,
CURRENT S&P 500 constituents -> survivorship-biased, D17). Features are MA5/10/20/30/close
divided by a per-stock train-period max; that scalar cancels in every window-relative input
and return. GICS sector codes come from universe.csv (current classification, not
point-in-time).
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hypershift.screening.data import sha256, forward_labels

src = Path('data/fresh/sp500_daily')
out = src/'screen_sp500_2015_2026.npz'
if out.exists():
    raise SystemExit('already assembled; refusing to overwrite')
z = np.load(src/'panel.npz')
tickers = (src/'tickers.txt').read_text().split()
universe = pd.read_csv(src/'universe.csv').set_index('ticker')
assert list(universe.index) == tickers or set(universe.index) == set(tickers)
sector_names = sorted(universe.sector.unique())
sectors = np.array([sector_names.index(universe.loc[t, 'sector']) for t in tickers], np.int16)
dates = pd.to_datetime(z['timestamps']).tz_localize('UTC').tz_convert('America/New_York')
dates = np.array(dates.date, dtype='datetime64[D]')
assert len(np.unique(dates)) == len(dates)
features, mask = z['features'].astype(np.float32), z['mask'].astype(bool)
y, ok = forward_labels(features[:, :, -1], mask)
tail = {'max_abs_h5': float(np.abs(y[ok]).max()), 'n_h5_above_100pct': int((np.abs(y[ok]) > 1).sum()),
        'n_h5_above_50pct': int((np.abs(y[ok]) > .5).sum()), 'n_labels': int(ok.sum())}
np.savez_compressed(out, features=features, mask=mask.astype(np.uint8), dates=dates,
                    tickers=np.array(tickers), identity_ok=np.ones(len(tickers), np.uint8), sectors=sectors)
dest = Path('docs/screening_20261019')
(dest/'SP500_INPUT_MANIFEST.json').write_text(json.dumps({
    'source_files': {str(p): sha256(p) for p in (src/'panel.npz', src/'tickers.txt', src/'universe.csv', src/'bars.csv.gz')},
    'panel_sha256': sha256(out), 'shape': list(features.shape), 'dates': [str(dates[0]), str(dates[-1])],
    'sector_names': sector_names, 'sector_counts': np.bincount(sectors).tolist(), 'h5_tail': tail,
    'bias': 'current constituents (survivorship); current GICS labels (not point-in-time)',
    'normalization': 'per-stock train-max scalar cancels in window ratios and returns'}, indent=2))
print(out, features.shape, dates[[0, -1]], tail, flush=True)
