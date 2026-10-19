"""Assemble local RSR NASDAQ input for the exploratory h5 screen; no download."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hypershift.data.rsr import load_rsr
from hypershift.screening.data import sha256

root = Path('data/raw/rsr/data')
out = root/'screen_nasdaq_2013_2017.npz'
if out.exists():
    raise SystemExit('already assembled; refusing to overwrite')
data = load_rsr(root, 'NASDAQ', norm='paper')
# Repository calendar convention: first 29 raw dates are the MA warm-up.
dates = np.loadtxt(root/'NASDAQ_aver_line_dates.csv', delimiter=',', dtype=str)[29:29+data.num_steps].astype('datetime64[D]')
assert str(dates[0]) == '2013-01-02' and str(dates[-1]) == '2017-12-08'
paths = [root/'NASDAQ_aver_line_dates.csv', root/'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv']
paths += [root/'2013-01-01'/f'NASDAQ_{t}_1.csv' for t in data.tickers]
manifest = {str(p): sha256(p) for p in paths}
np.savez_compressed(out, features=data.features, mask=data.mask.astype(np.uint8), dates=dates,
                    tickers=np.array(data.tickers), identity_ok=np.ones(data.num_nodes, np.uint8))
# Full-series price normalization cancels exactly in every window-relative input
# and return. No absolute normalized price enters the screening model.
dest = Path('docs/screening_20261019'); dest.mkdir(parents=True, exist_ok=True)
(dest/'RSR_INPUT_MANIFEST.json').write_text(json.dumps({'source_files': manifest, 'panel_sha256': sha256(out),
    'shape': list(data.features.shape), 'calendar_offset': 29,
    'normalization': 'full-series per-stock scalar cancels in window ratios and returns'}, indent=2))
print(out, data.features.shape, dates[[0, -1]], flush=True)
