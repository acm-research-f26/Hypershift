"""Re-run the published scalar feature spot checks against the new raw supplement."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path('.')
raw = root / 'data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101'
features = root / 'data/nasdaq_monthly_pit/extended_hours_features_20261024'
review = json.loads((features / 'REVIEW.json').read_text())
for name, expected in review['source_hashes'].items():
    file = raw.joinpath(*name.replace('\\', '/').split('/'))
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, name
records = json.loads((raw.parent / 'calendar_2018_2025.json').read_text())
cal = pd.DataFrame(records)
cal['date'] = pd.to_datetime(cal.date)
for field in ('open', 'close'):
    text = cal.date.dt.strftime('%Y-%m-%d') + ' ' + cal[field]
    cal[field + '_utc'] = pd.to_datetime(text).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
cal = cal.set_index('date')
source = (root / 'research/reporting/report_ndx_extended_hours.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'reference_summary')
module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
namespace = {'np': np, 'pd': pd}
exec(compile(module, '<published scalar reference>', 'exec'), namespace)
reference = namespace['reference_summary']
samples = pd.read_csv(root / 'docs/ndx_extended_hours_20261024/independent_feature_samples.csv')
cube = np.load(features / 'features.npz', allow_pickle=False)
dates = {v: i for i, v in enumerate(cube['dates'])}
tickers = {v: i for i, v in enumerate(cube['tickers'])}
cache = {}
max_error = 0.0
for row in samples.itertuples(index=False):
    year = int(row.date[:4])
    path = raw / 'bars' / row.symbol / f'{year}-01-01_{year+1}-01-01.csv.gz'
    if path not in cache:
        cache[path] = pd.read_csv(path)
    actual, count = reference(cache[path], cal, row.date, row.segment)
    saved = cube[row.segment][dates[row.date], tickers[row.security_id]]
    np.testing.assert_allclose(actual, saved, atol=2e-6, rtol=2e-6, equal_nan=True)
    assert count == row.observed
    finite = np.isfinite(actual) & np.isfinite(saved)
    if finite.any():
        max_error = max(max_error, float(np.max(np.abs(actual[finite] - saved[finite]))))
print(f"review_hashes={len(review['source_hashes'])} samples={len(samples)} "
      f"chunks_read={len(cache)} max_abs_error={max_error:.12g}")
