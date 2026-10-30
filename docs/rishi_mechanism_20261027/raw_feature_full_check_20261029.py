"""Reconstruct all saved extended-hours feature arrays without writing source artifacts."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('.')
raw = root / 'data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101'
features = root / 'data/nasdaq_monthly_pit/extended_hours_features_20261024'
cube = np.load(features / 'features.npz', allow_pickle=False)
dates = pd.DatetimeIndex(pd.to_datetime(cube['dates']))
names = cube['tickers'].tolist()
fields = cube['feature_names'].tolist()
manifest = json.loads((raw / 'manifest.json').read_text())
aliases = json.loads((root / 'data/nasdaq_monthly_pit/assembled/manifest.json').read_text())['alias_map']
rules = json.loads((root / 'config/alpaca_symbol_identity_20261019.json').read_text())['symbols']
cal = pd.DataFrame(json.loads((raw.parent / 'calendar_2018_2025.json').read_text()))
cal['date'] = pd.to_datetime(cal.date)
for field in ('open', 'close'):
    text = cal.date.dt.strftime('%Y-%m-%d') + ' ' + cal[field]
    cal[field + '_utc'] = pd.to_datetime(text).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
cal = cal.set_index('date')
np.testing.assert_array_equal(cal.index, dates)
source = (root / 'research/data/prepare_ndx_extended_hours.py').read_text()
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in ('windows', 'summarize')]
namespace = {'np': np, 'pd': pd, 'FIELDS': fields}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<source builder>', 'exec'), namespace)
summarize = namespace['summarize']
arrays = {s: np.full(cube[s].shape, np.nan, dtype='float32') for s in ('after_hours', 'premarket')}
counts = {s: np.zeros(cube[s + '_counts'].shape, dtype='int16') for s in arrays}
for a in arrays.values():
    a[..., 7] = 0
    a[..., 8:] = 1
seen = {s: set() for s in arrays}
for index, record in enumerate(manifest['chunks'], 1):
    symbol = record['query']['symbol']
    security = aliases.get(symbol, symbol)
    path = raw.joinpath(*record['file'].replace('\\', '/').split('/'))
    frame = pd.read_csv(path)
    assert len(frame) == record['rows'] and (not len(frame) or frame.symbol.eq(symbol).all())
    rule = rules.get(symbol, {})
    assert not rule or rule.get('security_id', security) == security
    if security not in names:
        continue
    node = names.index(security)
    for segment in arrays:
        summary = summarize(frame, cal, segment)
        if 'first_valid_date' in rule:
            summary = summary[summary.date >= rule['first_valid_date']]
        if 'end_valid_date' in rule:
            summary = summary[summary.date < rule['end_valid_date']]
        for date, observed, *values in summary[['date', 'observed'] + fields].itertuples(index=False, name=None):
            key = (security, date)
            assert key not in seen[segment], key
            seen[segment].add(key)
            day = dates.get_loc(pd.Timestamp(date))
            arrays[segment][day, node] = values
            counts[segment][day, node] = observed
    if index % 200 == 0:
        print(f'chunks {index}/{len(manifest["chunks"])}', flush=True)
for segment in arrays:
    np.testing.assert_array_equal(arrays[segment], cube[segment])
    np.testing.assert_array_equal(counts[segment], cube[segment + '_counts'])
    print(f'{segment}: {arrays[segment].size} feature cells and {counts[segment].size} counts exactly match; populated sessions {len(seen[segment])}')
print(f'complete: {len(manifest["chunks"])} chunks; saved arrays untouched')
