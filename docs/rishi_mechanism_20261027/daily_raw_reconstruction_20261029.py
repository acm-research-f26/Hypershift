"""Rebuild published daily panels in memory from raw sources and recorded event rules."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('data/nasdaq_monthly_pit')
assembled = root / 'assembled'
meta = json.loads((assembled / 'manifest.json').read_text())
raw_meta = json.loads((root / 'yahoo_raw/manifest.json').read_text())
members = pd.read_csv(root / 'ndx_preflight/monthly_members_PROVISIONAL.csv')
members = members[members.month >= '2019-07-01'].copy()
members['security_id'] = members.ticker.map(lambda x: raw_meta['aliases'].get(x, x))
assert not members.duplicated(['month', 'security_id']).any()
names = sorted(members.security_id.unique())
read_yahoo = lambda name: pd.read_csv(root / 'yahoo_raw' / (name + '.csv'), index_col=0, parse_dates=True)
raw_y, adj_y, vol_y = (read_yahoo(name) for name in ('close', 'adj_close', 'volume'))
dates = adj_y.index[adj_y.SPY.notna() & adj_y.QQQ.notna()]
raw = pd.DataFrame(index=dates, columns=names, dtype=float)
economic = raw.copy(); volume = raw.copy()
sources = {record['security_id']: record for record in meta['sources']}
for name in names:
    record = sources[name]
    if record['provider'] == 'Tiingo':
        path = root / 'tiingo_raw' / (name + '_prices.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        values = pd.DataFrame(json.loads(path.read_text()))
        values.index = pd.to_datetime(values.date, utc=True).dt.tz_localize(None)
        assert not values.index.has_duplicates
        values = values.reindex(dates)
        raw[name] = values.close; economic[name] = values.adjClose
        volume[name] = values.close * values.volume
    else:
        assert record['provider'] == 'Yahoo Finance'
        raw[name] = raw_y[name].reindex(dates); economic[name] = adj_y[name].reindex(dates)
        volume[name] = (raw_y[name] * vol_y[name]).reindex(dates)
membership = pd.DataFrame(False, index=dates, columns=names)
for month, rows in members.groupby('month'):
    membership.loc[dates.to_period('M') == pd.Timestamp(month).to_period('M'), rows.security_id] = True
tradable = raw.notna() & raw.gt(0)
for event in meta['test_settlements']:
    name = event['security_id']; day = pd.Timestamp(event['date'])
    before = raw.loc[raw.index < day, name].dropna().index[-1]
    factor = economic.at[before, name] / raw.at[before, name]
    stock = 0. if event['acquirer'] is None else event['stock_ratio'] * raw_y.at[day, event['acquirer']]
    payout = event['cash_per_raw_share'] + stock
    np.testing.assert_allclose([factor, payout], [event['adjusted_unit_factor'], event['payout_raw_dollars_at_completion']], rtol=1e-12, atol=1e-12)
    economic.loc[day:, name] = payout * factor
    raw.loc[day:, name] = np.nan; volume.loc[day:, name] = np.nan; tradable.loc[day:, name] = False
for name, day in meta['earlier_unpriced_mergers'].items():
    for panel in (raw, economic, volume): panel.loc[day:, name] = np.nan
    tradable.loc[day:, name] = False
panels = {'raw_close': raw, 'economic_close': economic, 'dollar_volume': volume,
          'tradable': tradable, 'membership': membership,
          'etf_adj_close': adj_y.loc[dates, ['SPY', 'QQQ']],
          'etf_dollar_volume': (raw_y * vol_y).loc[dates, ['SPY', 'QQQ']]}
for name, expected in panels.items():
    saved = pd.read_csv(assembled / (name + '.csv'), index_col=0, parse_dates=True)
    assert expected.index.equals(saved.index) and expected.columns.equals(saved.columns)
    if name in ('tradable', 'membership'):
        np.testing.assert_array_equal(expected.to_numpy(), saved.to_numpy())
        error = 0.
    else:
        np.testing.assert_allclose(expected.to_numpy(), saved.to_numpy(), rtol=1e-12, atol=1e-8, equal_nan=True)
        error = float(np.nanmax(np.abs(expected.to_numpy() - saved.to_numpy())))
    print(f'{name}: {expected.size} cells max absolute difference {error:.9g}', flush=True)
print(f'{len(names)} securities, {len(dates)} dates, {len(meta["test_settlements"])} settlement rules; no files overwritten')
