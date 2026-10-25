"""Review locked Alpaca bars and derive causal regular-session information."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pit_research_engine import load_data

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101'
CALENDAR = RAW.parent/'calendar_2018_2025.json'
OUT = ROOT/'data/nasdaq_monthly_pit/alpaca_features_20261019'
FEATURES = ['rth_return', 'rth_range', 'within_bar_realized_vol', 'first_hour_return',
            'last_hour_return', 'late_volume_share', 'close_to_vwap', 'intraday_coverage']
CACHE = ROOT/'.cache/alpaca_session_summaries'
IDENTITY = ROOT/'config/alpaca_symbol_identity_20261019.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def calendar_frame(records):
    frame = pd.DataFrame(records)
    frame['date'] = pd.to_datetime(frame.date)
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError('Invalid exchange calendar dates')
    for field in ['open', 'close']:
        text = frame.date.dt.strftime('%Y-%m-%d') + ' ' + frame[field]
        frame[field+'_utc'] = pd.to_datetime(text).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    frame['expected'] = ((frame.close_utc-frame.open_utc).dt.total_seconds()/900).astype(int)
    if not frame.expected.isin([14, 26]).all():
        raise ValueError('Unexpected session duration; review calendar')
    return frame.set_index('date')


def summarize_reference(frame, calendar):
    if frame.empty:
        return pd.DataFrame(columns=['date', 'observed', 'expected', 'complete', 'last_close']+FEATURES)
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp_utc, utc=True, format='ISO8601')
    frame['end'] = pd.to_datetime(frame.bar_end_utc, utc=True, format='ISO8601')
    frame['date'] = frame.timestamp.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    frame['session_open'] = frame.date.map(calendar.open_utc)
    frame['session_close'] = frame.date.map(calendar.close_utc)
    frame = frame[(frame.timestamp >= frame.session_open) & (frame.end <= frame.session_close)]
    if frame.timestamp.duplicated().any():
        raise ValueError('Duplicate bars in one ticker/chunk')
    result = []
    for date, group in frame.groupby('date', sort=True):
        group = group.sort_values('timestamp')
        expected = int(calendar.at[date, 'expected'])
        open_time, close_time = calendar.at[date, 'open_utc'], calendar.at[date, 'close_utc']
        grid = pd.date_range(open_time, close_time, freq='15min', inclusive='left')
        complete = len(group) == expected and np.array_equal(group.timestamp.to_numpy(), grid.to_numpy())
        row = dict(date=str(date.date()), observed=len(group), expected=expected, complete=bool(complete),
                   last_close=float(group.close.iloc[-1]), **{name: np.nan for name in FEATURES})
        row['intraday_coverage'] = len(group)/expected
        if complete:
            first, last = float(group.open.iloc[0]), float(group.close.iloc[-1])
            first_hour = group[group.end <= open_time+pd.Timedelta(hours=1)]
            last_hour = group[group.timestamp >= close_time-pd.Timedelta(hours=1)]
            total_volume = float(group.volume.sum())
            if total_volume <= 0:
                raise ValueError('Full session with no volume')
            vwap = float((group.vwap*group.volume).sum()/total_volume)
            row.update(rth_return=last/first-1, rth_range=float(group.high.max()/group.low.min()-1),
                       within_bar_realized_vol=float(np.sqrt(np.square(np.log(group.close/group.open)).sum())),
                       first_hour_return=float(first_hour.close.iloc[-1]/first-1),
                       last_hour_return=float(last/last_hour.open.iloc[0]-1),
                       late_volume_share=float(last_hour.volume.sum()/total_volume),
                       close_to_vwap=last/vwap-1)
        result.append(row)
    return pd.DataFrame(result, columns=['date', 'observed', 'expected', 'complete', 'last_close']+FEATURES)


def summarize(frame, calendar):
    """Vectorized equivalent of the explicit session-by-session reference above."""
    columns = ['date', 'observed', 'expected', 'complete', 'last_close']+FEATURES
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp_utc, utc=True, format='ISO8601')
    frame['end'] = pd.to_datetime(frame.bar_end_utc, utc=True, format='ISO8601')
    frame['date'] = frame.timestamp.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    frame['session_open'] = frame.date.map(calendar.open_utc)
    frame['session_close'] = frame.date.map(calendar.close_utc)
    frame = frame[(frame.timestamp>=frame.session_open)&(frame.end<=frame.session_close)].sort_values('timestamp')
    if frame.timestamp.duplicated().any():
        raise ValueError('Duplicate bars in one ticker/chunk')
    if frame.empty:
        return pd.DataFrame(columns=columns)
    ts = frame.timestamp.dt
    frame['on_grid'] = ((ts.minute%15==0)&(ts.second==0)&(ts.microsecond==0)&(ts.nanosecond==0)&
                        ((frame.end-frame.timestamp)==pd.Timedelta(minutes=15)))
    frame['squared_bar_return'] = np.square(np.log(frame.close/frame.open))
    frame['weighted_vwap'] = frame.vwap*frame.volume
    group = frame.groupby('date', sort=True)
    daily = group.agg(observed=('timestamp','size'), first_open=('open','first'), last_close=('close','last'),
                      maximum=('high','max'), minimum=('low','min'), volume=('volume','sum'),
                      squared_bar_return=('squared_bar_return','sum'),
                      weighted_vwap=('weighted_vwap','sum'), on_grid=('on_grid','all'))
    daily['expected'] = daily.index.map(calendar.expected)
    daily['complete'] = (daily.observed==daily.expected)&daily.on_grid
    if ((daily.volume<=0)&daily.complete).any():
        raise ValueError('Full session with no volume')
    first_hour = frame[frame.end<=frame.session_open+pd.Timedelta(hours=1)].groupby('date').close.last()
    last_hour = frame[frame.timestamp>=frame.session_close-pd.Timedelta(hours=1)].groupby('date')
    last_open, last_volume = last_hour.open.first(), last_hour.volume.sum()
    daily['rth_return'] = daily.last_close/daily.first_open-1
    daily['rth_range'] = daily.maximum/daily.minimum-1
    daily['within_bar_realized_vol'] = np.sqrt(daily.squared_bar_return)
    daily['first_hour_return'] = first_hour/daily.first_open-1
    daily['last_hour_return'] = daily.last_close/last_open-1
    daily['late_volume_share'] = last_volume/daily.volume
    daily['close_to_vwap'] = daily.last_close/(daily.weighted_vwap/daily.volume)-1
    daily['intraday_coverage'] = daily.observed/daily.expected
    daily.loc[~daily.complete, FEATURES[:-1]] = np.nan
    daily = daily.reset_index()
    daily['date'] = daily.date.dt.strftime('%Y-%m-%d')
    return daily[columns]


def load_summary(record, calendar):
    path = RAW/record['file']
    if sha(path) != record['sha256']:
        raise ValueError('Raw chunk hash mismatch')
    identity = record['sha256']+sha(CALENDAR)+sha(Path(__file__))
    key = hashlib.sha256(identity.encode()).hexdigest()
    cache, metadata = CACHE/(key+'.csv.gz'), CACHE/(key+'.json')
    if cache.exists() and metadata.exists():
        expected = json.loads(metadata.read_text())
        if sha(cache)==expected['sha256']:
            return pd.read_csv(cache, float_precision='round_trip')
    frame = pd.read_csv(path)
    symbol = record['query']['symbol']
    if len(frame)!=record['rows'] or (len(frame) and not frame.symbol.eq(symbol).all()):
        raise ValueError('Chunk row/symbol mismatch')
    daily = summarize(frame, calendar)
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
    daily.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(cache)
    temporary = metadata.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(sha256=sha(cache), source_sha256=record['sha256'])))
    temporary.replace(metadata)
    return daily


def cache_available():
    """Only precompute immutable completed chunks; never promote incomplete data."""
    calendar = calendar_frame(json.loads(CALENDAR.read_text()))
    records = [json.loads(p.read_text()) for p in sorted((RAW/'bars').glob('*/*.json'))]
    for number, record in enumerate(records, 1):
        load_summary(record, calendar)
        if number%40==0:
            print(f'Cached session summaries for {number}/{len(records)} available chunks', flush=True)
    print(f'Available-chunk cache complete: {len(records)}. Full data gate remains required.', flush=True)


def filter_identity(daily, symbol, rules):
    rule = rules.get(symbol, {})
    valid = pd.Series(True,index=daily.index)
    if 'first_valid_date' in rule:
        valid &= daily.date >= rule['first_valid_date']
    if 'end_valid_date' in rule:
        valid &= daily.date < rule['end_valid_date']
    return daily.loc[valid].copy(), daily.loc[~valid].copy()


def main():
    manifest = json.loads((RAW/'manifest.json').read_text())
    if manifest.get('download_status') != 'completed_with_review_required' or manifest['feed'] != 'sip':
        raise ValueError('Complete consolidated SIP download required before feature promotion')
    if manifest['completed_chunks'] != manifest['chunk_count']:
        raise ValueError('Incomplete chunk inventory')
    data = load_data()
    dates, tickers = data['panel'].dates, data['panel'].tickers
    daily_manifest = data['manifest']
    for name, expected in daily_manifest['hashes'].items():
        if sha(ROOT/daily_manifest['root']/name) != expected:
            raise ValueError('Changed daily input bytes')
    calendar = calendar_frame(json.loads(CALENDAR.read_text()))
    np.testing.assert_array_equal(calendar.index, dates)
    aliases = daily_manifest['alias_map']
    identity = json.loads(IDENTITY.read_text())
    result = np.full((len(dates), len(tickers), len(FEATURES)), np.nan, dtype='float32')
    result[..., -1] = 0.
    count = np.zeros((len(dates), len(tickers)), dtype='int16')
    complete = np.zeros_like(count, dtype=bool)
    seen, summaries, source_hashes, rejected = set(), [], {}, []
    for number, record in enumerate(manifest['chunks'], 1):
        path = RAW/record['file']
        source_hashes[record['file']] = record['sha256']
        symbol = record['query']['symbol']
        security = aliases.get(symbol, symbol)
        daily = load_summary(record, calendar)
        if symbol in identity['symbols'] and identity['symbols'][symbol]['security_id']!=security:
            raise ValueError('Identity rule disagrees with reviewed stable security ID')
        daily, outside_identity = filter_identity(daily,symbol,identity['symbols'])
        if len(outside_identity):
            rejected.append(outside_identity.assign(source_symbol=symbol,security_id=security,
                reason='Outside verified issuer/ticker validity interval; never a training input'))
        daily['source_symbol'], daily['security_id'] = symbol, security
        summaries.append(daily)
        if security in tickers:
            node = tickers.index(security)
            for row in daily.itertuples(index=False):
                key = (security, row.date)
                if key in seen:
                    raise ValueError('Overlapping canonical histories across aliases; identity review required')
                seen.add(key)
                day = dates.get_loc(pd.Timestamp(row.date))
                result[day, node] = [getattr(row, f) for f in FEATURES]
                count[day, node] = row.observed
                complete[day, node] = row.complete
        if number % 40 == 0:
            print(f'Validated {number}/{len(manifest["chunks"])} source chunks', flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    member_days = data['membership'] & data['tradable']
    active = member_days & (dates >= pd.Timestamp('2019-07-01'))[:, None]
    fraction = float(complete[active].mean())
    coverage = []
    for year in range(2019, 2026):
        annual = active & (dates.year == year)[:, None]
        for node, ticker in enumerate(tickers):
            eligible = annual[:, node]
            if eligible.any():
                coverage.append(dict(year=year, security_id=ticker, member_tradable_days=int(eligible.sum()),
                    complete_days=int(complete[eligible, node].sum()),
                    complete_fraction=float(complete[eligible, node].mean()),
                    average_coverage=float(result[eligible, node, -1].mean())))
    if OUT.exists() and (OUT/'REVIEW.json').exists():
        raise ValueError('Feature snapshot already reviewed; do not overwrite it')
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT/'session_summaries.csv.gz', index=False, compression='gzip')
    rejected_frame = pd.concat(rejected,ignore_index=True) if rejected else pd.DataFrame(columns=['date','source_symbol','security_id','reason'])
    rejected_frame.to_csv(OUT/'rejected_symbol_sessions.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage_by_security_year.csv', index=False)
    np.savez_compressed(OUT/'daily_features.npz', features=result,
        dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str), tickers=np.asarray(tickers, dtype=str),
        feature_names=np.asarray(FEATURES, dtype=str), observed_counts=count, complete=complete)
    review = dict(status='passed_research_input_gate' if fraction >= .90 else 'failed_insufficient_coverage',
        complete_member_session_fraction=fraction, member_tradable_days=int(active.sum()),
        complete_member_days=int(complete[active].sum()), raw_bars=manifest['rows'],
        source_chunks=len(source_hashes), calendar_sessions=len(calendar),
        early_close_sessions=int((calendar.expected==14).sum()),
        feature_sha256=sha(OUT/'daily_features.npz'), calendar_sha256=sha(CALENDAR),
        identity_rules_sha256=sha(IDENTITY),identity_rules=identity,
        rejected_symbol_sessions=len(rejected_frame),
        raw_manifest_sha256=sha(RAW/'manifest.json'), code_sha256=sha(Path(__file__)),
        source_hashes=source_hashes, source_aliases=aliases,
        no_eligibility_changes=True, missing_numeric_features='Training-only mean imputation, with observed fraction feature',
        limitations=['Uses prior reviewed security/alias identities; vendor bars are not independently issuer-certified.',
                     'Incomplete intraday sessions retained as missing features, never remove members or future outcomes.',
                     'Daily economic labels and corporate-action account remain unchanged.',
                     'Regular-session summaries are intraday-informed daily inputs, not a 15-minute trading policy.'])
    (OUT/'REVIEW.json').write_text(json.dumps(review, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in review.items() if k != 'source_hashes'}, indent=2), flush=True)
    if review['status'] != 'passed_research_input_gate':
        raise ValueError('Input coverage gate failed; augmented fitting is blocked')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-available', action='store_true')
    args = parser.parse_args()
    cache_available() if args.cache_available else main()
