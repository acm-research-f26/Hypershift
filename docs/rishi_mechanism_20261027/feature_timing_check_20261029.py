"""Check source summary cutoffs and all saved 16-session decision windows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('data/nasdaq_monthly_pit')
cal = pd.DataFrame(json.loads((root / 'alpaca_15min/calendar_2018_2025.json').read_text()))
dates = pd.DatetimeIndex(pd.to_datetime(cal.date))
assert dates.is_unique and dates.is_monotonic_increasing
close = pd.to_datetime(cal.date + ' ' + cal.close).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
summary = pd.read_csv(root / 'extended_hours_features_20261024/summaries.csv.gz', usecols=['date', 'segment', 'latest_bar_end'])
end = pd.to_datetime(summary.latest_bar_end, utc=True)
deadline = pd.to_datetime(summary.date + ' ' + summary.segment.map({'after_hours': '20:00', 'premarket': '09:15'})).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
assert summary.segment.isin(['after_hours', 'premarket']).all() and (end <= deadline).all()
last = summary.assign(last=end).groupby(['segment', 'date'])['last'].max()
timing = pd.read_csv('docs/ndx_extended_hours_20261024/TIMING.csv')
slots = 0; observed = 0; early = 0
for row in timing.itertuples(index=False):
    origin = int(row.origin); entry = int(row.entry)
    assert entry == origin + 1 and origin >= 15
    assert row.prior_regular_close == str(dates[origin].date())
    day = dates[entry]
    cutoff = pd.Timestamp(row.feature_cutoff_utc); decision = pd.Timestamp(row.decision_utc); fill = pd.Timestamp(row.fill_utc)
    assert cutoff == pd.Timestamp(str(day.date()) + ' 09:15', tz='America/New_York').tz_convert('UTC')
    assert decision - cutoff == pd.Timedelta(minutes=5) and fill == close.iloc[entry] and decision < fill
    early += int(cal.close.iloc[entry] != '16:00')
    for segment, indices in [('after_hours', range(origin-15, origin+1)), ('premarket', range(origin-14, origin+2))]:
        for i in indices:
            key = (segment, str(dates[i].date())); slots += 1
            if key in last.index:
                assert last.loc[key] <= cutoff
                observed += 1
assert len(timing) == 350 and slots == 11200
print(f'{len(summary)} summary rows, {len(timing)} decisions, {slots} feature-date slots ({observed} with bars), {early} early-close entries; all declared cutoffs pass')
