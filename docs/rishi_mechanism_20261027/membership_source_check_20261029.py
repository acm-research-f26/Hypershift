"""Independently reconstruct monthly membership from the published YAML events."""
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml

root = Path('data/nasdaq_monthly_pit')
source = root / 'ndx_source'
preflight = root / 'ndx_preflight'
manifest = json.loads((preflight / 'manifest.json').read_text())
class LiteralTickers(yaml.SafeLoader):
    yaml_implicit_resolvers = {key: [(tag, pattern) for tag, pattern in rules if tag != 'tag:yaml.org,2002:bool']
                               for key, rules in copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers).items()}

annual = {}
events = []
for name, expected in manifest['source_hashes'].items():
    path = source.joinpath(*name.replace('\\', '/').split('/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    data = yaml.load(path.read_text(encoding='utf-8'), Loader=LiteralTickers)
    year = data['year']; initial = data['tickers_on_Jan_1']
    assert len(initial) == len(set(initial))
    current = set(initial)
    changes = []
    for date, change in sorted((data.get('changes') or {}).items(), key=lambda x: str(x[0])):
        when = pd.Timestamp(date)
        remove = set(change.get('difference') or []); add = set(change.get('union') or [])
        assert when.year == year and remove <= current and not add & current and not add & remove
        current = (current - remove) | add
        changes.append((when, remove, add))
        if year < 2026:
            events.append((year, str(when.date()), '|'.join(sorted(remove)), '|'.join(sorted(add))))
    annual[year] = (set(initial), current, changes)
for year in range(2016, 2026):
    assert annual[year][1] == annual[year + 1][0]
expected = []
for month in pd.date_range('2017-01-01', '2025-12-01', freq='MS'):
    initial, _, changes = annual[month.year]
    current = set(initial); latest = pd.Timestamp(month.year, 1, 1)
    for date, remove, add in changes:
        if date <= month:
            current = (current - remove) | add; latest = date
    expected.extend((str(month.date()), ticker, str(latest.date()), 'provisional_not_training_input') for ticker in sorted(current))
actual = pd.read_csv(preflight / 'monthly_members_PROVISIONAL.csv', dtype=str).fillna('')
assert list(actual.columns) == ['month', 'ticker', 'latest_effective_event', 'review_status']
assert list(actual.itertuples(index=False, name=None)) == expected
actual_events = pd.read_csv(preflight / 'event_review.csv', dtype=str).fillna('')
observed_events = [(int(row.year), row.effective_date, row.removed, row.added) for row in actual_events.itertuples(index=False)]
assert observed_events == events
print(f'{len(annual)} source years, 10 year-end closures, {len(events)} events, {len(expected)} member rows and 108 monthly snapshots exactly match')
