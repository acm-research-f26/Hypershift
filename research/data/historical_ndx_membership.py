"""Validate and materialize research-grade historical Nasdaq-100 membership.

This is the user-authorized index fallback, not a historical market-cap ranking.
Raw third-party files remain unmodified. Output is PROVISIONAL until index events,
publication timing, stable identities, prices and corporate actions are reviewed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml

SOURCE = Path('data/nasdaq_monthly_pit/ndx_source')


class TickerLoader(yaml.SafeLoader):
    """Preserve ticker ON: YAML 1.1 otherwise interprets it as boolean True."""
    yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in resolvers
              if tag != 'tag:yaml.org,2002:bool']
        for key, resolvers in copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers).items()
    }


def read_year(path):
    data = yaml.load(path.read_text(encoding='utf-8'), Loader=TickerLoader)
    initial = data['tickers_on_Jan_1']
    if len(initial) != len(set(initial)):
        raise ValueError(f'Duplicate initial member in {path.name}')
    if not 100 <= len(initial) <= 110:
        raise ValueError(f'Unexpected initial security count in {path.name}')
    events = []
    names = set(initial)
    for when, event in sorted((data.get('changes') or {}).items(), key=lambda item: str(item[0])):
        date = pd.Timestamp(when)
        if date.year != data['year']:
            raise ValueError('Event assigned to wrong year')
        remove, add = set(event.get('difference') or []), set(event.get('union') or [])
        if remove - names:
            raise ValueError(f'Removal of absent member: {date}: {remove - names}')
        if add & names:
            raise ValueError(f'Addition of existing member: {date}: {add & names}')
        if remove & add:
            raise ValueError('Same ticker removed and added in one event: identity review needed')
        names.difference_update(remove)
        names.update(add)
        if not 100 <= len(names) <= 110:
            raise ValueError(f'Unexpected intermediate security count: {date}: {len(names)}')
        events.append(dict(effective_date=date, removed=sorted(remove), added=sorted(add)))
    return dict(year=data['year'], initial=set(initial), final=names, events=events)


def monthly_members(years, first='2017-01-01', last='2025-12-01'):
    rows = []
    for month in pd.date_range(first, last, freq='MS'):
        year = years[month.year]
        names = set(year['initial'])
        latest = pd.Timestamp(month.year, 1, 1)
        for event in year['events']:
            if event['effective_date'] <= month:
                names.difference_update(event['removed'])
                names.update(event['added'])
                latest = event['effective_date']
        rows.extend(dict(month=month.date().isoformat(), ticker=ticker,
                         latest_effective_event=latest.date().isoformat(),
                         review_status='provisional_not_training_input') for ticker in sorted(names))
    return pd.DataFrame(rows)


def build(source=SOURCE, out=Path('data/nasdaq_monthly_pit/ndx_preflight')):
    root = source / 'src/nasdaq_100_ticker_history'
    paths = [root / f'n100-ticker-changes-{y}.yaml' for y in range(2016, 2027)]
    years = {y: read_year(p) for y, p in zip(range(2016, 2027), paths)}
    for y in range(2016, 2026):
        if years[y]['final'] != years[y+1]['initial']:
            raise ValueError(f'Year-end closure mismatch: {y}')
    members = monthly_members(years)
    out.mkdir(parents=True, exist_ok=True)
    members.to_csv(out / 'monthly_members_PROVISIONAL.csv', index=False)
    events = [dict(year=y, effective_date=e['effective_date'].date().isoformat(),
                   removed='|'.join(e['removed']), added='|'.join(e['added']),
                   publication_date='', primary_source='', review_status='unreviewed')
              for y in range(2016, 2026) for e in years[y]['events']]
    pd.DataFrame(events).to_csv(out / 'event_review.csv', index=False)
    # Header presence does not prove price coverage or ticker identity.
    cache = set()
    for directory in ['recent_cohorts', 'largecap_20261010', 'random_trials_20261010']:
        cache.update(pd.read_csv(Path('data') / directory / 'adj_close.csv', nrows=0).columns[1:])
    inventory = members.groupby('ticker').month.agg(['min', 'max']).reset_index()
    inventory.columns = ['ticker', 'first_month', 'last_month']
    inventory['present_in_existing_cache_header'] = inventory.ticker.isin(cache)
    inventory['identity_reviewed'] = False
    inventory['complete_price_history_reviewed'] = False
    inventory.to_csv(out / 'price_identity_review.csv', index=False)
    manifest = dict(status='provisional_membership_only; training_not_started',
                    source='https://github.com/jmccarrell/n100tickers',
                    source_git_tree=json.loads((source / 'tree.json').read_text())['sha'],
                    monthly_snapshots=int(members.month.nunique()),
                    distinct_historical_symbols=int(members.ticker.nunique()),
                    test_symbols=int(members[members.month >= '2022-01-01'].ticker.nunique()),
                    securities_per_month=dict(min=int(members.groupby('month').size().min()),
                                              max=int(members.groupby('month').size().max())),
                    annual_closure_checks=10,
                    source_hashes={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    notes=['Index has 100 companies but can have more than 100 securities owing to share classes and spin-offs.',
                           'Monthly snapshots use only events effective by the month start; publication dates still require verification.',
                           'Contemporaneous ticker names are retained; no implicit current-symbol substitution.',
                           'Sources disagree on 2025 SOLS effective dates; this conflict remains open.',
                           'Membership continuity checks do not certify historical accuracy or price identity.',
                           'Do not discard missing former members, substitute acquirers, or carry disappeared prices indefinitely.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_hashes'}, indent=2))
    return members, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=Path('data/nasdaq_monthly_pit/ndx_preflight'))
    args = parser.parse_args()
    build(args.source, args.out)
