"""Describe monthly membership against the pinned source's daily event dates."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import yaml
from pit_research_engine import load_data, fold


class LiteralTickers(yaml.SafeLoader):
    yaml_implicit_resolvers = {k: [(tag, pattern) for tag, pattern in v if tag != 'tag:yaml.org,2002:bool']
                               for k, v in copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers).items()}


def main():
    data = load_data(); names = np.asarray(data['panel'].tickers); aliases = data['manifest']['alias_map']
    ids = np.concatenate([fold(data, y)[2] for y in (2022, 2023, 2024, 2025)])
    saved = np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz', allow_pickle=False)
    np.testing.assert_array_equal(ids, saved['ids']); np.testing.assert_array_equal(names, saved['security_ids'])
    root = Path('data/nasdaq_monthly_pit/ndx_source/src/nasdaq_100_ticker_history')
    years = {y: yaml.load((root / f'n100-ticker-changes-{y}.yaml').read_text(encoding='utf-8'), Loader=LiteralTickers)
             for y in (2022, 2023, 2024, 2025)}
    by_year = {}; differing = 0; eligible_extra = 0; missing_in_axis = set(); selected = {g: 0 for g in ('original_five','new_five','ten_seed')}
    for j, i in enumerate(ids):
        day = data['panel'].dates[data['entry'][i]]; annual = years[day.year]
        current = set(annual['tickers_on_Jan_1'])
        for date, event in sorted((annual.get('changes') or {}).items(), key=lambda v: str(v[0])):
            if pd.Timestamp(date) > day: break
            current.difference_update(event.get('difference') or []); current.update(event.get('union') or [])
        current = {aliases.get(name, name) for name in current}
        missing_in_axis.update(current - set(names))
        source_members = np.isin(names, list(current)); monthly = data['membership'][data['entry'][i]]
        extra = monthly & ~source_members; missing = source_members & ~monthly
        eligible = extra & data['mask'][i]
        changing = bool(extra.any() or missing.any()); differing += changing; eligible_extra += int(eligible.sum())
        row = by_year.setdefault(day.year, dict(origins=0, differing=0, extra=0, missing=0, eligible_extra=0))
        row['origins'] += 1; row['differing'] += changing; row['extra'] += int(extra.sum()); row['missing'] += int(missing.sum()); row['eligible_extra'] += int(eligible.sum())
        for group in selected:
            weights = saved[f'after_hours_premarket__{group}__issuer5'][j]
            selected[group] += int(np.count_nonzero(extra & (weights > 0)))
            for node in np.flatnonzero(extra & (weights > 0)):
                print('monthly-only selection', str(day.date()), group, names[node], float(weights[node]))
    for year, row in by_year.items(): print(year, row)
    print('differing origins', differing, 'eligible monthly-only origin/security rows', eligible_extra,
          'selected monthly-only target rows', selected, 'source names outside model axis', sorted(missing_in_axis))


if __name__ == '__main__': main()
