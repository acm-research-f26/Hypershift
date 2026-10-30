"""Identify next-session membership changes used by the origin-time eligibility mask."""
import numpy as np
import pandas as pd
from pit_research_engine import load_data, fold


def main():
    data = load_data()
    ids = np.concatenate([fold(data, y)[2] for y in (2022, 2023, 2024, 2025)])
    saved = np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz', allow_pickle=False)
    np.testing.assert_array_equal(ids, saved['ids'])
    names = np.asarray(data['panel'].tickers)
    np.testing.assert_array_equal(names, saved['security_ids'])
    changes = []; selected = []
    for j, i in enumerate(ids):
        origin, entry = data['origin'][i], data['entry'][i]
        old = data['membership'][origin]; new = data['membership'][entry]
        entered = new & ~old; exited = old & ~new
        if not (entered.any() or exited.any()): continue
        d0 = str(data['panel'].dates[origin].date()); d1 = str(data['panel'].dates[entry].date())
        for node in np.flatnonzero(entered | exited):
            changes.append((d0, d1, names[node], 'entered' if entered[node] else 'exited', bool(data['mask'][i, node])))
        for group in ('original_five', 'new_five', 'ten_seed'):
            w = saved[f'after_hours_premarket__{group}__issuer5'][j]
            for node in np.flatnonzero(entered & (w > 0)):
                selected.append((d0, d1, group, names[node], float(w[node])))
    print('test origins', len(ids), 'origins with changes', len({x[0] for x in changes}),
          'changed security rows', len(changes), 'eligible entrants', sum(x[3]=='entered' and x[4] for x in changes),
          'selected entrant rows', len(selected))
    for row in changes: print('CHANGE', *row)
    for row in selected: print('SELECTED', *row)


if __name__ == '__main__':
    main()
