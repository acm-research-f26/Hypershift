"""Describe accounting on the three already identified monthly-only targets."""
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data, fold


def main():
    data = load_data()
    ids = np.concatenate([fold(data, year)[2] for year in (2022, 2023, 2024, 2025)])
    dates = [str(data['panel'].dates[data['entry'][i]].date()) for i in ids]
    cases = [('original_five', 'NTES', '2022-12-30'), ('original_five', 'WBA', '2024-07-23'),
             ('ten_seed', 'WBA', '2024-07-23')]
    root = Path('results/RISHI_PARENT_OVERLAY_TRACE_20261028')
    for group, name, fill in cases:
        k = dates.index(fill); next_fill = dates[k + 1]
        with np.load(root / f'after_hours_premarket__{group}__issuer5__vol20.npz', allow_pickle=False) as z:
            j = list(z['security_ids']).index(name)
            mask = (z['dates'] >= fill) & (z['dates'] < next_fill)
            dates_in_window = z['dates'][mask].tolist()
            exposure = z['exposure'][mask, j]; fees = z['fees'][mask, j]; net = z['net'][mask, j]
            assert len(dates_in_window) and dates_in_window[0] == fill
            print(group, name, fill, 'next_fill', next_fill, 'days', len(dates_in_window),
                  'initial_exposure', float(exposure[0]), 'max_exposure', float(exposure.max()),
                  'net_contribution_sum', float(net.sum()), 'fees_sum', float(fees.sum()))


if __name__ == '__main__': main()
