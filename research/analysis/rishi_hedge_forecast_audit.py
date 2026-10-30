"""Reconstruct published hedge diagnostics from saved forecasts and labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as source


def main():
    doc = Path('docs/ndx_model_hedge_20261027')
    saved = pd.read_csv(doc / 'signals.csv').set_index(['group', 'fill_date'])
    calibration = pd.read_csv(doc / 'calibration.csv').set_index(['group', 'year', 'family'])
    data = source.load_data()
    target = np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz', allow_pickle=False)
    np.testing.assert_array_equal(target['security_ids'], np.asarray(data['panel'].tickers))
    ids = np.concatenate([source.fold(data, year)[2] for year in (2022, 2023, 2024, 2025)])
    np.testing.assert_array_equal(target['ids'], ids)
    folders = {'think': Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),
               'dynamic': Path('runs/ndx_reference_networks_20261024/dynamic_restricted')}
    errors = []; flags = 0; detail_rows = 0; bias_count = 0
    for group, seeds in source.GROUPS.items():
        weights = target[f'after_hours_premarket__{group}__issuer5']
        offset = 0
        for year in (2022, 2023, 2024, 2025):
            _, val, test = source.fold(data, year)
            assert data['exit'][val].max() < data['origin'][test].min()
            predictions = []; biases = []
            for family, folder in folders.items():
                v = []; t = []
                for seed in seeds:
                    with np.load(folder / str(year) / f'{seed}_predictions.npz', allow_pickle=False) as z:
                        v.append(z['validation'].astype(float)); t.append(z['test'].astype(float))
                v = sum(v) / len(v); t = sum(t) / len(t)
                per_date = [np.mean(v[j, data['labelmask'][i]] - 100 * data['y'][i, data['labelmask'][i]].astype(float))
                            for j, i in enumerate(val)]
                bias = float(np.mean(per_date))
                errors.append(abs(bias - calibration.loc[(group, year, family), 'bias_pp']))
                predictions.append(t); biases.append(bias); bias_count += 1
            for j, i in enumerate(test):
                active = data['mask'][i]; w = weights[offset + j]; assert np.isclose(w.sum(), 1)
                date = str(data['panel'].dates[data['entry'][i]].date())
                row = saved.loc[(group, date)]
                assert row.year == year and row.signal_date == str(data['panel'].dates[data['origin'][i]].date())
                for prefix, pair in [('raw', predictions), ('cal', [predictions[k] - biases[k] for k in range(2)])]:
                    a, b = (x[j] for x in pair); blended = (a + b) / 2
                    values = {'mean_pp': blended[active].mean(), 'basket_pp': np.dot(blended, w),
                              'negative_breadth': np.mean(blended[active] < 0),
                              'think_mean_pp': a[active].mean(), 'dynamic_mean_pp': b[active].mean()}
                    for name, value in values.items():
                        errors.append(abs(float(row[prefix + '_' + name]) - value))
                    decisions = {'mean': values['mean_pp'] < 0, 'basket': values['basket_pp'] < 0,
                                 'breadth60': values['negative_breadth'] >= .6,
                                 'agreement': values['think_mean_pp'] < 0 and values['dynamic_mean_pp'] < 0}
                    for name, decision in decisions.items():
                        assert bool(row[prefix + '_' + name]) == decision
                        flags += 1
                detail_rows += 1
            offset += len(test)
    assert len(saved) == detail_rows == 603 and bias_count == 24 and flags == 4824
    assert max(errors) < 1e-12
    print(f'{bias_count} calibration values, {detail_rows} forecasts, {flags} flags verified; max absolute error {max(errors):.3g}')


if __name__ == '__main__':
    main()
