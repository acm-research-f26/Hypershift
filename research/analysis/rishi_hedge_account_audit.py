"""Rebuild fixed published hedge targets and compare two account engines and metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as source
from ndx_drawdown_engine import daily_allocation, risk_inputs, exposure_series, market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq, simulate_signed
from audit_ndx_weak_market_short import value_replay
from rishi_mechanism_risk import POLICIES
from research_engine import performance


def main():
    doc = Path('docs/ndx_model_hedge_20261027')
    data = source.load_data()
    ids = np.concatenate([source.fold(data, year)[2] for year in (2022, 2023, 2024, 2025)])
    target = np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz', allow_pickle=False)
    np.testing.assert_array_equal(target['ids'], ids)
    np.testing.assert_array_equal(target['security_ids'], data['panel'].tickers)
    combined = append_qqq(data, etf_data(data, 'QQQ'))
    market = market_inputs(data)
    signals = pd.read_csv(doc / 'signals.csv')
    saved_targets = pd.read_csv(doc / 'targets.csv')
    metrics = pd.read_csv(doc / 'metrics.csv').set_index('id')
    max_day = 0.; max_metric = 0.; checked = 0
    for group in ('original_five', 'new_five', 'ten_seed'):
        w = target[f'after_hours_premarket__{group}__issuer5']
        risk = risk_inputs(data, daily_allocation(data, ids, w))
        long = exposure_series(POLICIES['vol20'], market, risk)[data['origin'][ids]]
        cap = .2 / np.maximum(market['qqqvol20'][data['origin'][ids]], .05)
        trend = (pd.Series(market['qqq']) < pd.Series(market['qqq']).rolling(200).mean()).to_numpy()[data['origin'][ids]]
        rows = signals[signals.group == group]
        assert len(rows) == len(ids)
        for recipe, flags in [('baseline', np.zeros(len(ids), dtype=bool)),
                              ('always/hedge50', np.ones(len(ids), dtype=bool)),
                              ('qqq_ma200/hedge50', trend),
                              ('cal_mean/hedge50', rows.cal_mean.to_numpy(dtype=bool))]:
            short = np.where(flags, np.minimum(np.minimum(.5, 1-long), cap), 0.)
            desired = np.column_stack((w*long[:, None], -short))
            recorded = saved_targets[(saved_targets.group == group) & (saved_targets.recipe == recipe)]
            assert recorded.fill_date.tolist() == [str(data['panel'].dates[data['entry'][i]].date()) for i in ids]
            np.testing.assert_allclose(recorded.long_target, desired[:, :-1].sum(1), rtol=0, atol=1e-12)
            np.testing.assert_allclose(recorded.short_target, short, rtol=0, atol=1e-12)
            calculated, path, _ = simulate_signed(combined, ids, desired, .03, 1.)
            independent = value_replay(combined, ids, desired, .03, 1.)
            delta = float(np.max(np.abs(independent-path))); max_day = max(max_day, delta)
            np.testing.assert_allclose(independent, path, rtol=0, atol=1e-12)
            key = f'{recipe}/{group}/c1/b0.03'; expected = metrics.loc[key]
            rebuilt = performance(independent)
            for name, value in rebuilt.items():
                error = abs(value-expected[name]); max_metric = max(max_metric, error)
                assert error < 1e-10, (key, name, error)
            checked += 1
    print(f'{checked} accounts: two engines, published targets and metrics match; max daily error {max_day:.3g}, max metric error {max_metric:.3g}')
    print('Published daily hedge paths absent locally; this checks independently rebuilt paths and published aggregate metrics.')


if __name__ == '__main__':
    main()
