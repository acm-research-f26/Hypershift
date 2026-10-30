"""Cross-check published hedge decision and summary tables; no account replay."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('docs/ndx_model_hedge_20261027')
signals = pd.read_csv(root / 'signals.csv')
metrics = pd.read_csv(root / 'metrics.csv')
uncertainty = pd.read_csv(root / 'uncertainty.csv')
names = ('mean', 'basket', 'breadth60', 'agreement')
for prefix in ('raw', 'cal'):
    expected = {
        'mean': signals[f'{prefix}_mean_pp'] < 0,
        'basket': signals[f'{prefix}_basket_pp'] < 0,
        'breadth60': signals[f'{prefix}_negative_breadth'] >= .60,
        'agreement': (signals[f'{prefix}_think_mean_pp'] < 0) & (signals[f'{prefix}_dynamic_mean_pp'] < 0),
    }
    for name in names:
        np.testing.assert_array_equal(signals[f'{prefix}_{name}'], expected[name])
assert len(signals) == 603 and not signals.duplicated(['group', 'fill_date']).any()
for group, rows in signals.groupby('group'):
    assert len(rows) == 201
    for name in names:
        assert not rows[f'raw_{name}'].any()
        np.testing.assert_array_equal(rows[f'cal_{name}'], rows.year == 2023)
assert len(metrics) == 495 and metrics.id.is_unique
for group in signals.group.unique():
    m = metrics[(metrics['group'] == group) & (metrics.cost_multiplier == 1) & (metrics.borrow_rate == .03)]
    baseline = m[m.recipe == 'baseline'].iloc[0]
    for recipe in [f'raw_{n}/{mode}' for n in names for mode in ('cash', 'hedge25', 'hedge50', 'switch50')]:
        row = m[m.recipe == recipe].iloc[0]
        for col in ('sharpe', 'annualized_return', 'annualized_volatility', 'maximum_drawdown', 'terminal_nav'):
            np.testing.assert_allclose(row[col], baseline[col], rtol=0, atol=1e-12)
primary = uncertainty[uncertainty.block_sessions == 20]
assert len(primary) == 120 and not (primary.simultaneous_active_low > 0).any()
print('603 decision rows, 4824 stored signal flags, 48 no-op account metric rows and 120 primary interval rows verified')
