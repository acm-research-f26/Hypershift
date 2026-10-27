"""Independent dollar accounting summaries, cohorts and preregistered gates."""
import numpy as np

from experiments.strategy_search.statistics import daily_rows, risk, bootstrap
from .engine import ORDER_COLUMNS


def dollar_daily(path, sessions, capital):
    rows = []
    for sid in np.unique(sessions):
        pos = np.flatnonzero(sessions == sid)
        a, b = pos[0], pos[-1]
        initial = capital if a == 0 else path[a - 1, 1]
        rows.append([sid, path[b, 1] / initial - 1, path[b, 1],
                     path[a:b + 1, 7].sum(), path[a:b + 1, 8:12].sum(), path[a:b + 1, 12].sum()])
    return np.asarray(rows)


def summarize(m, path, orders, failures, capital):
    daily = dollar_daily(path, m['sessions'], capital)
    mask = np.isin(daily[:, 0], m['evaluate_ids'])
    pos = np.flatnonzero(np.isin(m['sessions'], m['evaluate_ids']))
    start = capital if pos[0] == 0 else path[pos[0] - 1, 1]
    selected = path[pos]
    metrics = risk(daily[mask, 1], np.r_[start, selected[:, :2].ravel()])
    if len(orders):
        evaluated = orders[np.isin(m['sessions'][orders[:, 0].astype(int)], m['evaluate_ids'])]
    else:
        evaluated = orders
    intended = abs(evaluated[:, 4] * evaluated[:, 6]) if len(evaluated) else np.array([])
    total = float(intended.sum())
    covered = float(intended[evaluated[:, 9] == 1].sum()) if total else 0.
    fills = abs(evaluated[:, 5]) if len(evaluated) else np.array([])
    costs = selected[:, 8:12].sum(axis=0)
    turnover = float(selected[:, 7].sum())
    metrics.update(accounting_valid=bool(failures == 0 and np.isfinite(path).all()),
                   failures=int(failures), intended_notional=total,
                   quote_coverage=covered / total if total else 0.,
                   executed_orders=int((fills > 0).sum()), intended_attempts=len(evaluated),
                   fill_ratio_shares=float(fills.sum() / abs(evaluated[:, 4]).sum()) if len(evaluated) else 0.,
                   turnover_dollars=turnover, costs_dollars=dict(zip(('commission', 'exchange', 'clearing', 'regulatory'), map(float, costs))),
                   borrow_dollars=float(selected[:, 12].sum()), adverse_slippage_dollars=float(selected[:, 13].sum()),
                   explicit_cost_bps=float(costs.sum() / turnover * 10000) if turnover else None,
                   mean_gross=float(selected[:, 3].mean()), maximum_constraint_error=float(selected[:, 14].max()),
                   uncertainty=bootstrap(daily[mask, 1]),
                   top_five_daily_return_sum=float(np.sort(daily[mask, 1])[-5:].sum()))
    return metrics, daily


def normalized_summary(m, path, orders, failures, capital):
    return summarize(m, path, orders, failures, capital)


def gate(rows, protocol):
    """Missing data, unverified source regions and absent trials never pass."""
    primary = {r['scope']: r for r in rows if r['scenario'] == 'primary' and r['mode'] == 'quotes'}
    needed = ('earlier_OOF', 'initial_validation', 'later_development64')
    if not all(s in primary for s in needed):
        return dict(status='inconclusive', reason='Missing primary cohort results')
    if any(primary[s]['metrics']['quote_coverage'] < .99 for s in needed):
        return dict(status='inconclusive', reason='Executable-price coverage below 99% of intended notional')
    blocks = [b for s in needed[:2] for b in primary[s]['blocks'].values()]
    qualified = (len(blocks) == 5 and all(b['Sharpe'] is not None for b in blocks)
                 and np.median([b['Sharpe'] for b in blocks]) >= 1
                 and all(primary[s]['metrics']['total_return'] > 0 for s in needed)
                 and primary[needed[2]]['metrics']['Sharpe'] is not None
                 and primary[needed[2]]['metrics']['Sharpe'] >= 1
                 and all(primary[s]['metrics']['accounting_valid'] for s in needed))
    stresses = {r['scope']: r for r in rows if r['scenario'] == 'adverse_0p5' and r['mode'] == 'quotes'}
    qualified &= all(s in stresses and stresses[s]['metrics']['accounting_valid']
                     and stresses[s]['metrics']['total_return'] > 0 for s in needed)
    return dict(status='advance_to_forward_evaluation' if qualified else 'failed',
                median_development_block_Sharpe=float(np.median([b['Sharpe'] for b in blocks]))
                    if blocks and all(b['Sharpe'] is not None for b in blocks) else None,
                reason='Preregistered historical criteria satisfied' if qualified else 'Execution, net performance or stress criteria failed',
                deployment_ready=False, forward_validation_complete=False)
