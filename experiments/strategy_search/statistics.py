"""Daily Sharpe and minute drawdown; training sessions never enter selection."""
import numpy as np


def risk(daily, curve=None):
    daily = np.asarray(daily, float)
    if not len(daily) or not np.isfinite(daily).all() or np.any(daily <= -1):
        return dict(Sharpe=None, total_return=None, maximum_drawdown=None)
    sd = daily.std(ddof=1) if len(daily) > 1 else 0.
    if curve is None:
        curve = np.r_[1., np.cumprod(1 + daily)]
    curve = np.asarray(curve) / curve[0]
    return dict(sessions=len(daily), Sharpe=float(daily.mean() / sd * np.sqrt(252)) if sd > 1e-15 else None,
                total_return=float(np.prod(1 + daily) - 1), annualized_volatility=float(sd * np.sqrt(252)),
                maximum_drawdown=float(np.max(1 - curve / np.maximum.accumulate(curve))), risk_free=0.)


def daily_rows(path, sessions):
    rows = []
    for sid in np.unique(sessions):
        pos = np.flatnonzero(sessions == sid)
        a, b = pos[0], pos[-1]
        initial = 1. if a == 0 else path[a - 1, 1]
        rows.append([sid, path[b, 1] / initial - 1, path[b, 1], path[a:b + 1, 7].sum() / initial,
                     path[a:b + 1, 8].sum() / initial, path[a:b + 1, 9].sum() / initial,
                     path[a:b + 1, 10].sum()])
    return np.array(rows)


def summarize(path, sessions, ids, daily=None):
    daily = daily_rows(path, sessions) if daily is None else daily
    selected_daily = daily[np.isin(daily[:, 0], ids)]
    pos = np.flatnonzero(np.isin(sessions, ids))
    selected = path[pos]
    initial = 1. if pos[0] == 0 else path[pos[0] - 1, 1]
    result = risk(selected_daily[:, 1], np.r_[initial, selected[:, :2].ravel()])
    valid = (np.isfinite(selected).all() and selected[:, 12].max() <= 1e-7
             and selected[:, 14].max() <= 1e-9 and selected[:, 15:17].sum() == 0)
    turnover_dollars = selected[:, 7].sum()
    result.update(accounting_valid=bool(valid), active=bool(selected[:, 10].sum() > 0),
                  executed_asset_orders=int(selected[:, 10].sum()), turnover=float(selected_daily[:, 3].sum()),
                  turnover_per_day=float(selected_daily[:, 3].mean()), mean_gross=float(selected[:, 3].mean()),
                  mean_net=float(selected[:, 4].mean()), mean_stale_gross=float(selected[:, 11].mean()),
                  max_stock=float(selected[:, 5].max()), max_sector=float(selected[:, 6].max()),
                  posttrade_constraint_error=float(selected[:, 12].max()),
                  liquidation_failures=int((selected[:, 14] > 1e-9).sum()), action_failures=int(selected[:, 16].sum()),
                  transaction_cost=float(selected[:, 8].sum() / initial), borrow_cost=float(selected[:, 9].sum() / initial),
                  break_even_cost_bps_fixed_trade_tape=float((selected[-1, 1] - initial + selected[:, 8].sum())
                                                           / turnover_dollars * 10000) if turnover_dollars > 0 else None,
                  # Fee-inclusive P&L plus charged execution fees: still retains borrow.
                  gross_edge_bps_per_traded_dollar=float((selected[-1, 1] - initial + selected[:, 8].sum())
                                                        / turnover_dollars * 10000) if turnover_dollars > 0 else None)
    return result


def scales(daily):
    result = np.zeros(len(daily))
    for i in range(20, len(daily)):
        sd = np.std(daily[i - 20:i, 1], ddof=1) * np.sqrt(252)
        result[i] = min(1., .1 / sd) if sd > 1e-12 else 1.
    return result


def bootstrap(daily, repetitions=2000):
    rng = np.random.default_rng(20261026)
    daily = np.asarray(daily)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, max(1, len(daily) - 4), size=int(np.ceil(len(daily) / 5)))
        ix = np.concatenate([np.arange(i, min(i + 5, len(daily))) for i in starts])[:len(daily)]
        r = daily[ix]
        if r.std(ddof=1) > 1e-15:
            estimates.append(r.mean() / r.std(ddof=1) * np.sqrt(252))
    return dict(method='non-circular moving blocks of five daily sessions', repetitions=repetitions,
                Sharpe_CI95=np.quantile(estimates, [.025, .975]).tolist() if estimates else None)
