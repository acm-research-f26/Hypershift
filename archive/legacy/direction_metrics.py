"""Explicit classification, ranking, and delayed-execution backtest definitions."""
import numpy as np
import pandas as pd


def classification(y, p, threshold=0.5):
    y, p = np.asarray(y).ravel().astype(bool), np.asarray(p).ravel()
    predicted = p >= threshold
    tp, tn = int(np.sum(y & predicted)), int(np.sum(~y & ~predicted))
    fp, fn = int(np.sum(~y & predicted)), int(np.sum(y & ~predicted))
    divide = lambda a, b: float(a / b) if b else 0.0
    up_f1 = divide(2 * tp, 2 * tp + fp + fn)
    down_f1 = divide(2 * tn, 2 * tn + fp + fn)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    positives, negatives = int(y.sum()), int((~y).sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = ((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives)
           if positives and negatives else None)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {'accuracy': float((y == predicted).mean()),
            'macro_f1': (up_f1 + down_f1) / 2, 'up_f1': up_f1, 'down_f1': down_f1,
            'balanced_accuracy': (divide(tp, positives) + divide(tn, negatives)) / 2
                                 if positives and negatives else None,
            'mcc': divide(tp * tn - fp * fn, denominator),
            'roc_auc': float(auc) if auc is not None else None,
            'brier': float(np.mean((p - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(clipped) + (~y) * np.log(1 - clipped))),
            'actual_up_fraction': float(y.mean()), 'predicted_up_fraction': float(predicted.mean()),
            'confusion': {'true_up_pred_up': tp, 'true_down_pred_down': tn,
                          'true_down_pred_up': fp, 'true_up_pred_down': fn}, 'count': len(y)}


def select_threshold(y, p):
    # Fixed grid, validation data only. Ties favor the threshold nearest 0.5.
    grid = np.round(np.arange(0.2, 0.801, 0.025), 3)
    return float(max(grid, key=lambda t: (classification(y, p, t)['macro_f1'], -abs(t - 0.5))))


def tie_averaged_dcg(gains, scores, k):
    order = np.argsort(-scores, kind='stable')
    gains, scores = np.asarray(gains)[order], np.asarray(scores)[order]
    discounts = 1 / np.log2(np.arange(len(scores)) + 2)
    discounts[k:] = 0
    value, start = 0.0, 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        value += gains[start:end].mean() * discounts[start:end].sum()
        start = end
    return float(value)


def ndcg(actual_return_pp, scores, k=3):
    """Linear gain=max(realized return,0), log2 discounts, expected tied DCG.

    Dates with no positive return have zero ideal DCG and are excluded/reported.
    This declared choice is NOT a recovered setting from the THINK experiments.
    """
    values = []
    for actual, score in zip(actual_return_pp, scores):
        gain = np.maximum(actual, 0)
        ideal = tie_averaged_dcg(gain, gain, k)
        if ideal > 0:
            values.append(tie_averaged_dcg(gain, score, k) / ideal)
    return {'ndcg_at_k': float(np.mean(values)) if values else None,
            'k': k, 'dates_used': len(values), 'dates_excluded': len(scores) - len(values)}


def topk_weights(scores, k):
    """Equal top-k allocation, averaging allocations over ties at the cutoff.

    All-equal scores produce an equal-weight basket of all stocks. This avoids
    making a constant classifier look good merely because of ticker ordering.
    """
    if not 1 <= k <= len(scores):
        raise ValueError('k is outside the stock universe.')
    boundary = np.sort(scores)[-k]
    above, equal = scores > boundary, scores == boundary
    weights = above.astype(float)
    weights[equal] = (k - above.sum()) / equal.sum()
    return weights / k


def trading_backtest(prices, dates, scores, horizon, k=3, cost_bps=10):
    """Signal after close t, enter close t+1, exit close t+1+horizon.

    Fully liquidate each basket. New signal every horizon+1 sessions, giving one
    cash session between holdings. No overlapping capital, shorts, or leverage.
    Costs charged at both entry and exit. Risk-free rate is explicitly zero.
    """
    if horizon < 1 or not 0 <= cost_bps < 10000:
        raise ValueError('Invalid horizon or transaction cost.')
    values, positions = prices.to_numpy(), prices.index.get_indexer(dates)
    cost, net_nav, gross_nav = cost_bps / 10000, 1.0, 1.0
    net_curve, rows = [1.0], []
    for i in range(0, len(dates), horizon + 1):
        origin = positions[i]
        entry, exit_ = origin + 1, origin + 1 + horizon
        if origin < 0:
            raise ValueError('Signal date is missing from prices.')
        if exit_ >= len(prices):
            continue
        weights = topk_weights(np.asarray(scores[i]), k)
        path = (values[entry:exit_ + 1] / values[entry]) @ weights
        gross_return = float(path[-1] - 1)
        net_return = (1 - cost) ** 2 * (1 + gross_return) - 1
        held_nav = net_nav * (1 - cost) * path
        held_nav[-1] *= 1 - cost
        net_curve.extend(held_nav.tolist())
        net_nav, gross_nav = float(held_nav[-1]), gross_nav * (1 + gross_return)
        rows.append({'signal_date': str(dates[i].date()),
                     'entry_date': str(prices.index[entry].date()),
                     'exit_date': str(prices.index[exit_].date()),
                     'gross_return': gross_return, 'net_return': net_return,
                     'net_nav': net_nav, 'gross_nav': gross_nav,
                     'holdings': ';'.join(f'{ticker}:{w:.5f}' for ticker, w in zip(prices.columns, weights) if w)})
    returns = np.array([row['net_return'] for row in rows])
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std) if std > 1e-12 else None
    curve = np.asarray(net_curve)
    return {'periods': len(rows), 'gross_total_return': gross_nav - 1,
            'net_total_return': net_nav - 1, 'net_period_sharpe_rf0': sharpe,
            'net_annualized_sharpe_rf0': sharpe * np.sqrt(252 / (horizon + 1)) if sharpe is not None else None,
            'net_daily_close_max_drawdown': float(np.min(curve / np.maximum.accumulate(curve) - 1)),
            'cost_bps_per_side': cost_bps}, rows
