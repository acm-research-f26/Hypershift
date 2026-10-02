"""Portfolio and ranking metrics. SR follows STHAN-SR: mean/std(daily top-k return)*sqrt(252), no rf, no costs."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import ndcg_score


def _topk_sets(pred, mask, k):
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        yield d, idx[np.argsort(-pred[idx, d], kind="stable")[:k]]


def topk_daily_returns(pred, gt, mask, k=5) -> np.ndarray:
    out = np.zeros(pred.shape[1])
    for d, top in _topk_sets(pred, mask, k):
        out[d] = gt[top, d].mean() if len(top) else 0.0
    return out


def topk_daily_returns_net(pred, gt, mask, k, cost_bps) -> np.ndarray:
    """Equal-weight top-k; cost = 2 sides * bps * fraction of names replaced (first period = full)."""
    out, prev = np.zeros(pred.shape[1]), set()
    for d, top in _topk_sets(pred, mask, k):
        cur = set(top.tolist())
        turnover = 1.0 if not prev else len(cur - prev) / max(len(cur), 1)
        out[d] = (gt[top, d].mean() if len(top) else 0.0) - 2 * cost_bps * 1e-4 * turnover
        prev = cur
    return out


def daily_ic(pred, gt, mask) -> float:
    """Mean over days of the cross-sectional Pearson correlation of prediction and realised return (float64).
    A day whose predictions are constant (or whose correlation is not finite) counts as IC 0."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    ics = []
    for d in range(pred.shape[1]):
        i = mask[:, d] > 0.5
        p = pred[i, d]
        if i.sum() < 5 or np.ptp(p) == 0:
            ics.append(0.0)
            continue
        c = np.corrcoef(p, gt[i, d])[0, 1]
        ics.append(c if np.isfinite(c) else 0.0)
    return float(np.mean(ics))


def sharpe(r, periods_per_year=252) -> float:
    sd = float(np.std(r))
    return 0.0 if sd == 0 else float(np.mean(r)) / sd * math.sqrt(periods_per_year)


def irr(r) -> float:
    return float(np.sum(r))


def cumulative_return(r) -> float:
    return float(np.prod(1 + np.asarray(r)) - 1)


def max_drawdown(r) -> float:
    wealth = np.cumprod(1 + np.asarray(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1))


def ndcg_at_k(pred, gt, mask, k=5) -> float:
    """Mean over days of NDCG@k with relevance = true return shifted to be >= 0 that day."""
    vals = []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        if len(idx) < 2:
            continue
        rel = gt[idx, d] - gt[idx, d].min()
        if rel.max() <= 0:
            continue
        vals.append(ndcg_score(rel[None, :], pred[idx, d][None, :], k=k))
    return float(np.mean(vals)) if vals else 0.0


def ndcg_sthan_compat(pred, gt, mask, k=5) -> float:
    """Re-implementation of the (buggy) STHAN-SR evaluator: ticker-index sets, last day only. Reference only."""
    last = pred.shape[1] - 1
    idx = np.nonzero(mask[:, last] > 0.5)[0]
    gt_top = list(set(idx[np.argsort(-gt[idx, last], kind="stable")[:k]].tolist()))
    pr_top = list(set(idx[np.argsort(-pred[idx, last], kind="stable")[:k]].tolist()))
    if len(gt_top) != len(pr_top) or len(gt_top) < 2:
        return 0.0
    return float(ndcg_score(np.array(gt_top)[None, :], np.array(pr_top)[None, :]))


def masked_mse(pred, gt, mask) -> float:
    return float(np.sum(((pred - gt) * mask) ** 2) / max(np.sum(mask), 1.0))


def evaluate_all(pred, gt, mask, k=5, periods_per_year=252) -> dict:
    r = topk_daily_returns(pred, gt, mask, k)
    return {
        "sr": sharpe(r, periods_per_year),
        "irr": irr(r),
        "cumret": cumulative_return(r),
        "mdd": max_drawdown(r),
        "ann_vol": float(np.std(r) * math.sqrt(periods_per_year)),
        "ndcg5": ndcg_at_k(pred, gt, mask, k),
        "ndcg_sthan": ndcg_sthan_compat(pred, gt, mask, k),
        "mse": masked_mse(pred, gt, mask),
        "n_days": int(pred.shape[1]),
    }
