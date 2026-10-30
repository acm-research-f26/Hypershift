"""Causal target-retention and continuous QQQ hedging for saved forecasts."""
import numpy as np

from research_engine import target_weights


def retention_targets(scores, eligible, k=5, bonus=.02):
    """Reward the preceding decision's positive targets, not realized positions.

    State starts empty and continues across year boundaries. Current eligibility
    always wins. Securities (including share classes) occupy separate slots;
    boundary ties use the existing fractional-allocation convention.
    """
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if (scores.ndim != 2 or scores.shape != eligible.shape
            or not isinstance(k, (int, np.integer)) or k < 1
            or not np.isfinite(bonus) or bonus < 0
            or not np.isfinite(scores[eligible]).all()
            or np.any(eligible.sum(axis=1) == 0)):
        raise ValueError('Invalid retention inputs')
    weights = np.zeros_like(scores)
    for row in range(len(scores)):
        adjusted = scores[row].copy()
        if row:
            adjusted = np.where(
                eligible[row], adjusted + bonus * (weights[row-1] > 0), adjusted)
        weights[row] = target_weights(adjusted, eligible[row], k, 'equal')
    return weights


def hedged_targets(weights, long_exposure, qqq_volatility, cap=.75):
    """Short min(cap, unused gross budget, 20%/QQQ annualized volatility)."""
    weights = np.asarray(weights, dtype=float)
    long_exposure = np.asarray(long_exposure, dtype=float)
    qqq_volatility = np.asarray(qqq_volatility, dtype=float)
    if (weights.ndim != 2 or long_exposure.shape != (len(weights),)
            or qqq_volatility.shape != long_exposure.shape
            or not np.isfinite(cap) or not 0 <= cap <= 1
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-12)
            or not np.isfinite(long_exposure).all()
            or np.any((long_exposure < 0) | (long_exposure > 1))
            or not np.isfinite(qqq_volatility).all()
            or np.any(qqq_volatility < 0)):
        raise ValueError('Invalid hedge inputs')
    short = np.minimum(np.minimum(cap, 1-long_exposure),
                       .20/np.maximum(qqq_volatility, .05))
    return np.column_stack((weights*long_exposure[:, None], -short))
