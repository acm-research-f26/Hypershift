"""Validation-only ensemble calibration and rank-dispersion controls."""
import numpy as np
import pandas as pd

from research_engine import ranking_metrics, target_weights
from ndx_retention import retention_targets

METHODS = ('equal', 'validation_ic', 'validation_stable', 'stable_consensus')
ALLOCATIONS = ('retain5', 'retain10', 'soft10')


def percentile_cube(predictions, eligible):
    predictions = np.asarray(predictions, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if (predictions.ndim != 3 or predictions.shape[1:] != eligible.shape
            or np.any(eligible.sum(axis=1) < 2)
            or not np.isfinite(predictions[:, eligible]).all()):
        raise ValueError('Invalid forecast cube')
    result = np.zeros_like(predictions)
    for seed in range(len(predictions)):
        for day, mask in enumerate(eligible):
            result[seed, day, mask] = pd.Series(predictions[seed, day, mask]).rank(
                method='average').to_numpy()/mask.sum()
    return result


def validation_diagnostics(ranks, actual, eligible, observed):
    """Compute per-date ranking metrics; unknown outcomes never enter metrics."""
    actual = np.asarray(actual, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if (ranks.ndim != 3 or ranks.shape[1:] != actual.shape
            or eligible.shape != actual.shape or observed.shape != actual.shape
            or np.any(observed & ~eligible) or np.any(observed.sum(axis=1) < 2)
            or not np.isfinite(actual[observed]).all()):
        raise ValueError('Invalid validation labels')
    diagnostics = []; ic = []
    for seed_ranks in ranks:
        summary, frame = ranking_metrics(seed_ranks, actual, eligible, observed, [5, 10])
        # A constant forecast contributes no ranking evidence, not a good score.
        values = np.nan_to_num(frame.RankIC.to_numpy(), nan=0.)
        quarters = np.array([part.mean() for part in np.array_split(values, 4)])
        diagnostics.append(dict(rankic=float(values.mean()),
            quarter_rankic_sd=float(quarters.std(ddof=1)),
            ndcg5=summary['NDCG@5'], ndcg10=summary['NDCG@10']))
        ic.append(values)
    return diagnostics, np.asarray(ic)


def validation_weights(ic, method):
    """Shrink quality weights 75% toward equal weighting; never select seeds."""
    ic = np.asarray(ic, dtype=float)
    if (ic.ndim != 2 or ic.shape[0] < 2 or ic.shape[1] < 20
            or not np.isfinite(ic).all() or method not in METHODS):
        raise ValueError('Invalid validation evidence')
    n = len(ic)
    if method == 'equal':
        return np.full(n, 1/n)
    quality = ic.mean(axis=1)
    if method in ('validation_stable', 'stable_consensus'):
        quarters = np.stack([part.mean(axis=1) for part in np.array_split(ic, 4, axis=1)], axis=1)
        quality = quality-quarters.std(axis=1, ddof=1)
    scale = max(float(quality.std()), .01)
    logits = np.clip((quality-quality.mean())/scale, -2., 2.)
    softmax = np.exp(logits-logits.max()); softmax /= softmax.sum()
    return .75/n + .25*softmax


def blended_scores(think, dynamic, think_weights, dynamic_weights, penalty=0.):
    if think.shape != dynamic.shape or penalty < 0 or not np.isfinite(penalty):
        raise ValueError('Invalid blend')
    means = []; moments = []
    for cube, weights in ((think, think_weights), (dynamic, dynamic_weights)):
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (len(cube),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError('Invalid seed weights')
        means.append(cube.mean(axis=0) if np.array_equal(weights, np.full(len(cube), 1/len(cube)))
                     else np.einsum('s,stn->tn', weights, cube))
        moments.append(np.einsum('s,stn->tn', weights, cube*cube))
    # Preserve baseline arithmetic (mean over seeds, then mean over families).
    mean = .5*means[0]+.5*means[1]
    sd = np.sqrt(np.maximum(.5*moments[0]+.5*moments[1]-mean*mean, 0.))
    return mean-penalty*sd


def allocate(scores, eligible, kind):
    if kind not in ALLOCATIONS:
        raise ValueError('Unknown allocation')
    if kind.startswith('retain'):
        return retention_targets(scores, eligible, int(kind[6:]), .02)
    result = np.zeros_like(scores, dtype=float)
    for day, mask in enumerate(eligible):
        base = target_weights(scores[day], mask, 10, 'equal')
        selected = base > 0
        # Tie-aware top ten, with gradual positive score weights instead of equal.
        raw = base.copy()
        raw[selected] *= np.exp(4*(scores[day, selected]-scores[day, selected].max()))
        result[day] = raw/raw.sum()
    return result
