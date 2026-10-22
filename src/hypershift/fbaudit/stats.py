"""Fresh statistics for the FB audit: per-date Spearman, bootstrap variants, Newey-West, pooled fold means."""
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

FOLDS = (2019, 2020, 2021, 2022, 2023)


def spearman_daily(pred, y, mask, minimum=20):
    out = np.zeros(len(pred))
    for d in range(len(pred)):
        m = mask[d]
        if m.sum() < minimum:
            continue
        a, b = rankdata(pred[d, m]), rankdata(y[d, m])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out[d] = (a * b).sum() / den if den > 0 else 0.
    return out


def stationary_means(x, block, B, seed):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    restart = rng.random((B, n)) < 1. / block
    restart[:, 0] = True
    fresh = rng.integers(n, size=(B, n))
    t = np.arange(n)[None]
    start = np.maximum.accumulate(np.where(restart, t, 0), 1)
    idx = (np.take_along_axis(fresh, start, 1) + t - start) % n
    return x[idx].mean(1)


def mbb_means(x, block, B, seed):
    """Circular moving-block bootstrap with fixed block length."""
    x = np.asarray(x, float)
    n = len(x)
    block = max(1, min(int(block), n))
    k = -(-n // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(B, k))
    idx = ((starts[..., None] + np.arange(block)) % n).reshape(B, -1)[:, :n]
    return x[idx].mean(1)


def nw_se(x, lags):
    """Newey-West (Bartlett) standard error of the mean."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(np.sqrt(max(v, 0) / n))


def pooled(series_by_fold, method='stationary', block=10, B=5000, seed=0):
    """Pooled = mean of fold means. Bootstrap methods average fold bootstrap means; 'nw' combines fold SEs."""
    per = {f: float(np.mean(s)) for f, s in series_by_fold.items()}
    m = float(np.mean(list(per.values())))
    k = len(per)
    if method == 'nw':
        se = float(np.sqrt(sum(nw_se(s, block) ** 2 for s in series_by_fold.values())) / k)
        z95, z90 = norm.ppf(.975), norm.ppf(.95)
        return {'pooled': m, 'se': se, 'ci95': (m - z95 * se, m + z95 * se), 'ci90': (m - z90 * se, m + z90 * se),
                'p': float(2 * norm.sf(abs(m) / max(se, 1e-12))), 'per_fold': per}
    f = stationary_means if method == 'stationary' else mbb_means
    boots = np.mean([f(s, block, B, (seed, i)) for i, s in enumerate(series_by_fold.values())], 0)
    return {'pooled': m, 'se': float(boots.std(ddof=1)), 'ci95': tuple(np.quantile(boots, [.025, .975])),
            'ci90': tuple(np.quantile(boots, [.05, .95])), 'p': float(np.mean(np.abs(boots - m) >= abs(m))),
            'per_fold': per}


def nonoverlap(x, h):
    return [float(np.mean(x[o::h])) for o in range(h)]


def _dir(root, year, arm, seed):
    return Path(root) / str(year) / arm.replace(':', '__') / f'seed_{seed}'


def load_daily(root, year, arm, seed=7, key='neutral_RankIC'):
    z = np.load(_dir(root, year, arm, seed) / 'daily.npz')
    return np.asarray(z[key], float), np.asarray(z['dates']).astype(str)


def load_pred(root, year, arm, seed=7):
    z = np.load(_dir(root, year, arm, seed) / 'predictions.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}


def paired_by_fold(root, a, b=None, key='neutral_RankIC', seed_a=7, seed_b=7, folds=FOLDS):
    out = {}
    for f in folds:
        va, da = load_daily(root, f, a, seed_a, key)
        if b is None:
            out[f] = va
            continue
        vb, db = load_daily(root, f, b, seed_b, key)
        ib = {d: i for i, d in enumerate(db)}
        keep = [(i, ib[d]) for i, d in enumerate(da) if d in ib]
        out[f] = np.array([va[i] - vb[j] for i, j in keep])
    return out
