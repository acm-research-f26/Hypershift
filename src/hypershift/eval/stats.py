from __future__ import annotations

import numpy as np
from scipy import stats as sps

from hypershift.eval.metrics import sharpe


def wilcoxon_paired(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0):
        return 1.0
    return float(sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def wilcoxon_one_sample(x) -> float:
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    return float(sps.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < 1.0 / mean_block
    fresh = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = fresh[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def sharpe_contrast_ci(series, weights, mean_block=10, n_boot=5000, alpha=0.05, seed=0, periods_per_year=252) -> dict:
    """CI for sum_i w_i * Sharpe(series_i), resampling the same day-blocks for all series (keeps pairing)."""
    series = [np.asarray(s, float) for s in series]
    n = min(len(s) for s in series)
    series = [s[-n:] for s in series]                      # align on the last n days if lengths differ
    est = float(sum(w * sharpe(s, periods_per_year) for w, s in zip(weights, series)))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        boots[b] = sum(w * sharpe(s[idx], periods_per_year) for w, s in zip(weights, series))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"est": est, "lo": float(lo), "hi": float(hi), "p_boot": float(min(p, 1.0))}


def sharpe_diff_ci(ra, rb, **kw) -> dict:
    return sharpe_contrast_ci([ra, rb], [1.0, -1.0], **kw)


def verdict(p_holm: float, lo: float, hi: float) -> str:
    if p_holm < 0.01 and (lo > 0 or hi < 0):
        return "STRONG"
    if p_holm < 0.01:
        return "SEED-ROBUST ONLY"
    return "NO EVIDENCE"
