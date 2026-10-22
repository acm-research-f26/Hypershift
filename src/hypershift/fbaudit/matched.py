"""Matched-control tools for the research-closure review (plan 2026-10-22-matched-closure, M1/M3/M4)."""
import numpy as np
from scipy.stats import rankdata

from hypershift.featbench.data import FEATURES, transform
from hypershift.fbaudit.stats import spearman_daily

STYLE_COLS = ('beta', 'log_dvol_20', 'ret_60', 'vol_60')


def nested_inputs(fb, signals, which):
    """[S, N, F]: cs_z of the chosen FB feature columns + sector one-hot. 'style' columns are a subset of 'full'."""
    cols = FEATURES if which == 'full' else list(STYLE_COLS)
    idx = [FEATURES.index(c) for c in cols]
    xs = fb.X[:, signals][..., idx].transpose(1, 0, 2)
    z = transform(xs, fb.eligible[:, signals].T, 'cs_z', fb)
    sec = (fb.sectors[:, None] == np.unique(fb.sectors)[None]).astype(np.float32)
    return np.concatenate([z, np.broadcast_to(sec, (len(signals),) + sec.shape)], -1)


def encompass(full, base, y, mask):
    """Per-date Spearman of (rank(full) residualised on [1, rank(base)]) vs y: information in full beyond base."""
    res = np.zeros(full.shape, dtype=np.float64)
    for d in range(len(full)):
        m = mask[d]
        if m.sum() < 3:
            continue
        rf, rb = rankdata(full[d, m]), rankdata(base[d, m])
        A = np.column_stack([np.ones(m.sum()), rb])
        c, *_ = np.linalg.lstsq(A, rf, rcond=None)
        res[d, m] = rf - A @ c
    return spearman_daily(res, y, mask)


def synth_panel(scenario, seed, N=400, T=1250, n_sectors=10, b=.15, noise=10.):
    """Synthetic cross-section (M4). X [N, T, 10]: 4 persistent styles (AR(1) 0.98) then 6 other features (AR(1) 0.5).

    y [N, T] = f(styles) [+ b * x1 in scenario 'I'] + noise, where f is nonlinear with interactions:
    tanh(s1*s2) + (s3^2 - 1) * sign(s4) + c_sector * s4, standardised per date. X is identical across scenarios.
    """
    rng = np.random.default_rng(seed)
    sectors = rng.integers(n_sectors, size=N)
    X = np.zeros((N, T, 10))
    phi = np.r_[[.98] * 4, [.5] * 6]
    X[:, 0] = rng.normal(size=(N, 10))
    for t in range(1, T):
        X[:, t] = phi * X[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=(N, 10))
    c = rng.choice([-1., 1.], size=n_sectors)
    s1, s2, s3, s4 = (X[..., k] for k in range(4))
    f = np.tanh(s1 * s2) + (s3 ** 2 - 1) * np.sign(s4) + c[sectors][:, None] * s4
    f = (f - f.mean(0)) / f.std(0)
    eps = np.random.default_rng((seed, 1)).normal(size=(N, T)) * noise
    y = f + eps + (b * X[..., 4] if scenario == 'I' else 0.)
    return {'X': X, 'y': y, 'sectors': sectors, 'mask': np.ones((N, T), bool)}


def permute_within_bins(X, base, mask, bins=20, seed=0):
    """Conditional-permutation control: per date, permute rows of X [S, N, k] jointly among masked stocks that fall in
    the same quantile bin of `base` (e.g. the style-only prediction). Keeps the input dimension and, approximately,
    the association of X with style; destroys stock-specific information in X."""
    rng = np.random.default_rng(seed)
    out = np.array(X, copy=True)
    for d in range(len(X)):
        ix = np.flatnonzero(mask[d])
        if len(ix) < 2:
            continue
        r = np.argsort(np.argsort(base[d, ix], kind='stable'), kind='stable')
        b = np.minimum(r * bins // len(ix), bins - 1)
        for g in range(bins):
            sel = ix[b == g]
            if len(sel) > 1:
                out[d, sel] = X[d, rng.permutation(sel)]
    return out
