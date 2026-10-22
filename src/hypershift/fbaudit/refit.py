"""Matched refits for the FB audit: FB models with one thing changed (target kind or fixed rounds).

Uses featbench code read-only: models._features (cs_z inputs), models.fit_predict (FB selection on validation),
prospective.fit_fixed (fixed hyperparameters, no validation), data.target (FB target kinds).
"""
import numpy as np

from hypershift.featbench import data as fbdata
from hypershift.featbench.models import _features, fit_predict
from hypershift.featbench.prospective import fit_fixed
from hypershift.fbaudit.stats import spearman_daily


def target_values(fb, signals, kind):
    """[S, N] training target and mask. FB kinds pass through; adds demean, sector_beta, neutral."""
    if kind in ('raw', 'sector', 'residual'):
        return fbdata.target(fb, signals, kind)
    y = fb.y[:, signals].T.astype(np.float64)
    mask = (fb.y_ok[:, signals] & fb.eligible[:, signals]).T
    if kind == 'neutral':
        return np.where(mask, fb.neutral[:, signals].T.astype(np.float64), 0.), mask
    out = np.zeros_like(y)
    if kind == 'demean':
        for d in range(len(signals)):
            m = mask[d]
            if m.any():
                out[d, m] = y[d, m] - y[d, m].mean()
        return out, mask
    if kind == 'sector_beta':
        dummies = (fb.sectors[:, None] == np.unique(fb.sectors)[None]).astype(np.float64)
        for d, s in enumerate(signals):
            m = mask[d]
            if m.sum() < 30:
                continue
            b = fb.beta[:, s].astype(np.float64)[m]
            A = np.column_stack([dummies[m], (b - b.mean()) / max(b.std(), 1e-12)])
            coef, *_ = np.linalg.lstsq(A, y[d, m], rcond=None)
            out[d, m] = y[d, m] - A @ coef
        return out, mask
    raise ValueError(kind)


def run_arm(fb, model, features, target_kind, rounds=None, seed=7, alpha=None, threads=4):
    """Train on fb.train, score fb.evaluate. rounds/alpha None -> FB validation selection; else fixed, no validation."""
    ev = fb.evaluate
    Xtr = _features(fb, fb.train, features, 'cs_z')
    ytr, mtr = target_values(fb, fb.train, target_kind)
    mtr = mtr & fb.eligible[:, fb.train].T
    Xev = _features(fb, ev, features, 'cs_z')
    if rounds is None and alpha is None:
        Xva = _features(fb, fb.valid, features, 'cs_z')
        mva = (fb.y_ok[:, fb.valid] & fb.eligible[:, fb.valid]).T
        pred, info = fit_predict(model, Xtr, ytr, mtr, Xva, fb.neutral[:, fb.valid].T, mva, Xev, seed=seed,
                                 threads=threads)
    else:
        params = {'n_estimators': rounds} if model == 'lgbm' else {'alpha': alpha}
        pred, info = fit_fixed(model, Xtr, ytr, mtr, Xev, params, seed=seed, threads=threads), dict(params)
    mask = (fb.y_ok[:, ev] & fb.eligible[:, ev]).T
    daily = spearman_daily(pred, fb.neutral[:, ev].T, mask)
    return {'daily': daily, 'pred': pred, 'info': info, 'dates': fb.dates[ev].astype(str)}
