"""Prospective paper-trading components for the style-only book (PAPER_TRADING.md). No outcome is scored here.

style_scores: LightGBM (FB parameters, 50 rounds) on the four style characteristics + sector one-hot, trained on the
residual target over signals whose 5-day labels have matured by the as-of close; seed predictions rank-averaged.
"""
import numpy as np

from hypershift.featbench.prospective import fit_fixed
from hypershift.fbaudit.matched import nested_inputs
from hypershift.fbaudit.refit import target_values


def style_scores(fb, train_signals, asof, seeds=(7, 19, 42), rounds=50, threads=8):
    train_signals = np.asarray(train_signals)
    if (train_signals + fb.horizon > asof).any():
        raise ValueError('training labels must mature on or before the as-of date')
    Xtr = nested_inputs(fb, train_signals, 'style')
    ytr, mtr = target_values(fb, train_signals, 'residual')
    mtr = mtr & fb.eligible[:, train_signals].T
    Xev = nested_inputs(fb, np.array([asof]), 'style')
    e = fb.eligible[:, asof]
    ranks = []
    for s in seeds:
        p = fit_fixed('lgbm', Xtr, ytr, mtr, Xev, {'n_estimators': rounds}, seed=s, threads=threads)[0]
        r = np.full(len(p), np.nan)
        r[e] = np.argsort(np.argsort(p[e])) / max(e.sum() - 1, 1)
        ranks.append(r)
    return np.where(e, np.nanmean(ranks, 0), 0.)
