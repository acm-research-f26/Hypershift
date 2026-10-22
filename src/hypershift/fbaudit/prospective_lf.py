"""Prospective v1-LF (PROSPECTIVE_ADDENDUM.md): v1 with leak-free dollar volume. Same guard, model, rule, dates.

v1 code (featbench/prospective.py) is used read-only; only log_dvol_20 and amihud_20 are replaced, in the features
and therefore in the residual-target regressor. raw_npz: an auto_adjust=False download saved at the v1 fetch with
arrays tickers [N], dates [T'], close [N, T'] (unadjusted, split-adjusted Close) aligned to the panel's first T' dates.
"""
import numpy as np

from hypershift.featbench.data import FEATURES
from hypershift.featbench.prospective import load_range, score
from hypershift.fbaudit.leakfree import corrected_columns


def make_leakfree(fb, ohlcv, raw_close):
    """Replace log_dvol_20 / amihud_20 in fb.X (in place) with unadjusted-close versions; returns fb."""
    T = fb.X.shape[1]
    ldv, ami = corrected_columns(ohlcv[:, :T], fb.observed, fb.eligible, raw_close[:, :T])
    fb.X = fb.X.copy()
    fb.X[..., FEATURES.index('log_dvol_20')] = ldv
    fb.X[..., FEATURES.index('amihud_20')] = ami
    return fb


def load_range_lf(panel_path, ohlcv_path, raw_npz, horizon=5):
    fb = load_range(panel_path, ohlcv_path, horizon)
    z, o = np.load(panel_path), np.load(ohlcv_path)
    ident = z['identity_ok'].astype(bool)
    raw = np.load(raw_npz)
    assert (raw['tickers'] == fb.panel.tickers).all(), 'raw download tickers differ from the panel'
    n = len(raw['dates'])
    assert (raw['dates'] == fb.dates[:n].astype(str)).all(), 'raw download dates differ from the panel'
    rc = np.full(fb.X.shape[:2], np.nan)
    rc[:, :n] = raw['close'][:, :fb.X.shape[1]]
    return make_leakfree(fb, o['ohlcv'][ident].astype(np.float64), rc)


def score_lf(fb, manifest, threads=8):
    """Identical to v1 scoring (including the freeze guard), on a leak-free fb."""
    return score(fb, manifest, threads)
