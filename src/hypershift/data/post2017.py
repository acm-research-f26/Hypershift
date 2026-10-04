"""Post-2017 panel builder for the Phase 1.5b data gate. Contract: docs/phase1_5b/DATA_COMPATIBILITY.md.

Pure numpy/pandas. Nothing here computes strategy or portfolio returns; per-stock close-to-close returns are used
only for the identity / overlap test on the 2015-2017 overlap window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hypershift.data.rsr import FILL, MarketData

MA_WINDOWS = (5, 10, 20, 30)
TOL = 1e-4          # predeclared R5 return-match tolerance
MIN_MATCHED = 250   # predeclared identity test: minimum matched stock-days
IDENT_SHARE = 0.95  # predeclared identity / gate threshold
GAP_DAYS = 60       # predeclared continuity rule


class Features:
    """Per-stock feature block: features [T,5] (MA5/10/20/30/close, scaled), mask [T], gt [T], base_price [T]."""
    def __init__(self, features, mask, gt, base_price):
        self.features, self.mask, self.gt, self.base_price = features, mask, gt, base_price


def undo_splits(split_adjusted: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """Raw close = Yahoo (split-adjusted) Close times the product of all STRICTLY LATER split ratios.
    ratios[t] > 0 is the split ratio effective on day t (Yahoo 'Stock Splits', 2.0 = 2-for-1), 0 = none."""
    r = np.where(np.asarray(ratios, dtype=np.float64) > 0, ratios, 1.0)
    later = np.concatenate([np.cumprod(r[::-1])[::-1][1:], [1.0]])
    return np.asarray(split_adjusted, dtype=np.float64) * later


def returns_on_calendar(close: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """Close-to-close returns on `cal`: r[t] = c[t]/c[t-1]-1 when both calendar days are observed, else NaN (RSR parse_eod rule)."""
    c = close.reindex(cal).to_numpy(dtype=np.float64)
    r = np.full(len(c), np.nan)
    ok = np.isfinite(c[1:]) & np.isfinite(c[:-1]) & (c[:-1] != 0)
    r[1:][ok] = c[1:][ok] / c[:-1][ok] - 1
    return r


def match_share(r_new: np.ndarray, r_ref: np.ndarray, tol: float = TOL) -> tuple[int, int]:
    """(matched, total) stock-days where both returns are finite; matched = |r_new - r_ref| < tol."""
    both = np.isfinite(r_new) & np.isfinite(r_ref)
    return int((np.abs(r_new[both] - r_ref[both]) < tol).sum()), int(both.sum())


def identity_test(r_new, r_ref, tol: float = TOL, min_days: int = MIN_MATCHED, thr: float = IDENT_SHARE):
    """Per-ticker identity rule: (ok, share, n_matched_eligible). Rejects ticker reuse and splices."""
    m, n = match_share(np.asarray(r_new), np.asarray(r_ref), tol)
    share = m / n if n else 0.0
    return bool(n >= min_days and share >= thr), share, n


def truncate_gaps(obs: np.ndarray, gap: int = GAP_DAYS) -> np.ndarray:
    """Continuity rule: a run of >= gap unobserved days that is followed by a reappearance (and preceded by an
    observation) marks a reused symbol; everything from the start of that run on is unobserved."""
    obs = np.asarray(obs, dtype=bool).copy()
    n, i, seen = len(obs), 0, False
    while i < n:
        if obs[i]:
            seen, i = True, i + 1
            continue
        j = i
        while j < n and not obs[j]:
            j += 1
        if seen and j < n and j - i >= gap:
            obs[i:] = False
            break
        i = j
    return obs


def availability_mask(obs: np.ndarray) -> np.ndarray:
    """Eligibility from bar availability ONLY (no prices touched): a bar exists and at least 30 closes exist since the
    first bar (MA30 available). Used for 2018+ coverage counts, where no return may be computed."""
    obs = np.asarray(obs, dtype=bool)
    if not obs.any():
        return obs.copy()
    first = int(np.argmax(obs))
    out = obs.copy()
    out[: first + 29] = False
    return out


def build_features(close: np.ndarray, train_end: int) -> Features:
    """RSR-layout features for one stock from a calendar-aligned close vector (NaN = no bar).

    MA_w(t) = mean of the last w closes after forward-filling inside the series (INFERRED rule; RSR's own handling of a
    missing day inside an MA window is UNKNOWN). MA30 unavailable (fewer than 30 closes since the first bar) or no bar
    -> mask 0 and every feature cell = FILL. Scale = max observed close over indices <= train_end (fixed constant, so a
    later price cannot change an earlier feature). gt[t] = close return when t and t-1 are both observed, else 0 (mask
    is what makes a day usable; a delisted series is masked from the first missing day on, never zero-filled).
    """
    c = np.asarray(close, dtype=np.float64)
    T = len(c)
    obs = np.isfinite(c)
    feats = np.full((T, 5), FILL, dtype=np.float32)
    base = np.full(T, FILL, dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)
    gt = np.zeros(T, dtype=np.float32)
    if obs.any():
        first = int(np.argmax(obs))
        last = T - 1 - int(np.argmax(obs[::-1]))
        ff = pd.Series(c).ffill().to_numpy()
        head = c[: train_end + 1]
        scale = np.nanmax(head) if np.isfinite(head).any() else 1.0
        scale = scale if scale > 0 else 1.0
        ma = {}
        for w in MA_WINDOWS:
            m = np.full(T, np.nan)
            if T >= w:
                m[w - 1:] = np.convolve(np.nan_to_num(ff, nan=0.0), np.ones(w) / w, mode="valid")
            m[: first + w - 1] = np.nan          # window must start at or after the first bar
            m[last + 1:] = np.nan
            ma[w] = m
        ok = obs & np.isfinite(ma[30])
        for k, w in enumerate(MA_WINDOWS):
            feats[ok, k] = (ma[w][ok] / scale).astype(np.float32)
        feats[ok, 4] = (c[ok] / scale).astype(np.float32)
        base[ok] = (c[ok] / scale).astype(np.float32)
        mask[ok] = 1.0
        both = obs[1:] & obs[:-1]
        g = np.zeros(T - 1)
        g[both] = c[1:][both] / c[:-1][both] - 1
        gt[1:] = g.astype(np.float32)
    return Features(feats, mask, gt, base)


def build_panel(series: dict, order: list[str], cal: pd.DatetimeIndex, train_end: int) -> MarketData:
    """Panel in EXACT `order` (RSR ticker order). A ticker without a series keeps its slot, fully masked."""
    T = len(cal)
    parts = []
    for t in order:
        s = series.get(t)
        c = np.full(T, np.nan) if s is None else s.reindex(cal).to_numpy(dtype=np.float64)
        parts.append(build_features(c, train_end))
    return MarketData(
        tickers=list(order),
        features=np.stack([f.features for f in parts]),
        mask=np.stack([f.mask for f in parts]),
        gt=np.stack([f.gt for f in parts]),
        base_price=np.stack([f.base_price for f in parts]),
        valid_index=train_end + 1,
        test_index=T,
        timestamps=np.asarray(cal.values),
    )
