"""Eval panel for the locked 2018-2023 inference (Phase 1.5b Task 4; DATA_COMPATIBILITY.md A1/A2).

Features/mask/gt per `post2017.build_features` (RSR semantics, forward-fill MA INFERRED). Nodes keep the frozen RSR order;
a node without an accepted series is fully masked; a series is masked after its last bar (no terminal return is invented).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from hypershift.data import post2017 as P
from hypershift.data.rsr import MarketData


def assert_frozen_order(order, frozen) -> None:
    if list(order) != list(frozen):
        raise ValueError("ticker order differs from the frozen RSR order (node indices would be permuted)")


def calendar_from_bars(series: dict, start, end, frac: float = 0.5) -> pd.DatetimeIndex:
    """Trading days = dates on which >= frac of the median recent per-day ticker count have a bar (A2 item 7)."""
    cnt: dict = {}
    for s in series.values():
        for x in s.index:
            cnt[x] = cnt.get(x, 0) + 1
    cnt = pd.Series(cnt).sort_index()
    days = cnt.index[cnt.to_numpy() >= frac * np.median(cnt.to_numpy()[-500:])]
    return pd.DatetimeIndex(days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))])


def build_eval_panel(series: dict, order, frozen_order, cal: pd.DatetimeIndex, test_start, scale_end,
                     gap_days: int = P.GAP_DAYS, gap_from=pd.Timestamp("2017-12-08")) -> MarketData:
    assert_frozen_order(order, frozen_order)
    cal = pd.DatetimeIndex(cal)
    if cal.has_duplicates or not cal.is_monotonic_increasing:
        raise ValueError("calendar must be strictly increasing")
    g0 = max(int(cal.searchsorted(pd.Timestamp(gap_from))) - 1, 0)
    out = {}
    for t, s in series.items():
        if t not in order:
            raise ValueError(f"series for unknown ticker {t}")
        s = s[~s.index.duplicated(keep="last")].reindex(cal)
        obs = s.notna().to_numpy()
        keep = obs.copy()
        keep[g0:] = P.truncate_gaps(obs[g0:], gap_days)     # continuity rule: reused-symbol guard (A2 item 6)
        out[t] = s.where(keep)
    ti = int(cal.searchsorted(pd.Timestamp(test_start)))
    se = int(cal.searchsorted(pd.Timestamp(scale_end)))
    md = P.build_panel(out, list(order), cal, train_end=se)
    return dataclasses.replace(md, valid_index=ti, test_index=ti)


def target_dates(data: MarketData) -> np.ndarray:
    """Date of every test window's target day (window_offsets(test) target = offset + seq = test_index..T-1)."""
    return np.asarray(data.timestamps[data.test_index:])


def topk_names(pred_d, mask_d, k: int = 5):
    idx = np.nonzero(np.asarray(mask_d) > 0.5)[0]
    return idx[np.argsort(-np.asarray(pred_d)[idx], kind="stable")[:k]]


def daily_tie_flags(pred, mask, k: int = 5) -> np.ndarray:
    """True on days where the k-th and (k+1)-th eligible scores tie exactly (lowest-index tie-break decides the basket)."""
    flags = np.zeros(pred.shape[1], dtype=bool)
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        if len(idx) > k:
            v = np.sort(pred[idx, d])[::-1]
            flags[d] = bool(v[k - 1] == v[k])
    return flags
