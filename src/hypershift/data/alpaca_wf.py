"""Phase 1.5c walk-forward panels from the Alpaca 1.5b panel (docs/phase1_5c/SPEC.md).

One compact npz holds the full 2016-01-04..2023-12-29 panel (frozen RSR node order, non-identity nodes fully masked, series masked after
their last bar). `load_wf_panel(path, test_year)` slices it to the end of `test_year` and sets target-date split indices:
train targets < Jan 1 of test_year-1 (i.e. end Dec 31 of test_year-2), validation targets = test_year-1, test targets = test_year.
Features are causal functions of the closes with a fixed scale (1.5b panel), so truncation does not change any earlier value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypershift.data.rsr import MarketData

PANEL_NAME = "alpaca_panel_2016_2023.npz"
WF_YEARS = (2019, 2020, 2021, 2022, 2023)


def save_panel_npz(md: MarketData, identity_ok, path) -> None:
    np.savez_compressed(
        path, features=md.features.astype(np.float32), gt=md.gt.astype(np.float32), mask=md.mask.astype(np.uint8),
        dates=np.asarray(md.timestamps).astype("datetime64[D]"), tickers=np.asarray(md.tickers, dtype="U16"),
        identity_ok=np.asarray(identity_ok, dtype=np.uint8))


def split_indices(dates, test_year: int) -> tuple[int, int, int]:
    """(valid_index, test_index, end) for a calendar of np.datetime64 dates; end = number of days kept (<= Dec 31 of test_year)."""
    cal = pd.DatetimeIndex(np.asarray(dates))
    end = int(cal.searchsorted(pd.Timestamp(f"{test_year}-12-31"), side="right"))
    vi = int(cal.searchsorted(pd.Timestamp(f"{test_year - 1}-01-01")))
    ti = int(cal.searchsorted(pd.Timestamp(f"{test_year}-01-01")))
    if not (0 < vi < ti < end) or cal[-1] < pd.Timestamp(f"{test_year}-12-20"):
        raise ValueError(f"test_year {test_year}: panel does not cover the window (vi={vi}, ti={ti}, end={end})")
    return vi, ti, end


def load_wf_panel(path, test_year: int) -> MarketData:
    z = np.load(Path(path), allow_pickle=False)
    dates = z["dates"]
    vi, ti, end = split_indices(dates, test_year)
    mask = z["mask"][:, :end].astype(np.float32)
    ident = z["identity_ok"].astype(bool)
    if mask[~ident].any():
        raise ValueError("a non-identity-pass node has unmasked days")
    feats = z["features"][:, :end]
    return MarketData(
        tickers=[str(t) for t in z["tickers"]], features=feats, mask=mask, gt=z["gt"][:, :end],
        base_price=np.ascontiguousarray(feats[:, :, 4]), valid_index=vi, test_index=ti,
        timestamps=np.asarray(dates[:end]).astype("datetime64[ns]"))
