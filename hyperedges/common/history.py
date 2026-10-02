"""Availability-aware construction windows independent of predictor lookback."""

from dataclasses import replace
from numbers import Integral

import numpy as np
import pandas as pd

from data.types import ObservationPanel
from experiments.config import TimeSpan
from features.market import log_returns
from .types import ConstructionHistory, utc_timestamp


def validate_window(window):
    if (not isinstance(window, TimeSpan) or isinstance(window.value, bool)
            or not isinstance(window.value, Integral) or window.value < 1
            or window.unit not in {"steps", "minutes", "hours", "days", "weeks", "sessions"}):
        raise ValueError("history_window must be a positive TimeSpan")


def prepare_construction_history(panel, cutoff, window=TimeSpan(60, "sessions")):
    validate_window(window)
    cutoff = utc_timestamp(cutoff)
    if isinstance(panel, ObservationPanel):
        returns, valid = log_returns(panel)
        structural = np.zeros(len(panel.timestamps), dtype=bool)
        structural[1:] = ((panel.session_ids[1:] == panel.session_ids[:-1])
                          & (panel.bar_starts[1:] == panel.timestamps[:-1]))
        # Return availability includes both close observations, even when publication is delayed.
        available = panel.availability_times.tz_convert("UTC")
        if available.hasnans:
            raise ValueError("Observation availability must be present")
        # Pandas can store DatetimeIndex in microseconds; normalize before interpreting integers as ns.
        available_ns = available.as_unit("ns").asi8.copy()
        available_ns[1:] = np.maximum(available_ns[1:], available_ns[:-1])
        availability = pd.to_datetime(available_ns, utc=True)
        times = panel.timestamps.tz_convert("UTC")
        allowed = (times <= cutoff) & (availability <= cutoff)
        source = ConstructionHistory(
            pd.DataFrame(returns[allowed], index=times[allowed], columns=panel.node_ids),
            pd.DataFrame(valid[allowed], index=times[allowed], columns=panel.node_ids),
            structural[allowed], panel.session_ids[allowed], availability[allowed], cutoff,
            dict(panel.provenance),
            times.insert(0, pd.NaT)[:-1][allowed],
        )
    elif isinstance(panel, ConstructionHistory):
        # A supplied view can be narrowed, but observations may never be promoted to an earlier cutoff.
        times, availability = panel.returns.index, panel.availability_times
        allowed = np.asarray((times <= cutoff) & (availability <= cutoff))
        source = _subset(panel, allowed, cutoff)
    else:
        raise TypeError("Expected ObservationPanel or ConstructionHistory")
    times = source.returns.index
    selection = np.ones(len(times), dtype=bool)
    if window.unit == "sessions":
        sessions = source.session_ids.unique()[-window.value:]
        selection = np.asarray(source.session_ids.isin(sessions))
    elif window.unit == "steps":
        selection[:max(0, len(times) - window.value)] = False
    else:
        selection = np.asarray(times > cutoff - pd.Timedelta(**{window.unit: window.value}))
    result = _subset(source, selection, cutoff)
    return replace(result, provenance={**result.provenance, "construction_window": {"value": window.value, "unit": window.unit}})


def _subset(history, selection, cutoff):
    previous = (history.predecessor_times if history.predecessor_times is not None
                else history.returns.index.insert(0, pd.NaT)[:-1])
    return ConstructionHistory(
        history.returns.loc[selection].copy(), history.mask.loc[selection].copy(),
        history.structural_mask[selection].copy(), history.session_ids[selection].copy(),
        history.availability_times[selection].copy(), cutoff, dict(history.provenance), previous[selection].copy(),
    )



def make_construction_context(panel, cutoff, *, seed=0, metadata=None, fold_id="fold"):
    """Retain permitted source history so every plugin can select its own window."""
    from .types import ConstructionContext
    count = len(panel.timestamps) if isinstance(panel, ObservationPanel) else len(panel.returns)
    if count < 1:
        raise ValueError("Construction context requires source observations")
    history = prepare_construction_history(panel, cutoff, TimeSpan(count, "steps"))
    return ConstructionContext(tuple(history.returns.columns), history, history.cutoff, seed, metadata or {}, fold_id)
