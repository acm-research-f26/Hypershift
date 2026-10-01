"""Future return labels and the times at which they become available."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.market import log_returns


@dataclass(frozen=True)
class ReturnTargets:
    values: np.ndarray
    mask: np.ndarray
    start_times: pd.DatetimeIndex
    end_times: pd.DatetimeIndex
    availability_times: pd.DatetimeIndex
    units: str = "log_return"


# For 15-minute bars:
# $y_{10:00} = \log\left(\frac{C_{10:15}}{C_{10:00}}\right)$

def make_return_targets(panel) -> ReturnTargets:
    """Label each row with the next bar's log return, excluding overnight gaps."""
    returns, valid = log_returns(panel)
    times = panel.timestamps.tz_convert("UTC")
    available = panel.availability_times
    if (len(available) != len(times) or available.tz is None
            or available.hasnans):
        raise ValueError("Availability times must be present, aware, and aligned")
    available = available.tz_convert("UTC")
    if (available < times).any():
        raise ValueError("A completed bar cannot be available before its end")

    # The target at t is the return ending at t+1.
    values = np.full_like(returns, np.nan)
    mask = np.zeros_like(valid)
    values[:-1] = returns[1:] # The crux is this shift
    mask[:-1] = valid[1:]

    # The final row has no future endpoint. Empty panels stay empty.
    missing = pd.DatetimeIndex([pd.NaT], tz="UTC")[:min(1, len(times))]
    end_times = times[1:].append(missing)

    # Both endpoint prices must be available before the label can be known.
    known = available[:-1].where(
        available[:-1] >= available[1:], available[1:]
    )
    return ReturnTargets(
        values=values,
        mask=mask,
        start_times=times,
        end_times=end_times,
        availability_times=known.append(missing),
    )
