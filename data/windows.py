"""Select candidate forecast rows for intraday lookback windows."""

from numbers import Integral
from data.types import ForecastSample
from features.market import log_returns
from features.targets import make_return_targets


def iter_forecast_origins(panel, lookback):
    """Yield row indices with same-session history and an adjacent future bar."""
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 1:
        raise ValueError("lookback must be a positive integer number of bars")

    times, starts, sessions = panel.timestamps, panel.bar_starts, panel.session_ids
    if len(starts) != len(times) or len(sessions) != len(times) or sessions.hasnans:
        raise ValueError("Temporal metadata must be present and match the time axis")
    if any(index.tz is None or index.hasnans for index in (times, starts)):
        raise ValueError("Bar timestamps must be present and timezone-aware")
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Bar-end timestamps must be unique and sorted")

    # Reserve lookback rows ending at t, plus one future row for its target.
    for t in range(int(lookback) - 1, len(times) - 1):
        start = t - int(lookback) + 1

        # History and the future target endpoint stay inside one session.
        if not (sessions[start:t + 2] == sessions[t]).all():
            continue

        # Every neighboring pair must touch; never jump over a missing bin.
        if not (starts[start + 1:t + 2] == times[start:t + 1]).all():
            continue

        yield t
        
def make_forecast_samples(panel, lookback):
    """Yield same-session samples with one log-return feature per stock."""
    returns, return_mask = log_returns(panel)
    targets = make_return_targets(panel)
    times = panel.timestamps.tz_convert("UTC")
    available = panel.availability_times.tz_convert("UTC")

    for t in iter_forecast_origins(panel, lookback):
        start = t - int(lookback) + 1
        values = returns[start:t + 1, :, None].copy()
        mask = return_mask[start:t + 1, :, None].copy()

        # Initially require every historical return for a stock.
        eligible = mask.all(axis=(0, 2))
        if not eligible.any():
            continue

        # The earliest return also needs the close immediately before it.
        first_close = max(0, start - 1)
        origin = available[first_close:t + 1].max()
        if origin >= targets.end_times[t]:
            continue

        yield ForecastSample(
            origin_time=origin,
            history_times=times[start:t + 1].copy(),
            node_ids=panel.node_ids,
            feature_names=("log_return",),
            values=values,
            mask=mask,
            eligible_nodes=eligible,
            target=targets.values[t].copy(),
            target_mask=targets.mask[t].copy(),
            target_start=targets.start_times[t],
            target_end=targets.end_times[t],
            target_availability=targets.availability_times[t],
            target_units=targets.units,
        )
    
