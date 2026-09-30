"""Dataset adapters that compose archive loading and observation preparation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.alignment import align_nodes_and_times
from data.archive import BAR_COLUMNS, load_symbol_bars
from data.calendars import load_calendar, session_schedule
from data.resampling import (
    BarEligibilityPolicy,
    apply_bar_eligibility,
    resample_sessions,
)
from data.types import ObservationPanel


def load_alpaca_panel(group, window, nodes) -> ObservationPanel:
    """Load a small minute-derived panel over a half-open UTC instant window.

    Explicit nodes may be a subset of the archive universe. Windows must include
    whole regular sessions. Provider publication times are unavailable: use the
    configured availability delay after bar end as an explicit assumption.
    """
    # 1. Confirm that the configured source and sampler match this implementation.
    source, sampling = group.source.params, group.sampling.params
    if (
        group.domain != "finance"
        or group.source.name != "alpaca_archive"
        or group.sampling.name != "market_ohlcv"
    ):
        raise ValueError("Expected a finance Alpaca market_ohlcv dataset")
    supported_sampling = {
        "session": "regular",
        "timezone": "America/New_York",
        "intraday_anchor": "session_open",
        "missing_policy": "mask",
        "output_timestamp_role": "interval_end",
    }
    for key, expected in supported_sampling.items():
        if sampling.get(key, expected) != expected:
            raise ValueError(f"Unsupported sampling {key}")
    if source.get("timestamp_role") != "interval_start":
        raise ValueError("Source timestamp_role must be interval_start")
    expected_aggregation = dict(zip(BAR_COLUMNS, (
        "first", "max", "min", "last", "sum", "sum", "volume_weighted_mean",
    )))
    if sampling.get("aggregation", expected_aggregation) != expected_aggregation:
        raise ValueError("Unsupported OHLCV aggregation recipe")
    minute = pd.Timedelta("1min")
    try:
        native = pd.Timedelta(source["native_interval"])
        interval = pd.Timedelta(sampling["output_interval"])
        delay = pd.Timedelta(sampling.get("availability_delay", "0min"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Declare valid native, output, and availability intervals") from exc
    if native != minute:
        raise ValueError("This adapter requires a 1min source")
    if (
        pd.isna(interval)
        or interval <= pd.Timedelta(0)
        or interval % minute != pd.Timedelta(0)
    ):
        raise ValueError("Output interval must be positive whole minutes")
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError("Availability delay must be nonnegative")
    # 2. Preserve an explicit node order and require unambiguous UTC boundaries.
    if isinstance(nodes, str):
        raise ValueError("nodes must be an ordered sequence")
    node_ids = tuple(nodes)
    if (
        not node_ids
        or any(not isinstance(n, str) or not n for n in node_ids)
        or len(set(node_ids)) != len(node_ids)
    ):
        raise ValueError("nodes must be nonempty, unique symbol strings")
    start, end = map(pd.Timestamp, window)
    if any(pd.isna(t) or t.tzinfo is None for t in (start, end)):
        raise ValueError("Window endpoints must be timezone-aware instants")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start >= end or any(t != t.floor("min") for t in (start, end)):
        raise ValueError("Window must be increasing and minute-aligned")

    # 3. Record the manifest we expect the loader's partition checks to validate.
    root = Path(source["path"])
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    request = manifest["request"]
    if manifest["schema_version"] != 1 or request["timeframe"] != "1Min":
        raise ValueError("Archive manifest must describe schema 1, 1Min data")
    for key in ("feed", "adjustment"):
        if key not in source or request.get(key) != source[key]:
            raise ValueError(f"Archive {key} does not match configuration")
    if not set(node_ids).issubset(request["symbols"]):
        raise ValueError("Requested node is absent from the archive universe")

    # 4. Resolve exchange dates, then require complete sessions inside the window.
    local_start = start.tz_convert("America/New_York").tz_localize(None)
    local_end = end.tz_convert("America/New_York").tz_localize(None)
    end_date = local_end.normalize()
    if local_end != end_date:
        end_date += pd.Timedelta(days=1)
    schedule = session_schedule(
        load_calendar(sampling.get("calendar", "XNYS")),
        local_start.normalize(),
        end_date,
    )
    schedule = schedule.loc[
        (schedule["session_open"] < end) & (schedule["session_close"] > start)
    ].copy()
    if schedule.empty:
        raise ValueError("Window contains no regular market sessions")
    if ((schedule["session_open"] < start) | (schedule["session_close"] > end)).any():
        raise ValueError("Window cuts a session; request whole sessions")
    policy = BarEligibilityPolicy(
        minimum_coverage=sampling.get("minimum_coverage", 1.0),
        partial_bar_policy=sampling.get("partial_bar_policy", "keep_and_flag"),
    )

    # 5. Expected timestamps depend on the calendar, even if every bar is missing.
    empty = pd.DataFrame(
        columns=BAR_COLUMNS, dtype=float, index=pd.DatetimeIndex([], tz="UTC")
    )
    grid = resample_sessions(empty, schedule, interval)
    shape = (len(grid), len(node_ids), len(BAR_COLUMNS))
    values = np.full(shape, np.nan)
    observed = np.zeros(shape, dtype=bool)
    eligible = np.zeros(shape, dtype=bool)
    counts = np.zeros(shape[:2], dtype="int64")
    partitions = {}
    months = pd.date_range(
        start.normalize().replace(day=1), end, freq="MS", inclusive="left"
    )

    # 6. Load one symbol's requested window, prepare it, and fill its panel column.
    for n, node in enumerate(node_ids):
        raw = load_symbol_bars(root, node, start, end, expected_timeframe="1Min")
        bars = apply_bar_eligibility(resample_sessions(raw, schedule, interval), policy)
        v, o, e = align_nodes_and_times({node: bars}, (node,), grid.index)
        values[:, n, :] = v[:, 0, :]
        observed[:, n, :] = o[:, 0, :]
        eligible[:, n, :] = e[:, 0, :]
        counts[:, n] = bars.reindex(grid.index)["observed_minutes"].to_numpy(
            dtype="int64"
        )
        for month in months:
            relative = f"bars/{node}/{month:%Y-%m}.csv.gz"
            partitions[relative] = manifest["partitions"][relative]
    # 7. Do not return original provenance if selected source metadata changed.
    latest = json.loads(manifest_path.read_bytes())
    if (
        latest["request"] != request
        or any(latest["partitions"].get(k) != v for k, v in partitions.items())
    ):
        raise RuntimeError("Selected archive metadata changed while loading; retry")
    # 8. Preserve interpretation, coverage, and source identity with the values.
    provenance = {
        **schedule.attrs,
        "dataset_name": group.name,
        "archive_path": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_request": request,
        "partitions": partitions,
        "window": (start.isoformat(), end.isoformat()),
        "native_interval": "1min",
        "output_interval": str(interval),
        "minimum_coverage": policy.minimum_coverage,
        "partial_bar_policy": policy.partial_bar_policy,
        "timestamp_role": "interval_end",
        "availability_delay": str(delay),
        "availability_assumption": "bar_end + delay; provider publication times unavailable",
    }
    units = {f: "USD" for f in ("open", "high", "low", "close", "vwap")}
    units.update(volume="shares", trade_count="trades")
    return ObservationPanel(
        timestamps=grid.index,
        node_ids=node_ids,
        field_names=BAR_COLUMNS,
        values=values,
        observation_mask=observed,
        eligible_mask=eligible,
        session_ids=pd.DatetimeIndex(grid["session_id"]),
        bar_starts=pd.DatetimeIndex(grid["bar_start"]),
        availability_times=grid.index + delay,
        expected_minutes=grid["expected_minutes"].to_numpy(dtype="int64"),
        observed_minutes=counts,
        is_partial=grid["is_partial"].to_numpy(dtype=bool),
        units=units,
        provenance=provenance,
    )
