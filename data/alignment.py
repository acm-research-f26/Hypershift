"""Align per-node frames by explicit timestamps, node order, and field order."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from data.archive import BAR_COLUMNS


def align_nodes_and_times(frames, node_order, time_grid, field_names=BAR_COLUMNS):
    """Return values, observation masks, and eligibility masks shaped [T,N,F].

    Missing rows remain NaN. Each input frame must have a Boolean is_eligible
    column; field-level finiteness is combined with that bar-level policy here.
    """

    # Validate labels before converting DataFrames to unlabeled arrays.
    if isinstance(node_order, str) or isinstance(field_names, str):
        raise ValueError("Node and field orders must be sequences, not strings")
    nodes, fields = tuple(node_order), tuple(field_names)
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("Node order must be nonempty and unique")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Field order must be nonempty and unique")
    if set(frames) != set(nodes):
        raise ValueError("Frames must match the requested nodes exactly")
    if (
        not isinstance(time_grid, pd.DatetimeIndex)
        or time_grid.tz is None
        or time_grid.hasnans
        or not time_grid.is_unique
        or not time_grid.is_monotonic_increasing
    ):
        raise ValueError("Time grid must be aware, present, unique, and sorted")

    grid = time_grid.tz_convert("UTC")
    values = np.full((len(grid), len(nodes), len(fields)), np.nan)
    observed = np.zeros(values.shape, dtype=bool)
    eligible = np.zeros(values.shape, dtype=bool)
    for n, node in enumerate(nodes):
        frame = frames[node]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Each node frame must be a DataFrame")
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.tz is None
            or frame.index.hasnans
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError("Frame timestamps must be aware, unique, and sorted")
        if (
            not frame.columns.is_unique
            or not {*fields, "is_eligible"}.issubset(frame.columns)
        ):
            raise ValueError("Missing or duplicate frame columns")
        if (
            not is_bool_dtype(frame["is_eligible"])
            or frame["is_eligible"].isna().any()
        ):
            raise ValueError("is_eligible must contain nonmissing Booleans")

        # Match timestamps, leaving absent observations as NaN rather than
        # shifting later rows or filling prices from a neighboring timestamp.
        aligned = frame.reindex(grid)
        raw = aligned.loc[:, list(fields)].to_numpy(dtype=float, copy=True)
        usable = aligned["is_eligible"].fillna(False).to_numpy(dtype=bool)
        values[:, n, :] = raw
        observed[:, n, :] = np.isfinite(raw)
        eligible[:, n, :] = observed[:, n, :] & usable[:, None]

    return values, observed, eligible
