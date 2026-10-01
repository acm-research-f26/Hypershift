"""Connect chronological training samples to feature transformations."""

import numpy as np
import pandas as pd
from dataclasses import replace

from features.transforms import apply_standardizer

def training_feature_mask(feature_times, train_samples, *, node_ids, feature_names, fit_cutoff):
    """Select eligible training inputs once per time/node/feature coordinate."""
    times = pd.DatetimeIndex(feature_times)
    cutoff = pd.Timestamp(fit_cutoff)
    node_ids, feature_names = tuple(node_ids), tuple(feature_names)
    if (
        times.tz is None or times.hasnans
        or not times.is_unique or not times.is_monotonic_increasing
    ):
        raise ValueError("Feature times must be aware, present, unique, and sorted")
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("fit_cutoff must be present and timezone-aware")
    if (
        not node_ids or not feature_names
        or len(set(node_ids)) != len(node_ids)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("Node IDs and feature names must be nonempty and unique")

    times, cutoff = times.tz_convert("UTC"), cutoff.tz_convert("UTC")
    shape = (len(node_ids), len(feature_names))
    result = np.zeros((len(times), *shape), dtype=bool)
    previous = None
    for sample in train_samples:
        if tuple(sample.node_ids) != node_ids or tuple(sample.feature_names) != feature_names:
            raise ValueError("Training sample axes must match the feature axes")
        timing = [
            pd.Timestamp(value)
            for value in (sample.origin_time, sample.target_end, sample.target_availability)
        ]
        if any(pd.isna(time) or time.tzinfo is None for time in timing):
            raise ValueError("Sample information times must be present and aware")
        origin, end, known = (time.tz_convert("UTC") for time in timing)
        if not origin < end <= known <= cutoff:
            raise ValueError(
                "Require origin < target_end <= target_availability <= fit_cutoff"
            )
        if previous is not None and origin <= previous:
            raise ValueError("Training origins must be unique and increasing")
        previous = origin

        history = pd.DatetimeIndex(sample.history_times)
        if (
            history.tz is None or history.hasnans or len(history) == 0
            or not history.is_unique or not history.is_monotonic_increasing
        ):
            raise ValueError("History times must be nonempty, aware, unique, and sorted")
        history = history.tz_convert("UTC")
        if history[-1] > origin:
            raise ValueError("History cannot extend beyond prediction time")

        mask, eligible = np.asarray(sample.mask), np.asarray(sample.eligible_nodes)
        if mask.shape != (len(history), *shape) or mask.dtype.kind != "b":
            raise ValueError("Sample feature masks must match history and feature axes")
        if eligible.shape != (shape[0],) or eligible.dtype.kind != "b":
            raise ValueError("eligible_nodes must be Boolean with shape [node]")
        positions = times.get_indexer(history) # maps sample's historical timestamps to rows of the feature array 
        if (positions < 0).any():
            raise ValueError("Every historical feature time must occur on the time axis")

        # OR forms a union, so shared historical observations are selected once.
        # $$ \mathcal I_{n,f} = \bigcup_{s\in S_{\mathrm{train}}} \{t\in H_s:E_{s,n}\land M_{s,t,n,f}\}. $$
        
        # fit_standardizer estimates its mean and scale over this set.
        # mask & eligible[None, :, None] keeps only valid features belonging to eligeible stocks. 
        
        result[positions] |= mask & eligible[None, :, None]

    return result

def transform_forecast_sample(sample, state):
    """Standardize a single forecast sample using the fitted state & update prediction eligibility."""
    nodes = len(sample.node_ids)
    eligible = np.asarray(sample.eligible_nodes)
    target, target_mask = np.asarray(sample.target), np.asarray(sample.target_mask)
    if eligible.shape != (nodes,) or eligible.dtype.kind != "b":
        raise ValueError("eligible_nodes must be Boolean with shape [node]")
    if target.shape != (nodes,) or target_mask.shape != (nodes,) or target_mask.dtype.kind != "b":
        raise ValueError("target and target_mask must have shape [node]")
    value, mask = apply_standardizer(
        sample.values,
        sample.mask,
        state,
        node_ids=sample.node_ids,
        feature_names=sample.feature_names,
    )
    if len(sample.history_times) == 0 or len(sample.history_times) != value.shape[0]:
        raise ValueError("History times must be nonempty and match feature rows.")
    return replace(
        sample,
        values=value,
        mask=mask,
        eligible_nodes=eligible & mask.all(axis=(0,2)),
        history_times=sample.history_times.copy(deep=True),
        target=target.copy(),
        target_mask=target_mask.copy()
    )
    




# axes = dict(node_ids=panel.node_ids, feature_names=("log_return",))

# fit_mask = training_feature_mask(
#     panel.timestamps,
#     splits["train"],
#     fit_cutoff=fold["fit_cutoff"],
#     **axes,
# )

# state = fit_standardizer(
#     returns[:, :, None],
#     return_mask[:, :, None] & fit_mask,
#     fit_mask.any(axis=(1, 2)),
#     **axes,
# )
