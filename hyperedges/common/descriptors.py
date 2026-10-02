"""Historical stock descriptors shared by Mapper and Cover Learning."""

import numpy as np
import pandas as pd

__all__ = ["build_stock_descriptors"]


def build_stock_descriptors(history, descriptor_spec):
    if set(descriptor_spec) - {"features", "coverage_threshold", "standardize"}:
        raise ValueError("Unknown descriptor parameters")
    features = tuple(descriptor_spec["features"])
    supported = {"mean", "std", "q10", "q50", "q90", "downside_frequency", "upside_frequency", "lag1"}
    if not features or len(set(features)) != len(features) or not set(features) <= supported:
        raise ValueError("Specify unique supported return descriptors")
    threshold = descriptor_spec.get("coverage_threshold", 0.95)
    if not 0 < threshold <= 1 or not history.structural_mask.any():
        raise ValueError("Descriptor coverage must be in (0, 1] with valid return positions")
    structural = history.structural_mask
    coverage = history.mask.loc[structural].mean()
    rows, nodes = [], []
    for node in sorted(history.returns.columns):
        if coverage[node] < threshold:
            continue
        valid = history.mask[node].to_numpy() & structural
        values = history.returns.loc[valid, node].to_numpy()
        if len(values) < 2:
            continue
        summary = {"mean": values.mean(), "std": values.std(ddof=1),
                   "q10": np.quantile(values, 0.1), "q50": np.quantile(values, 0.5), "q90": np.quantile(values, 0.9),
                   "downside_frequency": np.mean(values < 0), "upside_frequency": np.mean(values > 0)}
        if "lag1" in features:
            adjacent = (valid[1:] & valid[:-1] & np.asarray(history.session_ids[1:] == history.session_ids[:-1]))
            if history.predecessor_times is not None:
                adjacent &= np.asarray(history.predecessor_times[1:] == history.returns.index[:-1])
            raw = history.returns[node].to_numpy()
            left, right = raw[:-1][adjacent], raw[1:][adjacent]
            summary["lag1"] = (np.corrcoef(left, right)[0, 1] if len(left) >= 2 and left.std() > 0 and right.std() > 0 else np.nan)
        row = [summary[feature] for feature in features]
        if np.isfinite(row).all():
            rows.append(row)
            nodes.append(node)
    frame = pd.DataFrame(rows, index=nodes, columns=features, dtype=float)
    if frame.empty:
        raise ValueError("No stocks have valid configured descriptors")
    mean, scale = frame.mean(), frame.std(ddof=0).replace(0, 1)
    if descriptor_spec.get("standardize", True):
        frame = (frame - mean) / scale
    frame.attrs["fit_state"] = {"mean": mean.to_dict(), "scale": scale.to_dict(), "recipe": descriptor_spec}
    return frame

