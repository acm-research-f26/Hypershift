"""Fit feature transformations using explicitly selected training inputs."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StandardizerState:
    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    mean: np.ndarray           # [node, feature]
    scale: np.ndarray          # [node, feature]; denominator for z-scores
    count: np.ndarray          # [node, feature]; valid training observations
    fitted_mask: np.ndarray    # [node, feature]; at least two observations


def fit_standardizer(values, mask, fit_window, *, node_ids, feature_names):
    """Fit one mean and population scale per node/feature using training rows."""
    # $\text{values} \in \mathbb{R}^{T \times N \times F}$
    values = np.asarray(values, dtype=float)
    
    # masks
    
    # $\text{mask} \in R^{TxNxF}$ Which observations are valid.
    mask = np.asarray(mask)
    
    # $\text{fit\_window} \in R^{T}$ Which time rows are used for fitting.
    fit_window = np.asarray(fit_window)
    
    node_ids, feature_names = tuple(node_ids), tuple(feature_names)

    if (
        values.ndim != 3
        or not node_ids
        or not feature_names
        or values.shape[1:] != (len(node_ids), len(feature_names))
    ):
        raise ValueError("Values must have shape [time, node, feature]")
    if (len(set(node_ids)) != len(node_ids) or len(set(feature_names)) != len(feature_names)):
        raise ValueError("Node IDs and feature names must be unique")
    if mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("mask must be Boolean and match values")
    if fit_window.shape != (values.shape[0],) or fit_window.dtype.kind != "b":
        raise ValueError("fit_window must be Boolean with shape [time]")

    selected = values[fit_window] # removes excluded time rows before fitting
    valid = mask[fit_window] & np.isfinite(selected) # Which observations are valid
    count = valid.sum(axis=0) # Count of valid training observations seperately for every stock and feature 
    fitted = count >= 2
    if not fitted.any():
        raise ValueError("No node/feature has two valid training observations")

    clean = np.where(valid, selected, 0.0) # Replace invalid observations with 0.0 to avoid NaN propagation in mean and variance calculations
    mean = np.divide(
        clean.sum(axis=0), count,
        out=np.full(count.shape, np.nan), where=fitted,
    )

    centered = np.zeros_like(clean)
    np.subtract(clean, mean, out=centered, where=valid & fitted[None, :, :])
    variance = np.divide(
        (centered ** 2).sum(axis=0), count,
        out=np.full(count.shape, np.nan), where=fitted,
    )
    scale = np.sqrt(variance)
    scale[fitted & (scale == 0.0)] = 1.0

    return StandardizerState(node_ids, feature_names, mean, scale, count, fitted)

#
def apply_standardizer(values, mask, state, *, node_ids, feature_names):
    """Apply saved training statistics; return standardized values and mask."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask)
    node_ids, feature_names = tuple(node_ids), tuple(feature_names)

    if node_ids != state.node_ids or feature_names != state.feature_names:
        raise ValueError("Node and feature ordering must match the fitted state")
    shape = (len(node_ids), len(feature_names))
    if values.ndim != 3 or values.shape[1:] != shape:
        raise ValueError("Values must have shape [time, node, feature]")
    if mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("mask must be Boolean and match values")

    mean, scale, count, fitted = (
        np.asarray(array)
        for array in (state.mean, state.scale, state.count, state.fitted_mask)
    )
    if any(array.shape != shape for array in (mean, scale, count, fitted)):
        raise ValueError("Fitted arrays must match the node and feature axes")
    if (
        fitted.dtype.kind != "b"
        or count.dtype.kind not in "iu"
        or (count < 0).any()
        or not np.array_equal(fitted, count >= 2)
    ):
        raise ValueError("Fitted counts and masks must be valid and consistent")
    if (
        not np.isfinite(mean[fitted]).all()
        or not np.isfinite(scale[fitted]).all()
        or (scale[fitted] <= 0).any()
    ):
        raise ValueError("Fitted means must be finite and scales finite and positive")

    valid = mask & np.isfinite(values) & fitted[None, :, :]
    standardized = np.full(values.shape, np.nan)
    with np.errstate(over="ignore"):
        np.subtract(values, mean, out=standardized, where=valid)
        np.divide(standardized, scale, out=standardized, where=valid)

    valid &= np.isfinite(standardized)
    standardized[~valid] = np.nan
    return standardized, valid