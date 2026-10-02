"""Covariance estimation, explicit similarities, and deterministic KNN groups."""

from numbers import Integral

import numpy as np
import pandas as pd

from ..common.incidence import incidence_from_groups
from ..common.validation import validate_identifiers


def _validate_returns(returns):
    if not isinstance(returns, pd.DataFrame) or isinstance(returns.columns, pd.MultiIndex):
        raise ValueError("returns must be a DataFrame with a flat stock axis")
    validate_identifiers(returns.columns, "node")
    times = returns.index
    if (not isinstance(times, pd.DatetimeIndex) or times.hasnans
            or not times.is_unique or not times.is_monotonic_increasing):
        raise ValueError("Return timestamps must be present, unique, and chronological")
    if returns.shape[0] < 2 or returns.shape[1] < 2:
        raise ValueError("At least two observations and two stocks are required")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
           or pd.api.types.is_complex_dtype(dtype) for dtype in returns.dtypes):
        raise ValueError("Returns must be real numeric observations")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns must be finite; select usable history before construction")


def estimate_covariance(complete_returns: pd.DataFrame) -> pd.DataFrame:
    _validate_returns(complete_returns)
    values = complete_returns.to_numpy(dtype=float)
    covariance = np.cov(values, rowvar=False, ddof=1)
    if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
        raise ValueError("Covariance requires nonconstant stock histories")
    return pd.DataFrame(covariance, index=complete_returns.columns, columns=complete_returns.columns)


def covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    _validate_square(covariance)
    values = covariance.to_numpy(dtype=float)
    variance = np.diag(values)
    if (variance <= 0).any():
        raise ValueError("Correlation requires positive variances")
    correlation = values / np.sqrt(np.outer(variance, variance))
    if (np.abs(correlation) > 1 + 1e-10).any():
        raise ValueError("Invalid covariance matrix")
    return pd.DataFrame(np.clip(correlation, -1, 1), index=covariance.index, columns=covariance.columns)


def _validate_square(matrix):
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Similarity must be a labeled DataFrame")
    validate_identifiers(matrix.index, "node")
    if not matrix.index.equals(matrix.columns) or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Matrix axes must match and values must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Matrix must be symmetric")


def knn_groups_from_similarity(similarity, neighbors=5, absolute=True):
    _validate_square(similarity)
    nodes = tuple(similarity.index)
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or not 0 < neighbors < len(nodes):
        raise ValueError("neighbors must be a positive integer smaller than the eligible universe")
    if not isinstance(absolute, (bool, np.bool_)):
        raise ValueError("absolute must be Boolean")
    values = similarity.abs() if absolute else similarity
    return {
        f"knn:{center}": (center, *sorted(
            (node for node in nodes if node != center),
            key=lambda node: (-values.loc[center, node], node),
        )[:int(neighbors)])
        for center in nodes
    }


def correlation_knn_hyperedges(returns: pd.DataFrame, *, neighbors: int = 5, absolute: bool = True) -> pd.DataFrame:
    similarity = covariance_to_correlation(estimate_covariance(returns))
    return incidence_from_groups(returns.columns, knn_groups_from_similarity(similarity, neighbors, absolute))


def build_covariance_knn_family(context, params):
    from .history import select_covariance_history
    from ..common.types import make_hyperedge_family

    returns, diagnostics = select_covariance_history(
        context.history, params.get("coverage_threshold", 0.95), params.get("min_rows", 30)
    )
    covariance = estimate_covariance(returns)
    absolute = params.get("absolute", True)
    if not isinstance(absolute, bool):
        raise ValueError("absolute must be Boolean")
    mode = params.get("similarity", "absolute_correlation" if absolute else "signed_correlation")
    if mode not in {"absolute_correlation", "signed_correlation", "absolute_covariance"}:
        raise ValueError(f"Unsupported similarity {mode!r}")
    if "absolute" in params and absolute != (mode != "signed_correlation"):
        raise ValueError("absolute and similarity parameters disagree")
    similarity = covariance if mode == "absolute_covariance" else covariance_to_correlation(covariance)
    groups = knn_groups_from_similarity(similarity, params.get("neighbors", 5), mode != "signed_correlation")
    return make_hyperedge_family(groups, context, diagnostics=diagnostics,
                                state={"covariance": covariance.to_numpy(), "eligible_nodes": list(returns.columns)})
