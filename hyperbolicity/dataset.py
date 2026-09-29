"In memory tests"

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .core import (
    euclidean_distances,
    exact_delta,
    relative_delta,
    sampled_delta,
    s_walk_adjacency,
    shortest_path_distances,
    validate_distances,
)


def prices_to_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Convert an aligned close-price table to timestamped log returns."""
    if not isinstance(close, pd.DataFrame):
        raise ValueError("close must be a pandas DataFrame.")
    if len(close.index) < 2:
        raise ValueError("At least two price rows are required.")
    if len(close.columns) == 0:
        raise ValueError("At least one stock column is required.")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Price rows must use a DatetimeIndex.")
    if close.index.has_duplicates:
        raise ValueError("Price timestamps must be unique.")
    if close.index.hasnans:
        raise ValueError("Price timestamps cannot contain missing values.")
    if not close.index.is_monotonic_increasing:
        raise ValueError("Price timestamps must be in chronological order.")
    if isinstance(close.columns, pd.MultiIndex):
        raise ValueError("Stock columns must use flat ticker identifiers.")
    if close.columns.has_duplicates:
        raise ValueError("Stock identifiers must be unique.")

    try:
        prices = close.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Close prices must be numeric.") from error

    if not np.isfinite(prices).all():
        raise ValueError("Close prices must be finite and complete.")
    if np.any(prices <= 0.0):
        raise ValueError("Close prices must be strictly positive.")

    log_returns = np.diff(np.log(prices), axis=0)
    return pd.DataFrame(
        log_returns,
        index=close.index[1:],
        columns=close.columns,
    )


def stock_feature_vectors(returns: pd.DataFrame) -> np.ndarray:
    """Return one independent floating-point return-history vector per stock."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Returns must contain at least one row and one stock.")

    try:
        values = returns.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Returns must be numeric.") from error

    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite values.")

    return values.T.copy()


def summarize_dataset(
    close: pd.DataFrame,
    incidence: pd.DataFrame,
    *,
    dataset: str = "market",
    s: int = 1,
    method: Literal["exact", "sampled"] = "sampled",
    n_samples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Calculate structural and temporal-feature hyperbolicity side by side."""
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)):
        raise ValueError("s must be a positive integer.")
    if s <= 0:
        raise ValueError("s must be a positive integer.")
    s = int(s)

    if method not in {"exact", "sampled"}:
        raise ValueError("method must be either 'exact' or 'sampled'.")
    if not isinstance(incidence, pd.DataFrame):
        raise ValueError("incidence must be a pandas DataFrame.")
    if len(incidence.columns) == 0:
        raise ValueError("Incidence must contain at least one hyperedge.")
    if incidence.index.has_duplicates:
        raise ValueError("Incidence stock identifiers must be unique.")

    returns = prices_to_log_returns(close)
    n_nodes = len(close.columns)
    if n_nodes < 4:
        raise ValueError("At least four stocks are required.")

    missing = close.columns[~close.columns.isin(incidence.index)]
    extra = incidence.index[~incidence.index.isin(close.columns)]
    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(
            "Incidence rows must contain exactly the price-table stock identifiers."
        )

    aligned_incidence = incidence.reindex(index=close.columns)
    if aligned_incidence.isna().to_numpy().any():
        raise ValueError("Incidence memberships cannot be missing.")

    membership_values = aligned_incidence.to_numpy(copy=True)
    if not np.isin(membership_values, [0, 1]).all():
        raise ValueError("Incidence memberships must be binary.")
    membership_values = membership_values.astype(np.int64, copy=False)

    adjacency = s_walk_adjacency(membership_values, s=s)
    hypergraph_distances = shortest_path_distances(adjacency)
    validate_distances(hypergraph_distances)

    feature_vectors = stock_feature_vectors(returns)
    feature_distances = euclidean_distances(feature_vectors)
    validate_distances(feature_distances)

    if method == "exact":
        delta_hg = exact_delta(hypergraph_distances)
        delta_features = exact_delta(feature_distances)
        reported_method = "exact"
        reported_n_samples: int | None = None
        reported_seed: int | None = None
    else:
        delta_hg = sampled_delta(
            hypergraph_distances,
            n_samples=n_samples,
            seed=seed,
        )
        delta_features = sampled_delta(
            feature_distances,
            n_samples=n_samples,
            seed=seed,
        )
        reported_method = "sampled_lower_bound"
        reported_n_samples = int(n_samples)
        reported_seed = int(seed)

    delta_rel = relative_delta(delta_features, feature_distances)

    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_timesteps": len(returns),
                "n_price_rows": len(close),
                "n_nodes": n_nodes,
                "delta_hg": delta_hg,
                "delta_rel": delta_rel,
                "delta_features": delta_features,
                "diameter_hg": float(np.max(hypergraph_distances)),
                "diameter_features": float(np.max(feature_distances)),
                "method": reported_method,
                "s": s,
                "n_samples": reported_n_samples,
                "seed": reported_seed,
            }
        ]
    )
