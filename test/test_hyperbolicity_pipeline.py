"""Later calculator checkpoints; intentionally red until the skeletons are filled.

Fixtures have independently known answers. No market-data request is made.
Run a checkpoint with pytest's -k shortest, -k validate, -k sampled,
-k prices, -k feature, or -k summary selector.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _call(module, name, *args, **kwargs):
    return getattr(import_module(f"hyperbolicity.{module}"), name)(*args, **kwargs)


def _square_distances():
    return np.array(
        [[0, 1, np.sqrt(2), 1], [1, 0, 1, np.sqrt(2)],
         [np.sqrt(2), 1, 0, 1], [1, np.sqrt(2), 1, 0]], dtype=float
    )


def _market_fixture():
    # Each STOCK is a corner of a square in two-dimensional return space.
    returns = np.array([[0, 0.1, 0.1, 0], [0, 0, 0.1, 0.1]])
    prices = 100 * np.exp(np.vstack([np.zeros(4), returns.cumsum(axis=0)]))
    close = pd.DataFrame(
        prices, index=pd.date_range("2024-01-02", periods=3), columns=list("ABCD")
    )
    incidence = pd.DataFrame(
        [[1, 0], [1, 1], [1, 1], [0, 1]],
        index=list("ABCD"), columns=["group_1", "group_2"],
    )
    return close, incidence, returns


def test_shortest_paths_use_hops_and_preserve_adjacency():
    adjacency = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=bool
    )
    original = adjacency.copy()
    result = _call("core", "shortest_path_distances", adjacency)
    np.testing.assert_array_equal(
        result, [[0, 1, 1, 2], [1, 0, 1, 1], [1, 1, 0, 1], [2, 1, 1, 0]]
    )
    np.testing.assert_array_equal(adjacency, original)


def test_shortest_paths_reject_disconnected_graph():
    adjacency = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    with pytest.raises(ValueError):
        _call("core", "shortest_path_distances", adjacency)


def test_relative_delta_uses_full_diameter_and_is_scale_invariant():
    distances = _square_distances()
    delta = np.sqrt(2) - 1
    expected = 2 - np.sqrt(2)
    for scale in (0.1, 1, 10):
        result = _call("core", "relative_delta", scale * delta, scale * distances)
        assert result == pytest.approx(expected)


def test_relative_delta_is_undefined_for_zero_diameter():
    assert _call("core", "relative_delta", 0.0, np.zeros((4, 4))) is None


def test_sampled_delta_with_four_points_always_finds_only_quadruple():
    for seed in (0, 4, 19):
        result = _call("core", "sampled_delta", _square_distances(), n_samples=1, seed=seed)
        assert result == pytest.approx(np.sqrt(2) - 1)


def test_sampled_delta_is_reproducible_and_a_lower_bound():
    # Center first; some quadruples give only half the global maximum.
    radius = np.sqrt(0.5)
    distances = np.zeros((5, 5))
    distances[1:, 1:] = _square_distances()
    distances[0, 1:] = distances[1:, 0] = radius
    first = _call("core", "sampled_delta", distances, n_samples=3, seed=7)
    second = _call("core", "sampled_delta", distances, n_samples=3, seed=7)
    assert first == second
    assert 0 <= first <= np.sqrt(2) - 1 + 1e-12
    assert first == pytest.approx((np.sqrt(2) - 1) / 2) or first == pytest.approx(np.sqrt(2) - 1)


@pytest.mark.parametrize(
    "distances",
    [np.zeros((2, 3)), np.array([[0, np.inf], [np.inf, 0]]),
     np.array([[0, np.nan], [np.nan, 0]]), np.array([[0, 1], [2, 0]]),
     np.array([[1, 1], [1, 0]]), np.array([[0, -1], [-1, 0]]),
     np.array([[0, 1, 3], [1, 0, 1], [3, 1, 0]])],
    ids=["nonsquare", "infinite", "nan", "asymmetric", "diagonal", "negative", "triangle"],
)
def test_validate_distances_rejects_invalid_metric(distances):
    with pytest.raises(ValueError):
        _call("core", "validate_distances", distances)


def test_validate_distances_permits_distinct_rows_at_same_location():
    distances = np.array([[0, 0, 2], [0, 0, 2], [2, 2, 0]], dtype=float)
    original = distances.copy()
    assert _call("core", "validate_distances", distances) is None
    np.testing.assert_array_equal(distances, original)


def test_prices_become_log_returns_with_remaining_timestamps():
    close, _, expected_values = _market_fixture()
    original = close.copy(deep=True)
    result = _call("dataset", "prices_to_log_returns", close)
    expected = pd.DataFrame(expected_values, index=close.index[1:], columns=close.columns)
    pd.testing.assert_frame_equal(result, expected, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(close, original)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_prices_reject_missing_nonfinite_or_nonpositive_values(invalid):
    close, _, _ = _market_fixture()
    close.iloc[1, 2] = invalid
    with pytest.raises(ValueError):
        _call("dataset", "prices_to_log_returns", close)


@pytest.mark.parametrize("problem", ["unsorted", "duplicate_time", "duplicate_stock", "not_datetime"])
def test_prices_reject_ambiguous_axes(problem):
    close, _, _ = _market_fixture()
    if problem == "unsorted":
        close = close.iloc[::-1]
    elif problem == "duplicate_time":
        close.index = [close.index[0], close.index[0], close.index[2]]
    elif problem == "duplicate_stock":
        close.columns = list("ABCC")
    else:
        close.index = [0, 1, 2]
    with pytest.raises(ValueError):
        _call("dataset", "prices_to_log_returns", close)


def test_stock_feature_vectors_make_stocks_rows_without_aliasing_input():
    returns = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["C", "A", "B"])
    result = _call("dataset", "stock_feature_vectors", returns)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_equal(result, [[1, 4], [2, 5], [3, 6]])
    result[0, 0] = 999
    assert returns.iloc[0, 0] == 1


def test_summary_separates_structural_and_feature_hyperbolicity():
    close, incidence, _ = _market_fixture()
    result = _call("dataset", "summarize_dataset", close, incidence, dataset="square", method="exact")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    expected_columns = {
        "dataset", "n_timesteps", "n_price_rows", "n_nodes", "delta_hg", "delta_rel",
        "delta_features", "diameter_hg", "diameter_features", "method", "s", "n_samples", "seed",
    }
    assert set(result.columns) == expected_columns
    row = result.iloc[0]
    assert row["dataset"] == "square"
    assert (row["n_timesteps"], row["n_price_rows"], row["n_nodes"]) == (2, 3, 4)
    assert row["delta_hg"] == pytest.approx(0.5)
    assert row["diameter_hg"] == pytest.approx(2.0)
    assert row["delta_features"] == pytest.approx(0.1 * (np.sqrt(2) - 1))
    assert row["diameter_features"] == pytest.approx(0.1 * np.sqrt(2))
    assert row["delta_rel"] == pytest.approx(2 - np.sqrt(2))
    assert row["method"] == "exact" and row["s"] == 1
    assert pd.isna(row["n_samples"]) and pd.isna(row["seed"])


def test_summary_is_invariant_to_incidence_row_order():
    close, incidence, _ = _market_fixture()
    first = _call("dataset", "summarize_dataset", close, incidence, method="exact")
    second = _call("dataset", "summarize_dataset", close, incidence.loc[list("DBAC")], method="exact")
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "nonbinary"])
def test_summary_rejects_incompatible_memberships(problem):
    close, incidence, _ = _market_fixture()
    if problem == "missing":
        incidence = incidence.drop(index="C")
    elif problem == "extra":
        incidence.loc["E"] = [1, 0]
    elif problem == "duplicate":
        incidence.index = list("ABCC")
    else:
        incidence = incidence.astype(float)
        incidence.iloc[0, 0] = 0.5
    with pytest.raises(ValueError):
        _call("dataset", "summarize_dataset", close, incidence, method="exact")


def test_summary_labels_sampling_as_lower_bound():
    close, incidence, _ = _market_fixture()
    result = _call(
        "dataset", "summarize_dataset", close, incidence,
        method="sampled", n_samples=1, seed=42,
    )
    row = result.iloc[0]
    assert row["method"] == "sampled_lower_bound"
    assert row["n_samples"] == 1 and row["seed"] == 42
    assert row["delta_hg"] == pytest.approx(0.5)
    assert row["delta_rel"] == pytest.approx(2 - np.sqrt(2))
