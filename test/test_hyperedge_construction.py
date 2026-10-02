"""Tests for the first hyperedge builders; no market downloads are needed.

Exercise common/incidence.py and covariance_knn/constructor.py. The KNN tests
use orthogonal return series with known correlations, not a second builder
as an oracle. These are baseline construction tests, not evidence that
pairwise correlation captures all higher-order relationships.
"""

from copy import deepcopy
from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _call(name, *args, **kwargs):
    return getattr(import_module("hyperedges"), name)(*args, **kwargs)


def _returns():
    return pd.DataFrame(
        {"A": [1, -1, 1, -1], "B": [-1, 1, -1, 1], "C": [1, 1, -1, -1]},
        index=pd.date_range("2024-01-02 09:30", periods=4, freq="15min"),
        dtype=float,
    )


def test_incidence_preserves_identifiers_overlap_and_uncovered_stocks():
    nodes = ["C", "A", "D", "B"]
    groups = {"ownership": ["A", "B"], "industry": ["B", "C"], "same_members": ["B", "A"]}
    original_nodes, original_groups = nodes.copy(), deepcopy(groups)
    result = _call("incidence_from_groups", nodes, groups)
    expected = pd.DataFrame(
        [[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]],
        index=nodes, columns=list(groups), dtype=bool,
    )
    pd.testing.assert_frame_equal(result, expected, check_names=False)
    assert nodes == original_nodes and groups == original_groups


@pytest.mark.parametrize("nodes", [[], ["A", "A"], ["A", 7]])
def test_incidence_rejects_invalid_node_identifiers(nodes):
    with pytest.raises(ValueError):
        _call("incidence_from_groups", nodes, {"group": ["A"]})


@pytest.mark.parametrize(
    "groups", [{}, {"empty": []}, {"duplicate": ["A", "A"]}, {"unknown": ["A", "Z"]}],
    ids=["no-groups", "empty-group", "duplicate-members", "unknown-member"],
)
def test_incidence_rejects_invalid_groups(groups):
    with pytest.raises(ValueError):
        _call("incidence_from_groups", ["A", "B"], groups)


@pytest.mark.parametrize(
    ("absolute", "expected"),
    [(True, [[1, 1, 1], [1, 1, 0], [0, 0, 1]]),
     (False, [[1, 0, 1], [0, 1, 0], [1, 1, 1]])],
    ids=["absolute-keeps-anticorrelated-neighbor", "signed-prefers-zero-to-negative"],
)
def test_correlation_knn_distinguishes_absolute_and_signed_similarity(absolute, expected):
    returns = _returns()
    result = _call("correlation_knn_hyperedges", returns, neighbors=1, absolute=absolute)
    pd.testing.assert_frame_equal(
        result, pd.DataFrame(expected, index=list("ABC"), columns=["knn:A", "knn:B", "knn:C"], dtype=bool),
        check_names=False,
    )


def test_correlation_knn_breaks_ties_by_ticker_not_input_order():
    # Every pair is orthogonal: all off-diagonal correlations are exactly zero.
    returns = pd.DataFrame(
        {"Z": [1, -1, 1, -1], "M": [1, 1, -1, -1], "A": [1, -1, -1, 1]},
        index=_returns().index,
    )
    result = _call("correlation_knn_hyperedges", returns, neighbors=1)
    expected = pd.DataFrame(
        [[1, 0, 0], [0, 1, 1], [1, 1, 1]],
        index=["Z", "M", "A"], columns=["knn:Z", "knn:M", "knn:A"], dtype=bool,
    )
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_correlation_knn_equivariance_under_stock_column_permutation():
    returns = _returns()
    first = _call("correlation_knn_hyperedges", returns, neighbors=1)
    reordered = _call("correlation_knn_hyperedges", returns[["C", "B", "A"]], neighbors=1)
    assert reordered.index.tolist() == ["C", "B", "A"]
    assert reordered.columns.tolist() == ["knn:C", "knn:B", "knn:A"]
    pd.testing.assert_frame_equal(first, reordered.reindex(index=first.index, columns=first.columns), check_names=False)


def test_correlation_knn_includes_center_and_exact_neighbor_count_without_mutation():
    returns = _returns()
    returns["D"] = returns["A"] + returns["C"]
    original = returns.copy(deep=True)
    result = _call("correlation_knn_hyperedges", returns, neighbors=2)
    assert result.shape == (4, 4)
    assert result.dtypes.eq(bool).all()
    assert result.sum(axis=0).eq(3).all()
    for ticker in returns.columns:
        assert result.loc[ticker, f"knn:{ticker}"]
    pd.testing.assert_frame_equal(returns, original)


def test_correlation_knn_rejects_invalid_neighbor_counts():
    for neighbors in (True, False, 1.0, 1.5, -1, 0, 3, 4):
        with pytest.raises(ValueError):
            _call("correlation_knn_hyperedges", _returns(), neighbors=neighbors)


@pytest.mark.parametrize("problem", ["values", "constant", "labels", "timestamps", "dimensions"])
def test_correlation_knn_rejects_ambiguous_or_undefined_return_windows(problem):
    invalid_windows = []
    if problem == "values":
        for value in (np.nan, np.inf):
            invalid = _returns()
            invalid.iloc[1, 0] = value
            invalid_windows.append(invalid)
        invalid_windows.append(_returns().astype(str))
    elif problem == "constant":
        invalid = _returns()
        invalid["C"] = 1.0
        invalid_windows.append(invalid)
    elif problem == "labels":
        for labels in (["A", "A", "C"], ["A", "B", 7], pd.MultiIndex.from_tuples([("price", x) for x in "ABC"])):
            invalid = _returns()
            invalid.columns = labels
            invalid_windows.append(invalid)
    elif problem == "timestamps":
        invalid_windows.append(_returns().iloc[::-1])
        for index in ([0, 1, 2, 3], [_returns().index[0]] * 4):
            invalid = _returns()
            invalid.index = index
            invalid_windows.append(invalid)
    else:
        invalid_windows.extend([_returns().iloc[:1], _returns().iloc[:0], _returns()[["A"]]])
    for invalid in invalid_windows:
        with pytest.raises(ValueError):
            _call("correlation_knn_hyperedges", invalid, neighbors=1)
