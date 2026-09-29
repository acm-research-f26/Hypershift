"""First structural task: turn a binary incidence matrix into s-walk adjacency.

Implement hyperbolicity.core.s_walk_adjacency(incidence, s=1).
Rows are stocks, columns are hyperedges. Two DISTINCT stocks are adjacent
when they share at least s hyperedges. Inputs here are valid; no validation
or shortest-path search is required at this checkpoint.
"""

from importlib import import_module

import numpy as np
import pytest


INCIDENCE = np.array([[1, 0], [1, 1], [1, 1], [0, 1]], dtype=bool)


def _adjacency(incidence: np.ndarray, s: int = 1) -> np.ndarray:
    try:
        module = import_module("hyperbolicity.core")
    except ModuleNotFoundError as error:
        if error.name not in {"hyperbolicity", "hyperbolicity.core"}:
            raise
        pytest.fail(
            "Create hyperbolicity/core.py and implement s_walk_adjacency(incidence, s=1).",
            pytrace=False,
        )
    function = getattr(module, "s_walk_adjacency", None)
    assert callable(function), "Implement s_walk_adjacency in hyperbolicity/core.py."
    result = function(incidence, s=s)
    assert isinstance(result, np.ndarray), "Return a NumPy adjacency matrix."
    return result


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        (1, [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]),
        (2, [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
    ],
)
def test_s_walk_adjacency_counts_shared_hyperedges(s, expected):
    np.testing.assert_array_equal(_adjacency(INCIDENCE, s), np.array(expected, dtype=bool))


def test_s_walk_adjacency_singleton_groups_do_not_create_self_loops():
    np.testing.assert_array_equal(
        _adjacency(np.eye(4, dtype=int)), np.zeros((4, 4), dtype=bool)
    )


def test_s_walk_adjacency_is_independent_of_hyperedge_order():
    expected = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(_adjacency(INCIDENCE[:, ::-1], s=2), expected)
