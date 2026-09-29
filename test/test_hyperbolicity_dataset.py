"""Next tasks: convert a point cloud to distances, then scan all quadruples.

Implement these functions in hyperbolicity/core.py:
    euclidean_distances(points: np.ndarray) -> np.ndarray
    exact_delta(distances: np.ndarray) -> float

For this stage, points is an already valid finite two-dimensional array with
one point per row. Distances is an already valid NxN metric matrix, N >= 4.
Input validation, graph construction, and sampling are separate later tasks.
These tests intentionally fail until the functions are implemented.
"""

from __future__ import annotations

from importlib import import_module
from itertools import permutations

import numpy as np
import pytest


# Point 0 is the center; points 1--4 are the unit square's corners in order.
# The maximum-delta quadruple excludes point 0, so checking only quadruples
# containing the first point gives the wrong answer.
SQUARE_WITH_CENTER_DISTANCES = np.array(
    [
        [0, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)],
        [np.sqrt(0.5), 0, 1, np.sqrt(2), 1],
        [np.sqrt(0.5), 1, 0, 1, np.sqrt(2)],
        [np.sqrt(0.5), np.sqrt(2), 1, 0, 1],
        [np.sqrt(0.5), 1, np.sqrt(2), 1, 0],
    ],
    dtype=float,
)


def _call(name: str, values: np.ndarray):
    """Make unfinished tasks readable failures instead of collection errors."""
    try:
        module = import_module("hyperbolicity.core")
    except ModuleNotFoundError as error:
        if error.name not in {"hyperbolicity", "hyperbolicity.core"}:
            raise
        pytest.fail(
            f"Create hyperbolicity/core.py and implement {name}.",
            pytrace=False,
        )

    function = getattr(module, name, None)
    assert callable(function), f"Implement {name} in hyperbolicity/core.py."
    return function(values)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        pytest.param(
            np.array([[0, 0], [3, 0], [0, 4]], dtype=float),
            np.array([[0, 3, 4], [3, 0, 5], [4, 5, 0]], dtype=float),
            id="three-points-two-features-3-4-5-triangle",
        ),
        pytest.param(
            np.array([[0, 0, 0], [1, 2, 2]], dtype=float),
            np.array([[0, 3], [3, 0]], dtype=float),
            id="two-points-three-features",
        ),
        pytest.param(
            np.array([[0.5, 0.5], [0, 0], [1, 0], [1, 1], [0, 1]]),
            SQUARE_WITH_CENTER_DISTANCES,
            id="square-with-center",
        ),
    ],
)
def test_euclidean_distances_between_rows(
    points: np.ndarray, expected: np.ndarray
) -> None:
    result = _call("euclidean_distances", points)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(points), len(points))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_euclidean_distances_preserves_input_coordinates() -> None:
    points = np.array([[2, -3], [5, -3], [2, 1]], dtype=float)
    original = points.copy()

    _call("euclidean_distances", points)

    np.testing.assert_array_equal(points, original)


def test_exact_delta_finds_worst_quadruple_excluding_first_point() -> None:
    # The four corners contribute sqrt(2)-1; any triple of corners plus the
    # center contributes only (sqrt(2)-1)/2.
    result = _call("exact_delta", SQUARE_WITH_CENTER_DISTANCES.copy())

    assert result == pytest.approx(np.sqrt(2) - 1)


def test_exact_delta_of_six_vertex_weighted_tree_is_zero() -> None:
    # A star with center 0 and five leaves, with edge lengths 1, 2, 3, 4, 5.
    distances = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 3, 4, 5, 6],
            [2, 3, 0, 5, 6, 7],
            [3, 4, 5, 0, 7, 8],
            [4, 5, 6, 7, 0, 9],
            [5, 6, 7, 8, 9, 0],
        ],
        dtype=float,
    )

    assert _call("exact_delta", distances) == pytest.approx(0.0)


def test_exact_delta_is_independent_of_point_order() -> None:
    for order in permutations(range(5)):
        distances = SQUARE_WITH_CENTER_DISTANCES[np.ix_(order, order)]
        assert _call("exact_delta", distances) == pytest.approx(np.sqrt(2) - 1), order


def test_exact_delta_preserves_input_distances() -> None:
    distances = SQUARE_WITH_CENTER_DISTANCES.copy()
    original = distances.copy()

    _call("exact_delta", distances)

    np.testing.assert_array_equal(distances, original)
