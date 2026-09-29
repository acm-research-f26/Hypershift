"""First task: implement hyperbolicity.core.four_point_delta(distances).

Input is an already valid 4x4 NumPy metric-distance matrix; output is its
four-point Gromov delta as a float. Validation and larger datasets come later.
These tests intentionally fail until you implement the function.
"""

from __future__ import annotations

from importlib import import_module
from itertools import permutations

import numpy as np
import pytest


CYCLE = np.array(
    [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]],
    dtype=float,
)
SQUARE = np.array(
    [[0, 1, np.sqrt(2), 1], [1, 0, 1, np.sqrt(2)],
     [np.sqrt(2), 1, 0, 1], [1, np.sqrt(2), 1, 0]],
    dtype=float,
)


def _delta(distances: np.ndarray) -> float:
    """Keep a missing implementation a readable failure, not a collection error."""
    try:
        module = import_module("hyperbolicity.core")
    except ModuleNotFoundError as error:
        if error.name not in {"hyperbolicity", "hyperbolicity.core"}:
            raise
        pytest.fail(
            "First task: create hyperbolicity/core.py and implement "
            "four_point_delta(distances: np.ndarray) -> float.",
            pytrace=False,
        )

    function = getattr(module, "four_point_delta", None)
    assert callable(function), "Implement four_point_delta in hyperbolicity/core.py."
    return function(distances)


@pytest.mark.parametrize(
    ("distances", "expected"),
    [
        pytest.param(
            np.abs(np.subtract.outer([0, 1, 3, 7], [0, 1, 3, 7])),
            0.0,
            id="points-on-a-line",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [1, 0, 3, 4],
                      [2, 3, 0, 5], [3, 4, 5, 0]], dtype=float),
            0.0,
            id="weighted-star-tree",
        ),
        pytest.param(np.ones((4, 4)) - np.eye(4), 0.0, id="equilateral"),
        pytest.param(CYCLE, 1.0, id="four-cycle-shortest-paths"),
        pytest.param(SQUARE, np.sqrt(2) - 1, id="square-euclidean-distances"),
    ],
)
def test_known_four_point_metrics(distances: np.ndarray, expected: float) -> None:
    assert _delta(distances) == pytest.approx(expected)


@pytest.mark.parametrize("scale", [0.25, 2.5, 10.0])
def test_scaling_distances_scales_delta(scale: float) -> None:
    assert _delta(scale * SQUARE) == pytest.approx(scale * (np.sqrt(2) - 1))


def test_relabeling_points_does_not_change_delta() -> None:
    for order in permutations(range(4)):
        relabeled = SQUARE[np.ix_(order, order)]
        assert _delta(relabeled) == pytest.approx(np.sqrt(2) - 1), order
