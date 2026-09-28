import numpy as np
import pytest
from hypershift.data.hypergraph import Hypergraph
from hypershift.geometry.hyperbolicity import (
    feature_hyperbolicity, gromov_delta, hypergraph_hyperbolicity, s_distance_matrix, sampled_delta,
)


def test_tree_has_zero_delta():
    path = Hypergraph(5, ((0, 1), (1, 2), (2, 3), (3, 4)))
    assert gromov_delta(s_distance_matrix(path)) == 0.0


def test_four_cycle_delta_is_one():
    c4 = Hypergraph(4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    assert gromov_delta(s_distance_matrix(c4)) == pytest.approx(1.0)


def test_single_hyperedge_is_clique_zero_delta():
    D = s_distance_matrix(Hypergraph(4, ((0, 1, 2, 3),)))
    assert D[0, 3] == 1 and gromov_delta(D) == 0.0


def test_s_parameter():
    hg = Hypergraph(3, ((0, 1), (0, 1, 2)))
    assert s_distance_matrix(hg, s=2)[0, 1] == 1
    assert np.isinf(s_distance_matrix(hg, s=2)[0, 2])


def test_hypergraph_hyperbolicity_uses_lcc():
    hg = Hypergraph(7, ((0, 1), (1, 2), (2, 3), (3, 0), (5, 6)))
    out = hypergraph_hyperbolicity(hg, sample=10, repeats=2)
    assert out["lcc_size"] == 4 and out["delta_max"] == pytest.approx(1.0)


def test_points_on_a_line_have_zero_rel_delta():
    X = np.linspace(0, 1, 30)[:, None]
    assert feature_hyperbolicity(X, sample=30, repeats=1)["delta_rel"] == pytest.approx(0.0, abs=1e-6)
