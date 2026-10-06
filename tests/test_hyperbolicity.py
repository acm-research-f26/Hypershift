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


def test_all_bases_delta_matches_known_values():
    from hypershift.geometry.hyperbolicity import gromov_delta_all_bases
    c4 = s_distance_matrix(Hypergraph(4, ((0, 1), (1, 2), (2, 3), (3, 0))))
    assert gromov_delta_all_bases(c4) == pytest.approx(1.0)
    tree = s_distance_matrix(Hypergraph(5, ((0, 1), (1, 2), (2, 3), (3, 4))))
    assert gromov_delta_all_bases(tree) == 0.0
    c5 = s_distance_matrix(Hypergraph(5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))))
    assert gromov_delta_all_bases(c5) == pytest.approx(0.5)


def test_s_walk_toy_distances():
    # edges A={0,1,2}, B={1,2,3}, C={3,4}: 0-1 share A (1); 0-3 share none -> via 1 or 2 (2); 0-4 -> 3
    hg = Hypergraph(5, ((0, 1, 2), (1, 2, 3), (3, 4)))
    D1 = s_distance_matrix(hg, 1)
    assert D1[0, 1] == 1 and D1[0, 3] == 2 and D1[0, 4] == 3
    D2 = s_distance_matrix(hg, 2)  # 1,2 share two hyperedges; 0-1 share only A
    assert D2[1, 2] == 1 and np.isinf(D2[0, 1])


def test_khrulkov_rel_delta_line_zero_and_circle_positive():
    from hypershift.geometry.hyperbolicity import khrulkov_delta_rel
    line = np.linspace(0, 1, 40)[:, None]
    assert khrulkov_delta_rel(line, 40, 2, replace=False)["mean"] == pytest.approx(0.0, abs=1e-6)
    t = np.linspace(0, 2 * np.pi, 41)[:-1]
    circ = np.stack([np.cos(t), np.sin(t)], 1)
    assert khrulkov_delta_rel(circ, 40, 2, replace=False)["mean"] > 0.1
