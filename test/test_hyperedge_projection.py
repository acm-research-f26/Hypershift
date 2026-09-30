"""A scientific limit of the THINK-style metric, not a calculator defect."""

import numpy as np

from hyperbolicity.core import exact_delta, s_walk_adjacency, shortest_path_distances


def test_s1_hyperbolicity_cannot_identify_irreducible_hyperedges():
    # First hypergraph: ABC is one triple, and CD is a pair.
    triple_and_pair = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    # Second hypergraph: only pairs AB, AC, BC, CD. No triple is present.
    pairs_only = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]])

    first_adjacency = s_walk_adjacency(triple_and_pair, s=1)
    second_adjacency = s_walk_adjacency(pairs_only, s=1)
    np.testing.assert_array_equal(first_adjacency, second_adjacency)

    first_distances = shortest_path_distances(first_adjacency)
    second_distances = shortest_path_distances(second_adjacency)
    np.testing.assert_array_equal(first_distances, second_distances)
    assert exact_delta(first_distances) == exact_delta(second_distances)
