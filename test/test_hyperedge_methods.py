"""Scientific fixtures: common witnesses, parity, pairwise nulls, and overlapping covers."""

import numpy as np
import pandas as pd
import pytest

from hyperedges.event_dowker import (build_event_dowker_family, measure_group_event_support, mine_recurring_groups)
from hyperedges.joint_information import (compare_joint_models, derive_pairwise_marginals, estimate_joint_distribution,
                                   fit_pairwise_maxent, generate_candidate_groups, score_joint_information,
                                   build_joint_information_family)
from hyperedges.mapper_cover import construct_mapper_cover, cluster_mapper_elements, fit_mapper_lens
from hyperedges.cover_learning import export_cover_memberships

# Reuse an explicit fixture, rather than obtaining market data.
from test.test_hyperedge_pipeline import context


def test_pairwise_triangle_is_not_a_common_witness():
    relation = pd.DataFrame([[True, True, False], [False, True, True], [True, False, True]], columns=list("ABC"))
    assert mine_recurring_groups(relation, 1, (3, 3)) == ()
    witnessed = pd.concat([relation, pd.DataFrame([[True, True, True]] * 2, columns=list("ABC"))], ignore_index=True)
    assert mine_recurring_groups(witnessed, 2, (3, 3)) == (("A", "B", "C"),)


def test_event_rates_use_group_specific_observability():
    relation = pd.DataFrame([[True, True, False], [True, False, True], [False, False, False]], columns=list("ABC"))
    validity = pd.DataFrame([[True, True, False], [True, False, True], [True, True, True]], columns=list("ABC"))
    results = measure_group_event_support([("A", "B"), ("A", "C"), ("A", "B", "C")], relation, validity)
    assert results[0]["event_rate"] == results[1]["event_rate"] == 0.5
    assert results[2]["joint_observations"] == 1 and results[2]["support"] == 0


def test_event_budget_fails_explicitly_instead_of_truncating_witnesses():
    relation = pd.DataFrame(True, index=range(3), columns=list("ABCDEF"))
    with pytest.raises(ValueError, match="budget exceeded"):
        mine_recurring_groups(relation, 1, (3, 5), candidate_budget=8)


def test_event_builder_preserves_evidence_and_direction(context):
    params = {"quantiles": [0.4, 0.6], "min_support": 2, "size_bounds": [3, 3], "edge_budget": 8, "direction": "down"}
    family = build_event_dowker_family(context, params)
    assert len(family.edge_ids) == 8
    assert all(item["direction"] == "down" and item["support"] >= 2 for item in family.attributes.values())


def test_exact_parity_has_one_bit_beyond_pairwise_marginals():
    states = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    full = estimate_joint_distribution(states, smoothing=0, cardinality=2)
    targets = derive_pairwise_marginals(full)
    model = fit_pairwise_maxent(targets, 1e-12, 100)
    assert model["converged"]
    assert score_joint_information(full, model) == pytest.approx(1)
    assert np.allclose(model["probability"], 1 / 8)


def test_pairwise_potential_model_has_no_population_residual():
    states = np.indices((2, 2, 2)) * 2 - 1
    energy = 0.8 * states[0] * states[1] - 0.3 * states[1] * states[2] + 0.4 * states[0] * states[2]
    probability = np.exp(energy)
    probability /= probability.sum()
    model = fit_pairwise_maxent(derive_pairwise_marginals(probability), 1e-12, 1000)
    assert model["converged"]
    assert score_joint_information(probability, model) == pytest.approx(0, abs=1e-10)
    for pair, target in derive_pairwise_marginals(probability).items():
        assert np.allclose(derive_pairwise_marginals(model["probability"])[pair], target, atol=1e-12)


def test_nonconverged_model_cannot_produce_an_information_score():
    probability = np.array([[[.1, .05], [.2, .1]], [[.05, .2], [.1, .2]]])
    model = fit_pairwise_maxent(derive_pairwise_marginals(probability), 1e-20, 1)
    assert not model["converged"]
    with pytest.raises(ValueError, match="nonconverged"):
        score_joint_information(probability, model)


def test_candidates_are_seeded_unrestricted_and_order_invariant():
    nodes = [f"n{i:03}" for i in range(250)]
    first = generate_candidate_groups(nodes, 3, 1000, 37)
    assert len(first) == len(set(first)) == 1000
    assert first == generate_candidate_groups(nodes[::-1], 3, 1000, 37)
    assert first != generate_candidate_groups(nodes, 3, 1000, 38)
    assert len(generate_candidate_groups(nodes[:20], 3, 2000, 37)) == 1140


def test_information_builder_discovers_parity_without_knn(context):
    history = context.history
    parity = np.tile(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 2 - 1, (25, 1))
    history.returns.iloc[:, :3] = parity
    params = {"bin_spec": {"kind": "binary_sign"}, "smoothing": 0.1, "min_observations": 30,
              "tolerance": 1e-9, "max_iterations": 1000, "candidate_budget": 100,
              "selection_spec": {"edge_budget": 5, "min_holdout_gain_bits": .1, "min_selection_observations": 10}}
    family = build_joint_information_family(context, params)
    target = "joint:" + "|".join(context.node_ids[:3])
    assert target in family.edge_ids
    assert family.attributes[target]["holdout_gain_bits"] > .9
    diagnostic = build_joint_information_family(context, {**params, "mode": "diagnose"})
    assert diagnostic.incidence.shape == (8, 0)
    assert diagnostic.diagnostics["results"]


def test_mapper_exports_overlapping_stock_sets_not_nerve_vertices():
    descriptors = pd.DataFrame({"x": [0., .4, .6, 1.]}, index=list("ABCD"))
    lens, _ = fit_mapper_lens(descriptors, {"kind": "feature", "feature": "x"})
    cover = construct_mapper_cover(lens, {"bins": 2, "overlap": .5})
    groups = cluster_mapper_elements(descriptors, cover, {"kind": "single_linkage", "radius": .5})
    assert set(groups.values()) == {("A", "B", "C"), ("B", "C", "D")}
    assert sum("B" in members for members in groups.values()) == 2


def test_lag_descriptors_do_not_bridge_filtered_history_rows(context):
    from hyperedges.common.history import _subset
    from hyperedges.common.descriptors import build_stock_descriptors
    history = context.history
    selection = np.ones(len(history.returns), dtype=bool)
    selection[[10, 31, 54, 75]] = False
    narrowed = _subset(history, selection, history.cutoff)
    frame = build_stock_descriptors(narrowed, {"features": ["lag1"], "standardize": False})
    node = context.node_ids[0]
    valid = narrowed.mask[node].to_numpy() & narrowed.structural_mask
    adjacent = valid[1:] & valid[:-1] & np.asarray(narrowed.session_ids[1:] == narrowed.session_ids[:-1])
    adjacent &= np.asarray(narrowed.predecessor_times[1:] == narrowed.returns.index[:-1])
    values = narrowed.returns[node].to_numpy()
    expected = np.corrcoef(values[:-1][adjacent], values[1:][adjacent])[0, 1]
    assert frame.loc[node, "lag1"] == pytest.approx(expected)


def test_cover_export_uses_explicit_threshold_and_preserves_overlap():
    class Fitted:
        cover_ = np.array([[1, 0], [.8, .8], [0, 1]])
    groups = export_cover_memberships(Fitted(), {"threshold": .5, "min_size": 2}, node_ids=("A", "B", "C"))
    assert groups == {"cover:0": ("A", "B"), "cover:1": ("B", "C")}


def test_optional_cover_backend_fails_clearly_when_missing(context):
    import importlib.util
    if importlib.util.find_spec("shapediscover") is not None:
        pytest.skip("Missing-backend fixture requires an environment without ShapeDiscover")
    from hyperedges.cover_learning import build_cover_neighborhood_graph
    points = pd.DataFrame(np.eye(3), index=list("ABC"))
    with pytest.raises(ImportError, match="not installed"):
        build_cover_neighborhood_graph(points, {"neighbors": 1, "algorithm": "umap", "backend_version": "fixture"})
