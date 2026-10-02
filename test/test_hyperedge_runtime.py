"""Binary membership gradients, masking, checkpoint recovery, and incidence consumers."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyperedges import (ConstructorSpec, build_hyperedge_snapshot, fit_hyperedge_pipeline,
                        load_hyperedge_snapshot, save_hyperedge_snapshot)
from hyperedges.learned_membership import (LearnedMembershipConstructor, freeze_learned_memberships,
                                 membership_regularization, summarize_learned_memberships)
from hyperedges.common.runtime import materialize_family_inputs
from hyperedges.common.types import make_hyperedge_family
from src.hypergraph import HyperedgeConsumer

from test.test_hyperedge_pipeline import context


def _pipeline(context, **parameters):
    return fit_hyperedge_pipeline(context, [ConstructorSpec("knn", "covariance_knn"),
                                            ConstructorSpec("learned", "learned_membership", {"slots": 3, **parameters})])


def test_learning_does_not_advance_global_random_state(context):
    before = torch.random.get_rng_state().clone()
    pipeline = _pipeline(context)
    module = pipeline.learned_modules["learned"]
    module(training=True)
    assert torch.equal(before, torch.random.get_rng_state())


def test_hard_memberships_have_surrogate_gradients_and_variable_sizes(context):
    pipeline = _pipeline(context)
    module = pipeline.learned_modules["learned"]
    sizes = set()
    for _ in range(20):
        family = module(training=True)
        assert set(family.incidence.detach().flatten().tolist()) <= {0., 1.}
        sizes.update(family.attributes["full_membership_sizes"])
    assert len(sizes) > 1
    weights = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    family = module(training=True)
    (family.incidence * weights).sum().backward()
    assert module.logits.grad is not None and torch.isfinite(module.logits.grad).all()
    assert module.logits.grad.abs().sum() > 0


def test_optimizer_changes_memberships_without_changing_fixed_gics(context):
    pipeline = fit_hyperedge_pipeline(context, [ConstructorSpec("gics", "gics"),
                                                ConstructorSpec("learned", "learned_membership", {"slots": 2})])
    snapshot = build_hyperedge_snapshot(pipeline)
    original = snapshot.families[0].incidence.copy()
    module = pipeline.learned_modules["learned"]
    with torch.no_grad():
        module.logits.fill_(-8)
        # An explicitly permitted cross-industry composition.
        module.logits[[0, 4, 7], 0] = 8
        module.logits[[1, 4, 6, 7], 1] = 8
    memberships = module(training=False).incidence
    assert memberships[0, 0] == memberships[4, 0] == memberships[7, 0] == 1
    assert memberships[4].sum() == 2
    assert snapshot.families[0].incidence.equals(original)


def test_forecast_loss_reaches_memberships_and_family_gates(context):
    pipeline = _pipeline(context)
    snapshot = build_hyperedge_snapshot(pipeline)
    model = HyperedgeConsumer(3, 7, family_ids=("knn", "learned"), learned_modules=pipeline.learned_modules)
    inputs = torch.tensor(np.random.default_rng(4).normal(size=(8, 3)), dtype=torch.float32)
    targets = inputs[:, :1].square()
    (model(inputs, snapshot) - targets).square().mean().backward()
    assert pipeline.learned_modules["learned"].logits.grad.abs().sum() > 0
    assert model.family_logits.grad is not None
    assert torch.isfinite(model.family_logits.grad).all()


def test_invalid_slots_are_suppressed_but_penalized(context):
    pipeline = _pipeline(context, min_size=3, max_size=5)
    module = pipeline.learned_modules["learned"]
    with torch.no_grad():
        module.logits.fill_(-10)
        module.logits[:, 1] = 10
        module.logits[:2, 2] = 10
    family = module(training=False)
    assert not family.edge_mask.any()
    assert not family.incidence.any()
    diagnostics = summarize_learned_memberships(module)
    assert diagnostics["invalid_slots"] == 3 and diagnostics["empty_slots"] == 1
    penalty = membership_regularization(torch.sigmoid(module.logits), module.settings)["total"]
    penalty.backward()
    assert module.logits.grad.abs().sum() > 0


def test_unseen_training_stocks_cannot_gain_learned_membership(context):
    history = deepcopy(context.history)
    history.mask.iloc[:, -1] = False
    pipeline = _pipeline(replace(context, history=history))
    module = pipeline.learned_modules["learned"]
    with torch.no_grad():
        module.logits[-1] = 10
    assert not module(training=False).incidence[-1].any()


def test_runtime_masks_recompute_degrees_and_suppress_single_member_edges(context):
    family = make_hyperedge_family({"first": context.node_ids[:3], "second": context.node_ids[2:5]}, context)
    snapshot = replace(build_hyperedge_snapshot(fit_hyperedge_pipeline(context)), families=(family,))
    active = np.zeros(8, dtype=bool)
    active[:2] = True
    runtime, = materialize_family_inputs(snapshot, active_nodes=active)
    assert runtime.edge_mask.tolist() == [True, False]
    assert runtime.edge_degrees.tolist() == [2, 0]
    assert runtime.node_degrees.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert family.incidence["second"].sum() == 3


def test_temporal_only_and_uncovered_nodes_have_finite_predictions(context):
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context))
    model = HyperedgeConsumer(3, 4)
    inputs = torch.randn(8, 3)
    torch.testing.assert_close(model(inputs, snapshot), model.head(model.temporal_projection(inputs)))
    assert torch.isfinite(model(inputs, snapshot, np.zeros(8, dtype=bool))).all()


def test_masked_nonfinite_inputs_cannot_contaminate_other_stocks(context):
    pipeline = fit_hyperedge_pipeline(context, [ConstructorSpec("knn", "covariance_knn")])
    snapshot = build_hyperedge_snapshot(pipeline)
    model = HyperedgeConsumer(3, 4, family_ids=("knn",))
    inputs = torch.randn(8, 3)
    inputs[-1] = torch.nan
    active = np.ones(8, dtype=bool)
    active[-1] = False
    assert torch.isfinite(model(inputs, snapshot, active)).all()
    with pytest.raises(ValueError, match="Active temporal"):
        model(inputs, snapshot)


def test_frozen_checkpoint_masking_keeps_direct_path(context):
    pipeline = _pipeline(context)
    snapshot = build_hyperedge_snapshot(pipeline)
    model = HyperedgeConsumer(3, 5, family_ids=("knn", "learned"), learned_modules=pipeline.learned_modules)
    model.eval()
    inputs = torch.randn(8, 3)
    prediction = model(inputs, snapshot, disabled_families=("knn", "learned"))
    torch.testing.assert_close(prediction, model.head(model.temporal_projection(inputs)))


def test_freezing_is_deterministic_and_roundtrips_rng_and_parameters(context, tmp_path):
    pipeline = _pipeline(context)
    module = pipeline.learned_modules["learned"]
    module(training=True)
    state = module.export_state()
    restored = LearnedMembershipConstructor.from_export_state(state)
    torch.testing.assert_close(module(training=True).incidence, restored(training=True).incidence)
    family = freeze_learned_memberships(module)
    assert family.incidence.dtypes.eq(bool).all()
    assert not module.logits.requires_grad
    with pytest.raises(ValueError, match="Frozen"):
        module(training=True)
    first, second = module(training=False), module(training=False)
    torch.testing.assert_close(first.incidence, second.incidence)
    snapshot = build_hyperedge_snapshot(pipeline)
    recovered = load_hyperedge_snapshot(save_hyperedge_snapshot(snapshot, tmp_path / "learned.json"))
    actual = materialize_family_inputs(snapshot, training=False)
    expected = materialize_family_inputs(recovered, training=False)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left.incidence, right.incidence)


def test_context_values_reach_forecast_and_invalid_values_are_masked(context):
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context))
    model = HyperedgeConsumer(3, 4, context_channels=2)
    inputs = torch.randn(8, 3)
    first = model(inputs, snapshot, context_values=[0., np.nan], context_mask=[True, False])
    second = model(inputs, snapshot, context_values=[10., np.nan], context_mask=[True, False])
    assert torch.isfinite(first).all() and not torch.allclose(first, second)
