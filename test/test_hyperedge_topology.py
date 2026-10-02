"""Actual persistence summaries and temporal context controls, without optional dependencies."""

from dataclasses import replace

import numpy as np
import pytest

from hyperedges import ConstructorSpec, build_context_features, build_hyperedge_snapshot, fit_hyperedge_pipeline
from hyperedges.ph_context import (compute_persistence, summarize_persistence, fit_context_transform,
                                 transform_context_features, build_joint_state_cloud)
from hyperedges.common.types import ContextFeatures

from test.test_hyperedge_pipeline import context


def _ph_spec(context):
    return ConstructorSpec("ph", "ph_context", {
        "node_ids": context.node_ids[:3], "cloud_spec": {"max_points": 12, "min_rows": 10},
        "backend_spec": {"name": "native_rips", "version": "1", "max_simplices": 1000},
        "dimensions": [0, 1], "summary_spec": ["finite_count", "total_persistence", "max_persistence"],
    })


def test_square_has_persistent_h1_interval():
    cloud = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    diagrams = compute_persistence(cloud, {"name": "native_rips", "version": "1", "max_simplices": 100}, [0, 1])
    assert diagrams[1].shape == (1, 2)
    assert np.allclose(diagrams[1][0], [1, np.sqrt(2)])
    names, values = summarize_persistence(diagrams, ["finite_count", "total_persistence"])
    assert names == ("H0:finite_count", "H0:total_persistence", "H1:finite_count", "H1:total_persistence")
    assert np.allclose(values, [3, 3, 1, np.sqrt(2) - 1])


def test_native_simplex_budget_is_enforced():
    with pytest.raises(ValueError, match="exceeding budget"):
        compute_persistence(np.eye(20), {"name": "native_rips", "version": "1", "max_simplices": 10}, [0, 1])


def test_ph_features_are_present_and_do_not_change_knn(context):
    knn = ConstructorSpec("knn", "covariance_knn")
    first = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, [knn]))
    second = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, [knn], [_ph_spec(context)]))
    assert first.families[0].incidence.equals(second.families[0].incidence)
    feature, = second.context_features
    assert feature.values.shape == (6,) and feature.mask.all()
    assert feature.values[0] > 0
    assert len(second.families) == 1


def test_missing_joint_cloud_yields_masked_context_not_fake_membership(context):
    pipeline = fit_hyperedge_pipeline(context, (), [_ph_spec(context)])
    history = replace(context.history, mask=context.history.mask.copy())
    history.mask.iloc[:, :3] = False
    features, = build_context_features(pipeline, replace(context, history=history))
    assert not features.mask.any()
    assert pipeline.constructors == {}


def test_context_transform_rejects_future_fit_and_preserves_missingness(context):
    first = ContextFeatures("ph", ("feature",), np.array([2.]), np.array([True]), {"cutoff": context.cutoff})
    second = replace(first, values=np.array([4.]))
    state = fit_context_transform([first, second], fit_cutoff=context.cutoff)
    assert transform_context_features(first, state).values[0] == -1
    assert not transform_context_features(replace(first, mask=np.array([False])), state).mask.any()
    future = replace(first, provenance={"cutoff": context.cutoff + np.timedelta64(1, "D")})
    with pytest.raises(ValueError, match="after the training cutoff"):
        fit_context_transform([future], fit_cutoff=context.cutoff)
