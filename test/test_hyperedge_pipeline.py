"""Construction integration contracts using small local histories and supplied metadata."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from data.types import ObservationPanel
from experiments.config import TimeSpan
from hyperedges import (ConstructionContext, ConstructionHistory, ConstructorSpec, HyperedgePipelineConfig,
                        build_hyperedge_snapshot, fit_hyperedge_pipeline, load_hyperedge_snapshot,
                        prepare_construction_history, save_hyperedge_snapshot, select_snapshot_for_origin)
from hyperedges.covariance_knn import covariance_to_correlation, estimate_covariance, knn_groups_from_similarity
from hyperedges.gics import build_gics_family, load_classification_records, select_classifications_at_cutoff
from hyperedges.covariance_knn.history import select_covariance_history
from hyperedges.common.registry import (derive_component_seed, register_constructor, register_context_provider,
                                resolve_pipeline_config, validate_pipeline_config)
from hyperedges.common.types import ContextFeatures, align_family_to_nodes, make_hyperedge_family


@pytest.fixture
def context():
    sessions = pd.date_range("2024-01-02", periods=5, freq="B")
    times = pd.DatetimeIndex([session.tz_localize("UTC") + pd.Timedelta(hours=15, minutes=15 * bar)
                              for session in sessions for bar in range(20)])
    nodes = tuple(f"stock:{index}" for index in range(8))
    returns = pd.DataFrame(np.random.default_rng(23).normal(size=(len(times), len(nodes))), index=times, columns=nodes)
    structural = np.tile([False, *([True] * 19)], len(sessions))
    cutoff = times[-1] + pd.Timedelta(minutes=2)
    history = ConstructionHistory(returns, pd.DataFrame(True, index=times, columns=nodes), structural,
                                  pd.DatetimeIndex(np.repeat(sessions, 20)), times + pd.Timedelta(minutes=1), cutoff)
    metadata = pd.DataFrame({"node_id": nodes, "industry_group": ["1010"] * 4 + ["2020"] * 3 + [None],
                             "source": "fixture GICS vendor", "effective_from": "2020-01-01T00:00:00Z",
                             "available_at": "2020-01-01T00:00:00Z"})
    return ConstructionContext(nodes, history, cutoff, seed=37, metadata={"gics": metadata})


def test_default_knn_has_five_neighbors_and_normalized_covariance(context):
    pipeline = fit_hyperedge_pipeline(context, [ConstructorSpec("knn", "covariance_knn")])
    family, = build_hyperedge_snapshot(pipeline).families
    assert family.incidence.sum().eq(6).all()
    assert family.node_ids == context.node_ids
    assert family.state["covariance"].shape == (8, 8)


def test_covariance_and_correlation_have_distinct_volatility_behavior(context):
    returns = context.history.returns
    covariance = estimate_covariance(returns)
    scaled = returns.copy()
    scaled.iloc[:, 0] *= 100
    changed = estimate_covariance(scaled)
    pd.testing.assert_frame_equal(covariance_to_correlation(covariance), covariance_to_correlation(changed))
    assert not np.allclose(covariance, changed)
    assert knn_groups_from_similarity(covariance, 2) != knn_groups_from_similarity(changed, 2)


def test_fixed_families_do_not_depend_on_other_methods_order_or_seeds(context):
    knn = ConstructorSpec("knn", "covariance_knn")
    gics = ConstructorSpec("gics", "gics")
    families = []
    for specs in ((knn,), (knn, gics), (gics, knn)):
        snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, specs))
        families.append(next(family for family in snapshot.families if family.instance_id == "knn"))
    for family in families[1:]:
        pd.testing.assert_frame_equal(family.incidence, families[0].incidence)
        assert family.provenance == families[0].provenance
    assert derive_component_seed(37, "knn", "construction") != derive_component_seed(37, "events", "construction")


def test_repeated_constructor_types_and_empty_pipeline(context):
    specs = (ConstructorSpec("absolute", "covariance_knn"),
             ConstructorSpec("signed", "covariance_knn", {"similarity": "signed_correlation"}))
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, specs))
    assert [family.instance_id for family in snapshot.families] == ["absolute", "signed"]
    assert not snapshot.families[0].incidence.equals(snapshot.families[1].incidence)
    assert build_hyperedge_snapshot(fit_hyperedge_pipeline(context)).families == ()


def test_legacy_absolute_parameter_and_seed_ids_are_order_independent(context):
    from types import SimpleNamespace
    from experiments.config import ComponentConfig
    signed = ComponentConfig("correlation_knn", {"absolute": False})
    absolute = ComponentConfig("correlation_knn", {"absolute": True})
    configs = [resolve_pipeline_config(SimpleNamespace(hyperedge_builders=builders, hyperedge_learning=ComponentConfig("fixed")))
               for builders in ((signed, absolute), (absolute, signed))]
    assert {spec.instance_id for spec in configs[0].constructors} == {spec.instance_id for spec in configs[1].constructors}
    snapshots = [build_hyperedge_snapshot(fit_hyperedge_pipeline(context, config)) for config in configs]
    for family in snapshots[0].families:
        other = next(item for item in snapshots[1].families if item.instance_id == family.instance_id)
        pd.testing.assert_frame_equal(family.incidence, other.incidence)


def test_registered_custom_method_needs_no_orchestrator_change(context):
    class Custom:
        def __init__(self, spec, seed):
            self.spec, self.seed = spec, seed
        def fit(self, ctx):
            self.family = make_hyperedge_family({"custom": ctx.node_ids[:3]}, ctx)
        def build(self, ctx):
            return self.family
    register_constructor("fixture_custom", Custom)
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, [ConstructorSpec("plugin", "fixture_custom")]))
    assert snapshot.families[0].incidence["custom"].sum() == 3
    assert snapshot.families[0].provenance["seed"] == derive_component_seed(context.seed, "plugin", "construction")


def test_registered_context_is_separate_from_membership(context):
    class Provider:
        def __init__(self, spec, seed):
            self.spec = spec
        def fit(self, ctx):
            pass
        def build(self, ctx):
            return ContextFeatures(self.spec.instance_id, ("value",), np.array([2.]), np.array([True]), {"cutoff": ctx.cutoff})
    register_context_provider("fixture_context", Provider)
    knn = ConstructorSpec("knn", "covariance_knn")
    first = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, [knn]))
    second = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, [knn], [ConstructorSpec("features", "fixture_context")]))
    pd.testing.assert_frame_equal(first.families[0].incidence, second.families[0].incidence)
    assert len(second.context_features) == 1 and len(second.families) == 1


@pytest.mark.parametrize("specs", [
    (ConstructorSpec("same", "gics"), ConstructorSpec("same", "covariance_knn")),
    (ConstructorSpec("x", "missing_plugin"),),
    (ConstructorSpec("x", "covariance_knn", {"neigbors": 2}),),
    (ConstructorSpec("x", "event_dowker"),),
    (ConstructorSpec("x", "covariance_knn", history_window=TimeSpan(0, "sessions")),),
])
def test_invalid_configs_fail_before_construction(specs):
    with pytest.raises(ValueError):
        validate_pipeline_config(HyperedgePipelineConfig(specs))


def test_coverage_exclusion_does_not_remove_canonical_stock(context):
    history = deepcopy(context.history)
    history.mask.iloc[:20, -1] = False
    frame, diagnostics = select_covariance_history(history)
    assert context.node_ids[-1] not in frame
    assert diagnostics["excluded_nodes"] == [context.node_ids[-1]]
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(replace(context, history=history), [ConstructorSpec("knn", "covariance_knn")]))
    assert snapshot.node_ids == context.node_ids
    assert not snapshot.families[0].incidence.loc[context.node_ids[-1]].any()


def test_coverage_threshold_and_complete_row_count(context):
    history = deepcopy(context.history)
    positions = np.flatnonzero(history.structural_mask)
    history.mask.iloc[positions[:4], 0] = False  # 91/95 >= .95
    frame, diagnostics = select_covariance_history(history)
    assert context.node_ids[0] in frame and diagnostics["complete_rows"] == 91
    history.mask.iloc[positions[4], 0] = False
    frame, _ = select_covariance_history(history)
    assert context.node_ids[0] not in frame
    with pytest.raises(ValueError, match="jointly complete"):
        select_covariance_history(history, min_rows=100)


def test_gics_missing_labels_singletons_and_changes(context):
    family = build_gics_family(replace(context, instance_id="gics", method="gics"), {})
    assert family.incidence.sum().tolist() == [4, 3]
    assert family.diagnostics["missing_labels"] == [context.node_ids[-1]]
    records = context.metadata["gics"].copy()
    records.loc[6, "industry_group"] = "3030"
    future = records.iloc[[0]].copy()
    future["industry_group"] = "9999"
    future["effective_from"] = future["available_at"] = "2025-01-01T00:00:00Z"
    selected = select_classifications_at_cutoff(load_classification_records(pd.concat([records, future])), context.node_ids, context.cutoff)
    assert selected.loc[context.node_ids[0], "industry_group"] == "1010"
    family = build_gics_family(replace(context, metadata={"gics": records}), {})
    assert "gics:industry_group:3030" in family.diagnostics["omitted_small_groups"]


def test_gics_requires_metadata_and_explicit_temporal_scope(context):
    with pytest.raises(ValueError, match="supplied classification"):
        build_gics_family(replace(context, metadata={}), {})
    records = context.metadata["gics"].drop(columns=["effective_from", "available_at"])
    with pytest.raises(ValueError, match="Point-in-time"):
        build_gics_family(replace(context, metadata={"gics": records}), {})
    family = build_gics_family(replace(context, metadata={"gics": records}), {"metadata_protocol": "retrospective_static"})
    assert family.diagnostics["metadata_protocol"] == "retrospective_static"


def test_snapshot_roundtrip_integrity_and_determinism(context, tmp_path):
    pipeline = fit_hyperedge_pipeline(context, [ConstructorSpec("gics", "gics"), ConstructorSpec("knn", "covariance_knn")])
    first, second = build_hyperedge_snapshot(pipeline), build_hyperedge_snapshot(pipeline)
    assert first.snapshot_id == second.snapshot_id
    path = save_hyperedge_snapshot(first, tmp_path / "snapshot.json")
    restored = load_hyperedge_snapshot(path)
    assert restored.snapshot_id == first.snapshot_id
    assert restored.configuration == first.configuration
    for left, right in zip(first.families, restored.families, strict=True):
        pd.testing.assert_frame_equal(left.incidence, right.incidence)
        assert left.provenance == right.provenance
    envelope = json.loads(path.read_text())
    envelope["snapshot"]["snapshot_id"] = "tampered"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        load_hyperedge_snapshot(path)


def test_snapshot_availability_and_frozen_vs_rolling(context):
    lag = pd.Timedelta(minutes=10)
    frozen = fit_hyperedge_pipeline(context, [ConstructorSpec("knn", "covariance_knn")])
    snapshot = build_hyperedge_snapshot(frozen, available_at=context.cutoff + lag)
    with pytest.raises(ValueError, match="No snapshot"):
        select_snapshot_for_origin([snapshot], context.cutoff)
    assert select_snapshot_for_origin([snapshot], context.cutoff + lag) is snapshot
    earlier_history = prepare_construction_history(context.history, context.history.returns.index[59] + lag, TimeSpan(60, "sessions"))
    earlier = replace(context, history=earlier_history, cutoff=earlier_history.cutoff)
    with pytest.raises(ValueError, match="Frozen membership"):
        build_hyperedge_snapshot(frozen, earlier)
    rolling = fit_hyperedge_pipeline(earlier, [ConstructorSpec("knn", "covariance_knn")], protocol="rolling")
    with pytest.raises(ValueError, match="explicit build"):
        build_hyperedge_snapshot(rolling, context)
    current = build_hyperedge_snapshot(rolling, context, available_at=context.cutoff + lag)
    previous = build_hyperedge_snapshot(rolling, earlier, available_at=earlier.cutoff + lag)
    repeated = build_hyperedge_snapshot(rolling, earlier, available_at=earlier.cutoff + lag)
    assert previous.snapshot_id == repeated.snapshot_id != current.snapshot_id


def test_rolling_snapshot_cannot_backdate_a_learned_checkpoint(context):
    rolling = fit_hyperedge_pipeline(context, [ConstructorSpec("learned", "learned_membership", {"slots": 2})], protocol="rolling")
    earlier_history = prepare_construction_history(context.history, context.history.returns.index[59] + pd.Timedelta(minutes=2), TimeSpan(60, "sessions"))
    earlier = replace(context, history=earlier_history, cutoff=earlier_history.cutoff)
    with pytest.raises(ValueError, match="fitted after snapshot availability"):
        build_hyperedge_snapshot(rolling, earlier, available_at=earlier.cutoff)


def _panel(context):
    history = context.history
    times = history.returns.index
    closes = np.exp(history.returns.to_numpy().cumsum(axis=0) / 100)[:, :, None]
    shape = closes.shape
    return ObservationPanel(times, context.node_ids, ("close",), closes, np.ones(shape, dtype=bool), np.ones(shape, dtype=bool),
                            history.session_ids, times - pd.Timedelta(minutes=15), history.availability_times,
                            np.full(len(times), 15), np.full(shape[:2], 15), np.zeros(len(times), dtype=bool), {}, {})


def test_history_uses_sessions_masks_and_both_close_availability(context):
    panel = _panel(context)
    cutoff = panel.timestamps[60] + pd.Timedelta(minutes=2)
    history = prepare_construction_history(panel, cutoff, TimeSpan(2, "sessions"))
    assert history.session_ids.nunique() == 2
    assert history.returns.index.max() <= cutoff
    assert not history.mask.loc[~history.structural_mask].to_numpy().any()
    available = panel.availability_times.copy().to_numpy()
    available[40] = cutoff + pd.Timedelta(hours=1)
    changed = replace(panel, availability_times=pd.DatetimeIndex(available))
    delayed = prepare_construction_history(changed, cutoff, TimeSpan(60, "sessions"))
    assert panel.timestamps[40] not in delayed.returns.index
    assert panel.timestamps[41] not in delayed.returns.index


def test_future_perturbation_cannot_change_training_snapshot(context):
    panel = _panel(context)
    cutoff = panel.timestamps[59] + pd.Timedelta(minutes=2)
    snapshots = []
    for source in (panel, replace(panel, values=panel.values.copy())):
        if source is not panel:
            source.values[60:] *= 10
        history = prepare_construction_history(source, cutoff)
        ctx = replace(context, history=history, cutoff=cutoff)
        snapshots.append(build_hyperedge_snapshot(fit_hyperedge_pipeline(ctx, [ConstructorSpec("knn", "covariance_knn")])))
    assert snapshots[0].snapshot_id == snapshots[1].snapshot_id
