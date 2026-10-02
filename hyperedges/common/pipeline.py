"""Fit independently seeded plugins and construct reusable fold or rolling snapshots."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter

import numpy as np

from .history import prepare_construction_history
from .registry import _derive_component_seed, _instantiate_component, resolve_pipeline_config
from .types import (ConstructionContext, HyperedgePipelineConfig, HyperedgeSnapshot,
                    align_family_to_nodes, _utc_timestamp, validate_hyperedge_family)

__all__ = [
    "FittedHyperedgePipeline", "fit_hyperedge_pipeline", "build_hyperedge_snapshot",
    "build_context_features", "summarize_hyperedge_families",
]


@dataclass
class FittedHyperedgePipeline:
    config: HyperedgePipelineConfig
    context: ConstructionContext
    constructors: dict = field(default_factory=dict)
    learned_modules: dict = field(default_factory=dict)
    context_providers: dict = field(default_factory=dict)


def _component_context(context, spec):
    # Copies isolate plugin mutation and preserve independent preprocessing.
    return replace(
        context, history=prepare_construction_history(context.history, context.cutoff, spec.history_window),
        metadata=deepcopy(context.metadata),
        instance_id=spec.instance_id, method=spec.method, params=deepcopy(spec.params),
        seed=_derive_component_seed(context.seed, spec.instance_id, "construction"),
    )


def fit_hyperedge_pipeline(context, specs=(), context_specs=(), *, protocol="fold_frozen"):
    config = (resolve_pipeline_config(specs) if isinstance(specs, HyperedgePipelineConfig) or hasattr(specs, "hyperedge_builders")
              else resolve_pipeline_config(HyperedgePipelineConfig(tuple(specs), tuple(context_specs), protocol)))
    pipeline = FittedHyperedgePipeline(config, deepcopy(context))
    for spec in config.constructors:
        component_context = _component_context(context, spec)
        constructor = _instantiate_component(spec, component_context.seed)
        constructor.fit(component_context)
        destination = pipeline.learned_modules if getattr(constructor, "trainable_membership", False) else pipeline.constructors
        destination[spec.instance_id] = constructor
    for spec in config.context_providers:
        component_context = _component_context(context, spec)
        provider = _instantiate_component(spec, component_context.seed, context_provider=True)
        provider.fit(component_context)
        pipeline.context_providers[spec.instance_id] = provider
    return pipeline


def build_hyperedge_snapshot(fitted_pipeline, context=None, *, available_at=None, effective_time=None):
    from .storage import _artifact_digest

    context = fitted_pipeline.context if context is None else context
    fitted = fitted_pipeline.context
    if tuple(context.node_ids) != tuple(fitted.node_ids) or context.fold_id != fitted.fold_id:
        raise ValueError("Snapshot stock axes and fold must match fitted pipeline")
    cutoff = _utc_timestamp(context.cutoff)
    if fitted_pipeline.config.protocol == "fold_frozen" and cutoff != _utc_timestamp(fitted.cutoff):
        raise ValueError("Frozen membership snapshots use the training cutoff; rebuild through the rolling protocol")
    if fitted_pipeline.config.protocol == "rolling" and available_at is None:
        raise ValueError("Rolling snapshots require an explicit build available_at time")
    available = cutoff if available_at is None else _utc_timestamp(available_at)
    effective = available if effective_time is None else _utc_timestamp(effective_time)
    if not cutoff <= available <= effective:
        raise ValueError("Require cutoff <= available_at <= effective_time")
    families = []
    for spec in fitted_pipeline.config.constructors:
        if spec.instance_id not in fitted_pipeline.constructors:
            continue
        component_context = _component_context(context, spec)
        constructor = fitted_pipeline.constructors[spec.instance_id]
        start = perf_counter()
        if fitted_pipeline.config.protocol == "rolling":
            # A fresh historical fit does not change the trained membership modules.
            constructor = _instantiate_component(spec, component_context.seed)
            constructor.fit(component_context)
        family = align_family_to_nodes(constructor.build(component_context), context.node_ids)
        validate_hyperedge_family(family)
        if family.instance_id != spec.instance_id:
            raise ValueError("Constructor returned a different family instance ID")
        family = replace(family, diagnostics={**family.diagnostics, "materialization_seconds": perf_counter() - start})
        families.append(family)
    references = {
        name: module.export_state() for name, module in fitted_pipeline.learned_modules.items()
    }
    features = tuple(provider.build(_component_context(context, spec))
                     for spec in fitted_pipeline.config.context_providers
                     for provider in (fitted_pipeline.context_providers[spec.instance_id],))
    for name, reference in references.items():
        if _utc_timestamp(reference["provenance"]["cutoff"]) > available:
            raise ValueError(f"Learned checkpoint {name!r} was fitted after snapshot availability")
    for feature, spec in zip(features, fitted_pipeline.config.context_providers, strict=True):
        if feature.instance_id != spec.instance_id:
            raise ValueError("Context provider returned a different instance ID")
        fitted_at = feature.provenance.get("scaling_fit_cutoff")
        if fitted_at is not None and _utc_timestamp(fitted_at) > available:
            raise ValueError("Context scaling was fitted after snapshot availability")
    configuration = asdict(fitted_pipeline.config)
    configuration["availability_policy"] = "declared" if available_at is not None else "offline_cutoff_assumption"
    identity = {
        "nodes": context.node_ids, "cutoff": cutoff, "available_at": available, "effective_time": effective,
        "fold_id": context.fold_id, "configuration": configuration, "learned": references,
        "families": [(f.instance_id, f.method, f.incidence, f.attributes, f.provenance, f.state,
                      {key: value for key, value in f.diagnostics.items() if key not in {"construction_seconds", "materialization_seconds"}})
                     for f in families],
        "context": features,
    }
    return HyperedgeSnapshot(_artifact_digest(identity), tuple(context.node_ids), tuple(families),
                             cutoff, available, effective, fitted_pipeline.config.protocol,
                             context.fold_id, configuration, references, features)


def build_context_features(fitted_pipeline, context):
    """Context history can move while memberships stay frozen."""
    if tuple(context.node_ids) != tuple(fitted_pipeline.context.node_ids):
        raise ValueError("Context stock axes differ from the training axes")
    return tuple(fitted_pipeline.context_providers[spec.instance_id].build(_component_context(context, spec))
                 for spec in fitted_pipeline.config.context_providers)


def summarize_hyperedge_families(families):
    result = {}
    for family in families:
        values = family.incidence.to_numpy()
        sizes, degrees = values.sum(axis=0), values.sum(axis=1)
        memberships = [tuple(np.flatnonzero(values[:, i])) for i in range(values.shape[1])]
        result[family.instance_id] = {
            "edges": values.shape[1], "covered_nodes": int((degrees > 0).sum()),
            "overlapping_nodes": int((degrees > 1).sum()), "sizes": sizes.tolist(),
            "duplicate_memberships": len(memberships) - len(set(memberships)),
            "diagnostics": deepcopy(family.diagnostics),
        }
    return result
