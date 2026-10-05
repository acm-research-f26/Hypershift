"""Lazy plugin resolution; constructor count and names do not affect orchestration."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from numbers import Integral
from time import perf_counter

from .history import _validate_window
from .types import ConstructorSpec, HyperedgePipelineConfig

__all__ = ["register_constructor", "register_context_provider", "resolve_pipeline_config"]

_CONSTRUCTORS = {}
_CONTEXT_PROVIDERS = {}


def _register(registry, name, factory, required, allowed):
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("Registration requires a name and callable factory")
    if name in registry:
        raise ValueError(f"Already registered: {name}")
    registry[name] = (factory, frozenset(required), None if allowed is None else frozenset(allowed))


def register_constructor(name, factory, *, required_params=(), allowed_params=None):
    """Factory(spec, seed) returns a fit/build plugin or an nn.Module membership plugin."""
    _register(_CONSTRUCTORS, name, factory, required_params, allowed_params)


def register_context_provider(name, factory, *, required_params=(), allowed_params=None):
    """Factory(spec, seed) returns a provider with fit(context), build(context)."""
    _register(_CONTEXT_PROVIDERS, name, factory, required_params, allowed_params)


def _derive_component_seed(base_seed, instance_id, purpose):
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral) or base_seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    digest = sha256(f"{int(base_seed)}\0{instance_id}\0{purpose}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class _HistoricalBuilder:
    """Adapt a pure family builder to the fit/build lifecycle."""

    def __init__(self, spec, seed, builder):
        self.spec, self.seed, self.builder = spec, seed, builder
        self.family = None

    def fit(self, context):
        start = perf_counter()
        family = self.builder(context, deepcopy(self.spec.params))
        self.family = replace(family, diagnostics={**family.diagnostics, "construction_seconds": perf_counter() - start})

    def build(self, context):
        if self.family is None:
            raise RuntimeError("Constructor has not been fitted")
        return deepcopy(self.family)


def _historical(module, function):
    def _factory(spec, seed):
        builder = getattr(import_module(f"hyperedges.{module}"), function)
        return _HistoricalBuilder(spec, seed, builder)
    return _factory


def _class_factory(module, name):
    def _factory(spec, seed):
        return getattr(import_module(f"hyperedges.{module}"), name)(spec, seed)
    return _factory


_KNN_PARAMS = {"neighbors", "similarity", "absolute", "coverage_threshold", "min_rows"}
register_constructor("covariance_knn", _historical("covariance_knn.constructor", "build_covariance_knn_family"), allowed_params=_KNN_PARAMS)
# Compatibility method name; new specifications use covariance_knn.
register_constructor("correlation_knn", _historical("covariance_knn.constructor", "build_covariance_knn_family"), allowed_params=_KNN_PARAMS)
register_constructor("gics", _historical("gics.constructor", "build_gics_family"),
                     allowed_params={"source", "level", "min_size", "metadata_protocol"})
register_constructor("event_dowker", _historical("event_dowker.constructor", "build_event_dowker_family"),
                     required_params={"quantiles", "min_support", "size_bounds", "edge_budget"},
                     allowed_params={"quantiles", "min_support", "size_bounds", "edge_budget", "direction", "candidate_budget", "candidate_mode"})
register_constructor("joint_information", _historical("joint_information.constructor", "build_joint_information_family"),
                     required_params={"bin_spec", "smoothing", "min_observations", "tolerance", "max_iterations", "candidate_budget", "selection_spec"},
                     allowed_params={"bin_spec", "smoothing", "min_observations", "tolerance", "max_iterations", "candidate_budget", "selection_spec", "order", "estimation_fraction", "mode"})
register_constructor("mapper_cover", _historical("mapper_cover.constructor", "build_mapper_cover_family"),
                     required_params={"descriptor_spec", "lens_spec", "cover_spec", "clustering_spec"},
                     allowed_params={"descriptor_spec", "lens_spec", "cover_spec", "clustering_spec", "min_size", "edge_budget"})
register_constructor("cover_learning", _historical("cover_learning.constructor", "build_cover_learning_family"),
                     required_params={"descriptor_spec", "graph_spec", "backend_spec", "objective_spec", "membership_rule"},
                     allowed_params={"descriptor_spec", "graph_spec", "backend_spec", "objective_spec", "membership_rule", "graph_representation"})
register_constructor("ph_localized", _historical("ph_localized.constructor", "build_ph_localized_family"))
register_constructor("learned_membership", _class_factory("learned_membership.constructor", "LearnedMembershipConstructor"),
                     allowed_params={"slots", "initial_size", "selected_probability", "unselected_probability", "temperature", "min_size", "max_size", "size_weight", "duplicate_weight", "confidence_weight", "duplicate_threshold"})
register_context_provider("ph_context", _class_factory("ph_context.provider", "PHContextProvider"),
                          required_params={"node_ids", "cloud_spec", "backend_spec", "dimensions", "summary_spec"},
                          allowed_params={"node_ids", "cloud_spec", "backend_spec", "dimensions", "summary_spec"})


def _validate_pipeline_config(config):
    if not isinstance(config, HyperedgePipelineConfig) or config.protocol not in {"fold_frozen", "rolling"}:
        raise ValueError("Expected pipeline configuration with fold_frozen or rolling protocol")
    identifiers = set()
    for specs, registry in ((config.constructors, _CONSTRUCTORS), (config.context_providers, _CONTEXT_PROVIDERS)):
        for spec in specs:
            if not isinstance(spec, ConstructorSpec) or not isinstance(spec.instance_id, str) or not spec.instance_id.strip():
                raise ValueError("Every component requires a ConstructorSpec and nonempty instance_id")
            if spec.instance_id in identifiers:
                raise ValueError(f"Duplicate component instance_id: {spec.instance_id}")
            identifiers.add(spec.instance_id)
            if spec.method not in registry:
                raise ValueError(f"Unregistered component: {spec.method}")
            if not isinstance(spec.params, dict):
                raise ValueError("Component parameters must be a dictionary")
            _, required, allowed = registry[spec.method]
            missing = required - spec.params.keys()
            extra = set() if allowed is None else spec.params.keys() - allowed
            if missing or extra:
                raise ValueError(f"{spec.instance_id}: missing parameters {sorted(missing)}; unknown parameters {sorted(extra)}")
            _validate_window(spec.history_window)
    return config


def resolve_pipeline_config(experiment_config):
    """Validate canonical configuration or adapt the legacy experiment fields."""
    if isinstance(experiment_config, HyperedgePipelineConfig):
        return _validate_pipeline_config(deepcopy(experiment_config))
    configured = getattr(experiment_config, "hyperedge_pipeline", None)
    if configured is not None:
        return _validate_pipeline_config(deepcopy(configured))
    specs = []
    from .storage import _artifact_digest
    for component in experiment_config.hyperedge_builders:
        params = deepcopy(component.params)
        identifier = params.pop("instance_id", None)
        if identifier is None:
            identifier = f"{component.name}:{_artifact_digest(params)[:12]}"
        specs.append(ConstructorSpec(identifier, component.name, params))
    learning = experiment_config.hyperedge_learning
    if learning.name != "fixed":
        if learning.name != "learned_membership":
            raise ValueError(f"Unsupported legacy hyperedge_learning: {learning.name}")
        specs.append(ConstructorSpec("learned", learning.name, deepcopy(learning.params)))
    return _validate_pipeline_config(HyperedgePipelineConfig(tuple(specs)))


def _instantiate_component(spec, seed, *, context_provider=False):
    registry = _CONTEXT_PROVIDERS if context_provider else _CONSTRUCTORS
    return registry[spec.method][0](deepcopy(spec), seed)
