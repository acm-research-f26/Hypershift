"""Independent stock hyperedge families and separately toggled context providers.

Only lightweight construction contracts are imported here; Torch and optional
research backends are loaded by their owning modules when requested.
"""

from .common.incidence import incidence_from_groups
from .covariance_knn import correlation_knn_hyperedges, select_covariance_history
from .common.history import make_construction_context, prepare_construction_history
from .common.pipeline import build_context_features, build_hyperedge_snapshot, fit_hyperedge_pipeline, summarize_hyperedge_families
from .common.registry import register_constructor, register_context_provider, resolve_pipeline_config
from .common.schedule import attach_snapshot_to_sample, select_snapshot_for_origin
from .common.storage import load_hyperedge_snapshot, save_hyperedge_snapshot
from .common.types import (ConstructionContext, ConstructionHistory, ConstructorSpec, ContextFeatures,
                    HyperedgeFamily, HyperedgePipelineConfig, HyperedgeSnapshot)

__all__ = [
    "ConstructionContext", "ConstructionHistory", "ConstructorSpec", "ContextFeatures",
    "HyperedgeFamily", "HyperedgePipelineConfig", "HyperedgeSnapshot",
    "incidence_from_groups", "correlation_knn_hyperedges", "select_covariance_history",
    "make_construction_context", "prepare_construction_history",
    "fit_hyperedge_pipeline", "build_hyperedge_snapshot", "build_context_features",
    "summarize_hyperedge_families", "register_constructor", "register_context_provider",
    "resolve_pipeline_config", "attach_snapshot_to_sample", "select_snapshot_for_origin",
    "load_hyperedge_snapshot", "save_hyperedge_snapshot",
]
