"""Labeled contracts shared by independent construction plugins."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
import pandas as pd

from experiments.config import TimeSpan
from .incidence import incidence_from_groups
from .validation import _validate_identifiers

__all__ = [
    "ConstructorSpec", "HyperedgePipelineConfig", "ConstructionHistory", "ConstructionContext",
    "HyperedgeFamily", "ContextFeatures", "HyperedgeSnapshot", "FamilyTensor", "HistoricalConstructor",
    "make_hyperedge_family", "validate_hyperedge_family", "align_family_to_nodes",
]


def _utc_timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Information times must be present and timezone-aware")
    return value.tz_convert("UTC")


@dataclass(frozen=True)
class ConstructorSpec:
    instance_id: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    history_window: TimeSpan = field(default_factory=lambda: TimeSpan(60, "sessions"))


@dataclass(frozen=True)
class HyperedgePipelineConfig:
    constructors: tuple[ConstructorSpec, ...] = ()
    context_providers: tuple[ConstructorSpec, ...] = ()
    protocol: str = "fold_frozen"


@dataclass(frozen=True)
class ConstructionHistory:
    returns: pd.DataFrame
    mask: pd.DataFrame
    structural_mask: np.ndarray
    session_ids: pd.DatetimeIndex
    availability_times: pd.DatetimeIndex
    cutoff: pd.Timestamp
    provenance: dict[str, Any] = field(default_factory=dict)
    predecessor_times: pd.DatetimeIndex | None = None

    def __post_init__(self):
        values, mask = self.returns, self.mask
        _validate_identifiers(values.columns, "node")
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in values.dtypes):
            raise ValueError("Historical returns must have real numeric dtypes")
        if (not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None
                or values.index.hasnans or not values.index.is_unique or not values.index.is_monotonic_increasing):
            raise ValueError("History timestamps must be aware, unique, and chronological")
        if not mask.index.equals(values.index) or not mask.columns.equals(values.columns) or not mask.dtypes.eq(bool).all():
            raise ValueError("History masks must be Boolean and aligned")
        structural = np.asarray(self.structural_mask)
        if structural.shape != (len(values),) or structural.dtype.kind != "b":
            raise ValueError("Structural mask must be Boolean with one entry per timestamp")
        if len(self.session_ids) != len(values) or self.session_ids.hasnans:
            raise ValueError("History session IDs must match timestamps")
        available = pd.DatetimeIndex(self.availability_times)
        if available.tz is None or available.hasnans or len(available) != len(values):
            raise ValueError("History availability must be aware and aligned")
        if (available < values.index).any():
            raise ValueError("Return availability cannot precede the observation timestamp")
        cutoff = _utc_timestamp(self.cutoff)
        if len(values) and ((values.index > cutoff).any() or (available > cutoff).any()):
            raise ValueError("Construction history includes future or unavailable observations")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric[mask.to_numpy()]).all():
            raise ValueError("Valid historical observations must be finite")
        if self.predecessor_times is not None:
            previous = pd.DatetimeIndex(self.predecessor_times)
            if previous.tz is None or len(previous) != len(values):
                raise ValueError("Predecessor timestamps must be aware and aligned with the return rows")
            valid_previous = ~previous.isna()
            if (previous[valid_previous] >= values.index[valid_previous]).any():
                raise ValueError("Return predecessor timestamps must precede their observations")


@dataclass(frozen=True)
class ConstructionContext:
    node_ids: tuple[str, ...]
    history: ConstructionHistory
    cutoff: pd.Timestamp
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    fold_id: str = "fold"
    instance_id: str = "constructor"
    method: str = "custom"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_identifiers(self.node_ids, "node")
        if tuple(self.history.returns.columns) != tuple(self.node_ids):
            raise ValueError("Context and history stock axes must match")
        if _utc_timestamp(self.cutoff) != _utc_timestamp(self.history.cutoff):
            raise ValueError("Context and history cutoffs must match")


@dataclass(frozen=True)
class HyperedgeFamily:
    """Encapsulates incidence matrix for a constructor, edge attributes, diagnostics, provenance, and fitted state."""

    instance_id: str
    method: str
    incidence: pd.DataFrame
    attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def node_ids(self):
        return tuple(self.incidence.index)

    @property
    def edge_ids(self):
        return tuple(self.incidence.columns)


@dataclass(frozen=True)
class ContextFeatures:
    instance_id: str
    names: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_identifiers(self.names, "feature")
        if self.values.shape != (len(self.names),) or self.mask.shape != self.values.shape or self.mask.dtype.kind != "b":
            raise ValueError("Context feature axes and Boolean masks must match")
        if not np.isfinite(self.values[self.mask]).all():
            raise ValueError("Valid context features must be finite")


@dataclass(frozen=True)
class HyperedgeSnapshot:
    snapshot_id: str
    node_ids: tuple[str, ...]
    families: tuple[HyperedgeFamily, ...]
    cutoff: pd.Timestamp
    available_at: pd.Timestamp
    effective_time: pd.Timestamp
    protocol: str = "fold_frozen"
    fold_id: str = "fold"
    configuration: dict[str, Any] = field(default_factory=dict)
    learned_references: dict[str, Any] = field(default_factory=dict)
    context_features: tuple[ContextFeatures, ...] = ()


@dataclass
class FamilyTensor:
    """Mask adjusted tensor view of a hyperedge family"""
    instance_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    incidence: Any
    edge_weights: Any
    active_nodes: Any
    edge_mask: Any
    node_degrees: Any
    edge_degrees: Any
    attributes: dict[str, Any] = field(default_factory=dict)
    regularization: Any = None


class HistoricalConstructor(Protocol):
    def fit(self, context: ConstructionContext) -> None: ...
    def build(self, context: ConstructionContext) -> HyperedgeFamily: ...


def make_hyperedge_family(groups, context, attributes=None, *, diagnostics=None, state=None):
    incidence = (incidence_from_groups(context.node_ids, groups) if groups
                 else pd.DataFrame(index=context.node_ids, columns=[], dtype=bool))
    history = context.history
    provenance = {
        "builder_version": 1, "parameters": context.params, "seed": context.seed,
        "cutoff": _utc_timestamp(context.cutoff), "fold_id": context.fold_id,
        "history_start": history.returns.index[0] if len(history.returns) else None,
        "history_end": history.returns.index[-1] if len(history.returns) else None,
        "history": history.provenance,
    }
    family = HyperedgeFamily(context.instance_id, context.method, incidence,
                             attributes or {}, diagnostics or {}, provenance, state or {})
    validate_hyperedge_family(family)
    return family


def validate_hyperedge_family(family):
    _validate_identifiers((family.instance_id, ), "instance")
    _validate_identifiers((family.method, ), "method")
    incidence = family.incidence
    if not isinstance(incidence, pd.DataFrame):
        raise ValueError("Family incidence must be a DataFrame")
    _validate_identifiers(incidence.index, "node")
    _validate_identifiers(incidence.columns, "edge", allow_empty=True)
    if not incidence.dtypes.eq(bool).all() or (incidence.sum(axis=0) == 0).any():
        raise ValueError("Family incidence must be Boolean with no zero-member edges")
    if not set(family.attributes) <= set(incidence.columns):
        raise ValueError("Attributes refer to unknown edges")
    for attributes in family.attributes.values():
        weight = attributes.get("weight", 1.0)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Model edge weights must be finite and nonnegative")


def align_family_to_nodes(family, node_ids):
    validate_hyperedge_family(family)
    node_ids = _validate_identifiers(node_ids, "node")
    if not set(family.node_ids) <= set(node_ids):
        raise ValueError("Family contains stocks outside the canonical axis")
    return replace(family, incidence=family.incidence.reindex(index=node_ids, fill_value=False))
