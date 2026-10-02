"""Historical joint-state persistence as context, never automatic stock membership."""

from dataclasses import replace
from itertools import combinations
from math import comb

import numpy as np

from ..common.backends import _require_backend
from ..common.validation import _validate_positive_integer
from ..common.types import ContextFeatures, _utc_timestamp

_SUMMARIES = {"finite_count", "total_persistence", "max_persistence", "mean_persistence"}


def build_joint_state_cloud(history, node_ids, scaling_state, cloud_spec):
    if set(cloud_spec) != {"max_points", "min_rows"}:
        raise ValueError("cloud_spec requires max_points and min_rows")
    _validate_positive_integer(cloud_spec["max_points"], "cloud point budget")
    _validate_positive_integer(cloud_spec["min_rows"], "minimum cloud observations")
    if tuple(scaling_state["node_ids"]) != tuple(node_ids) or not set(node_ids) <= set(history.returns.columns):
        raise ValueError("PH scaling and stock coordinate axes must match")
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("PH coordinates must be nonempty and unique")
    valid = history.mask.loc[:, list(node_ids)].all(axis=1).to_numpy() & history.structural_mask
    observations = history.returns.loc[valid, list(node_ids)].to_numpy()
    if len(observations) < cloud_spec["min_rows"]:
        return np.empty((0, len(node_ids)))
    mean, scale = np.asarray(scaling_state["mean"]), np.asarray(scaling_state["scale"])
    if mean.shape != (len(node_ids),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("PH coordinate scaling must be finite with positive scales")
    if len(observations) > cloud_spec["max_points"]:
        indices = np.linspace(0, len(observations) - 1, cloud_spec["max_points"], dtype=int)
        observations = observations[indices]
    return (observations - mean) / scale


def _native_rips(cloud, dimensions, max_simplices):
    """Exact small-cloud Vietoris--Rips persistence over F2 through H1."""
    if max(dimensions) > 1:
        raise ValueError("native_rips supports H0/H1; select an explicit external backend for higher dimensions")
    _validate_positive_integer(max_simplices, "simplex budget")
    order = max(dimensions) + 2
    count = sum(comb(len(cloud), size) for size in range(1, min(order, len(cloud)) + 1))
    if count > max_simplices:
        raise ValueError(f"Native Rips requires {count} simplices, exceeding budget {max_simplices}")
    distances = np.linalg.norm(cloud[:, None] - cloud[None, :], axis=2)
    simplices = []
    for size in range(1, min(order, len(cloud)) + 1):
        for vertices in combinations(range(len(cloud)), size):
            diameter = max((distances[left, right] for left, right in combinations(vertices, 2)), default=0.0)
            simplices.append((float(diameter), size - 1, vertices))
    simplices.sort()
    positions = {vertices: index for index, (_, _, vertices) in enumerate(simplices)}
    pivots, births, paired = {}, set(), set()
    diagrams = {dimension: [] for dimension in dimensions}
    for column, (death, dimension, vertices) in enumerate(simplices):
        boundary = ({positions[face] for face in combinations(vertices, len(vertices) - 1)} if dimension else set())
        while boundary and max(boundary) in pivots:
            boundary ^= pivots[max(boundary)]
        if not boundary:
            births.add(column)
        else:
            pivot = max(boundary)
            pivots[pivot] = boundary
            paired.add(pivot)
            birth, birth_dimension, _ = simplices[pivot]
            if birth_dimension in diagrams and death > birth:
                diagrams[birth_dimension].append((birth, death))
    for column in sorted(births - paired):
        birth, dimension, _ = simplices[column]
        if dimension in diagrams:
            diagrams[dimension].append((birth, float("inf")))
    return {dimension: np.asarray(intervals, dtype=float).reshape(-1, 2) for dimension, intervals in diagrams.items()}


def compute_persistence(cloud, backend_spec, dimensions=(0, 1)):
    cloud = np.asarray(cloud, dtype=float)
    dimensions = tuple(dimensions)
    if (cloud.ndim != 2 or not len(cloud) or not np.isfinite(cloud).all() or not dimensions
            or len(set(dimensions)) != len(dimensions)
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in dimensions)):
        raise ValueError("Persistence requires finite observations and unique nonnegative homology dimensions")
    name = backend_spec.get("name")
    if name == "native_rips":
        if set(backend_spec) != {"name", "version", "max_simplices"} or backend_spec["version"] != "1":
            raise ValueError("Native backend requires version='1' and max_simplices")
        return _native_rips(cloud, dimensions, backend_spec["max_simplices"])
    if name == "ripser":
        if set(backend_spec) != {"name", "version"}:
            raise ValueError("Ripser backend requires an explicit version")
        module = _require_backend("ripser", backend_spec["version"])
        diagrams = module.ripser(cloud, maxdim=max(dimensions))["dgms"]
        return {dimension: np.asarray(diagrams[dimension]) for dimension in dimensions}
    raise ValueError("Persistence backend must be native_rips or ripser")


def summarize_persistence(diagrams, summary_spec):
    summaries = tuple(summary_spec)
    if not summaries or len(set(summaries)) != len(summaries) or not set(summaries) <= _SUMMARIES:
        raise ValueError(f"Choose unique persistence summaries from {sorted(_SUMMARIES)}")
    names, values = [], []
    for dimension, diagram in sorted(diagrams.items()):
        diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
        finite = diagram[np.isfinite(diagram).all(axis=1)]
        lifetimes = finite[:, 1] - finite[:, 0]
        statistics = {"finite_count": len(lifetimes), "total_persistence": lifetimes.sum(),
                      "max_persistence": lifetimes.max() if len(lifetimes) else 0,
                      "mean_persistence": lifetimes.mean() if len(lifetimes) else 0}
        for summary in summaries:
            names.append(f"H{dimension}:{summary}")
            values.append(statistics[summary])
    return tuple(names), np.asarray(values, dtype=float)


def _build_ph_context(context, fitted_state, params):
    cloud = build_joint_state_cloud(context.history, tuple(params["node_ids"]), fitted_state, params["cloud_spec"])
    dimensions = tuple(params["dimensions"])
    diagrams = (compute_persistence(cloud, params["backend_spec"], dimensions) if len(cloud)
                else {dimension: np.empty((0, 2)) for dimension in dimensions})
    names, values = summarize_persistence(diagrams, params["summary_spec"])
    return ContextFeatures(
        context.instance_id, names, values, np.full(len(values), bool(len(cloud)), dtype=bool),
        {"cutoff": context.cutoff, "scaling_fit_cutoff": fitted_state["fit_cutoff"], "parameters": params,
         "cloud_points": len(cloud), "sampling": "evenly_spaced_valid_timestamps",
         "essential_intervals": {str(dim): int(np.isinf(diagram[:, 1]).sum()) for dim, diagram in diagrams.items()}},
        {"coordinate_scaling": fitted_state, "diagrams": {str(key): value for key, value in diagrams.items()}},
    )


class PHContextProvider:
    def __init__(self, spec, seed=0):
        self.spec, self.seed, self.scaling_state = spec, seed, None

    def fit(self, context):
        nodes = tuple(self.spec.params["node_ids"])
        if not nodes or len(set(nodes)) != len(nodes) or not set(nodes) <= set(context.node_ids):
            raise ValueError("Specify known, unique PH stock coordinates")
        observations = context.history.returns.loc[:, list(nodes)].where(
            context.history.mask.loc[:, list(nodes)] & context.history.structural_mask[:, None])
        mean, scale = observations.mean().to_numpy(), observations.std(ddof=0).to_numpy()
        scale = np.where(scale == 0, 1, scale)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("PH coordinates must have valid training observations")
        self.scaling_state = {"node_ids": nodes, "mean": mean, "scale": scale, "fit_cutoff": context.cutoff}
        # Validate the declared recipe/backend during fitting instead of failing silently at prediction.
        self.build(context)

    def build(self, context):
        if self.scaling_state is None:
            raise RuntimeError("Fit PH coordinate scaling first")
        return _build_ph_context(context, self.scaling_state, self.spec.params)


def fit_context_transform(training_contexts, *, fit_cutoff):
    contexts = tuple(training_contexts)
    cutoff = _utc_timestamp(fit_cutoff)
    if not contexts:
        raise ValueError("Context scaling requires training contexts")
    names = contexts[0].names
    instance = contexts[0].instance_id
    for context in contexts:
        if context.names != names or context.instance_id != instance:
            raise ValueError("Training context axes must match")
        if _utc_timestamp(context.provenance["cutoff"]) > cutoff:
            raise ValueError("Context scaling includes observations after the training cutoff")
    values = np.stack([context.values for context in contexts])
    mask = np.stack([context.mask for context in contexts])
    count = mask.sum(axis=0)
    means = np.divide(np.where(mask, values, 0).sum(axis=0), count, out=np.zeros(len(names)), where=count > 0)
    variance = np.divide(np.where(mask, (values - means) ** 2, 0).sum(axis=0), count, out=np.zeros(len(names)), where=count > 0)
    scales = np.sqrt(variance)
    return {"instance_id": instance, "names": names, "mean": means, "scale": np.where(scales > 0, scales, 1),
            "valid_features": count > 0, "fit_cutoff": cutoff}


def transform_context_features(context, state):
    if context.names != tuple(state["names"]) or context.instance_id != state["instance_id"]:
        raise ValueError("Context transformation axes must match")
    valid = context.mask & state["valid_features"]
    values = np.where(valid, (context.values - state["mean"]) / state["scale"], 0)
    return replace(context, values=values, mask=valid, state={**context.state, "feature_transform": state})
