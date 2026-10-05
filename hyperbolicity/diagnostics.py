"""Geometry of prepared stock features and hyperedge incidence, with coverage."""
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .core import (basepoint_bounds, euclidean_distances, exact_delta_details,
                   relative_delta, s_walk_adjacency, sampled_delta, validate_distances)


def summarize_metric(distances, *, exact_limit=250, seed=0, basepoints=4):
    d = np.asarray(distances, dtype=float)
    validate_distances(d)
    started = perf_counter()
    result = (exact_delta_details(d) if len(d) <= exact_limit
              else basepoint_bounds(d, n_basepoints=basepoints, seed=seed))
    if len(d) > exact_limit and len(d) >= 4:
        sampled = sampled_delta(d, n_samples=10000, seed=seed)
        result["sampled_lower_bound"] = sampled
        result["lower_bound"] = max(result["lower_bound"], sampled)
    result.update(points=len(d), diameter=float(d.max()), seconds=perf_counter() - started,
                  relative_delta=relative_delta(result["delta"], d) if result["delta"] is not None else None,
                  relative_lower=relative_delta(result["lower_bound"], d),
                  relative_upper=relative_delta(result["upper_bound"], d))
    return result


def summarize_feature_geometry(features, mask=None, *, node_ids=None, exact_limit=250, seed=0):
    """Nodes are points. Imputation+mask embedding is an actual Euclidean metric.

    Input is [nodes, channels] or [time, nodes, features]. Any normalization of
    values must already have been fitted on permitted training observations.
    """
    x = np.asarray(features, dtype=float)
    valid = np.isfinite(x) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(x)
    if x.shape != valid.shape or x.ndim not in (2, 3):
        raise ValueError("Aligned node vectors or time/node/features required")
    if x.ndim == 3:
        x, valid = x.transpose(1, 0, 2).reshape(x.shape[1], -1), valid.transpose(1, 0, 2).reshape(x.shape[1], -1)
    active = valid.any(axis=1)
    if not active.any():
        return {"status": "undefined", "reason": "no_observed_features", "nodes": len(x)}
    embedded = np.concatenate([np.where(valid[active], x[active], 0), valid[active].astype(float) * .25], axis=1)
    result = summarize_metric(euclidean_distances(embedded), exact_limit=exact_limit, seed=seed)
    result.update(status="ok", total_nodes=len(x), observed_nodes=int(active.sum()),
                  coverage=float(active.mean()), missing_fraction=float(1 - valid.mean()),
                  embedding="prepared_values_zero_imputation_plus_0.25_mask", seed=seed,
                  node_ids=list(np.asarray(tuple(range(len(x))) if node_ids is None else node_ids)[active]),
                  distinct_points=len(np.unique(embedded, axis=0)),
                  geometry_sha256=sha256(embedded.tobytes()).hexdigest())
    return result


def graph_geometry(adjacency, *, node_ids=None, exact_limit=250, seed=0):
    a = np.asarray(adjacency, dtype=bool)
    if a.ndim != 2 or not len(a) or a.shape[0] != a.shape[1] or not np.array_equal(a, a.T):
        raise ValueError("Undirected nonempty square adjacency required")
    a = a.copy()
    np.fill_diagonal(a, False)
    graph = csr_matrix(a)
    count, labels = connected_components(graph, directed=False)
    components = []
    for component in range(count):
        indices = np.flatnonzero(labels == component)
        d = shortest_path(graph[indices][:, indices], directed=False, unweighted=True)
        item = summarize_metric(d, exact_limit=exact_limit, seed=seed + component)
        item["node_indices"] = indices.tolist()
        components.append(item)
    n = len(a)
    sizes = [item["points"] for item in components]
    result = {"status": "connected" if count == 1 else "disconnected",
              "global_delta": components[0]["delta"] if count == 1 else None,
              "global_relative_delta": components[0]["relative_delta"] if count == 1 else None,
              "nodes": n, "components": components, "component_sizes": sizes,
              "isolates": int((a.sum(axis=1) == 0).sum()),
              "connected_pair_coverage": sum(k * (k - 1) for k in sizes) / (n * (n - 1)) if n > 1 else 0,
              "largest_component_fraction": max(sizes) / n,
              "max_component_lower_bound": max(item["lower_bound"] for item in components),
              "adjacency_sha256": sha256(a.tobytes()).hexdigest()}
    if node_ids is not None:
        result["node_ids"] = list(node_ids)
    return result


def summarize_structural_geometry(snapshot_or_incidence, *, node_ids=None, exact_limit=250,
                                 seed=0, include_bipartite=True):
    if hasattr(snapshot_or_incidence, "families"):
        node_ids = snapshot_or_incidence.node_ids
        arrays = [f.incidence.to_numpy(dtype=bool) for f in snapshot_or_incidence.families]
        h = np.concatenate(arrays, axis=1) if arrays else np.zeros((len(node_ids), 0), bool)
    else:
        h = np.asarray(snapshot_or_incidence)
    if h.ndim != 2 or not len(h) or not np.isin(h, [0, 1]).all():
        raise ValueError("Binary stock-by-edge incidence required")
    h = h.astype(bool)
    unique = np.unique(h, axis=1)
    unique = unique[:, unique.sum(axis=0) >= 2]
    sizes, degrees = h.sum(axis=0), h.sum(axis=1)
    result = {"nodes": len(h), "edges": h.shape[1], "unique_edges_size_ge_2": unique.shape[1],
              "duplicate_edges": h.shape[1] - np.unique(h, axis=1).shape[1],
              "sizes": sizes.tolist(), "degrees": degrees.tolist(),
              "covered_fraction": float((degrees > 0).mean()), "seed": seed,
              "primary_membership_policy": "deduplicated_binary_size_ge_2",
              "projections": {}, "incidence_sha256": sha256(h.tobytes()).hexdigest()}
    for s in (1, 2, 3):
        result["projections"][str(s)] = graph_geometry(s_walk_adjacency(unique, s), node_ids=node_ids,
                                                      exact_limit=exact_limit, seed=seed)
        if result["duplicate_edges"] and s > 1:
            result.setdefault("multiset_sensitivity", {})[str(s)] = graph_geometry(s_walk_adjacency(h, s), exact_limit=0, seed=seed)
    if include_bipartite:
        b = np.zeros((len(h) + unique.shape[1],) * 2, dtype=bool)
        b[:len(h), len(h):] = unique
        b[len(h):, :len(h)] = unique.T
        result["bipartite"] = graph_geometry(b, exact_limit=exact_limit, seed=seed)
    if unique.shape[1] > 1:
        select = np.random.default_rng(seed).choice(unique.shape[1], min(1000, unique.shape[1]), replace=False)
        sub = unique[:, select].astype(np.int32)
        intersection = sub.T @ sub
        union = sub.sum(axis=0)[:, None] + sub.sum(axis=0)[None, :] - intersection
        overlap = (intersection / np.maximum(union, 1))[np.triu_indices(len(select), 1)]
        result["edge_jaccard_quantiles"] = np.quantile(overlap, [0, .5, .9, 1]).tolist()
    return result
