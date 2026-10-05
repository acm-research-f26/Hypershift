"""Explicit adapter to the published ShapeDiscover cover-learning implementation."""

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from unittest.mock import patch

import numpy as np

from ..common.backends import _isolated_random_state, _require_backend
from ..common.validation import _validate_positive_integer
from ..common.descriptors import build_stock_descriptors
from ..common.types import make_hyperedge_family

_GRAPH_INJECTION_LOCK = RLock()


@dataclass
class CoverGraph:
    graph: object
    recipe: dict
    seed: int


def build_cover_neighborhood_graph(descriptors, graph_spec):
    required = {"neighbors", "algorithm", "backend_version"}
    if required != set(graph_spec):
        raise ValueError(f"graph_spec requires {sorted(required)}")
    _validate_positive_integer(graph_spec["neighbors"], "graph neighbors")
    if graph_spec["neighbors"] >= len(descriptors):
        raise ValueError("Cover graph neighbors must be smaller than its stock universe")
    _require_backend("shapediscover", graph_spec["backend_version"])
    seed = int(descriptors.attrs.get("seed", 0))
    graph_module = import_module("shapediscover.weighted_graph")
    with _isolated_random_state(seed):
        if graph_spec["algorithm"] == "umap":
            # At most 250 stock points: exact neighbors avoid approximate-search
            # thread pools and make the fuzzy UMAP graph reproducible.
            from scipy.spatial.distance import cdist
            from umap.umap_ import fuzzy_simplicial_set
            points = descriptors.to_numpy()
            distances = cdist(points, points)
            neighbors = np.argsort(distances, axis=1, kind="stable")[:, :graph_spec["neighbors"]]
            weights, _, _ = fuzzy_simplicial_set(points, graph_spec["neighbors"], np.random.RandomState(seed), "euclidean",
                knn_indices=neighbors, knn_dists=np.take_along_axis(distances, neighbors, axis=1))
            graph = graph_module.WeightedGraph(weights)
        else:
            graph = graph_module.graph_from_pointcloud(descriptors.to_numpy(), n_neighbors=graph_spec["neighbors"], algorithm=graph_spec["algorithm"])
    return CoverGraph(graph, dict(graph_spec), seed)


def fit_published_cover(descriptors, graph, backend_spec, objective_spec):
    if set(backend_spec) != {"name", "version"} or backend_spec["name"] != "shapediscover":
        raise ValueError("Published cover backend must explicitly name shapediscover and its version")
    if backend_spec["version"] != graph.recipe["backend_version"]:
        raise ValueError("Cover graph and optimization backend versions must agree")
    _require_backend("shapediscover", backend_spec["version"])
    if set(objective_spec) != {"preset", "parameters"} or objective_spec["preset"] != "ShapeDiscover":
        raise ValueError("Specify the published ShapeDiscover preset and all desired constructor parameters")
    parameters = dict(objective_spec["parameters"])
    if not {"n_cover", "loss_weights", "initialization_algorithm", "model", "n_max_iter"} <= parameters.keys():
        raise ValueError("Declare cover count, objective weights, initialization, model, and iteration budget")
    if {"knn", "graph_algorithm"} & parameters.keys():
        raise ValueError("Neighborhood choices belong in graph_spec")
    weights = np.asarray(parameters["loss_weights"], dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("ShapeDiscover requires four nonnegative finite loss weights")
    cls = import_module("shapediscover.shapediscover").ShapeDiscover
    module = import_module("shapediscover.shapediscover")
    def supplied_graph(points, **kwargs):
        if len(points) != graph.graph.n_vertices():
            raise ValueError("Supplied graph vertex order must match optimization descriptors")
        return graph.graph
    # This published revision reads lazy cache attributes before initializing
    # them. Initialize its documented adjacency cache without changing weights.
    if not hasattr(graph.graph, "flat_neighbors_"):
        graph.graph.flat_neighbors_ = None
    # The pinned published fit has one graph-building hook. Replace only that
    # hook under a lock; the published initialization/objective remain intact.
    with _GRAPH_INJECTION_LOCK, _isolated_random_state(graph.seed), patch.object(module, "graph_from_pointcloud", supplied_graph):
        fitted = cls(knn=graph.recipe["neighbors"], graph_algorithm=graph.recipe["algorithm"], **parameters)
        fitted.fit(descriptors.to_numpy(), seed=graph.seed, verbose=False, plot_loss_curve=False)
    if fitted.graph_ is not graph.graph:
        raise RuntimeError("Published cover optimizer did not consume the supplied graph")
    return fitted


def export_cover_memberships(fitted_cover, membership_rule, *, node_ids):
    if set(membership_rule) != {"threshold", "min_size"}:
        raise ValueError("Cover membership rule requires threshold and min_size")
    threshold, minimum = membership_rule["threshold"], membership_rule["min_size"]
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Cover threshold must be in (0, 1]")
    _validate_positive_integer(minimum, "minimum cover size")
    values = stock_cover_matrix(fitted_cover, len(node_ids))
    if values.ndim != 2 or values.shape[0] != len(node_ids) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Published cover must return a finite stock-by-cover membership matrix in [0, 1]")
    return {f"cover:{index}": tuple(node_ids[row] for row in np.flatnonzero(values[:, index] >= threshold))
            for index in range(values.shape[1]) if int((values[:, index] >= threshold).sum()) >= minimum}


def stock_cover_matrix(fitted_cover, node_count):
    values = np.asarray(fitted_cover.cover_)
    # The pinned published backend stores [cover slots, stock vertices].
    if values.ndim == 2 and values.shape[1] == node_count:
        values = values.T
    return values


def build_cover_learning_family(context, params):
    descriptors = build_stock_descriptors(context.history, params["descriptor_spec"])
    descriptors.attrs["seed"] = context.seed
    graph_descriptors = descriptors
    if params.get("graph_representation", "descriptors") == "return_pca":
        import pandas as pd
        from sklearn.decomposition import PCA
        values = context.history.returns.loc[:, descriptors.index].to_numpy().T
        valid = context.history.mask.loc[:, descriptors.index].to_numpy().T & context.history.structural_mask[None,:]
        values = np.where(valid, values, 0)
        values /= np.maximum(np.sqrt(np.mean(values ** 2, axis=1, keepdims=True)), 1e-12)
        points = PCA(n_components=min(16, len(values) - 1, values.shape[1]), random_state=context.seed).fit_transform(values)
        graph_descriptors = pd.DataFrame(points, index=descriptors.index)
        graph_descriptors.attrs["seed"] = context.seed
    elif params.get("graph_representation", "descriptors") != "descriptors":
        raise ValueError("Unknown Cover Learning graph representation")
    graph = build_cover_neighborhood_graph(graph_descriptors, params["graph_spec"])
    fitted = fit_published_cover(descriptors, graph, params["backend_spec"], params["objective_spec"])
    groups = export_cover_memberships(fitted, params["membership_rule"], node_ids=tuple(descriptors.index))
    return make_hyperedge_family(groups, context,
                                diagnostics={"backend": params["backend_spec"], "objective": params["objective_spec"],
                                             "graph_recipe": graph.recipe, "forecasting_trained": False,
                                             "supplied_graph_consumed": fitted.graph_ is graph.graph},
                                state={"descriptors": descriptors, "descriptor_fit": descriptors.attrs["fit_state"],
                                       "cover": np.asarray(fitted.cover_), "graph_recipe": graph.recipe,
                                       "graph_adjacency": np.asarray(graph.graph.adjacency_matrix().toarray()),
                                       "graph_node_ids": tuple(descriptors.index),
                                       "graph_representation": params.get("graph_representation", "descriptors"),
                                       "loss_history": np.asarray(fitted.main_optimization_losses_),
                                       "backend": params["backend_spec"], "objective": params["objective_spec"]})
