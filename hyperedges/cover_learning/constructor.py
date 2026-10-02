"""Explicit adapter to the published ShapeDiscover cover-learning implementation."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from ..common.backends import isolated_random_state, require_backend
from ..common.validation import validate_positive_integer
from ..common.descriptors import build_stock_descriptors
from ..common.types import make_hyperedge_family


@dataclass
class CoverGraph:
    graph: object
    recipe: dict
    seed: int


def build_cover_neighborhood_graph(descriptors, graph_spec):
    required = {"neighbors", "algorithm", "backend_version"}
    if required != set(graph_spec):
        raise ValueError(f"graph_spec requires {sorted(required)}")
    validate_positive_integer(graph_spec["neighbors"], "graph neighbors")
    if graph_spec["neighbors"] >= len(descriptors):
        raise ValueError("Cover graph neighbors must be smaller than its stock universe")
    require_backend("shapediscover", graph_spec["backend_version"])
    seed = int(descriptors.attrs.get("seed", 0))
    graph_from_pointcloud = import_module("shapediscover.weighted_graph").graph_from_pointcloud
    with isolated_random_state(seed):
        graph = graph_from_pointcloud(descriptors.to_numpy(), n_neighbors=graph_spec["neighbors"], algorithm=graph_spec["algorithm"])
    return CoverGraph(graph, dict(graph_spec), seed)


def fit_published_cover(descriptors, graph, backend_spec, objective_spec):
    if set(backend_spec) != {"name", "version"} or backend_spec["name"] != "shapediscover":
        raise ValueError("Published cover backend must explicitly name shapediscover and its version")
    if backend_spec["version"] != graph.recipe["backend_version"]:
        raise ValueError("Cover graph and optimization backend versions must agree")
    require_backend("shapediscover", backend_spec["version"])
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
    with isolated_random_state(graph.seed):
        fitted = cls(knn=graph.recipe["neighbors"], graph_algorithm=graph.recipe["algorithm"], **parameters)
        # The published API reconstructs its graph internally from this same declared recipe.
        fitted.fit(descriptors.to_numpy(), seed=graph.seed, verbose=False, plot_loss_curve=False)
    return fitted


def export_cover_memberships(fitted_cover, membership_rule, *, node_ids):
    if set(membership_rule) != {"threshold", "min_size"}:
        raise ValueError("Cover membership rule requires threshold and min_size")
    threshold, minimum = membership_rule["threshold"], membership_rule["min_size"]
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Cover threshold must be in (0, 1]")
    validate_positive_integer(minimum, "minimum cover size")
    values = np.asarray(fitted_cover.cover_)
    if values.ndim != 2 or values.shape[0] != len(node_ids) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Published cover must return a finite stock-by-cover membership matrix in [0, 1]")
    return {f"cover:{index}": tuple(node_ids[row] for row in np.flatnonzero(values[:, index] >= threshold))
            for index in range(values.shape[1]) if int((values[:, index] >= threshold).sum()) >= minimum}


def build_cover_learning_family(context, params):
    descriptors = build_stock_descriptors(context.history, params["descriptor_spec"])
    descriptors.attrs["seed"] = context.seed
    graph = build_cover_neighborhood_graph(descriptors, params["graph_spec"])
    fitted = fit_published_cover(descriptors, graph, params["backend_spec"], params["objective_spec"])
    groups = export_cover_memberships(fitted, params["membership_rule"], node_ids=tuple(descriptors.index))
    return make_hyperedge_family(groups, context,
                                diagnostics={"backend": params["backend_spec"], "objective": params["objective_spec"],
                                             "graph_recipe": graph.recipe, "forecasting_trained": False},
                                state={"descriptors": descriptors, "descriptor_fit": descriptors.attrs["fit_state"],
                                       "cover": np.asarray(fitted.cover_), "graph_recipe": graph.recipe,
                                       "backend": params["backend_spec"], "objective": params["objective_spec"]})
