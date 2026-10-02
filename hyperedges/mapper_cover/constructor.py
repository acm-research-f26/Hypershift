"""Stock-descriptor Mapper covers with explicit lens, cover, and clustering choices."""

from itertools import product

import numpy as np
import pandas as pd

from ..common.validation import _validate_positive_integer
from ..common.types import make_hyperedge_family
from ..common.descriptors import build_stock_descriptors



def fit_mapper_lens(descriptors, lens_spec):
    kind = lens_spec.get("kind")
    if kind == "feature":
        if set(lens_spec) != {"kind", "feature"} or lens_spec["feature"] not in descriptors:
            raise ValueError("Feature lens requires a descriptor column")
        values = descriptors[[lens_spec["feature"]]].copy()
        state = dict(lens_spec)
    elif kind == "pca":
        if set(lens_spec) != {"kind", "components"}:
            raise ValueError("PCA lens requires components")
        count = lens_spec["components"]
        _validate_positive_integer(count, "PCA components")
        if count > min(descriptors.shape):
            raise ValueError("Too many PCA components")
        mean = descriptors.mean().to_numpy()
        centered = descriptors.to_numpy() - mean
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        axes = vectors[:count].copy()
        for axis in axes:
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0:
                axis *= -1
        values = pd.DataFrame(centered @ axes.T, index=descriptors.index, columns=[f"pc:{i}" for i in range(count)])
        state = {"kind": kind, "mean": mean, "axes": axes}
    else:
        raise ValueError("Lens kind must be feature or pca")
    return values, state


def construct_mapper_cover(lens_values, cover_spec):
    if set(cover_spec) - {"bins", "overlap", "max_elements"} or not {"bins", "overlap"} <= set(cover_spec):
        raise ValueError("Cover requires bins and overlap")
    bins, overlap = cover_spec["bins"], cover_spec["overlap"]
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Cover overlap must be in [0, 1)")
    bins = (bins,) * lens_values.shape[1] if isinstance(bins, int) else tuple(bins)
    if len(bins) != lens_values.shape[1]:
        raise ValueError("One bin count per lens coordinate is required")
    for count in bins:
        _validate_positive_integer(count, "bins")
    maximum = cover_spec.get("max_elements", 4096)
    _validate_positive_integer(maximum, "max cover elements")
    if np.prod(bins) > maximum:
        raise ValueError("Cover element budget exceeded")
    intervals = []
    values = lens_values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Lens values must be finite")
    for axis, count in enumerate(bins):
        low, high = values[:, axis].min(), values[:, axis].max()
        if low == high:
            intervals.append([(low, high)])
        else:
            width = (high - low) / (count - (count - 1) * overlap)
            step = width * (1 - overlap)
            intervals.append([(low + i * step, high if i == count - 1 else low + i * step + width) for i in range(count)])
    cover = {}
    for index, bounds in enumerate(product(*intervals)):
        selected = np.ones(len(values), dtype=bool)
        for axis, (low, high) in enumerate(bounds):
            selected &= (values[:, axis] >= low) & (values[:, axis] <= high)
        cover[f"bin:{index}"] = tuple(lens_values.index[selected])
    return cover


def cluster_mapper_elements(descriptors, cover, clustering_spec):
    if set(clustering_spec) != {"kind", "radius"} or clustering_spec["kind"] != "single_linkage":
        raise ValueError("Clustering requires kind='single_linkage' and an explicit radius")
    radius = clustering_spec["radius"]
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Clustering radius must be finite and nonnegative")
    groups = {}
    for bin_id, members in cover.items():
        nodes = sorted(members)
        if not nodes:
            continue
        points = descriptors.loc[nodes].to_numpy()
        adjacency = np.linalg.norm(points[:, None] - points[None, :], axis=2) <= radius
        remaining, cluster_index = set(range(len(nodes))), 0
        while remaining:
            seed = min(remaining)
            reached, pending = {seed}, [seed]
            while pending:
                point = pending.pop()
                for neighbor in np.flatnonzero(adjacency[point]):
                    neighbor = int(neighbor)
                    if neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)
            remaining -= reached
            groups[f"mapper:{bin_id}:cluster:{cluster_index}"] = tuple(nodes[index] for index in sorted(reached))
            cluster_index += 1
    return groups


def build_mapper_cover_family(context, params):
    descriptors = build_stock_descriptors(context.history, params["descriptor_spec"])
    lens, lens_state = fit_mapper_lens(descriptors, params["lens_spec"])
    cover = construct_mapper_cover(lens, params["cover_spec"])
    clusters = cluster_mapper_elements(descriptors, cover, params["clustering_spec"])
    minimum = params.get("min_size", 2)
    _validate_positive_integer(minimum, "minimum size")
    groups = {edge: members for edge, members in clusters.items() if len(members) >= minimum}
    if "edge_budget" in params:
        _validate_positive_integer(params["edge_budget"], "edge budget")
        groups = dict(sorted(groups.items())[:params["edge_budget"]])
    return make_hyperedge_family(groups, context,
                                diagnostics={"descriptor_nodes": list(descriptors.index), "cover_elements": len(cover)},
                                state={"descriptors": descriptors, "descriptor_fit": descriptors.attrs["fit_state"],
                                       "lens": lens_state, "cover": cover})
