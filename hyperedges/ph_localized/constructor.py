"""Experimental coordinate-ablation localization of observed persistent features.

Cloud rows are timestamps. Coordinate groups are stocks, including their missing
observation indicators. Diagonal-aware diagram matching never confuses cocycle
vertices with stock identifiers. This does not identify every type of dependence.
"""
from collections import Counter
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from ..common.types import make_hyperedge_family

DEFAULTS = {"max_points": 64, "min_rows": 32, "max_features": 8,
            "max_size": 16, "min_size": 2, "null_replicates": 32,
            "landmark_replicates": 3, "coverage_threshold": .95,
            "row_coverage": .90, "materiality": .05, "retention": .5,
            "stability_jaccard": .6, "dimensions": [1, 2], "block_size": 5}


def persistence(cloud):
    from ripser import ripser
    distances = squareform(pdist(np.asarray(cloud, dtype=np.float64)))
    diagrams = ripser(distances, distance_matrix=True, maxdim=2)["dgms"]
    return {dim: diagram[np.isfinite(diagram).all(axis=1)] for dim, diagram in enumerate(diagrams)}


def match_diagrams(reference, changed):
    """Optimal sum of L-infinity costs, allowing either diagram to hit diagonal."""
    ref, alt = np.asarray(reference).reshape(-1, 2), np.asarray(changed).reshape(-1, 2)
    n, m = len(ref), len(alt)
    if not n:
        return np.empty(0, int), np.empty(0)
    costs = np.full((n + m, n + m), 1e12, dtype=float)
    if m:
        costs[:n, :m] = np.abs(ref[:, None] - alt[None]).max(axis=2)
        costs[n:, :m] = np.diag((alt[:, 1] - alt[:, 0]) / 2)
        off = ~np.eye(m, dtype=bool)
        costs[n:, :m][off] = 1e12
    costs[:n, m:] = np.diag((ref[:, 1] - ref[:, 0]) / 2)
    costs[:n, m:][~np.eye(n, dtype=bool)] = 1e12
    costs[n:, m:] = 0
    rows, cols = linear_sum_assignment(costs)
    chosen = cols[rows < n]
    return np.where(chosen < m, chosen, -1), costs[np.arange(n), chosen]


def _landmarks(x, budget, rng):
    if len(x) <= budget:
        return np.arange(len(x))
    distances = squareform(pdist(x))
    selected = [int(rng.integers(len(x)))]
    nearest = distances[selected[0]].copy()
    for _ in range(budget - 1):
        nearest[selected] = -np.inf
        index = int(nearest.argmax())
        selected.append(index)
        nearest = np.minimum(nearest, distances[index])
    return np.sort(selected)


def _permute_blocks(x, groups, rng, block_size):
    blocks = [np.arange(i, min(i + block_size, len(x))) for i in range(0, len(x), block_size)]
    result = x.copy()
    for columns in groups:
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        result[:, columns] = x[order][:, columns]
    return result


def _one_localization(x, coordinate_groups, settings, seed):
    rng = np.random.default_rng(seed)
    rows = _landmarks(x, settings["max_points"], rng)
    cloud = x[rows]
    # Keep this normalization fixed for every removal and counterfactual.
    scale = np.median(pdist(cloud))
    cloud = cloud / (scale if scale > 0 else 1)
    reference = persistence(cloud)
    null_maxima = {dim: [] for dim in settings["dimensions"]}
    for _ in range(settings["null_replicates"]):
        diagrams = persistence(_permute_blocks(cloud, coordinate_groups, rng, settings["block_size"]))
        for dim in null_maxima:
            life = np.diff(diagrams[dim], axis=1).ravel()
            null_maxima[dim].append(float(life.max()) if len(life) else 0)
    candidates = []
    for dim in settings["dimensions"]:
        threshold = np.quantile(null_maxima[dim], .95)
        for index, (birth, death) in enumerate(reference[dim]):
            if death - birth > max(threshold, 1e-8):
                candidates.append((float(death - birth), dim, index))
    candidates = sorted(candidates, reverse=True)[:settings["max_features"]]
    if not candidates:
        return [], {"rows": rows, "diagrams": reference, "null_maxima": null_maxima, "features": []}
    influence = np.zeros((len(candidates), len(coordinate_groups)))
    all_columns = np.arange(cloud.shape[1])
    for stock, columns in enumerate(coordinate_groups):
        kept = np.setdiff1d(all_columns, columns)
        changed = persistence(cloud[:, kept])
        for dim in settings["dimensions"]:
            matches, costs = match_diagrams(reference[dim], changed[dim])
            for f, (life, dimension, index) in enumerate(candidates):
                if dimension != dim:
                    continue
                remaining = (float(changed[dim][matches[index], 1] - changed[dim][matches[index], 0]) if matches[index] >= 0 else 0)
                influence[f, stock] = max(costs[index] * 2 / life, (life - remaining) / life, 0)
    results, feature_records = [], []
    for f, (life, dim, index) in enumerate(candidates):
        chosen = np.flatnonzero(influence[f] >= settings["materiality"])
        chosen = sorted(chosen, key=lambda node: (-influence[f, node], int(node)))[:settings["max_size"]]

        def survives(nodes):
            if not nodes:
                return False, 0.0
            columns = np.concatenate([coordinate_groups[node] for node in nodes])
            diagram = persistence(cloud[:, columns])[dim]
            matched, costs = match_diagrams(reference[dim], diagram)
            position = matched[index]
            remaining = float(diagram[position, 1] - diagram[position, 0]) if position >= 0 else 0
            return remaining >= life * settings["retention"] and costs[index] <= life * .5, remaining

        ok, retained = survives(chosen)
        if ok:
            for node in sorted(chosen, key=lambda node: (influence[f, node], int(node))):
                if len(chosen) <= settings["min_size"]:
                    break
                reduced = [other for other in chosen if other != node]
                keep, value = survives(reduced)
                if keep:
                    chosen, retained = reduced, value
        accepted = ok and len(chosen) >= settings["min_size"]
        null_p = None
        if accepted:
            columns = np.concatenate([coordinate_groups[node] for node in chosen])
            subset = cloud[:, columns]
            subset_groups = [np.arange(i, i + len(coordinate_groups[node])) for i, node in
                             zip(np.cumsum([0] + [len(coordinate_groups[n]) for n in chosen[:-1]]), chosen)]
            maxima = []
            for _ in range(settings["null_replicates"]):
                diagram = persistence(_permute_blocks(subset, subset_groups, rng, settings["block_size"]))[dim]
                maxima.append(float(np.diff(diagram, axis=1).max()) if len(diagram) else 0)
            null_p = (1 + np.sum(np.asarray(maxima) >= retained)) / (1 + len(maxima))
            accepted = null_p <= .05
        record = {"dimension": dim, "feature_index": index, "persistence": life,
                  "retained_persistence": retained, "coordinates": sorted(map(int, chosen)),
                  "accepted": bool(accepted), "subset_null_p": null_p,
                  "influence": influence[f]}
        feature_records.append(record)
        if accepted:
            results.append(record)
    return results, {"rows": rows, "diagrams": reference, "null_maxima": null_maxima, "features": feature_records}


def localize_cloud(cloud, *, coordinate_groups=None, seed=0, params=None):
    settings = {**DEFAULTS, **(params or {})}
    x = np.asarray(cloud, dtype=float)
    if x.ndim != 2 or len(x) < settings["min_rows"] or not np.isfinite(x).all():
        raise ValueError("Localization requires sufficient finite timestamp observations")
    groups = ([np.array([i]) for i in range(x.shape[1])] if coordinate_groups is None
              else [np.asarray(group, dtype=int) for group in coordinate_groups])
    if sorted(np.concatenate(groups).tolist()) != list(range(x.shape[1])):
        raise ValueError("Each coordinate must belong to exactly one stock")
    runs = [_one_localization(x, groups, settings, seed + i * 1009)
            for i in range(settings["landmark_replicates"])]
    accepted = []
    for feature in runs[0][0]:
        members = set(feature["coordinates"])
        matches = [feature]
        for results, _ in runs[1:]:
            eligible = [other for other in results if other["dimension"] == feature["dimension"]]
            other = max(eligible, key=lambda item: len(members & set(item["coordinates"])) /
                        max(1, len(members | set(item["coordinates"]))), default=None)
            if other is not None and len(members & set(other["coordinates"])) / max(1, len(members | set(other["coordinates"]))) >= settings["stability_jaccard"]:
                matches.append(other)
        if len(matches) >= (len(runs) // 2 + 1):
            counts = Counter(node for item in matches for node in item["coordinates"])
            consensus = sorted(node for node, count in counts.items() if count >= (len(runs) // 2 + 1))
            if len(consensus) >= settings["min_size"] and not any(
                    item["dimension"] == feature["dimension"] and item["coordinates"] == consensus for item in accepted):
                accepted.append({**feature, "coordinates": consensus, "stability": len(matches) / len(runs)})
    records = [{**record, "diagrams": {str(k): v for k, v in record["diagrams"].items()},
                "null_maxima": {str(k): v for k, v in record["null_maxima"].items()}} for _, record in runs]
    return accepted, {"parameters": settings, "runs": records, "seed": seed}


def build_ph_localized_family(context, params):
    settings = {**DEFAULTS, **params}
    started = perf_counter()
    values = context.history.returns.to_numpy()
    valid = context.history.mask.to_numpy() & context.history.structural_mask[:, None]
    structural = context.history.structural_mask
    coverage = valid[structural].mean(axis=0) if structural.any() else np.zeros(values.shape[1])
    stocks = np.flatnonzero(coverage >= settings["coverage_threshold"])
    rows = structural & (valid[:, stocks].mean(axis=1) >= settings["row_coverage"]) if len(stocks) else np.zeros(len(values), bool)
    if len(stocks) < 2 or rows.sum() < settings["min_rows"]:
        return make_hyperedge_family({}, context, diagnostics={"reason": "insufficient_PH_coverage", "eligible_stocks": len(stocks)})
    observations, mask = values[rows][:, stocks], valid[rows][:, stocks]
    mean = np.where(mask, observations, 0).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    centered = np.where(mask, observations - mean, 0)
    pooled = np.sqrt(np.square(centered).sum() / max(mask.sum(), 1)) or 1
    x = np.concatenate([centered / pooled, .25 * mask], axis=1)
    coordinate_groups = [np.array([i, i + len(stocks)]) for i in range(len(stocks))]
    # Farthest-point selection only needs a bounded source pool, not a T*T matrix.
    source = np.linspace(0, len(x) - 1, min(len(x), 256), dtype=int)
    localized, state = localize_cloud(x[source], coordinate_groups=coordinate_groups, seed=context.seed, params=settings)
    groups = {f"PH:H{item['dimension']}:{i}": tuple(context.node_ids[stocks[node]] for node in item["coordinates"])
              for i, item in enumerate(localized)}
    attrs = {name: {"dimension": item["dimension"], "persistence": item["persistence"],
                    "stability": item["stability"], "subset_null_p": item["subset_null_p"]}
             for name, item in zip(groups, localized)}
    return make_hyperedge_family(groups, context, attrs,
                                diagnostics={"eligible_stocks": len(stocks), "eligible_rows": int(rows.sum()),
                                             "construction_seconds": perf_counter() - started,
                                             "localization": "coordinate_ablation_diagonal_matching", "higher_order_fraction":
                                             float(np.mean([len(g) >= 3 for g in groups.values()])) if groups else 0},
                                state={**state, "stock_coordinates": [context.node_ids[i] for i in stocks],
                                       "mean": mean, "pooled_scale": pooled, "source_times": context.history.returns.index[rows][source].astype(str).tolist()})
