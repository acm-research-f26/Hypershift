"""Bounded recurring common-witness groups; missing observations remain unknown."""

from itertools import combinations

import numpy as np
import pandas as pd

from ..common.types import make_hyperedge_family
from ..common.validation import validate_positive_integer



def fit_event_thresholds(history, quantiles):
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("quantiles must specify lower and upper probabilities in (0, 1)")
    valid = history.mask & history.structural_mask[:, None]
    thresholds = history.returns.where(valid).quantile(quantiles).T
    thresholds.columns = ["lower", "upper"]
    return thresholds


def build_stock_event_relation(history, thresholds, direction="down"):
    if direction not in {"down", "up", "absolute"}:
        raise ValueError("Event direction must be down, up, or absolute")
    if not thresholds.index.equals(history.returns.columns):
        raise ValueError("Threshold and stock axes must match")
    down = history.returns.le(thresholds.lower, axis=1)
    up = history.returns.ge(thresholds.upper, axis=1)
    participation = down if direction == "down" else up if direction == "up" else down | up
    return participation & history.mask & history.structural_mask[:, None]


def mine_recurring_groups(relation, min_support, size_bounds, *, candidate_budget=100_000):
    validate_positive_integer(min_support, "min_support")
    validate_positive_integer(candidate_budget, "candidate_budget")
    minimum, maximum = size_bounds
    validate_positive_integer(minimum, "minimum group size")
    validate_positive_integer(maximum, "maximum group size")
    if minimum > maximum or not relation.dtypes.eq(bool).all():
        raise ValueError("Require ordered group sizes and Boolean participation")
    nodes = sorted(relation.columns)
    witnesses = {node: set(np.flatnonzero(relation[node].to_numpy())) for node in nodes}
    frequent = {(node,): timestamps for node, timestamps in witnesses.items() if len(timestamps) >= min_support}
    output = list(frequent) if minimum == 1 else []
    examined = len(nodes)
    if examined > candidate_budget:
        raise ValueError("Event mining candidate budget exceeded; tighten support or group-size settings")
    for size in range(2, min(maximum, len(nodes)) + 1):
        next_frequent = {}
        buckets = {}
        for group in sorted(frequent):
            buckets.setdefault(group[:-1], []).append(group)
        for bucket in buckets.values():
            for left, right in combinations(bucket, 2):
                candidate = (*left, right[-1])
                examined += 1
                if examined > candidate_budget:
                    raise ValueError("Event mining candidate budget exceeded; tighten support or group-size settings")
                if not all(tuple(subset) in frequent for subset in combinations(candidate, size - 1)):
                    continue
                shared = frequent[left] & frequent[right]
                if len(shared) >= min_support:
                    next_frequent[candidate] = shared
        frequent = next_frequent
        if not frequent:
            break
        if size >= minimum:
            output.extend(sorted(frequent))
    return tuple(output)


def measure_group_event_support(groups, relation, validity):
    if not relation.index.equals(validity.index) or not relation.columns.equals(validity.columns):
        raise ValueError("Event relation and validity axes must match")
    statistics = []
    for members in groups:
        support = int(relation.loc[:, list(members)].all(axis=1).sum())
        observable = int(validity.loc[:, list(members)].all(axis=1).sum())
        statistics.append({"members": tuple(members), "support": support, "joint_observations": observable,
                           "event_rate": support / observable if observable else 0.0})
    return statistics


def select_event_groups(statistics, edge_budget):
    validate_positive_integer(edge_budget, "edge_budget")
    return sorted(statistics, key=lambda item: (-item["support"], -item["event_rate"], item["members"]))[:edge_budget]


def build_event_dowker_family(context, params):
    thresholds = fit_event_thresholds(context.history, params["quantiles"])
    direction = params.get("direction", "down")
    relation = build_stock_event_relation(context.history, thresholds, direction)
    groups = mine_recurring_groups(relation, params["min_support"], params["size_bounds"],
                                  candidate_budget=params.get("candidate_budget", 100_000))
    valid = context.history.mask & context.history.structural_mask[:, None]
    selected = select_event_groups(measure_group_event_support(groups, relation, valid), params["edge_budget"])
    memberships, attributes = {}, {}
    for item in selected:
        edge = f"event:{direction}:" + "|".join(item["members"])
        memberships[edge] = item["members"]
        attributes[edge] = {**item, "direction": direction}
    return make_hyperedge_family(memberships, context, attributes,
                                diagnostics={"candidate_groups": len(groups), "selected_groups": len(selected)},
                                state={"thresholds": thresholds})
