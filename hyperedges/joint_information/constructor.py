"""Independent triple search and connected information beyond complete pairwise marginals."""

from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np
import pandas as pd

from ..common.validation import validate_identifiers
from ..common.validation import validate_positive_integer
from ..common.history import _subset
from ..common.types import make_hyperedge_family


def generate_candidate_groups(node_ids, order=3, budget=1000, seed=0):
    nodes = sorted(validate_identifiers(node_ids, "node"))
    validate_positive_integer(order, "order")
    validate_positive_integer(budget, "candidate budget")
    if order > len(nodes):
        return ()
    total = comb(len(nodes), order)
    if total <= budget:
        return tuple(combinations(nodes, order))
    # Sample combination ranks without generating the full search space.
    rng = np.random.default_rng(seed)
    ranks = set()
    for high in range(total - budget, total):
        draw = int(rng.integers(high + 1))
        ranks.add(high if draw in ranks else draw)
    candidates = []
    for rank in sorted(ranks):
        indices, lower = [], 0
        for position in range(order):
            remaining = order - position - 1
            for index in range(lower, len(nodes) - remaining):
                block = comb(len(nodes) - index - 1, remaining)
                if rank < block:
                    indices.append(index)
                    lower = index + 1
                    break
                rank -= block
        candidates.append(tuple(nodes[index] for index in indices))
    return tuple(sorted(candidates))


def split_estimation_selection_history(history, fraction=0.8):
    if not 0 < fraction < 1:
        raise ValueError("Estimation fraction must be in (0, 1)")
    sessions = history.session_ids.unique()
    if len(sessions) < 2:
        raise ValueError("Joint-information selection requires at least two historical sessions")
    count = max(1, min(len(sessions) - 1, int(len(sessions) * fraction)))
    earlier = np.asarray(history.session_ids.isin(sessions[:count]))
    return _subset(history, earlier, history.cutoff), _subset(history, ~earlier, history.cutoff)


@dataclass(frozen=True)
class StateEncoder:
    cutpoints: pd.DataFrame
    cardinality: int
    recipe: dict


def fit_state_encoder(estimation_history, bin_spec):
    if set(bin_spec) - {"kind", "quantiles"}:
        raise ValueError("Unknown state encoder parameters")
    kind = bin_spec.get("kind", "quantiles")
    nodes = estimation_history.returns.columns
    if kind == "binary_sign":
        points = pd.DataFrame(0.0, index=nodes, columns=["boundary:0"])
    elif kind == "quantiles":
        quantiles = tuple(bin_spec["quantiles"])
        if not quantiles or any(not 0 < x < 1 for x in quantiles) or tuple(sorted(set(quantiles))) != quantiles:
            raise ValueError("State quantiles must be unique, increasing probabilities in (0, 1)")
        valid = estimation_history.mask & estimation_history.structural_mask[:, None]
        points = estimation_history.returns.where(valid).quantile(quantiles).T
        points.columns = [f"boundary:{index}" for index in range(len(quantiles))]
    else:
        raise ValueError("State encoder kind must be quantiles or binary_sign")
    return StateEncoder(points, len(points.columns) + 1, dict(bin_spec))


def encode_joint_states(history, encoder, members):
    members = tuple(members)
    if not set(members) <= set(history.returns.columns) or not set(members) <= set(encoder.cutpoints.index):
        raise ValueError("Unknown candidate member")
    if not np.isfinite(encoder.cutpoints.loc[list(members)].to_numpy()).all():
        return np.empty((0, len(members)), dtype=int)
    valid = history.mask.loc[:, list(members)].all(axis=1).to_numpy() & history.structural_mask
    values = history.returns.loc[valid, list(members)].to_numpy()
    output = np.empty(values.shape, dtype=int)
    for column, member in enumerate(members):
        output[:, column] = np.searchsorted(encoder.cutpoints.loc[member].to_numpy(), values[:, column], side="right")
    return output


def estimate_joint_distribution(states, smoothing=0.5, *, cardinality=None):
    states = np.asarray(states)
    if states.ndim != 2 or not len(states) or states.dtype.kind not in "iu" or (states < 0).any():
        raise ValueError("Joint states must be a nonempty matrix of nonnegative integers")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Smoothing must be finite and nonnegative")
    cardinality = int(states.max()) + 1 if cardinality is None else cardinality
    validate_positive_integer(cardinality, "state cardinality")
    if (states >= cardinality).any():
        raise ValueError("State exceeds declared cardinality")
    shape = (cardinality,) * states.shape[1]
    if cardinality ** states.shape[1] > 1_000_000:
        raise ValueError("Joint state space exceeds the table budget")
    counts = np.full(shape, smoothing, dtype=float)
    np.add.at(counts, tuple(states.T), 1)
    return counts / counts.sum()


def _validate_probability(probability):
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.isclose(probability.sum(), 1, atol=1e-10):
        raise ValueError("Probability table must be finite, nonnegative, and normalized")
    return probability


def derive_pairwise_marginals(joint_distribution):
    probability = _validate_probability(joint_distribution)
    if probability.ndim < 3:
        raise ValueError("At least three variables are required")
    return {pair: probability.sum(axis=tuple(axis for axis in range(probability.ndim) if axis not in pair))
            for pair in combinations(range(probability.ndim), 2)}


def fit_pairwise_maxent(marginals, tolerance=1e-8, max_iterations=1000):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive and finite")
    validate_positive_integer(max_iterations, "max_iterations")
    if not marginals:
        raise ValueError("Pairwise constraints are required")
    dimensions = max(max(pair) for pair in marginals) + 1
    if set(marginals) != set(combinations(range(dimensions), 2)):
        raise ValueError("Supply every canonical pairwise marginal")
    shape = [None] * dimensions
    targets = {}
    for pair, marginal in marginals.items():
        target = _validate_probability(marginal)
        if target.ndim != 2:
            raise ValueError("Pairwise constraints must be matrices")
        for axis, size in zip(pair, target.shape, strict=True):
            if shape[axis] is not None and shape[axis] != size:
                raise ValueError("Incompatible marginal dimensions")
            shape[axis] = size
        targets[pair] = target
    if np.prod(shape) > 1_000_000:
        raise ValueError("Pairwise model exceeds the probability table budget")
    probability = np.full(shape, 1 / np.prod(shape), dtype=float)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        for pair, target in sorted(targets.items()):
            current = probability.sum(axis=tuple(axis for axis in range(dimensions) if axis not in pair))
            if ((current == 0) & (target > 0)).any():
                return {"probability": probability, "converged": False, "iterations": iteration,
                        "residual": float("inf"), "failure": "incompatible support"}
            factor = np.divide(target, current, out=np.zeros_like(target), where=current > 0)
            broadcast = [1] * dimensions
            for axis in pair:
                broadcast[axis] = shape[axis]
            probability *= factor.reshape(broadcast)
        mass = probability.sum()
        if mass <= 0 or not np.isfinite(mass):
            return {"probability": probability, "converged": False, "iterations": iteration,
                    "residual": float("inf"), "failure": "invalid probability mass"}
        probability /= mass
        residual = max(float(np.max(np.abs(
            probability.sum(axis=tuple(axis for axis in range(dimensions) if axis not in pair)) - target
        ))) for pair, target in targets.items())
        if residual <= tolerance:
            return {"probability": probability, "converged": True, "iterations": iteration, "residual": residual}
    return {"probability": probability, "converged": False, "iterations": max_iterations, "residual": residual,
            "failure": "iteration limit"}


def score_joint_information(joint_distribution, pairwise_model):
    if not pairwise_model["converged"]:
        raise ValueError("A nonconverged model cannot establish a joint-information score")
    probability = _validate_probability(joint_distribution)
    null = _validate_probability(pairwise_model["probability"])
    if probability.shape != null.shape:
        raise ValueError("Joint and pairwise probability axes must match")
    positive = probability > 0
    if (null[positive] == 0).any():
        return float("inf")
    return float(np.sum(probability[positive] * np.log2(probability[positive] / null[positive])))


def compare_joint_models(selection_states, joint_model, pairwise_model):
    states = np.asarray(selection_states)
    if not pairwise_model["converged"] or not len(states):
        return {"valid": False, "selection_observations": len(states)}
    full = joint_model[tuple(states.T)]
    pairwise = pairwise_model["probability"][tuple(states.T)]
    if (full <= 0).any() or (pairwise <= 0).any():
        return {"valid": False, "selection_observations": len(states), "failure": "unseen holdout support"}
    full_score, pair_score = np.log2(full), np.log2(pairwise)
    return {"valid": True, "selection_observations": len(states),
            "joint_log_likelihood_bits": float(full_score.mean()), "pairwise_log_likelihood_bits": float(pair_score.mean()),
            "holdout_gain_bits": float((full_score - pair_score).mean())}


def select_information_groups(results, selection_spec):
    required = {"edge_budget", "min_holdout_gain_bits", "min_selection_observations"}
    if required != set(selection_spec):
        raise ValueError(f"selection_spec must contain {sorted(required)}")
    validate_positive_integer(selection_spec["edge_budget"], "edge budget")
    validate_positive_integer(selection_spec["min_selection_observations"], "min selection observations")
    threshold = selection_spec["min_holdout_gain_bits"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Minimum holdout gain must be finite and nonnegative")
    eligible = [item for item in results if item.get("converged") and item.get("valid")
                and item["selection_observations"] >= selection_spec["min_selection_observations"]
                and item["holdout_gain_bits"] > threshold and np.isfinite(item["joint_information_bits"])]
    return sorted(eligible, key=lambda item: (-item["holdout_gain_bits"], item["members"]))[:selection_spec["edge_budget"]]


def build_joint_information_family(context, params):
    if params.get("order", 3) != 3:
        raise ValueError("The first joint-information constructor supports triples")
    mode = params.get("mode", "construct")
    if mode not in {"construct", "diagnose"}:
        raise ValueError("Information mode must be construct or diagnose")
    validate_positive_integer(params["min_observations"], "min observations")
    estimation, selection = split_estimation_selection_history(context.history, params.get("estimation_fraction", 0.8))
    encoder = fit_state_encoder(estimation, params["bin_spec"])
    candidates = generate_candidate_groups(context.node_ids, 3, params["candidate_budget"], context.seed)
    results = []
    for members in candidates:
        states = encode_joint_states(estimation, encoder, members)
        result = {"members": members, "estimation_observations": len(states), "converged": False}
        if len(states) >= params["min_observations"]:
            full = estimate_joint_distribution(states, params["smoothing"], cardinality=encoder.cardinality)
            null = fit_pairwise_maxent(derive_pairwise_marginals(full), params["tolerance"], params["max_iterations"])
            result.update({key: value for key, value in null.items() if key != "probability"})
            if null["converged"]:
                result["joint_information_bits"] = score_joint_information(full, null)
                result["observed_states"] = len(np.unique(states, axis=0))
                result.update(compare_joint_models(encode_joint_states(selection, encoder, members), full, null))
        results.append(result)
    selected = select_information_groups(results, params["selection_spec"])
    groups, attributes = {}, {}
    if mode == "construct":
        for result in selected:
            edge = "joint:" + "|".join(result["members"])
            groups[edge], attributes[edge] = result["members"], result
    return make_hyperedge_family(groups, context, attributes,
                                diagnostics={"mode": mode, "candidate_count": len(candidates), "results": results,
                                             "score_units": "bits", "significance_test": None},
                                state={"cutpoints": encoder.cutpoints, "cardinality": encoder.cardinality,
                                       "candidate_groups": candidates})
