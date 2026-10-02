"""Small frozen-fold forecasting comparisons over prepared ForecastSamples.

The common temporal representation is the flattened prepared input history.
Callers retain responsibility for training-only feature preprocessing. This is
an offline construction experiment; it does not alter the existing GCN runner.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

from hyperedges.common.pipeline import build_hyperedge_snapshot, fit_hyperedge_pipeline, summarize_hyperedge_families
from hyperedges.common.registry import resolve_pipeline_config
from hyperedges.common.types import _utc_timestamp
from src.hypergraph import HyperedgeConsumer


def _prepared_sample(sample, node_ids):
    if tuple(sample.node_ids) != tuple(node_ids):
        raise ValueError("All forecast samples must use the construction stock axis")
    values, mask = np.asarray(sample.values), np.asarray(sample.mask)
    eligible = np.asarray(sample.eligible_nodes)
    target, target_mask = np.asarray(sample.target), np.asarray(sample.target_mask)
    if (values.ndim != 3 or values.shape[1] != len(node_ids) or mask.shape != values.shape
            or mask.dtype.kind != "b" or eligible.shape != (len(node_ids),) or eligible.dtype.kind != "b"):
        raise ValueError("Sample values, masks, and node eligibility must have aligned axes")
    if target.shape != (len(node_ids),) or target_mask.shape != target.shape or target_mask.dtype.kind != "b":
        raise ValueError("Sample targets and Boolean target masks must match the stock axis")
    if len(sample.history_times) != len(values) or len(values) == 0 or max(sample.history_times) > sample.origin_time:
        raise ValueError("Sample history must be nonempty and end before prediction time")
    active = eligible & mask.all(axis=(0, 2))
    x = np.where(mask, values, 0).transpose(1, 0, 2).reshape(len(node_ids), -1)
    if not np.isfinite(x[active]).all():
        raise ValueError("Active samples must have finite features")
    support = active & target_mask & np.isfinite(target)
    return torch.as_tensor(x, dtype=torch.float32), active, support


def _context_input(features, sample, snapshot):
    if not snapshot.context_features:
        return {}
    if features is None or sample.origin_time not in features:
        raise ValueError("PH/context experiments require actual history-derived features for every origin")
    contexts = tuple(features[sample.origin_time])
    expected = [(item.instance_id, item.names) for item in snapshot.context_features]
    if [(item.instance_id, item.names) for item in contexts] != expected:
        raise ValueError("Per-origin context feature axes differ from fitted providers")
    for item in contexts:
        if _utc_timestamp(item.provenance["cutoff"]) > _utc_timestamp(sample.origin_time):
            raise ValueError("Context features contain information after prediction time")
    return {"context_values": np.concatenate([item.values for item in contexts]),
            "context_mask": np.concatenate([item.mask for item in contexts])}


def _evaluate(model, samples, snapshot, context_features=None, disabled_families=()):
    model.eval()
    squared, absolute, correct, count = 0., 0., 0, 0
    forecasts = []
    with torch.no_grad():
        for sample in samples:
            x, active, support = _prepared_sample(sample, snapshot.node_ids)
            prediction = model(x, snapshot, active, disabled_families=disabled_families,
                               **_context_input(context_features, sample, snapshot)).flatten().cpu().numpy()
            errors = prediction[support] - sample.target[support]
            squared += float(np.square(errors).sum())
            absolute += float(np.abs(errors).sum())
            correct += int(((prediction[support] >= 0) == (sample.target[support] >= 0)).sum())
            count += int(support.sum())
            forecasts.append({"origin_time": sample.origin_time, "prediction": prediction, "support": support.copy()})
    if not count:
        raise ValueError("Evaluation samples have no eligible targets")
    return {"mse": squared / count, "mae": absolute / count, "directional_accuracy": correct / count,
            "target_count": count}, forecasts


def run_hyperedge_ablation(context, config, train_samples, validation_samples, test_samples, *,
                          seed=1001, hidden_channels=32, epochs=100, patience=15,
                          learning_rate=1e-3, context_features=None):
    config = resolve_pipeline_config(config)
    if config.protocol != "fold_frozen":
        raise ValueError("This runner implements frozen-fold ablations; rolling snapshots use the separate pipeline protocol")
    train_samples, validation_samples, test_samples = map(tuple, (train_samples, validation_samples, test_samples))
    if not train_samples or not validation_samples or not test_samples or epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Require nonempty chronological splits and positive training settings")
    cutoff = _utc_timestamp(context.cutoff)
    axes = (train_samples[0].feature_names, train_samples[0].values.shape)
    for split, samples in (("train", train_samples), ("validation", validation_samples), ("test", test_samples)):
        previous = None
        for sample in samples:
            if (sample.feature_names, sample.values.shape) != axes:
                raise ValueError("All ablation samples must have the same temporal and feature axes")
            origin, end, known = map(_utc_timestamp, (sample.origin_time, sample.target_end, sample.target_availability))
            if not origin < end <= known or (previous is not None and origin <= previous):
                raise ValueError("Forecast samples must be chronological with valid label availability")
            previous = origin
            if split == "train" and (origin >= cutoff or known > cutoff):
                raise ValueError("Training features and labels must be available by the construction cutoff")
            if split != "train" and origin < cutoff:
                raise ValueError("Evaluation must follow the training cutoff")
    if validation_samples[-1].origin_time >= test_samples[0].origin_time:
        raise ValueError("Validation and test origin intervals must not overlap")
    if any(sample.target_availability > test_samples[0].origin_time for sample in validation_samples):
        raise ValueError("Validation labels must be available before model selection for test")
    start = perf_counter()
    pipeline = fit_hyperedge_pipeline(context, config)
    snapshot = build_hyperedge_snapshot(pipeline)
    construction_seconds = perf_counter() - start
    first, _, _ = _prepared_sample(train_samples[0], snapshot.node_ids)
    context_channels = sum(len(item.names) for item in snapshot.context_features)
    # Keep the caller's RNG unchanged and align shared consumer initialization across ablations.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HyperedgeConsumer(first.shape[1], hidden_channels,
                                  family_ids=tuple(spec.instance_id for spec in config.constructors),
                                  learned_modules=pipeline.learned_modules, context_channels=context_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_state, best_mse, stalled, losses = None, float("inf"), 0, []
    start = perf_counter()
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for sample in train_samples:
            x, active, support = _prepared_sample(sample, snapshot.node_ids)
            if not support.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, snapshot, active, **_context_input(context_features, sample, snapshot)).flatten()
            target = torch.as_tensor(sample.target, dtype=prediction.dtype)
            supervised = (prediction[support] - target[support]).square().mean()
            loss = supervised + model.membership_penalty()
            loss.backward()
            optimizer.step()
            total += float(supervised.detach()) * int(support.sum())
            count += int(support.sum())
        if not count:
            raise ValueError("Training samples have no eligible targets")
        metrics, _ = _evaluate(model, validation_samples, snapshot, context_features)
        losses.append({"epoch": epoch + 1, "train_mse": total / count, "validation_mse": metrics["mse"]})
        if metrics["mse"] < best_mse:
            best_state, best_mse, stalled = deepcopy(model.state_dict()), metrics["mse"], 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    learned = model.freeze_memberships()
    snapshot = build_hyperedge_snapshot(pipeline)
    metrics, forecasts = _evaluate(model, test_samples, snapshot, context_features)
    return {"model": model, "pipeline": pipeline, "snapshot": snapshot, "test_metrics": metrics,
            "forecasts": forecasts, "training_history": losses, "construction_seconds": construction_seconds,
            "training_seconds": perf_counter() - start, "parameter_count": sum(p.numel() for p in model.parameters()),
            "families": summarize_hyperedge_families((*snapshot.families, *learned)), "seed": seed,
            "protocol": "retrained_frozen_fold", "representation": "flattened_prepared_history"}


def compare_hyperedge_ablations(context, configurations, train_samples, validation_samples, test_samples, **training_settings):
    """Retrain on identical splits/support and report one result per named configuration."""
    splits = tuple(tuple(samples) for samples in (train_samples, validation_samples, test_samples))
    results = {name: run_hyperedge_ablation(context, config, *splits, **training_settings)
               for name, config in configurations.items()}
    supports = [tuple(forecast["support"].tobytes() for forecast in result["forecasts"]) for result in results.values()]
    if supports and any(support != supports[0] for support in supports[1:]):
        raise RuntimeError("Ablation evaluation support differs")
    return results


def evaluate_frozen_family_masking(result, samples, disabled_families, *, context_features=None):
    metrics, forecasts = _evaluate(result["model"], samples, result["snapshot"], context_features, disabled_families)
    return {"protocol": "frozen_checkpoint_masking", "test_metrics": metrics, "forecasts": forecasts,
            "disabled_families": tuple(disabled_families)}
