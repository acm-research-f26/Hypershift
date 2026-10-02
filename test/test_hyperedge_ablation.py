"""Small forecasting comparisons distinguish retraining from frozen family masking."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.types import ForecastSample
from experiments.hyperedge_ablation import compare_hyperedge_ablations, evaluate_frozen_family_masking, run_hyperedge_ablation
from hyperedges import ConstructorSpec, HyperedgePipelineConfig

from test.test_hyperedge_pipeline import context


def _sample(context, row=None, origin=None, seed=0):
    if row is None:
        values = np.random.default_rng(seed).normal(size=(3, 8, 1))
        times = context.history.returns.index[-3:]
    else:
        values = context.history.returns.iloc[row - 2:row + 1].to_numpy()[:, :, None]
        times = context.history.returns.index[row - 2:row + 1]
        origin = times[-1] + pd.Timedelta(minutes=1)
    return ForecastSample(origin, times, context.node_ids, ("log_return",), values, np.ones(values.shape, dtype=bool),
                          np.ones(8, dtype=bool), .2 * values[-1, :, 0], np.ones(8, dtype=bool),
                          origin, origin + pd.Timedelta(minutes=14), origin + pd.Timedelta(minutes=15))


def _splits(context):
    train = tuple(_sample(context, row=row) for row in range(43, 47))
    validation = tuple(_sample(context, origin=context.cutoff + pd.Timedelta(hours=i + 1), seed=100 + i) for i in range(2))
    test = tuple(_sample(context, origin=context.cutoff + pd.Timedelta(hours=i + 5), seed=200 + i) for i in range(3))
    return train, validation, test


def test_core_forecast_ablations_share_support_and_train(context):
    knn, gics = ConstructorSpec("knn", "covariance_knn"), ConstructorSpec("gics", "gics")
    learned = ConstructorSpec("learned", "learned_membership", {"slots": 2})
    configurations = {"temporal": HyperedgePipelineConfig(), "knn": HyperedgePipelineConfig((knn,)),
                      "gics": HyperedgePipelineConfig((gics,)), "both": HyperedgePipelineConfig((gics, knn)),
                      "learned": HyperedgePipelineConfig((gics, knn, learned))}
    results = compare_hyperedge_ablations(context, configurations, *_splits(context), epochs=2, patience=2, hidden_channels=4)
    assert all(result["test_metrics"]["target_count"] == 24 for result in results.values())
    assert all(np.isfinite(result["test_metrics"]["mse"]) for result in results.values())
    assert all(len(result["training_history"]) == 2 for result in results.values())
    assert results["learned"]["families"]["learned"]["diagnostics"]["frozen"]
    original_logits = results["learned"]["pipeline"].learned_modules["learned"].logits.detach().clone()
    masked = evaluate_frozen_family_masking(results["learned"], _splits(context)[2], ("gics", "knn", "learned"))
    assert masked["protocol"] == "frozen_checkpoint_masking" and masked["test_metrics"]["target_count"] == 24
    assert np.array_equal(original_logits, results["learned"]["pipeline"].learned_modules["learned"].logits.detach())


def test_runner_rejects_future_training_labels_and_overlapping_splits(context):
    train, validation, test = _splits(context)
    bad = replace(train[0], target_availability=context.cutoff + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="available by the construction cutoff"):
        run_hyperedge_ablation(context, HyperedgePipelineConfig(), (bad, *train[1:]), validation, test, epochs=1)
    with pytest.raises(ValueError, match="not overlap"):
        run_hyperedge_ablation(context, HyperedgePipelineConfig(), train, validation, validation, epochs=1)
