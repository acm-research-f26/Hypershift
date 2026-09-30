"""Configuration integration tests; no adapters, training, or downloads run."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments import datasets
from experiments.config import (
    AnalysisConfig,
    ComponentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TaskConfig,
    TimeSpan,
)
from experiments.presets import (
    RETURN_FEATURES,
    RETURN_VOLUME_FEATURES,
    SMOKE_SEEDS,
)


def test_feature_comparison_retains_task_and_analysis_when_serialized():
    """A saved comparison must retain its horizon and inference assumptions."""
    task = TaskConfig(
        kind="regression",
        target=ComponentConfig(
            "future_log_return",
            {"horizon": TimeSpan(1, "hours")},
        ),
        output=ComponentConfig("point_prediction"),
        evaluation_scale="original",
    )
    evaluation = EvaluationConfig(
        metrics=(ComponentConfig("mse"),),
        analyses=(
            AnalysisConfig(
                name="paired_loss_difference",
                statistic=ComponentConfig("mean_loss_difference"),
                aggregation=ComponentConfig("equal_weight_nodes_per_origin"),
                uncertainty=ComponentConfig(
                    "paired_block_bootstrap",
                    {"block_length": TimeSpan(5, "sessions")},
                ),
                hypothesis=ComponentConfig("two_sided", {"null_value": 0}),
                inference_scope="fixed_forecasts",
            ),
        ),
        alignment=ComponentConfig("common_eligible_observations"),
        seed_reporting=ComponentConfig("separate_from_temporal_uncertainty"),
    )
    experiment = ExperimentConfig(
        name="returns_comparison",
        dataset=datasets.ALPACA_DATA_15_MIN_INTERVALS_250_GROUP,
        features=RETURN_FEATURES,
        task=task,
        hyperedge_features=RETURN_FEATURES,
        hyperedge_builders=(ComponentConfig("correlation_knn"),),
        hyperedge_learning=ComponentConfig("fixed"),
        model=ComponentConfig("euclidean_hgnn"),
        geometry=ComponentConfig("euclidean"),
        initialization=ComponentConfig("model_default"),
        optimizer=ComponentConfig("adamw", {"lr": 0.001}),
        objective=ComponentConfig("mse"),
        scheduler=None,
        training=ComponentConfig("early_stopping"),
        validation=ComponentConfig("expanding_window"),
        tuning=ComponentConfig("disabled"),
        evaluation=evaluation,
        diagnostics=(),
        portfolio=None,
        seeds=SMOKE_SEEDS,
        output_directory=str(datasets.LOCAL_DATA / "runs"),
    )
    variant = replace(
        experiment,
        name="returns_volume_comparison",
        features=RETURN_VOLUME_FEATURES,
    )
    original = json.loads(json.dumps(asdict(experiment)))
    saved = json.loads(json.dumps(asdict(variant)))

    assert saved["task"]["target"]["params"]["horizon"] == {
        "value": 1,
        "unit": "hours",
    }
    assert saved["features"][0]["params"]["volatility_window"] == {
        "value": 20,
        "unit": "steps",
    }
    analysis = saved["evaluation"]["analyses"][0]
    assert analysis["uncertainty"]["params"]["block_length"] == {
        "value": 5,
        "unit": "sessions",
    }
    assert analysis["inference_scope"] == "fixed_forecasts"
    assert saved["portfolio"] is None
    assert saved["scheduler"] is None
    assert not {"target", "metrics", "uncertainty"}.intersection(saved)
    changed_fields = {
        key for key in original if original[key] != saved[key]
    }
    assert changed_fields == {"name", "features"}


@pytest.fixture
def local_catalog():
    catalog_path = datasets.LOCAL_DATA / "catalog.json"
    if not catalog_path.is_file():
        pytest.skip("Ignored local data catalog is not available in this checkout")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "identifier",
    [
        "ALPACA_DATA_15_MIN_INTERVALS_250_GROUP",
        "ALPACA_DATA_HOURLY_250_GROUP",
        "ALPACA_DATA_WEEKLY_250_GROUP",
    ],
)
def test_available_views_use_catalog_archive_and_universe(identifier, local_catalog):
    group = getattr(datasets, identifier)
    catalog_group = local_catalog["dataset_groups"][identifier]
    archive = local_catalog["datasets"][catalog_group["source_dataset"]]
    universe = local_catalog["universes"][archive["universe_id"]]
    source_path = Path(group.source.params["path"])
    universe_path = Path(group.nodes.params["path"])

    assert datasets.DATASET_GROUPS[catalog_group["name"]] is group
    assert source_path.resolve() == (
        datasets.LOCAL_DATA / archive["path"]
    ).resolve()
    assert universe_path.resolve() == (
        datasets.LOCAL_DATA / universe["path"]
    ).resolve()
    assert source_path.is_dir()
    assert (source_path / "bars").is_dir()
    assert universe_path.is_file()
    assert group.nodes.params["expected_count"] == universe["node_count"]
    assert group.source.params["native_interval"] == archive["native_interval"]
    assert group.sampling.params["output_interval"] == catalog_group["output_interval"]


def test_minute_data_is_a_distinct_native_source():
    """A declaration cannot claim that 15-minute bars contain minute data."""
    minute = datasets.ALPACA_DATA_1_MIN_INTERVALS_250_GROUP
    quarter_hour = datasets.ALPACA_DATA_15_MIN_INTERVALS_250_GROUP

    assert minute.source.params["native_interval"] == "1min"
    assert minute.sampling.params["output_interval"] == "1min"
    assert Path(minute.source.params["path"]).resolve() != Path(
        quarter_hour.source.params["path"]
    ).resolve()
    assert minute.nodes.params["path"] == quarter_hour.nodes.params["path"]
    # No existence check: the minute archive has only been reserved, not fetched.


def test_minute_source_is_inside_catalog_reserved_directory(local_catalog):
    identifier = "ALPACA_DATA_1_MIN_INTERVALS_250_GROUP"
    catalog_group = local_catalog["dataset_groups"][identifier]
    archive = local_catalog["datasets"][catalog_group["source_dataset"]]
    reserved_path = (datasets.LOCAL_DATA / archive["reserved_directory"]).resolve()
    configured_path = Path(
        datasets.ALPACA_DATA_1_MIN_INTERVALS_250_GROUP.source.params["path"]
    ).resolve()

    assert configured_path.is_relative_to(reserved_path)
