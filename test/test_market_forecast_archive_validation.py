from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from experiments.market_ablation.config import SweepConfig, core_variants
from experiments.market_ablation.dataset import MarketDataset
from experiments.market_ablation.forecast_archive_validation import verify_shared_archive, verify_fit_artifacts
from experiments.market_ablation.storage import atomic_json, atomic_npz
from hyperedges.common.types import HyperedgeSnapshot


@pytest.fixture
def saved_forecasts(tmp_path, monkeypatch):
    from experiments.market_ablation import training
    from experiments.market_ablation.runner import archive_shared
    config = replace(SweepConfig(), output=str(tmp_path), lookback=3, epochs=2, device="cpu")
    symbols = tuple(f"S{i}" for i in range(250))
    times = pd.date_range("2025-01-02 14:30", periods=24, freq="min", tz="UTC").as_unit("ns").asi8
    bars = np.ones((24, 250, 5), np.float32)
    bars[:, :, 3] = 100 + np.arange(24)[:, None] * .1
    valid = np.ones((24, 250), bool)
    valid[13, 0] = False
    bars[13, 0, 3] = np.nan
    data = SimpleNamespace(interval="1m", config=config, symbols=symbols, times=times, bars=bars, valid=valid,
                           features=np.random.default_rng(8).normal(size=(24, 250, 10)).astype(np.float32),
                           feature_mask=np.ones((24, 250, 10), bool), sessions=np.zeros(24, np.int32),
                           session_dates=np.array(["2025-01-02"]))
    split = {"train": np.arange(3, 11), "validation": np.arange(12, 16), "test": np.arange(16, 23)}
    data.split = lambda fold: split
    moments = {"mean": np.zeros((250, 10), np.float32), "std": np.ones((250, 10), np.float32),
               "target_scale": np.array(.01, np.float32), "price_target_scale": np.array(.1, np.float32)}
    data.moments = lambda fold: moments
    data.batch = lambda origins, variant, values: MarketDataset.batch(data, origins, variant, values)
    data.tuning_origins = lambda origins: origins
    data.epoch_origins = lambda origins, epoch, **kwargs: origins[epoch % config.minute_training_stride::config.minute_training_stride]
    fold = {"id": 0, "fit_cutoff": pd.Timestamp(times[11], tz="UTC").isoformat()}
    variant = next(v for v in core_variants() if v.name == "T")
    snapshot = HyperedgeSnapshot("archive-test", symbols, (), pd.Timestamp(fold["fit_cutoff"]),
                                pd.Timestamp(fold["fit_cutoff"]), pd.Timestamp(fold["fit_cutoff"]))
    monkeypatch.setattr(training, "cached_components", lambda *args: (snapshot, {}))
    settings = {"hidden": 8, "learning_rate": .001, "weight_decay": 0., "batch_size": 4}
    atomic_json(config.root / "settings/1m.json", settings)
    atomic_npz(config.root / "normalization/1m/fold-0.npz", **moments)
    archive_shared(config, data, fold)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.fit(config, data, fold, variant, 1001, settings)
    finally:
        torch.set_num_threads(previous)
    job = {"id": "1m/fold-0/T/seed-1001", "variant": variant.__dict__, "seed": 1001,
           "interval": "1m", "fold": 0, "scope": "core"}
    return config, data, fold, job


def mutate_npz(path, key, index, value):
    with np.load(path) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays[key][index] = value
    np.savez_compressed(path, **arrays)


def test_verifies_written_labels_checkpoints_and_hashes(saved_forecasts):
    config, data, fold, job = saved_forecasts
    labels = verify_shared_archive(config, data, fold)
    fit = verify_fit_artifacts(config, data, fold, job)
    assert labels["status"] == fit["status"] == "verified"
    assert len(labels["files"]) == 2 and len(fit["files"]) == 7
    assert all(len(f["sha256"]) == 64 for f in labels["files"] + fit["files"])


@pytest.mark.parametrize("key,index,value,match", [
    ("mask", (0, 1), False, "eligibility"), ("target", (0, 1), .2, "target values"),
    ("actual_price", (0, 1), 150., "prices"), ("target_times", 0, 0, "timestamps"),
])
def test_rejects_corrupted_shared_labels(saved_forecasts, key, index, value, match):
    config, data, fold, _ = saved_forecasts
    path = config.root / "shared/1m/fold-0/validation/2025-01.npz"
    mutate_npz(path, key, index, value)
    with pytest.raises(ValueError, match=match):
        verify_shared_archive(config, data, fold)


@pytest.mark.parametrize("damage,match", [
    ("model", "model values"), ("optimizer", "optimizer/RNG"),
    ("best_epoch", "Selected checkpoint"), ("resume", "final resume checkpoint"),
    ("score", "session contributions"), ("missing", "Missing fit recovery"),
])
def test_rejects_corrupted_fit_recovery(saved_forecasts, damage, match):
    config, data, fold, job = saved_forecasts
    directory = config.root / "fits" / job["id"]
    if damage == "score":
        path = directory / "result.json"
        result = json.loads(path.read_text())
        result["full_validation"] += .01
        atomic_json(path, result)
    elif damage == "missing":
        (directory / "best.pt").unlink()
    else:
        path = directory / ("best.pt" if damage in ("model", "best_epoch") else "latest.pt")
        state = torch.load(path, weights_only=False)
        if damage == "model":
            state["model"]["temporal_projection.weight"][0, 0] = float("nan")
        elif damage == "optimizer":
            next(v["exp_avg"] for v in state["optimizer"]["state"].values() if v["exp_avg"].numel()).flatten()[0] = float("nan")
        elif damage == "best_epoch":
            state["epoch"] += 2
        else:
            state["batch_offset"] = 1
        torch.save(state, path)
    with pytest.raises(ValueError, match=match):
        verify_fit_artifacts(config, data, fold, job)


@pytest.mark.parametrize("damage", [None, "shared_target", "resume_state"])
def test_final_report_requires_recoverable_shared_and_fit_archives(saved_forecasts, monkeypatch, damage):
    from experiments.market_ablation import reporting, archive_validation, relative_price_inference, feature_pair_inference
    config, data, fold, job = saved_forecasts
    config = replace(config, intervals=("1m",))
    monkeypatch.setattr(reporting, "manifest", lambda _: {"jobs": [job]})
    monkeypatch.setattr(reporting, "folds", lambda: [fold])
    monkeypatch.setattr(reporting, "MarketDataset", lambda *_: data)
    monkeypatch.setattr(reporting, "analysis_worker_count", lambda: 1)
    monkeypatch.setattr(reporting, "analyze_jobs", lambda *_: iter([(0, {"job": job["id"]}, [])]))
    for name in ("statistical_catalog", "comparisons", "figures"):
        monkeypatch.setattr(reporting, name, lambda *_: None)
    monkeypatch.setattr(relative_price_inference, "relative_price_report", lambda *_: [])
    def feature_pairs_stub(c, _):
        payload = {"groups_count": 0}
        atomic_json(c.root / "report/feature_context_pair_inference.json", payload)
        return payload
    monkeypatch.setattr(feature_pair_inference, "feature_pair_report", feature_pairs_stub)
    monkeypatch.setattr(feature_pair_inference, "feature_pair_figures", lambda *_: [])
    monkeypatch.setattr(archive_validation, "verify_portfolio_archive", lambda *_: {"files": []})
    monkeypatch.setattr(reporting, "archive_execution_sources",
                        lambda c: atomic_json(c.root / "execution_sources/latest.json", {"identity": "a" * 64}))
    atomic_json(config.root / "corporate_actions/manifest.json", {"complete": True, "records": 0})
    atomic_json(config.root / "corporate_actions/records.json", [])
    if damage == "shared_target":
        mutate_npz(config.root / "shared/1m/fold-0/test/2025-01.npz", "target", (0, 0), .2)
    elif damage == "resume_state":
        path = config.root / "fits" / job["id"] / "latest.pt"
        state = torch.load(path, weights_only=False)
        state["batch_offset"] = 1
        torch.save(state, path)
    if damage:
        with pytest.raises(ValueError):
            reporting.report(config)
        assert json.loads((config.root / "report/completion.json").read_text())["complete"] is False
    else:
        reporting.report(config)
        result = json.loads((config.root / "report/completion.json").read_text())
        assert result["complete"] and result["shared_archive_files"] == 2 and result["recovery_artifact_files"] == 7
        certificate = json.loads((config.root / "report/recovery_archive_verification.json").read_text())
        assert len(certificate["jobs"]) == len(certificate["shared_cohorts"]) == 1
