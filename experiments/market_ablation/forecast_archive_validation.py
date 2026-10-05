"""Verify shared labels and recoverable fit artifacts without fitting models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from hyperedges.common.storage import load_hyperedge_snapshot
from .config import FEATURE_PACKS
from .storage import digest_file


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file(path, root):
    return {"path": str(Path(path).relative_to(root)), "bytes": Path(path).stat().st_size,
            "sha256": digest_file(path)}


def verify_shared_archive(config, dataset, fold):
    """Compare every shared label/mask value with the prepared historical data."""
    normalization = config.root / "normalization" / dataset.interval / f"fold-{fold['id']}.npz"
    with np.load(normalization, allow_pickle=False) as saved:
        scale = saved["target_scale"]
        _require(np.isfinite(scale).all() and scale > 0, f"Invalid target scale: {normalization}")
    checks, sections = [], []
    split = dataset.split(fold)
    schema = {"origins", "target", "mask", "active", "current_price", "actual_price",
              "origin_times", "target_times", "sessions"}
    for section in ("validation", "test"):
        directory = config.root / "shared" / dataset.interval / f"fold-{fold['id']}" / section
        positions, observations = [], 0
        files = sorted(directory.glob("*.npz"))
        expected_months = set(pd.to_datetime(dataset.times[split[section] + 1], utc=True).strftime("%Y-%m"))
        _require({p.stem for p in files} == expected_months, f"Incomplete shared months: {directory}")
        for path in files:
            with np.load(path, allow_pickle=False) as saved:
                _require(set(saved.files) == schema, f"Unexpected shared-label schema: {path}")
                arrays = {key: saved[key] for key in saved.files}
            origins = arrays["origins"]
            shape = (len(origins), len(dataset.symbols))
            _require(origins.dtype == np.int64 and origins.ndim == 1, f"Invalid shared origins: {path}")
            _require(all(arrays[k].shape == shape for k in ("target", "mask", "active", "current_price", "actual_price")),
                     f"Invalid shared stock axes: {path}")
            _require(arrays["target"].dtype == np.float32 and arrays["mask"].dtype == np.bool_
                     and arrays["active"].dtype == np.bool_, f"Invalid shared label/mask dtype: {path}")
            _require(np.array_equal(arrays["origin_times"], dataset.times[origins])
                     and np.array_equal(arrays["target_times"], dataset.times[origins + 1])
                     and np.array_equal(arrays["sessions"], dataset.sessions[origins]), f"Shared timestamps/sessions changed: {path}")
            _require(np.all(arrays["target_times"] > arrays["origin_times"]), f"Nonfuture shared targets: {path}")
            if dataset.interval != "1d":
                _require(np.array_equal(dataset.sessions[origins], dataset.sessions[origins + 1]),
                         f"Cross-session intraday target: {path}")
            if dataset.interval == "1m":
                _require(np.all(arrays["target_times"] - arrays["origin_times"] == 60_000_000_000),
                         f"Minute horizon changed: {path}")
            for begin in range(0, len(origins), 256):
                selection = slice(begin, begin + 256)
                batch = origins[selection]
                history = batch[:, None] + np.arange(1 - config.lookback, 1)[None, :]
                active = dataset.valid[batch] & (dataset.feature_mask[history, :, 0].sum(axis=1) >= 2)
                mask = active & dataset.valid[batch + 1]
                current, actual = dataset.bars[batch, :, 3], dataset.bars[batch + 1, :, 3]
                # Reproduce the saved float32 log-return target and scale round trip.
                target = (np.log(np.divide(actual, current, out=np.ones_like(actual), where=mask)) / scale).astype(np.float32) * scale
                _require(np.array_equal(arrays["active"][selection], active)
                         and np.array_equal(arrays["mask"][selection], mask), f"Shared eligibility differs from history: {path}")
                _require(np.array_equal(arrays["current_price"][selection], current, equal_nan=True)
                         and np.array_equal(arrays["actual_price"][selection], actual, equal_nan=True), f"Shared prices changed: {path}")
                _require(np.array_equal(arrays["target"][selection], target, equal_nan=True)
                         and np.isfinite(target[mask]).all(), f"Shared target values changed: {path}")
                observations += int(mask.sum())
            positions.append(origins)
            checks.append({"section": section, **_file(path, config.root)})
        _require(positions and np.array_equal(np.concatenate(positions), split[section]), f"Incomplete shared forecast origins: {directory}")
        sections.append({"section": section, "origins": len(split[section]), "eligible_stock_observations": observations})
    return {"interval": dataset.interval, "fold": fold["id"], "status": "verified", "files": checks,
            "normalization": _file(normalization, config.root), "sections": sections}


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(child) for child in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def verify_fit_artifacts(config, dataset, fold, job):
    """Verify model/optimizer/RNG recovery, selected epoch, snapshots and contributions."""
    directory = config.root / "fits" / job["id"]
    paths = [directory / name for name in ("result.json", "learning_curve.json", "best.pt", "latest.pt",
                                           "frozen_snapshot.json", "validation_contributions.npz", "test_contributions.npz")]
    _require(all(path.is_file() and path.stat().st_size > 0 for path in paths), f"Missing fit recovery artifacts: {job['id']}")
    result = json.loads(paths[0].read_text())
    curves = json.loads(paths[1].read_text())
    settings = json.loads((config.root / "settings" / f"{dataset.interval}.json").read_text())
    variant = json.loads(json.dumps(job["variant"]))
    _require(result["variant"] == variant and result["seed"] == job["seed"]
             and result["settings"] == settings and result["tuning"] is False
             and result["training_loss"] == config.training_loss, f"Fit identity/settings changed: {job['id']}")
    _require(len(curves) == result["epochs"] and 1 <= len(curves) <= config.epochs,
             f"Incomplete learning curve: {job['id']}")
    _require(result["still_improving_at_cap"] == (len(curves) == config.epochs and curves[-1]["improved"]),
             f"Incorrect epoch-cap flag: {job['id']}")
    for epoch, row in enumerate(curves, 1):
        _require(row["epoch"] == epoch and row["training_origins"] > 0 and _finite_tree(row), f"Invalid learning curve: {job['id']}")
        offset = (epoch - 1) % (config.minute_training_stride if dataset.interval == "1m" else 1)
        _require(row["sampling_offset"] == offset, f"Training-origin rotation changed: {job['id']}")
    snapshot = load_hyperedge_snapshot(paths[4])
    _require(tuple(snapshot.node_ids) == tuple(dataset.symbols) and not snapshot.learned_references,
             f"Frozen snapshot stock axes/memberships invalid: {job['id']}")
    _require(pd.Timestamp(snapshot.cutoff) <= pd.Timestamp(fold["fit_cutoff"]), f"Snapshot uses later history: {job['id']}")
    best_epoch = min(range(len(curves)), key=lambda i: curves[i]["validation_persistence_relative_mse"])
    width = config.lookback * len(FEATURE_PACKS[variant["features"]]) * 2
    for name in ("best.pt", "latest.pt"):
        checkpoint = torch.load(directory / name, map_location="cpu", weights_only=False)
        _require(checkpoint["settings"] == settings and checkpoint["seed"] == job["seed"]
                 and json.loads(json.dumps(checkpoint["variant"])) == variant, f"Checkpoint identity changed: {job['id']}/{name}")
        _require(_finite_tree(checkpoint["model"])
                 and tuple(checkpoint["model"]["temporal_projection.weight"].shape) == (settings["hidden"], width),
                 f"Invalid checkpoint model values/feature axes: {job['id']}/{name}")
        context = checkpoint["model"].get("context_projection.weight")
        _require((context is not None) == variant["ph_context"]
                 and (context is None or tuple(context.shape) == (settings["hidden"], 15)), f"Checkpoint PH feature axes changed: {job['id']}/{name}")
        if name == "best.pt":
            _require(checkpoint["epoch"] == best_epoch and checkpoint["score"] == result["best_sampled_validation"]
                     == curves[best_epoch]["validation_persistence_relative_mse"], f"Selected checkpoint differs from early stopping: {job['id']}")
        else:
            _require(checkpoint["epoch"] == result["epochs"] and checkpoint["batch_offset"] == 0
                     and checkpoint["curves"] == curves and checkpoint["training_loss"] == config.training_loss
                     and checkpoint["snapshot_id"] == snapshot.snapshot_id, f"Incomplete final resume checkpoint: {job['id']}")
            _require(_finite_tree(checkpoint["optimizer"]) and checkpoint["torch_rng"].dtype == torch.uint8
                     and checkpoint["torch_rng"].ndim == 1 and checkpoint["torch_rng"].numel() > 0
                     and isinstance(checkpoint["cuda_rng"], list)
                     and all(r.dtype == torch.uint8 and r.ndim == 1 and r.numel() > 0 for r in checkpoint["cuda_rng"]),
                     f"Invalid optimizer/RNG recovery: {job['id']}")
    contribution_schema = {"loss_sum", "persistence_loss_sum", "count", "price_loss_sum", "persistence_price_loss_sum"}
    for section in ("validation", "test"):
        with np.load(directory / f"{section}_contributions.npz", allow_pickle=False) as saved:
            _require(set(saved.files) == contribution_schema, f"Invalid session contributions: {job['id']}/{section}")
            values = {key: saved[key] for key in saved.files}
        _require(all(v.shape == (len(dataset.session_dates),) and np.isfinite(v).all() and np.all(v >= 0) for v in values.values()),
                 f"Invalid session loss/count values: {job['id']}/{section}")
        comparable = (values["count"] > 0) & (values["persistence_price_loss_sum"] > 0)
        expected = np.mean(values["price_loss_sum"][comparable] / values["persistence_price_loss_sum"][comparable])
        score = result["full_validation"] if section == "validation" else result["test_price_persistence_relative_mse"]
        if config.training_loss == "raw_price_MSE":
            _require(np.isclose(expected, score, rtol=1e-12, atol=1e-12), f"Stored raw-price score differs from session contributions: {job['id']}/{section}")
    if variant["ph_context"]:
        path = directory / "context_scaling.npz"
        with np.load(path, allow_pickle=False) as saved:
            _require(set(saved.files) == {"mean", "std"} and saved["mean"].shape == saved["std"].shape == (15,)
                     and np.isfinite(saved["mean"]).all() and np.isfinite(saved["std"]).all()
                     and np.all(saved["std"] > 0), f"Invalid causal PH scaling: {job['id']}")
        paths.append(path)
    return {"job": job["id"], "status": "verified", "selected_epoch": best_epoch + 1,
            "snapshot_id": snapshot.snapshot_id, "files": [_file(path, config.root) for path in paths]}
