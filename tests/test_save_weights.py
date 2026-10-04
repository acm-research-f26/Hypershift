"""R2 (post-2017 plan): opt-in best_state.pt + per-epoch predictions."""
import numpy as np
import torch

from hypershift.config import RunConfig
from hypershift.train.loop import (build_model, input_transform, predict_split, prepare, train_one_run)


def _cfg(tmp_path, label, **kw):
    base = dict(exp="t", label=label, seq=8, kernel=2, hidden=8, epochs=4, patience=50, device="cpu",
                out_root=str(tmp_path), lr=5e-3, input_mode="relative", batch_days=2)
    base.update(kw)
    return RunConfig(**base)


def _files(d):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def test_default_is_off():
    assert RunConfig().save_weights is False


def test_flag_off_file_set_unchanged(synthetic_market, synthetic_hypergraph, tmp_path):
    cfg = _cfg(tmp_path, "off")
    train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    assert _files(cfg.run_dir()) == sorted(
        ["config.json", "history.jsonl", "metrics.json", "test_daily.npy", "test_gt.npy",
         "test_mask.npy", "test_pred.npy"])


def test_flag_on_reproduces_test_pred_and_epoch_files(synthetic_market, synthetic_hypergraph, tmp_path):
    cfg = _cfg(tmp_path, "on", save_weights=True)
    m = train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    out = cfg.run_dir()
    assert (out / "best_state.pt").exists()
    n = m["epochs_run"]
    for kind in ("val", "test"):
        fs = sorted((out / "epoch_preds").glob(f"{kind}_e*.npy"))
        assert len(fs) == n
        assert fs[0].name == f"{kind}_e000.npy"
        assert np.load(fs[0]).dtype == np.float32
    # best epoch's saved test preds match test_pred.npy
    be = m["best_epoch"]
    np.testing.assert_allclose(np.load(out / f"epoch_preds/test_e{be:03d}.npy"), np.load(out / "test_pred.npy"),
                               atol=1e-6)
    # fresh model + best_state.pt reproduces test_pred.npy on CPU
    data, hg = prepare(cfg, synthetic_market, synthetic_hypergraph)
    dev = torch.device("cpu")
    model = build_model(cfg, data.features.shape[2], data).to(dev)
    model.load_state_dict(torch.load(out / "best_state.pt", map_location=dev))
    tp, _, _ = predict_split(model, data, hg.to_torch(dev), cfg, "test", dev, input_transform(cfg, data))
    np.testing.assert_allclose(tp, np.load(out / "test_pred.npy"), atol=1e-6)
