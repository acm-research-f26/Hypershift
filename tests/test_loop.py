import json
import numpy as np
import pytest
import torch
from dataclasses import replace
from hypershift.config import RunConfig, apply_overrides, config_from_dict
from hypershift.train.loop import gather_batch, prepare, train_one_run, window_offsets
from hypershift.train.loss import rank_mse_loss


def small_cfg(tmp_path, **kw):
    base = dict(exp="t", label="x", seq=8, kernel=2, hidden=8, epochs=4, patience=50, device="cpu",
                out_root=str(tmp_path), lr=5e-3)
    base.update(kw)
    return RunConfig(**base)


def test_window_offsets_no_leak(synthetic_market):
    d, seq = synthetic_market, 8
    tr, va, te = (window_offsets(d, seq, s) for s in ("train", "val", "test"))
    assert (tr + seq).max() < d.valid_index                     # train targets strictly before validation
    assert (va + seq).min() == d.valid_index and (va + seq).max() == d.test_index - 1
    assert (te + seq).min() == d.test_index and (te + seq).max() == d.num_steps - 1
    assert tr.min() == 0


def test_gather_batch_alignment(synthetic_market):
    d, seq = synthetic_market, 8
    offs = np.array([40, 45])
    x, m, b, g = gather_batch(d, offs, seq)
    assert x.shape == (2, d.num_nodes, seq, 5)
    np.testing.assert_allclose(x[1, :, -1], d.features[:, 45 + seq - 1])   # last input day
    np.testing.assert_allclose(g[1], d.gt[:, 45 + seq])                   # target = next day
    np.testing.assert_allclose(b[1], d.base_price[:, 45 + seq - 1])
    assert m[1, 0] == 0.0          # stock 0 missing on day 50, inside window 45..53
    assert m[0, 0] == 1.0          # window 40..48 is clean


def test_rank_mse_loss():
    gt = torch.tensor([[0.03, 0.01, -0.02]])
    m = torch.ones_like(gt)
    _, _, rank_good = rank_mse_loss(gt * 2, gt, m, 1.0)
    _, _, rank_bad = rank_mse_loss(-gt, gt, m, 1.0)
    assert rank_good.item() == 0.0 and rank_bad.item() > 0
    m2 = torch.tensor([[1.0, 1.0, 0.0]])
    loss, reg, _ = rank_mse_loss(torch.tensor([[0.03, 0.01, 9.0]]), gt, m2, 0.0)
    assert reg.item() == pytest.approx(0.0)


def test_overrides_and_roundtrip():
    d = apply_overrides({"label": "a"}, ["lr=0.01", "sources=industry", "shuffle_train_labels=true", "epochs=3"])
    cfg = config_from_dict(d)
    assert cfg.lr == 0.01 and cfg.sources == ("industry",) and cfg.shuffle_train_labels is True and cfg.epochs == 3
    assert config_from_dict(cfg.to_dict()) == cfg
    with pytest.raises(KeyError):
        config_from_dict({"nope": 1})


def test_prepare_structures(synthetic_market, synthetic_hypergraph, tmp_path):
    _, hg = prepare(small_cfg(tmp_path, structure="clique"), synthetic_market, synthetic_hypergraph)
    assert all(len(e) == 2 for e in hg.edges)
    _, hg = prepare(small_cfg(tmp_path, drop_hub_degree=2), synthetic_market, synthetic_hypergraph)
    assert 2 not in {v for e in hg.edges for v in e}               # node 2 has degree 2
    d, hg = prepare(small_cfg(tmp_path, universe_size=6, universe_seed=1), synthetic_market, synthetic_hypergraph)
    assert d.num_nodes == 6 and hg.num_nodes == 6
    d, _ = prepare(small_cfg(tmp_path, shuffle_train_labels=True), synthetic_market, synthetic_hypergraph)
    assert not np.allclose(d.gt[:, : d.valid_index], synthetic_market.gt[:, : d.valid_index])
    np.testing.assert_allclose(d.gt[:, d.valid_index:], synthetic_market.gt[:, d.valid_index:])
    # cross-sectional shuffle: each training day keeps the same multiset of returns
    np.testing.assert_allclose(np.sort(d.gt[:, : d.valid_index], axis=0),
                               np.sort(synthetic_market.gt[:, : d.valid_index], axis=0))


def test_train_one_run_end_to_end_and_resume(synthetic_market, synthetic_hypergraph, tmp_path):
    cfg = small_cfg(tmp_path)
    m = train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    out = cfg.run_dir()
    for f in ("config.json", "history.jsonl", "metrics.json", "test_pred.npy", "test_daily.npy"):
        assert (out / f).exists(), f
    assert len((out / "history.jsonl").read_text().splitlines()) == 4
    assert {"best_epoch", "val", "test", "test_oracle_sr", "sec_per_epoch", "covered_frac"} <= set(m)
    assert m["covered_frac"] == pytest.approx(11 / 12)                   # node 11 is isolated
    assert np.load(out / "test_pred.npy").shape == (12, 20)
    mtime = (out / "metrics.json").stat().st_mtime
    train_one_run(cfg, synthetic_market, synthetic_hypergraph)            # resume: no retrain
    assert (out / "metrics.json").stat().st_mtime == mtime


@pytest.mark.parametrize("temporal,spatial", [("hyp", "hyp"), ("euc", "euc")])
def test_train_loss_decreases(synthetic_market, synthetic_hypergraph, tmp_path, temporal, spatial):
    cfg = small_cfg(tmp_path, epochs=15, temporal=temporal, spatial=spatial, label=temporal + spatial)
    train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    hist = [json.loads(l) for l in (cfg.run_dir() / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_micro_batch_matches_full_batch(synthetic_market, synthetic_hypergraph, tmp_path):
    """Gradient accumulation must be mathematically identical to the unsplit batch (D15)."""
    runs = {}
    for label, micro in (("full", 0), ("micro", 1)):
        cfg = small_cfg(tmp_path, label=label, batch_days=4, micro_batch_days=micro, epochs=3)
        train_one_run(cfg, synthetic_market, synthetic_hypergraph)
        runs[label] = [json.loads(l)["train_loss"] for l in (cfg.run_dir() / "history.jsonl").read_text().splitlines()]
    np.testing.assert_allclose(runs["micro"], runs["full"], rtol=1e-4)


def test_train_raises_on_nan(synthetic_market, synthetic_hypergraph, tmp_path):
    bad = replace(synthetic_market, features=synthetic_market.features.copy())
    bad.features[:, 5, :] = np.nan
    with pytest.raises(FloatingPointError):
        train_one_run(small_cfg(tmp_path, label="nan"), bad, synthetic_hypergraph)
