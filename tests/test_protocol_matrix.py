import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import protocol_matrix as pm  # noqa: E402
from hypershift.eval.metrics import evaluate_all, ndcg_at_k, ndcg_sthan_compat, topk_daily_returns  # noqa: E402


def panel(n=40, d=30, seed=0, signal=0.5):
    rng = np.random.default_rng(seed)
    gt = rng.normal(0, 0.02, (n, d))
    pred = signal * gt + rng.normal(0, 0.02, (n, d))
    mask = np.ones((n, d))
    mask[:5, ::3] = 0
    return pred, gt, mask


def test_topk_returns_match_repo_and_k1():
    pred, gt, mask = panel()
    r = pm.topk_returns(pred, gt, mask)
    assert np.allclose(r[5], topk_daily_returns(pred, gt, mask, 5))
    assert np.allclose(r[10], topk_daily_returns(pred, gt, mask, 10))
    assert np.allclose(r[1], topk_daily_returns(pred, gt, mask, 1))


def test_topk_perfect_and_masked():
    gt = np.array([[0.1, 0.0], [0.3, 0.5], [0.2, -0.1], [9.0, 9.0]])
    mask = np.ones_like(gt)
    mask[3] = 0                      # masked stock must never be picked
    r = pm.topk_returns(gt.copy(), gt, mask, ks=(1, 2))
    assert np.allclose(r[1], [0.3, 0.5]) and np.allclose(r[2], [0.25, 0.25])


def test_sharpe_variants_definitions():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    v = pm.sharpe_variants(r, rf=0.0)
    assert math.isclose(v["unann"], r.mean() / r.std())
    assert math.isclose(v["ours"], v["unann"] * math.sqrt(252))
    assert math.isclose(v["unann_rf"], v["unann"])             # rf = 0
    v2 = pm.sharpe_variants(r, rf=0.001)
    assert math.isclose(v2["unann_rf"], (r.mean() - 0.001) / r.std())
    assert v2["unann_rf"] < v2["unann"]
    assert pm.sharpe_variants(np.zeros(5))["ours"] == 0.0     # sd = 0 -> 0, like the repo


def test_variants_from_hist_roundtrip():
    pred, gt, mask = panel(seed=3)
    r = pm.topk_returns(pred, gt, mask)[5]
    m = evaluate_all(pred, gt, mask)
    a = pm.variants_from_hist(m["sr"], m["ann_vol"], rf=2e-4)
    b = pm.sharpe_variants(r, rf=2e-4)
    assert all(math.isclose(a[k], b[k], rel_tol=1e-9) for k in a)


def test_ndcg_perfect_and_buggy_reference():
    pred, gt, mask = panel(seed=4, signal=1.0)
    assert math.isclose(ndcg_at_k(gt.copy(), gt, mask, 5), 1.0)
    # buggy evaluator on a perfect prediction: identical sets -> 1.0
    assert math.isclose(ndcg_sthan_compat(gt.copy(), gt, mask, 5), 1.0)


def test_oracle_and_val_selection():
    assert pm.oracle_pick([0.1, 0.9, 0.9, 0.3]) == 1             # first max
    hist = [{"val": {"sr": s}} for s in (1.0, 2.0, 2.0, 0.5)]
    assert pm.val_epoch(hist) == 1                               # strict >, like the loop


def _write_run(root, pred_epochs, gt, mask, val_srs):
    root.mkdir(parents=True)
    hist = []
    for e, p in enumerate(pred_epochs):
        m = evaluate_all(p, gt, mask)
        hist.append({"epoch": e, "val": {"sr": val_srs[e]}, "test": m})
    be = pm.val_epoch(hist)
    (root / "history.jsonl").write_text("\n".join(json.dumps(h) for h in hist))
    (root / "metrics.json").write_text(json.dumps({"best_epoch": be}))
    np.save(root / "test_pred.npy", pred_epochs[be])
    np.save(root / "test_gt.npy", gt)
    np.save(root / "test_mask.npy", mask)
    return be, hist


def test_extract_run_exact_vs_history_oracle(tmp_path):
    pred0, gt, mask = panel(seed=5, signal=0.0)
    preds = [pred0, panel(seed=5, signal=1.0)[0], panel(seed=6, signal=0.2)[0]]
    # exact: epoch_preds saved
    be, hist = _write_run(tmp_path / "a", preds, gt, mask, [3.0, 1.0, 0.5])
    (tmp_path / "a" / "epoch_preds").mkdir()
    for e, p in enumerate(preds):
        np.save(tmp_path / "a" / "epoch_preds" / f"test_e{e:03d}.npy", p.astype(np.float32))
    _write_run(tmp_path / "b", preds, gt, mask, [3.0, 1.0, 0.5])     # history only
    ra, rb = pm.extract_run(tmp_path / "a"), pm.extract_run(tmp_path / "b")
    assert ra["validation"]["epoch"] == 0 == be
    # oracle picks the planted-signal epoch (1), not the val-selected epoch (0)
    assert ra["oracle"]["sharpe_5_ours"]["epoch"] == 1
    assert ra["oracle"]["sharpe_5_ours"]["value"] > ra["validation"]["sharpe_5"]["ours"]
    # history-only oracle: k=5 matches the exact one (float32 rounding), k=1/10 are NA
    assert math.isclose(rb["oracle"]["sharpe_5_unann"]["value"], ra["oracle"]["sharpe_5_unann"]["value"], rel_tol=1e-3)
    assert rb["oracle"]["sharpe_1_ours"] is None and rb["oracle"]["sharpe_10_unann_rf"] is None
    assert ra["oracle"]["sharpe_1_ours"] is not None and ra["exact_oracle"] and not rb["exact_oracle"]
    # NDCG oracle = max over epochs of history
    assert rb["oracle"]["ndcg_correct"]["value"] == max(h["test"]["ndcg5"] for h in hist)
    # rf variant is below the unannualised one for positive rf
    assert ra["validation"]["sharpe_5"]["unann_rf"] < ra["validation"]["sharpe_5"]["unann"]


def test_compare_and_closeness():
    cells = {"THINK": {"mean": 1.2, "sd": 0.1, "n": 5}, "TCONV+DHHAN": {"mean": 1.1, "sd": 0.1, "n": 5}, "STHGCN": {"mean": 1.3, "sd": 0.1, "n": 5}}
    o = pm.compare(cells)
    assert o["think_gt_eh"] and not o["eh_gt_sthgcn"] and not o["full_order"]
    assert pm.compare({"THINK": cells["THINK"], "TCONV+DHHAN": None, "STHGCN": None})["full_order"] is None
    c = pm.closeness("sharpe_5_ours", cells)
    assert c["THINK"]["match"] and not pm.closeness("sharpe_5_ours", {"THINK": {"mean": 3.0, "sd": 0.1, "n": 5}})["THINK"]["match"]
    assert pm.paper_for("ndcg_buggy", "THINK") == (0.86, 9e-4)
