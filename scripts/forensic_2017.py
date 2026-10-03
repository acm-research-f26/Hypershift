"""Phase 1.5a forensic pipeline (CPU only). Run from repo root:
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage inventory
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hypershift.eval import forensics as F  # noqa: E402

PRIMARY = ("R5_f2_alpha0_train", "HH")
REFERENCES = (("R5_f_train", "HH"), ("R5_f_train", "EH"), ("R5_f2_alpha0_train", "EH"), ("R5_f_paper", "HH"))
SEEDS = (0, 1, 2, 3, 4)
DOCS = ROOT / "docs" / "phase1_5a"
FIGS = ROOT / "docs" / "figures"
DATA_ROOT = "data/raw/rsr/data"
EVIDENCE_KEYS = ("weight_decay", "input_mode", "alpha", "norm", "spatial_residual", "topk", "seq", "seed")


def stage_inventory(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    dates = [ln.strip()[:10] for ln in open(ROOT / DATA_ROOT / "NYSE_aver_line_dates.csv") if ln.strip()]
    test_dates = [dates[29 + 1008 + j] for j in range(237)]
    assert test_dates[0] == "2017-01-03" and test_dates[-1] == "2017-12-08", (test_dates[0], test_dates[-1])
    from hypershift.data.rsr import read_ticker_file
    tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    assert len(tickers) == 1737
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    inv = {"git_head": head, "test_dates": [test_dates[0], test_dates[-1]], "runs": {}, "n_verified": 0}
    for exp, label in (PRIMARY,) + REFERENCES:
        rows = []
        for s in SEEDS:
            run = f"{exp}/{label}"
            ar = F.load_run(exp, label, s, ROOT / "results")
            r, baskets = F.portfolio(ar.pred, ar.gt, ar.mask)
            diff = float(np.abs(r - ar.daily).max())
            sr = F.perf(r)["sr"]
            be = ar.metrics["best_epoch"]
            if diff > 1e-7 or abs(sr - ar.metrics["test"]["sr"]) > 1e-5 or abs(sr - ar.history[be]["test"]["sr"]) > 1e-5:
                raise SystemExit(f"{run} seed {s}: max|diff|={diff} sr={sr} metrics={ar.metrics['test']['sr']}")
            inv["n_verified"] += 1
            files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ar.path.iterdir())}
            dtype = str(np.load(ar.path / "test_pred.npy", mmap_mode="r").dtype)
            inv["runs"][f"{run}/seed_{s}"] = {
                "sha256": files, "pred_dtype": dtype, "best_epoch": be, "test_oracle_epoch": ar.metrics.get("test_oracle_epoch"),
                "epochs_run": ar.metrics.get("epochs_run"), "config": {k: ar.config.get(k) for k in EVIDENCE_KEYS},
                "sr_recomputed": sr, "max_abs_daily_diff": diff}
            # tidy exports
            b = F.boundary_stats(ar.pred, ar.mask)
            ha = F.hold_all(ar.gt, ar.mask)
            to = F.turnover(baskets)
            recs = []
            for d in range(237):
                idx = np.nonzero(ar.mask[:, d])[0]
                o = idx[np.argsort(-ar.pred[idx, d], kind="stable")]
                rank = np.empty(len(o), int)
                rank[:] = np.arange(1, len(o) + 1)
                top5 = set(baskets[d].tolist())
                recs.append(pd.DataFrame({
                    "date": test_dates[d], "day": d, "ticker": [tickers[i] for i in o], "index": o,
                    "score": ar.pred[o, d], "ret_next": ar.gt[o, d], "rank": rank,
                    "in_top5": [int(i in top5) for i in o], "s5": b["s_k"][d], "s6": b["s_k1"][d],
                    "margin5": b["margin"][d], "exact_tie5": bool(b["exact_tie"][d]), "seed": s}))
            pd.concat(recs).to_csv(out / f"stockday_{exp}_{label}_s{s}.csv.gz", index=False)
            rows.append(pd.DataFrame({"date": test_dates, "day": np.arange(237), "seed": s, "ret_gross": r,
                                      "ret_hold_all": ha, "turnover": to, "exact_tie5": b["exact_tie"],
                                      "margin5": b["margin"], "score_sd": b["sd"], "n_valid": b["n_valid"]}))
        pd.concat(rows).to_csv(out / f"portday_{exp}_{label}.csv", index=False)
    inv["not_recoverable"] = [
        "model weights / checkpoints (none saved; loop.py has no torch.save)",
        "per-stock predictions for non-selected epochs (overwritten; only the last val-improving epoch is on disk)",
        "through-model index permutation of the trained model",
        "frozen-model evaluation outside 2017"]
    (DOCS / "inventory.json").write_text(json.dumps(inv, indent=1))
    print(f"inventory: {inv['n_verified']} run-seeds verified")


STAGES = {"inventory": stage_inventory}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--out", default="results/forensics_1_5a")
    ap.add_argument("--quick", action="store_true", help="debug only: tiny B/R")
    a = ap.parse_args()
    os.chdir(ROOT)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        STAGES[name](a)


if __name__ == "__main__":
    main()
