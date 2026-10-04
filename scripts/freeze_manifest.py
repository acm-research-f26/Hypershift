"""Phase 1.5b Task 3: freeze manifest + predeclared R7 2017 replication check. CPU only, reads only pre-2018 artifacts.
Usage: freeze_manifest.py write   -> docs/phase1_5b/FREEZE_MANIFEST.md
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

NEW, OLD = Path("results/R5_f3_alpha0_train"), Path("results/R5_f2_alpha0_train")
ARMS, SEEDS = ("HH", "EH"), range(5)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sr(root, arm):
    return np.array([json.loads((root / arm / f"seed_{s}" / "metrics.json").read_text())["test"]["sr"] for s in SEEDS])


def main():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    L = ["# Freeze manifest (Phase 1.5b, plan Task 3)", "",
         f"Written before any 2018+ inference. git HEAD at writing: `{head}`. Experiment `R5_f3_alpha0_train`, arms HH and EH, seeds 0-4, 100 epochs, trained on Kaggle GPU with `save_weights=true`.",
         "Selection rule: epoch with the best pre-2017 validation Sharpe (`best_epoch`, validation = 2016). No 2017 or 2018+ return affected selection.", "",
         "## Runs", "", "| arm | seed | best_epoch | epochs_run | val SR | 2017 test SR | sha256 config.json | sha256 best_state.pt | sha256 test_pred.npy |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for s in SEEDS:
            d = NEW / a / f"seed_{s}"
            m = json.loads((d / "metrics.json").read_text())
            n = len(list((d / "epoch_preds").glob("*.npy")))
            assert n == 200 and m["epochs_run"] == 100, (a, s, n)
            L.append(f"| {a} | {s} | {m['best_epoch']} | {m['epochs_run']} | {m['val']['sr']:.4f} | {m['test']['sr']:.4f} | `{sha(d/'config.json')}` | `{sha(d/'best_state.pt')}` | `{sha(d/'test_pred.npy')}` |")
    L += ["", "## Graph and data inputs (sha256)", ""]
    files = [("graph v2 cache (hypergraph used by all runs)", "data/raw/rsr/data/hypergraph_cache/NYSE_industry-wiki.json"),
             ("RSR ticker order", "data/raw/rsr/data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"),
             ("RSR calendar", "data/raw/rsr/data/NYSE_aver_line_dates.csv"),
             ("Alpaca raw bars (all adjustments)", "data/raw/alpaca_post2017/bars.pkl.gz"),
             ("Alpaca download meta", "data/raw/alpaca_post2017/download_meta.json"),
             ("Alpaca identity-pass list (per-ticker audit table)", "docs/phase1_5b/alpaca_audit_tickers.csv"),
             ("Alpaca audit results", "docs/phase1_5b/alpaca_audit_results.json")]
    L += ["| input | path | sha256 |", "|---|---|---|"]
    for n, p in files:
        L.append(f"| {n} | `{p}` | `{sha(p)}` |")
    L += ["", "## R7: 2017 replication check (predeclared, plan R7)", "",
          "Rule: pass if the new mean HH test Sharpe over seeds 0-4 lies in [min - SD, max + SD] of the historical `R5_f2_alpha0_train/HH` seeds (SD = sample SD, ddof=1, of the historical seeds). Inference proceeds either way.", ""]
    oh, nh = sr(OLD, "HH"), sr(NEW, "HH")
    oe, ne = sr(OLD, "EH"), sr(NEW, "EH")
    sd = oh.std(ddof=1); lo, hi = oh.min() - sd, oh.max() + sd
    ok = lo <= nh.mean() <= hi
    L += [f"- Historical HH seeds: {np.round(oh,3).tolist()}; mean {oh.mean():.3f}, min {oh.min():.3f}, max {oh.max():.3f}, SD {sd:.3f}. Interval [{lo:.3f}, {hi:.3f}].",
          f"- New HH seeds: {np.round(nh,3).tolist()}; mean **{nh.mean():.3f}**.",
          f"- **R7 {'PASS' if ok else 'FAIL'}** (new HH mean {nh.mean():.3f} {'inside' if ok else 'outside'} [{lo:.3f}, {hi:.3f}]).",
          f"- EH (descriptive only): historical {np.round(oe,3).tolist()} mean {oe.mean():.3f}, SD {oe.std(ddof=1):.3f}; new {np.round(ne,3).tolist()} mean {ne.mean():.3f}, SD {ne.std(ddof=1):.3f}.",
          f"- Also descriptive: HH-EH 2017 mean difference, new {nh.mean()-ne.mean():+.3f}, historical {oh.mean()-oe.mean():+.3f}.", "",
          "Note: this is a replication of the recipe, not of the historical weights (those do not exist); GPU nondeterminism makes the runs differ.", ""]
    Path("docs/phase1_5b/FREEZE_MANIFEST.md").write_text("\n".join(L), encoding="utf-8")
    print("R7", "PASS" if ok else "FAIL", round(nh.mean(), 3), round(lo, 3), round(hi, 3), "EH", round(ne.mean(), 3), round(oe.mean(), 3))


main()
