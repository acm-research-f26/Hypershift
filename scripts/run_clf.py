"""python scripts/run_clf.py  -> 25 seeds x {HH, EH, EE} on NASDAQ; macro- and micro-F1 (paper's averaging is ambiguous).

Resumable: train_clf_run skips seeds whose results/E11_clf/<label>/seed_<k>/metrics.json exists, so a crash
loses at most the seed in flight. Per-seed results are echoed as they finish; the summary is re-derived from disk.
"""
import argparse

import numpy as np

from hypershift.config import RunConfig, apply_overrides
from hypershift.experiments.grid import _geo
from hypershift.train.clf import train_clf_run

SEEDS = range(25)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E11_clf", help="output folder results/<exp> (use a new name per graph version, e.g. E11_clf_g2)")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and exit")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds (first-seed .. first-seed+n-1)")
    ap.add_argument("--first-seed", type=int, default=0, help="first seed (default 0); lets Kaggle split the 25 seeds into chunks")
    ap.add_argument("--arms", nargs="*", default=["HH", "EH", "EE"], help="subset of HH EH EE")
    ap.add_argument("--set", nargs="*", default=None, metavar="K=V", dest="overrides",
                    help="RunConfig overrides applied after the arm's config (same parser as hypershift.run --set), e.g. input_mode=relative weight_decay=0")
    a = ap.parse_args()
    ov = apply_overrides({}, a.overrides) if a.overrides else {}
    def mk(g, s):
        return RunConfig(exp=a.exp, label=g, seed=s, **{**_geo(g, market="NASDAQ", alpha=0.1), **ov})
    if a.dry_run:
        for g in a.arms:
            c = mk(g, a.first_seed)
            print(g, c.run_dir(), "attn_score", c.attn_score, "attn_dist", c.attn_dist, "temporal", c.temporal, "spatial", c.spatial,
                  "input_mode", c.input_mode, "weight_decay", c.weight_decay, "epochs", c.epochs, "seeds", a.first_seed, "..", a.first_seed + a.seeds - 1)
        raise SystemExit(0)
    for g in a.arms:
        rows = []
        for s in range(a.first_seed, a.first_seed + a.seeds):
            m = train_clf_run(mk(g, s))
            rows.append(m)
            print(f"seed {s} {g}: macro-F1 {m['test_f1']:.3f} micro-F1 {m.get('test_micro_f1', float('nan')):.3f}", flush=True)
        ma = [m["test_f1"] for m in rows]
        mi = [m.get("test_micro_f1", float("nan")) for m in rows]
        print(g, f"macro-F1 {np.mean(ma):.3f} ± {np.std(ma):.3f} | micro-F1 {np.nanmean(mi):.3f} ± {np.nanstd(mi):.3f}", flush=True)
