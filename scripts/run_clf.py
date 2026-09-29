"""python scripts/run_clf.py  -> 25 seeds x {HH, EH, EE} on NASDAQ; macro- and micro-F1 (paper's averaging is ambiguous).

Resumable: train_clf_run skips seeds whose results/E11_clf/<label>/seed_<k>/metrics.json exists, so a crash
loses at most the seed in flight. Per-seed results are echoed as they finish; the summary is re-derived from disk.
"""
import argparse

import numpy as np

from hypershift.config import RunConfig
from hypershift.experiments.grid import _geo
from hypershift.train.clf import train_clf_run

SEEDS = range(25)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E11_clf", help="output folder results/<exp> (use a new name per graph version, e.g. E11_clf_g2)")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and exit")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds (0..n-1)")
    a = ap.parse_args()
    if a.dry_run:
        for g in ("HH", "EH", "EE"):
            c = RunConfig(exp=a.exp, label=g, seed=0, **_geo(g, market="NASDAQ", alpha=0.1))
            print(g, c.run_dir(), "attn_score", c.attn_score, "attn_dist", c.attn_dist, "temporal", c.temporal, "spatial", c.spatial,
                  "seeds", a.seeds)
        raise SystemExit(0)
    for g in ("HH", "EH", "EE"):
        rows = []
        for s in range(a.seeds):
            m = train_clf_run(RunConfig(exp=a.exp, label=g, seed=s, **_geo(g, market="NASDAQ", alpha=0.1)))
            rows.append(m)
            print(f"seed {s} {g}: macro-F1 {m['test_f1']:.3f} micro-F1 {m.get('test_micro_f1', float('nan')):.3f}", flush=True)
        ma = [m["test_f1"] for m in rows]
        mi = [m.get("test_micro_f1", float("nan")) for m in rows]
        print(g, f"macro-F1 {np.mean(ma):.3f} ± {np.std(ma):.3f} | micro-F1 {np.nanmean(mi):.3f} ± {np.nanstd(mi):.3f}", flush=True)
