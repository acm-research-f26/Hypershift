"""Paired-by-seed THINK vs EH (paper's TCONV+DHHAN) on R1/R2/R3: mean diff (EH - THINK), Wilcoxon p, seed counts.

  python scripts/compare_eh.py [results/R1_tennis results/R2_chickenpox results/R3_windmill]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def mse(root: Path, proto: str, arm: str) -> dict[int, float]:
    return {int(d.name.split("_")[1]): json.loads((d / "metrics.json").read_text())["test_mse"]
            for d in (root / proto / arm).glob("seed_*") if (d / "metrics.json").exists()}


def main(roots):
    for r in roots:
        root = Path(r)
        for proto in ("pygt", "leakfree"):
            a, b = mse(root, proto, "THINK"), mse(root, proto, "EH")
            s = sorted(set(a) & set(b))
            if not s:
                continue
            t, e = np.array([a[i] for i in s]), np.array([b[i] for i in s])
            d = e - t
            p = wilcoxon(e, t).pvalue if len(s) > 1 and np.any(d != 0) else float("nan")
            print(f"{root.name:14s} {proto:8s} n={len(s):2d} THINK {t.mean():.4f}+-{t.std(ddof=1):.4f} "
                  f"EH {e.mean():.4f}+-{e.std(ddof=1):.4f} diff(EH-THINK) {d.mean():+.4f} wilcoxon p={p:.4f} "
                  f"EH better {(d < 0).sum()}/{len(s)}, THINK better {(d > 0).sum()}/{len(s)}, ties {(d == 0).sum()}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/R1_tennis", "results/R2_chickenpox", "results/R3_windmill"])
