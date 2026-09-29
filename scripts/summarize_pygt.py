"""Summarize results/R{1_tennis,2_chickenpox,3_windmill}/<protocol>/<label>/seed_k/metrics.json.

  python scripts/summarize_pygt.py results/R1_tennis
Prints mean +- std (ddof=1) of test_mse (val-selected epoch), the best-epoch oracle if present, n seeds, and the
baselines (seed_0/metrics.json of the `baselines` label). Also a paired-by-seed comparison of arms vs THINK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(root: Path):
    out = {}
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for adir in sorted(a for a in pdir.iterdir() if a.is_dir()):
            ms = {}
            for sd in sorted(adir.glob("seed_*")):
                f = sd / "metrics.json"
                if f.exists():
                    ms[int(sd.name.split("_")[1])] = json.loads(f.read_text())
            if ms:
                out[(pdir.name, adir.name)] = ms
    return out


def main(root):
    res = load(Path(root))
    protocols = sorted({k[0] for k in res})
    for pr in protocols:
        print(f"\n## protocol {pr}")
        for (p, arm), ms in res.items():
            if p != pr:
                continue
            if arm.startswith("baselines"):
                print("baselines:", {k: round(v, 4) for k, v in ms[0].items()})
                continue
            v = np.array([m["test_mse"] for m in ms.values()])
            o = [m["test_oracle_mse"] for m in ms.values() if "test_oracle_mse" in m]
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            extra = f"  oracle {np.mean(o):.4f}" if o else ""
            ep = np.mean([m["best_epoch"] for m in ms.values()])
            print(f"{arm:10s} n={len(v):2d}  test {v.mean():.4f} +- {sd:.4f}  (best_ep {ep:.1f}, "
                  f"hyperedges {list(ms.values())[0]['num_hyperedges']}){extra}")
        base = res.get((pr, "THINK"))
        if base:
            for (p, arm), ms in res.items():
                if p != pr or arm in ("THINK",) or arm.startswith("baselines") or arm.endswith("_raw"):
                    continue
                common = sorted(set(base) & set(ms))
                if len(common) >= 2:
                    d = np.array([ms[s]["test_mse"] - base[s]["test_mse"] for s in common])
                    print(f"  paired {arm} - THINK over {len(common)} seeds: mean {d.mean():+.4f}, "
                          f"{(d < 0).sum()}/{len(d)} seeds arm better")


if __name__ == "__main__":
    main(sys.argv[1])
