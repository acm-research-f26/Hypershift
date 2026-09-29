"""Search feature definitions for delta_rel that match the paper (NYSE 0.087, NASDAQ 0.107). Run from repo root."""
import itertools, json, sys
import numpy as np
from scipy.spatial.distance import cdist
from hypershift.data.rsr import load_rsr
from hypershift.geometry.hyperbolicity import gromov_delta

sys.stdout.reconfigure(encoding="utf-8")
TARGET = {"NYSE": 0.087, "NASDAQ": 0.107}
R = "data/raw/rsr/data"

def delta_rel(X, n=1500, trials=3, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(trials):
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        D = cdist(X[idx], X[idx])
        vals.append(2 * gromov_delta(D, base=0) / D.max())
    return float(np.mean(vals)), float(np.max(vals))

def zs(X):
    s = X.std(0); return (X - X.mean(0)) / np.where(s > 0, s, 1)

def candidates(d):
    vi, T = d.valid_index, d.num_steps
    f, g, m = d.features, d.gt, d.mask
    out = {}
    for per, sl in [("train", slice(1, vi)), ("all", slice(1, T))]:
        out[f"returns[{per}]"] = np.where(m[:, sl] > 0, g[:, sl], 0)
        out[f"close[{per}]"] = f[:, sl, -1]
        out[f"5feat-series[{per}]"] = f[:, sl, :].reshape(len(f), -1)
        out[f"returns[{per}] z-scored"] = zs(out[f"returns[{per}]"])
        out[f"close[{per}] z-scored"] = zs(out[f"close[{per}]"])
    for per, sl in [("train", slice(0, vi)), ("all", slice(0, T))]:
        pts = f[:, sl, :][m[:, sl] > 0]                          # each (stock, day) = one 5-dim point
        out[f"(stock,day) 5-feat points[{per}]"] = pts
        out[f"(stock,day) 5-feat points[{per}] z-scored"] = zs(pts)
    for name, t0 in [("first", 0), ("last-train", vi - 16), ("last-test", T - 16)]:
        out[f"16-day window[{name}]"] = f[:, t0:t0 + 16, :].reshape(len(f), -1)
    return out

res = {}
for mkt in TARGET:
    for norm in ("paper", "train"):
        d = load_rsr(R, mkt, norm)
        for k, X in candidates(d).items():
            mean_v, max_v = delta_rel(X.astype(np.float64))
            res[(norm, k, "mean")] = res.get((norm, k, "mean"), {}) | {mkt: mean_v}
            res[(norm, k, "max")] = res.get((norm, k, "max"), {}) | {mkt: max_v}
        print(f"done {mkt} norm={norm}", flush=True)

rows = []
for (norm, k, agg), v in res.items():
    err = sum(abs(v[mk] - TARGET[mk]) / TARGET[mk] for mk in TARGET) / 2
    rows.append((err, norm, k, agg, v["NYSE"], v["NASDAQ"]))
rows.sort()
print(f"\n{'rel.err':>7} | norm  | feature definition | agg | NYSE (0.087) | NASDAQ (0.107)")
for r in rows:
    print(f"{r[0]:7.2f} | {r[1]:5s} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f}")
json.dump([dict(zip(["rel_err","norm","feature","agg","NYSE","NASDAQ"], r)) for r in rows], open("results/delta_rel_search.json", "w"), indent=1)
