"""Phase 1.5a forensic calculations (CPU, pure numpy). Grids are predeclared; do not edit after inspecting returns."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS_ABS = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
EPS_FRAC = (0.001, 0.01, 0.05, 0.10, 0.25)
JITTER_FRAC = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
K_GRID = (1, 5, 10, 20, 50)
K_PRIMARY = 5
COST_BPS = (0, 5, 10, 25)
MARGIN_BUCKETS = 5
DROP_DAYS = (1, 3, 5, 10, 20)
DROP_STOCKS = (1, 2, 3, 5, 10)
TOP_FREQ = (1, 5, 10, 20)
LOCAL_IC_Q = (0.05, 0.10, 0.20)
CALIB_BINS = 10
B_NULL, B_PERM, R_TIE, R_JIT, N_BOOT, BLOCK, BLOCK_SENS = 10_000, 2_000, 1_000, 200, 5_000, 10, (5, 20)
SEED_RNG = 20261003
RNG_OFFSETS = {"tie": 1, "eps": 2, "jitter": 3, "null_random": 4, "null_fixed": 5, "null_perm": 6,
               "null_matched": 7, "boot": 8, "index_perm": 9}


def rng_for(name: str, extra: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED_RNG + 1000 * RNG_OFFSETS[name] + extra)


@dataclass
class RunArrays:
    pred: np.ndarray
    gt: np.ndarray
    mask: np.ndarray
    daily: np.ndarray
    metrics: dict
    config: dict
    history: list
    path: Path


def load_run(exp: str, label: str, seed: int, root: Path | str = Path("results")) -> RunArrays:
    p = Path(root) / exp / label / f"seed_{seed}"
    pred = np.load(p / "test_pred.npy").astype(np.float64)
    gt = np.load(p / "test_gt.npy").astype(np.float64)
    mask = np.load(p / "test_mask.npy") > 0.5
    if not (np.isfinite(pred[mask]).all() and np.isfinite(gt[mask]).all()):
        raise ValueError(f"non-finite pred/gt in valid cells: {p}")
    hist = [json.loads(line) for line in open(p / "history.jsonl") if line.strip()]
    return RunArrays(pred, gt, mask, np.load(p / "test_daily.npy"), json.loads((p / "metrics.json").read_text()),
                     json.loads((p / "config.json").read_text()), hist, p)


def select(scores_d, idx, k, order="stable", rng=None):
    s = scores_d[idx]
    if order == "stable":
        o = np.argsort(-s, kind="stable")
    elif order == "reverse":
        o = np.lexsort((-np.arange(len(s)), -s))           # last key primary: score desc, then index desc
    elif order == "random":
        o = np.lexsort((rng.random(len(s)), -s))
    else:
        raise ValueError(order)
    return idx[o[:k]]


def portfolio(pred, gt, mask, k=K_PRIMARY, order="stable", rng=None):
    r, baskets = np.zeros(pred.shape[1]), []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        top = select(pred[:, d], idx, k, order, rng) if len(idx) else idx
        baskets.append(top)
        r[d] = gt[top, d].mean() if len(top) else 0.0
    return r, baskets


def hold_all(gt, mask):
    return np.array([gt[mask[:, d], d].mean() if mask[:, d].any() else 0.0 for d in range(gt.shape[1])])


def boundary_stats(pred, mask, k=K_PRIMARY):
    D = pred.shape[1]
    out = {key: np.zeros(D) for key in ("s_k", "s_k1", "margin", "sd", "ptp")}
    out["exact_tie"], out["n_valid"] = np.zeros(D, bool), np.zeros(D, int)
    for d in range(D):
        s = np.sort(pred[mask[:, d], d])[::-1]
        out["n_valid"][d] = len(s)
        if len(s) == 0:
            continue
        out["sd"][d], out["ptp"][d] = s.std(), s[0] - s[-1]
        if len(s) > k:
            out["s_k"][d], out["s_k1"][d] = s[k - 1], s[k]
            out["margin"][d] = s[k - 1] - s[k]
            out["exact_tie"][d] = s[k - 1] == s[k]
    return out


def turnover(baskets):
    to, prev = np.zeros(len(baskets)), set()
    for d, b in enumerate(baskets):
        cur = set(b.tolist())
        to[d] = 1.0 if not prev else len(cur - prev) / max(len(cur), 1)
        prev = cur
    return to


def net(r, to, bps):
    return r - 2 * bps * 1e-4 * to


def perf(r):
    r = np.asarray(r, dtype=np.float64)
    sd = r.std()
    w = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return {"mean": float(r.mean()), "vol_d": float(sd), "sr": 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(252)),
            "cumret": float(w[-1] - 1) if len(w) else 0.0, "mdd": float((w / peak - 1).min()) if len(w) else 0.0,
            "n": int(len(r))}
