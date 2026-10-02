"""Known-signal learning check (Phase 1.5, task D1).

Plants a learnable signal in a synthetic market and runs the FULL pipeline (train_one_run: windows -> model ->
rank_mse_loss -> evaluator) to see whether predictions recover it.

Data-generating process (daily returns, all stocks):
    r[n,t] = phi * r[n,t-1] + gamma * nbr[n,t-1] + eps[n,t],   eps ~ N(0, sigma^2)
    nbr[n,t] = (A r[:,t])[n], A = Dv^-1 H De^-1 H^T the hypergraph random-walk operator (mean of hyperedge means),
i.e. an own-lag signal (visible in the price window of stock n) plus a group effect (visible only through the
hyperedge co-members' windows). Prices are cumprod(1 + r); features = (ma5, ma10, ma20, ma30, close) normalised by the
per-stock max over the training period, the real mask / dates / splits / graph are kept.

Reference predictors (the best any model could do): oracle = phi*r + gamma*nbr, own-only = phi*r.

Usage (CPU):
  CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/known_signal.py --levels mid --arms HH_hyper HH_none \
      --modes level relative --seeds 0 1 --epochs 15 --jobs 6
  ... --summarize          # -> prints a markdown table from results/known_signal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if "--device" not in sys.argv or sys.argv[sys.argv.index("--device") + 1] != "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # CPU unless --device cuda is requested

from hypershift.data.hypergraph import Hypergraph
from hypershift.data.rsr import FILL, MarketData
from hypershift.eval.metrics import daily_ic, evaluate_all, sharpe, topk_daily_returns  # noqa: F401

# (phi, gamma, sigma): own-lag coefficient, group-lag coefficient, daily noise sd
LEVELS = {
    "none": (0.0, 0.0, 0.01),       # no signal: null control
    "own": (0.15, 0.0, 0.01),       # own-lag only: graph should not help
    "group": (0.0, 0.50, 0.01),     # group effect dominates: graph arms should beat none
    "low": (0.05, 0.10, 0.01),
    "mid": (0.10, 0.20, 0.01),
    "high": (0.20, 0.40, 0.01),
}
ARMS = {
    "HH_hyper": dict(temporal="hyp", spatial="hyp", structure="hyper"),
    "EH_hyper": dict(temporal="euc", spatial="hyp", structure="hyper"),
    "EE_hyper": dict(temporal="euc", spatial="euc", structure="hyper"),
    "HH_none": dict(temporal="hyp", spatial="hyp", structure="none"),
    "EE_none": dict(temporal="euc", spatial="euc", structure="none"),
}


def neighbour_operator(hg: Hypergraph) -> np.ndarray:
    """Row-stochastic [N,N] hypergraph random-walk operator A = Dv^-1 H De^-1 H^T: a stock's neighbour mean is the
    average, over its hyperedges, of each hyperedge's mean (small hyperedges give a cross-sectionally informative group
    signal; a union-of-co-members mean collapses to the market factor on the large hyperedges). Isolated stocks keep
    only themselves."""
    N = hg.num_nodes
    H = np.zeros((N, len(hg.edges)), dtype=np.float64)
    for k, e in enumerate(hg.edges):
        H[list(e), k] = 1.0
    deg = H.sum(1)
    A = (H / np.maximum(deg, 1)[:, None]) @ (H / H.sum(0)[None, :]).T
    iso = deg == 0
    A[iso, iso] = 1.0
    return A.astype(np.float32)


def plant_signal(real: MarketData, hg: Hypergraph, phi: float, gamma: float, sigma: float, seed: int):
    """Return (synthetic MarketData with the real mask/dates/splits, neighbour operator, return panel)."""
    rng = np.random.default_rng(10_000 + seed)
    N, T = real.num_nodes, real.num_steps
    A = neighbour_operator(hg)
    r = np.zeros((N, T), dtype=np.float64)
    r[:, 0] = rng.normal(0, sigma, N)
    for t in range(1, T):
        r[:, t] = phi * r[:, t - 1] + gamma * (A @ r[:, t - 1]) + rng.normal(0, sigma, N)
    close = np.cumprod(1.0 + r, axis=1)
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(close, axis=1)], axis=1)

    def ma(w):
        out = np.empty_like(close)
        for t in range(T):
            lo = max(0, t - w + 1)
            out[:, t] = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)
        return out

    feats = np.stack([ma(5), ma(10), ma(20), ma(30), close], axis=2)
    scale = close[:, : real.valid_index].max(axis=1)           # norm "train": train-period max
    feats = feats / scale[:, None, None]
    base = close / scale[:, None]
    ok = real.mask > 0
    gt = np.zeros((N, T), dtype=np.float64)
    gt[:, 1:] = (base[:, 1:] - base[:, :-1]) / base[:, :-1]
    both = ok[:, 1:] & ok[:, :-1]                               # same convention as parse_eod
    gt[:, 1:] = np.where(both, gt[:, 1:], 0.0)
    feats[~ok] = FILL
    base[~ok] = FILL
    syn = replace(real, features=feats.astype(np.float32), mask=real.mask.copy(), gt=gt.astype(np.float32),
                  base_price=base.astype(np.float32))
    return syn, A


def reference_metrics(syn: MarketData, A: np.ndarray, phi: float, gamma: float, topk: int = 5) -> dict:
    """Metrics of the oracle / own-only predictors and of buy-and-hold on the test split."""
    te = slice(syn.test_index, syn.num_steps)
    prev = np.zeros_like(syn.gt)
    prev[:, 1:] = syn.gt[:, :-1]
    own = phi * prev
    orc = phi * prev + gamma * (A @ prev)
    gt, m = syn.gt[:, te], syn.mask[:, te]
    out = {}
    for name, p in (("oracle", orc), ("own_only", own)):
        p = p[:, te]
        out[name] = {"sr": evaluate_all(p, gt, m)["sr"], "ic": daily_ic(p, gt, m)}
    hold = (gt * m).sum(0) / np.maximum(m.sum(0), 1)
    out["hold"] = {"sr": sharpe(hold)}
    return out


def make_cfg(exp, label, seed, arm, mode, epochs, device="cpu", out_root="results", **kw):
    from hypershift.config import RunConfig
    base = dict(exp=exp, label=label, seed=seed, market="NYSE", batch_days=8, epochs=epochs, patience=10,
                input_mode=mode, device=device, out_root=out_root, **ARMS[arm])
    return RunConfig(**{**base, **kw})


def run_cell(real, hg, level, arm, mode, seed, epochs, exp="known_signal", out_root="results", device="cpu", **kw):
    """Train one (level, arm, mode, seed) run; return its metrics dict."""
    from hypershift.train.loop import train_one_run
    phi, gamma, sigma = LEVELS[level]
    syn, A = plant_signal(real, hg, phi, gamma, sigma, seed)
    cfg = make_cfg(f"{exp}_{level}_{mode}", arm, seed, arm, mode, epochs, device, out_root, **kw)
    m = train_one_run(cfg, syn, hg)
    d = cfg.run_dir()
    tp, tg, tm = (np.load(d / f"test_{k}.npy") for k in ("pred", "gt", "mask"))
    res = {"level": level, "arm": arm, "mode": mode, "seed": seed, "best_epoch": m["best_epoch"],
           "epochs_run": m["epochs_run"], "val_sr": m["val"]["sr"], "test_sr": m["test"]["sr"],
           "test_oracle_sr": m["test_oracle_sr"], "test_ic": daily_ic(tp, tg, tm),
           "pred_std": float(tp[tm > 0.5].std()), "mse_ratio": m["test"]["mse"] / float((tg[tm > 0.5] ** 2).mean())}
    res["ref"] = reference_metrics(syn, A, phi, gamma)
    (d / "known_signal.json").write_text(json.dumps(res, indent=2))
    return res


def _worker(args):
    import torch
    torch.set_num_threads(int(os.environ.get("KS_THREADS", "2")))
    level, arm, mode, seed, epochs, exp, device, kw = args
    if device == "cuda":
        import torch
        assert torch.cuda.is_available(), "--device cuda but no GPU"
    from hypershift.train.loop import load_market  # noqa: F401
    real, hg = _universe()
    return run_cell(real, hg, level, arm, mode, seed, epochs, exp, device=device, **kw)


def _universe():
    sys.path.insert(0, str(Path(__file__).parent))
    from poc_sectors import universe       # the small-scale universe: 309 stocks, 558-edge g2 graph, real dates
    return universe()


def summarize(exp="known_signal", root="results"):
    rows = [json.loads(p.read_text()) for p in sorted(Path(root).glob(f"{exp}_*/*/seed_*/known_signal.json"))]
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["level"], r["mode"], r["arm"])].append(r)
    print("| level | input | arm | n | test IC | test Sharpe | oracle-pred Sharpe / IC | own-only Sharpe / IC | hold Sharpe | val Sharpe | MSE/zero-MSE |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (lv, mo, arm), rs in sorted(g.items()):
        f = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        ref = lambda a, k: np.mean([r["ref"][a][k] for r in rs])  # noqa: E731
        print(f"| {lv} | {mo} | {arm} | {len(rs)} | {f('test_ic'):+.4f} | {f('test_sr'):+.2f} | "
              f"{ref('oracle','sr'):.2f} / {ref('oracle','ic'):.3f} | {ref('own_only','sr'):.2f} / {ref('own_only','ic'):.3f} | "
              f"{ref('hold','sr'):.2f} | {f('val_sr'):+.2f} | {f('mse_ratio'):.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["mid"], choices=list(LEVELS))
    ap.add_argument("--arms", nargs="+", default=["HH_hyper", "HH_none"], choices=list(ARMS))
    ap.add_argument("--modes", nargs="+", default=["level"], choices=["level", "relative"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--exp", default="known_signal")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="extra RunConfig overrides k=v")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.exp)
        sys.exit(0)
    from hypershift.config import apply_overrides
    kw = apply_overrides({}, a.set)
    jobs = [(lv, arm, mo, s, a.epochs, a.exp, a.device, kw)
            for lv in a.levels for mo in a.modes for arm in a.arms for s in a.seeds]
    if a.dry_run:
        for j in jobs:
            print("DRY", j[:6])
        sys.exit(0)
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(_worker, jobs):
                print(r["level"], r["mode"], r["arm"], r["seed"], f"IC {r['test_ic']:+.4f} SR {r['test_sr']:+.2f}", flush=True)
    else:
        for j in jobs:
            r = _worker(j)
            print(r["level"], r["mode"], r["arm"], r["seed"], f"IC {r['test_ic']:+.4f} SR {r['test_sr']:+.2f}", flush=True)
