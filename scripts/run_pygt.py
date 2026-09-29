"""R1/R2/R3 (PyG-T tasks): train THINK arms / baselines on tennis, chickenpox, windmill. CPU-friendly.

  python scripts/run_pygt.py --dataset chickenpox|tennis|windmill --protocol pygt|leakfree       --arm THINK|EE|EH|none|clique|baselines --seeds 0-9 [--hg topk|quantile] [--label NAME]
Writes results/R{1_tennis,2_chickenpox,3_windmill}/<protocol>/<label>/seed_k/metrics.json (skipped if present).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hypershift.data.pygt import (load_chickenpox, load_tennis, load_windmill, make_windows,  # noqa: E402
                                  make_windows_tennis, neighbourhood_hypergraph, pairwise_hypergraph,
                                  quantile_hypergraph, tennis_neighbourhood, tennis_pairs, topk_hypergraph,
                                  topk_pairs)
from hypershift.models.think import THINK  # noqa: E402

ARMS = {  # temporal, spatial, structure, hypergraph
    "THINK": ("hyp", "hyp", "hyper", "nbhd"),
    "EE": ("euc", "euc", "hyper", "nbhd"),
    "EH": ("euc", "hyp", "hyper", "nbhd"),   # paper's TCONV+DHHAN: Euclidean temporal conv + hyperbolic hypergraph attention
    "none": ("hyp", "hyp", "none", "nbhd"),
    "clique": ("hyp", "hyp", "clique", "pair"),
    "EE_none": ("euc", "euc", "none", "nbhd"),
}


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def split_data(x: np.ndarray, protocol: str, lags: int, per_node: bool = False):
    """Return dict of (inputs, targets) for train/val/test. x [N,T] (chickenpox/windmill).

    pygt: stored series as-is (chickenpox) or the PyG-T per-node full-series z-score (windmill, per_node=True).
    leakfree: chronological 70/10/20 on target time; z-score from the train period only (global scalars for
    chickenpox, per node for windmill).
    """
    t = x.shape[1]
    w = t - lags
    if protocol == "pygt":
        # PyG-T temporal_signal_split(train_ratio=0.8) over the W windows; the last 10% of the train
        # windows are carved out as validation for early stopping.
        n_tr = int(0.8 * w)
        n_va = int(0.1 * n_tr)
        bounds = (0, n_tr - n_va, n_tr, w)
        xs = x
        if per_node:
            xs = ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-10)).astype(np.float32)
    else:
        tr_end, va_end = int(0.7 * t), int(0.8 * t)
        if per_node:
            mu, sd = x[:, :tr_end].mean(1, keepdims=True), x[:, :tr_end].std(1, keepdims=True) + 1e-10
        else:
            mu, sd = x[:, :tr_end].mean(), x[:, :tr_end].std()
        xs = ((x - mu) / sd).astype(np.float32)
        bounds = (0, tr_end - lags, va_end - lags, w)   # window w has target time w+lags
    inp, tgt = make_windows(xs, lags)
    b = bounds
    return {"train": (inp[b[0]:b[1]], tgt[b[0]:b[1]]), "val": (inp[b[1]:b[2]], tgt[b[1]:b[2]]),
            "test": (inp[b[2]:b[3]], tgt[b[2]:b[3]])}


def split_tennis(data, protocol: str, lags: int):
    """Tennis: window w = snapshots w..w+lags-1 (target time = its last snapshot e=w+lags-1). No scaling in either
    protocol (one-hot features, log1p target are pointwise), so protocols differ only in the split."""
    inp, tgt, hist = make_windows_tennis(data, lags)
    w, t = len(inp), data.x.shape[1]
    if protocol == "pygt":
        n_tr = int(0.8 * w)
        n_va = int(0.1 * n_tr)
        b = (0, n_tr - n_va, n_tr, w)
    else:
        tr_end, va_end = int(0.7 * t), int(0.8 * t)
        b = (0, tr_end - lags + 1, va_end - lags + 1, w)   # train windows have last snapshot < tr_end
    sp = {k: (inp[b[i]:b[i + 1]], tgt[b[i]:b[i + 1]]) for i, k in enumerate(("train", "val", "test"))}
    sp["hist"] = {k: hist[b[i]:b[i + 1]] for i, k in enumerate(("train", "val", "test"))}
    # snapshots [0, end) are inside the training windows (pygt: val carve-out counts as train; leakfree: strict train)
    sp["train_end_time"] = (b[2] if protocol == "pygt" else b[1]) + lags - 1
    return sp


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def baselines(sp) -> dict:
    """mean / persistence / AR(lags) on the lagged series. For tennis the lagged series are the loader targets of
    earlier snapshots (labels), which the neural models do NOT see, so these baselines get privileged inputs."""
    (xtr, ytr), (xte, yte) = sp["train"], sp["test"]
    if "hist" in sp:
        htr, hte = sp["hist"]["train"], sp["hist"]["test"]
        ok = ~np.isnan(htr).any(axis=(1, 2))
        htr, ytr = htr[ok], ytr[ok]
    else:
        htr, hte = xtr[..., 0], xte[..., 0]
    out = {"mean": mse(ytr.mean(), yte), "persistence": mse(hte[:, :, -1], yte)}
    if "hist" in sp:
        out["per_node_mean"] = mse(ytr.mean(0, keepdims=True), yte)
    lags = htr.shape[2]
    a1 = np.c_[htr.reshape(-1, lags), np.ones(htr.shape[0] * htr.shape[1])]
    coef = np.linalg.lstsq(a1, ytr.reshape(-1), rcond=None)[0]
    pte = np.c_[hte.reshape(-1, lags), np.ones(hte.shape[0] * hte.shape[1])] @ coef
    out["ar_pooled"] = mse(pte, yte.reshape(-1))
    per = []
    for n in range(htr.shape[1]):
        c = np.linalg.lstsq(np.c_[htr[:, n, :], np.ones(len(htr))], ytr[:, n], rcond=None)[0]
        per.append(((np.c_[hte[:, n, :], np.ones(len(hte))] @ c - yte[:, n]) ** 2).mean())
    out["ar_per_node"] = float(np.mean(per))
    return out


def train_one(arm, sp, hg, seed, hidden, lags, kernel, lr, epochs, patience, bs, wd, in_dim=1, max_train=0,
              oracle=True, eval_bs=256, val_stride=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tmp, spa, struct, _ = ARMS[arm]
    model = THINK(in_dim=in_dim, hidden=hidden, seq=lags, kernel=kernel, temporal=tmp, spatial=spa,
                  structure=struct, out_dim=1)
    thg = hg.to_torch("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    T = {k: (torch.as_tensor(sp[k][0]), torch.as_tensor(sp[k][1])) for k in ("train", "val", "test")}
    if val_stride > 1:   # epoch selection on every val_stride-th validation window (cost cap)
        T["val"] = (T["val"][0][::val_stride], T["val"][1][::val_stride])

    def evaluate(k):
        model.eval()
        x, y = T[k]
        se = 0.0
        with torch.no_grad():
            for i in range(0, len(x), eval_bs):
                se += float(((model(x[i:i + eval_bs], thg) - y[i:i + eval_bs]) ** 2).sum())
        return se / y.numel()

    best, best_ep, best_state, hist, bad = np.inf, -1, None, [], 0
    xtr, ytr = T["train"]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        if max_train and max_train < len(perm):
            perm = perm[:max_train]
        tl = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = ((model(xtr[j], thg) - ytr[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss) * len(j)
        va = evaluate("val")
        te = evaluate("test") if oracle else float("nan")
        hist.append({"epoch": ep, "train_mse": tl / len(perm), "val_mse": va, "test_mse": te})
        if not np.isfinite(va):
            break
        if va < best - 1e-6:
            best, best_ep, bad = va, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    out = {"val_mse": best, "test_mse": evaluate("test"), "best_epoch": best_ep, "epochs_run": len(hist)}
    if oracle:
        out["test_oracle_mse"] = min(h["test_mse"] for h in hist)
    return out, hist


DATASETS = {  # out dir, default data path
    "chickenpox": ("results/R2_chickenpox", "data/raw/pygt/chickenpox.json"),
    "tennis": ("results/R1_tennis", "data/raw/pygt/twitter_tennis_rg17.json"),
    "windmill": ("results/R3_windmill", "data/raw/pygt/windmill_output.json"),
}


def build(dataset, args):
    """-> (data, split dict, hypergraph builder(kind) -> Hypergraph, in_dim)."""
    if dataset == "chickenpox":
        d = load_chickenpox(args.data)
        sp = split_data(d.x, args.protocol, args.lags)
        return d, sp, (lambda kind: neighbourhood_hypergraph(d) if kind == "nbhd" else pairwise_hypergraph(d)), 1
    if dataset == "windmill":
        d = load_windmill(args.data)
        sp = split_data(d.x, args.protocol, args.lags, per_node=True)

        def hg(kind):
            if kind == "pair":
                return topk_pairs(d, args.k)
            return topk_hypergraph(d, args.k) if args.hg == "topk" else quantile_hypergraph(d, args.top_frac)
        return d, sp, hg, 1
    d = load_tennis(args.data, args.feature_mode, args.target)
    sp = split_tennis(d, args.protocol, args.lags)
    end = sp["train_end_time"]
    return d, sp, (lambda kind: tennis_neighbourhood(d, end) if kind == "nbhd" else tennis_pairs(d, end)), d.x.shape[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chickenpox", choices=list(DATASETS))
    ap.add_argument("--protocol", choices=["pygt", "leakfree"], required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", default=None, help="results sub-folder name (default = arm)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--lags", type=int, default=None, help="default 4 (chickenpox, tennis), 8 (windmill: PyG-T default)")
    ap.add_argument("--kernel", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=0, help="cap on training windows sampled per epoch (0 = all)")
    ap.add_argument("--val_stride", type=int, default=1, help="use every k-th validation window for selection")
    ap.add_argument("--oracle", type=int, default=1, help="evaluate test every epoch (diagnostic oracle MSE)")
    ap.add_argument("--hg", choices=["topk", "quantile"], default="topk", help="windmill hyperedges")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top_frac", type=float, default=0.10)
    ap.add_argument("--feature_mode", default="encoded", help="tennis: encoded | raw")
    ap.add_argument("--target", default="log1p", choices=["log1p", "raw"], help="tennis target")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    out_dir, data_path = DATASETS[a.dataset]
    a.out = a.out or out_dir
    a.data = a.data or data_path
    a.lags = a.lags or (8 if a.dataset == "windmill" else 4)
    torch.set_num_threads(2)
    data, sp, hg_of, in_dim = build(a.dataset, a)
    root = Path(a.out) / a.protocol / (a.label or a.arm)
    if a.arm == "baselines":
        (root / "seed_0").mkdir(parents=True, exist_ok=True)
        m = baselines(sp)
        (root / "seed_0" / "metrics.json").write_text(json.dumps(m, indent=1))
        print(a.protocol, m)
        return
    hg = hg_of(ARMS[a.arm][3])
    cfg = dict(vars(a))
    for s in parse_seeds(a.seeds):
        d = root / f"seed_{s}"
        if (d / "metrics.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        m, hist = train_one(a.arm, sp, hg, s, a.hidden, a.lags, a.kernel, a.lr, a.epochs, a.patience, a.bs, a.wd,
                            in_dim=in_dim, max_train=a.max_train, oracle=bool(a.oracle), val_stride=a.val_stride)
        m["seconds"] = time.time() - t0
        m["num_hyperedges"] = len(hg.edges)
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "history.jsonl").write_text("\n".join(json.dumps(h) for h in hist))
        (d / "metrics.json").write_text(json.dumps(m, indent=1))
        print(a.protocol, a.arm, s, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
