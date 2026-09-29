"""R2 (PyG-T tasks): train THINK arms / baselines on Hungary Chickenpox. CPU-friendly.

  python scripts/run_pygt.py --dataset chickenpox --protocol pygt|leakfree --arm THINK|EE|none|clique|baselines --seeds 0-9
Writes results/R2_chickenpox/<protocol>/<arm>/seed_k/metrics.json (skipped if present).
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
from hypershift.data.pygt import (load_chickenpox, make_windows, neighbourhood_hypergraph,  # noqa: E402
                                  pairwise_hypergraph)
from hypershift.models.think import THINK  # noqa: E402

ARMS = {  # temporal, spatial, structure, hypergraph
    "THINK": ("hyp", "hyp", "hyper", "nbhd"),
    "EE": ("euc", "euc", "hyper", "nbhd"),
    "none": ("hyp", "hyp", "none", "nbhd"),
    "clique": ("hyp", "hyp", "clique", "pair"),
    "EE_none": ("euc", "euc", "none", "nbhd"),
}


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def split_data(x: np.ndarray, protocol: str, lags: int):
    """Return dict of (inputs, targets) for train/val/test."""
    t = x.shape[1]
    w = t - lags
    if protocol == "pygt":
        # PyG-T temporal_signal_split(train_ratio=0.8) over the W windows; the last 10% of the train
        # windows are carved out as validation for early stopping. Targets as stored (full-series z-score).
        n_tr = int(0.8 * w)
        n_va = int(0.1 * n_tr)
        bounds = (0, n_tr - n_va, n_tr, w)
        xs = x
    else:
        # chronological 70/10/20 on TARGET time; z-score with train-period mean/std (global scalars).
        tr_end, va_end = int(0.7 * t), int(0.8 * t)
        mu, sd = x[:, :tr_end].mean(), x[:, :tr_end].std()
        xs = ((x - mu) / sd).astype(np.float32)
        bounds = (0, tr_end - lags, va_end - lags, w)   # window w has target time w+lags
    inp, tgt = make_windows(xs, lags)
    b = bounds
    return {"train": (inp[b[0]:b[1]], tgt[b[0]:b[1]]), "val": (inp[b[1]:b[2]], tgt[b[1]:b[2]]),
            "test": (inp[b[2]:b[3]], tgt[b[2]:b[3]])}


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def baselines(sp) -> dict:
    (xtr, ytr), (xte, yte) = sp["train"], sp["test"]
    out = {"mean": mse(ytr.mean(), yte), "persistence": mse(xte[:, :, -1, 0], yte)}
    lags = xtr.shape[2]
    a1 = np.c_[xtr[..., 0].reshape(-1, lags), np.ones(xtr.shape[0] * xtr.shape[1])]
    coef = np.linalg.lstsq(a1, ytr.reshape(-1), rcond=None)[0]
    pte = np.c_[xte[..., 0].reshape(-1, lags), np.ones(xte.shape[0] * xte.shape[1])] @ coef
    out["ar_pooled"] = mse(pte, yte.reshape(-1))
    per = []
    for n in range(xtr.shape[1]):
        c = np.linalg.lstsq(np.c_[xtr[:, n, :, 0], np.ones(len(xtr))], ytr[:, n], rcond=None)[0]
        per.append(((np.c_[xte[:, n, :, 0], np.ones(len(xte))] @ c - yte[:, n]) ** 2).mean())
    out["ar_per_node"] = float(np.mean(per))
    return out


def train_one(arm, sp, hg, seed, hidden, lags, kernel, lr, epochs, patience, bs, wd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tmp, spa, struct, _ = ARMS[arm]
    model = THINK(in_dim=1, hidden=hidden, seq=lags, kernel=kernel, temporal=tmp, spatial=spa,
                  structure=struct, out_dim=1)
    thg = hg.to_torch("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    T = {k: (torch.as_tensor(v[0]), torch.as_tensor(v[1])) for k, v in sp.items()}

    def evaluate(k):
        model.eval()
        with torch.no_grad():
            return float(((model(T[k][0], thg) - T[k][1]) ** 2).mean())

    best, best_ep, best_state, hist, bad = np.inf, -1, None, [], 0
    xtr, ytr = T["train"]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        tl = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = ((model(xtr[j], thg) - ytr[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss) * len(j)
        va, te = evaluate("val"), evaluate("test")
        hist.append({"epoch": ep, "train_mse": tl / len(xtr), "val_mse": va, "test_mse": te})
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
    return {"val_mse": best, "test_mse": evaluate("test"), "best_epoch": best_ep,
            "test_oracle_mse": min(h["test_mse"] for h in hist), "epochs_run": len(hist)}, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chickenpox")
    ap.add_argument("--protocol", choices=["pygt", "leakfree"], required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="results/R2_chickenpox")
    ap.add_argument("--data", default="data/raw/pygt/chickenpox.json")
    a = ap.parse_args()
    if a.dataset != "chickenpox":
        raise SystemExit("only chickenpox is implemented")
    torch.set_num_threads(2)
    data = load_chickenpox(a.data)
    sp = split_data(data.x, a.protocol, a.lags)
    root = Path(a.out) / a.protocol / a.arm
    if a.arm == "baselines":
        (root / "seed_0").mkdir(parents=True, exist_ok=True)
        m = baselines(sp)
        (root / "seed_0" / "metrics.json").write_text(json.dumps(m, indent=1))
        print(a.protocol, m)
        return
    hg = neighbourhood_hypergraph(data) if ARMS[a.arm][3] == "nbhd" else pairwise_hypergraph(data)
    cfg = dict(vars(a))
    for s in parse_seeds(a.seeds):
        d = root / f"seed_{s}"
        if (d / "metrics.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        m, hist = train_one(a.arm, sp, hg, s, a.hidden, a.lags, a.kernel, a.lr, a.epochs, a.patience, a.bs, a.wd)
        m["seconds"] = time.time() - t0
        m["num_hyperedges"] = len(hg.edges)
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "history.jsonl").write_text("\n".join(json.dumps(h) for h in hist))
        (d / "metrics.json").write_text(json.dumps(m, indent=1))
        print(a.protocol, a.arm, s, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
