"""Analyse the known-signal grid (scripts/known_signal.py) locally on CPU, single thread.

Reads results/known_signal_<level>_<mode>/<arm>/seed_<k>/{known_signal.json,test_pred.npy,test_gt.npy,test_mask.npy},
regenerates each seed's planted market (deterministic, same code as the Kaggle run) to compute references:
  oracle      = phi*r + gamma*(A r)                      (best possible with the graph)
  no-graph    = (phi + gamma*diag(A)) * r                (best possible for a model that sees only its own window)
  graph-only  = gamma*(A - diag(A)) r                    (the part only hyperedge co-members reveal)
and a pooled ridge probe on the real window inputs (level / relative). Prints markdown tables.

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/known_signal_analysis.py [--exp known_signal]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import known_signal as ks  # noqa: E402
from hypershift.eval.metrics import sharpe, topk_daily_returns  # noqa: E402
from hypershift.train.loop import apply_input_mode, gather_batch, window_offsets  # noqa: E402


def ic_of(pred, gt, mask):
    return ks.daily_ic(pred, gt, mask)


def excess_sharpe(pred, gt, mask, k=5):
    top = topk_daily_returns(pred, gt, mask, k)
    hold = (gt * mask).sum(0) / np.maximum(mask.sum(0), 1)
    return sharpe(top - hold)


def references(syn, A, phi, gamma):
    te = slice(syn.test_index, syn.num_steps)
    prev = np.zeros_like(syn.gt)
    prev[:, 1:] = syn.gt[:, :-1]
    d = np.diag(A).astype(np.float64)
    preds = {"oracle": phi * prev + gamma * (A @ prev),
             "nograph": (phi + gamma * d)[:, None] * prev,
             "graphonly": gamma * ((A - np.diag(d)) @ prev),
             "lag1": prev}
    gt, m = syn.gt[:, te], syn.mask[:, te]
    return {k: {"ic": ic_of(p[:, te], gt, m), "xsr": excess_sharpe(p[:, te], gt, m)} for k, p in preds.items()}


def probe(syn, mode, seq=16, ridge=1.0, max_rows=150_000, seed=0):
    """Pooled ridge regression gt ~ flattened window (after the pipeline's own window construction + input mode)."""
    rng = np.random.default_rng(seed)

    def design(split):
        offs = window_offsets(syn, seq, split)
        X, Y, M = [], [], []
        for i in range(0, len(offs), 64):
            x, m, _, g = gather_batch(syn, offs[i:i + 64], seq)
            x = apply_input_mode(x, mode)
            X.append(x.reshape(x.shape[0], x.shape[1], -1)); Y.append(g); M.append(m)
        return np.concatenate(X), np.concatenate(Y), np.concatenate(M)

    Xtr, Ytr, Mtr = design("train")
    Xte, Yte, Mte = design("test")
    xf, yf = Xtr[Mtr > 0.5], Ytr[Mtr > 0.5]
    if len(xf) > max_rows:
        sel = rng.choice(len(xf), max_rows, replace=False)
        xf, yf = xf[sel], yf[sel]
    mu, sd = xf.mean(0), xf.std(0) + 1e-8
    Z = (xf - mu) / sd
    w = np.linalg.solve(Z.T @ Z + ridge * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ (yf - yf.mean()))
    pred = (((Xte - mu) / sd) @ w)                      # [days, N]
    return ic_of(pred.T, Yte.T, Mte.T), excess_sharpe(pred.T, Yte.T, Mte.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="known_signal")
    ap.add_argument("--root", default="results")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    real, hg = ks._universe()
    A0 = ks.neighbour_operator(hg)
    iso = int((hg.node_degree() == 0).sum())
    print(f"universe: {real.num_nodes} stocks, {len(hg.edges)} hyperedges, isolated stocks {iso}, "
          f"mean diag(A) {np.diag(A0).mean():.3f} (min {np.diag(A0).min():.3f}, max {np.diag(A0).max():.3f})\n")
    runs = defaultdict(dict)       # (level, mode) -> arm -> seed -> dict
    for p in sorted(Path(a.root).glob(f"{a.exp}_*/*/seed_*/known_signal.json")):
        r = json.loads(p.read_text())
        d = p.parent
        tp, tg, tm = (np.load(d / f"test_{k}.npy") for k in ("pred", "gt", "mask"))
        r["xsr"] = excess_sharpe(tp, tg, tm)
        r["test_ic"] = ic_of(tp, tg, tm)          # recomputed in float64 (the Kaggle json value can be NaN for near-constant predictions)
        r["n_ties"] = float(np.mean([len(np.unique(tp[:, d][tm[:, d] > 0.5])) < 0.9 * (tm[:, d] > 0.5).sum() for d in range(tp.shape[1])]))
        runs[(r["level"], r["mode"])].setdefault(r["arm"], {})[r["seed"]] = r
    refs, probes = {}, {}
    for (lv, mo), arms in sorted(runs.items()):
        phi, gamma, sigma = ks.LEVELS[lv]
        for s in sorted({s for v in arms.values() for s in v}):
            if (lv, s) not in refs:
                syn, A = ks.plant_signal(real, hg, phi, gamma, sigma, s)
                refs[(lv, s)] = references(syn, A, phi, gamma)
                if not a.no_probe:
                    for m in ("level", "relative"):
                        probes[(lv, m, s)] = probe(syn, m)
    out = {"runs": {f"{k[0]}|{k[1]}": {arm: {str(s): {x: v[x] for x in ("test_ic", "xsr", "test_sr", "val_sr", "pred_std", "best_epoch", "mse_ratio")}
                                          for s, v in d.items()} for arm, d in arms.items()} for k, arms in runs.items()},
           "refs": {f"{k[0]}|{k[1]}": v for k, v in refs.items()},
           "probes": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in probes.items()}}
    Path(a.root, f"{a.exp}_analysis.json").write_text(json.dumps(out, indent=1))

    def ms(x):
        x = np.asarray(x, float)
        return f"{x.mean():+.3f} +- {x.std(ddof=1) / np.sqrt(len(x)):.3f}" if len(x) > 1 else f"{x.mean():+.3f}"

    print("### Table A: references and probes (mean over seeds)\n")
    print("| level | oracle IC | no-graph IC | graph-only IC | lag-1 IC | oracle excess SR | no-graph excess SR | ridge probe IC level / relative |")
    print("|---|---|---|---|---|---|---|---|")
    for lv in [l for l in ks.LEVELS if any(k[0] == l for k in refs)]:
        ss = sorted(s for (l, s) in refs if l == lv)
        f = lambda k, m: np.mean([refs[(lv, s)][k][m] for s in ss])  # noqa: E731
        pr = ""
        if probes:
            pr = " / ".join(f"{np.mean([probes[(lv, m, s)][0] for s in ss]):+.3f}" for m in ("level", "relative"))
        print(f"| {lv} | {f('oracle','ic'):.3f} | {f('nograph','ic'):.3f} | {f('graphonly','ic'):.3f} | {f('lag1','ic'):.3f} | "
              f"{f('oracle','xsr'):.1f} | {f('nograph','xsr'):.1f} | {pr} |")
    print("\n### Table B: model test IC at the validation-selected epoch (mean +- s.e. over seeds), fraction of oracle IC, excess SR\n")
    print("| level | input | arm | n | test IC | IC / oracle IC | IC / no-graph IC | excess Sharpe (top5 - hold) | pred sd |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (lv, mo), arms in sorted(runs.items()):
        for arm, d in sorted(arms.items()):
            ss = sorted(d)
            ic = [d[s]["test_ic"] for s in ss]
            o = np.mean([refs[(lv, s)]["oracle"]["ic"] for s in ss])
            n = np.mean([refs[(lv, s)]["nograph"]["ic"] for s in ss])
            frac = lambda ref: f"{np.mean(ic) / ref:.2f}" if abs(ref) > 0.02 else "n/a"  # noqa: E731
            print(f"| {lv} | {mo} | {arm} | {len(ss)} | {ms(ic)} | {frac(o)} | {frac(n)} | {ms([d[s]['xsr'] for s in ss])} | "
                  f"{np.mean([d[s]['pred_std'] for s in ss]):.2e} |")
    print("\n### Table C: graph arm minus its no-graph counterpart, paired by seed (test IC)\n")
    print("| level | input | pair | n | mean IC diff | s.e. | seeds with diff > 0 | graph-only IC (reference gap) |")
    print("|---|---|---|---|---|---|---|---|")
    pairs = [("HH_hyper", "HH_none"), ("EE_hyper", "EE_none"), ("EH_hyper", "EE_none")]
    for (lv, mo), arms in sorted(runs.items()):
        for g, n in pairs:
            if g in arms and n in arms:
                ss = sorted(set(arms[g]) & set(arms[n]))
                if not ss:
                    continue
                df = np.array([arms[g][s]["test_ic"] - arms[n][s]["test_ic"] for s in ss])
                gap = np.mean([refs[(lv, s)]["oracle"]["ic"] - refs[(lv, s)]["nograph"]["ic"] for s in ss])
                se = df.std(ddof=1) / np.sqrt(len(df)) if len(df) > 1 else float("nan")
                print(f"| {lv} | {mo} | {g} - {n} | {len(ss)} | {df.mean():+.4f} | {se:.4f} | {(df > 0).sum()}/{len(df)} | {gap:+.4f} |")


if __name__ == "__main__":
    main()
