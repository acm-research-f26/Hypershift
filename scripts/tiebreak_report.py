"""Random tie-break robustness column (Phase 1.5 A follow-up). CPU only; reads saved test_{pred,gt,mask}.npy.
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/tiebreak_report.py R5_g2 known_signal_high_level f_wd0_high_relative [--draws 20]
For every results/<exp>/<label>/seed_*: default (stable, lowest-index tie-break) vs random tie-break Sharpe, excess Sharpe
(top-k minus hold) and NDCG@5, plus the share of days with an exact tie at the top-k boundary. The default evaluator is unchanged."""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, "src")
from hypershift.eval.metrics import evaluate_all, evaluate_random_ties, sharpe, topk_daily_returns


def boundary_tie_share(p, m, k=5):
    n = tot = 0
    for d in range(p.shape[1]):
        v = np.sort(p[m[:, d] > 0.5, d])[::-1]
        if len(v) > k:
            tot += 1
            n += int(v[k - 1] == v[k])
    return n / max(tot, 1)


def one(d, draws):
    p, g, m = (np.load(f"{d}/test_{n}.npy").astype(np.float64) if n == "pred" else np.load(f"{d}/test_{n}.npy") for n in ("pred", "gt", "mask"))
    base = evaluate_all(p, g, m)
    hold = np.array([g[m[:, t] > 0.5, t].mean() for t in range(g.shape[1])])
    rt = evaluate_random_ties(p, g, m, draws=draws)
    return dict(sr=base["sr"], sr_rt=rt["sr_rt"], xsr=sharpe(topk_daily_returns(p, g, m) - hold), xsr_rt=rt["xsr_rt"],
                nd=base["ndcg5"], nd_rt=rt["ndcg5_rt"], ties=boundary_tie_share(p, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    print("| exp | label | n | tie days | SR default | SR random | delta | max abs seed delta | xSR default | xSR random | NDCG@5 default | NDCG@5 random |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for exp in a.exps:
        for ld in sorted(glob.glob(f"results/{exp}/*/")):
            ds = [d for d in sorted(glob.glob(ld + "seed_*")) if os.path.exists(d + "/test_pred.npy")]
            if not ds:
                continue
            R = [one(d, a.draws) for d in ds]
            f = lambda k: np.array([r[k] for r in R])
            ms = lambda k: f"{f(k).mean():.3f} ± {f(k).std():.3f}"
            dl = f("sr_rt") - f("sr")
            print(f"| {exp} | {os.path.basename(ld.rstrip('/\\'))} | {len(R)} | {100*f('ties').mean():.0f}% | {ms('sr')} | {ms('sr_rt')} | {dl.mean():+.3f} | {np.abs(dl).max():.3f} | "
                  f"{ms('xsr')} | {ms('xsr_rt')} | {ms('nd')} | {ms('nd_rt')} |", flush=True)


if __name__ == "__main__":
    main()
