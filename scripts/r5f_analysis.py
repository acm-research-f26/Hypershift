"""Phase 1.5 F analysis of the corrected full-NYSE reruns (preset 10 = R5_f_*, preset 11 = R8_f_*). CPU only.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/r5f_analysis.py [--root DIR] [--out-md docs/phase1_5/F_r5f_results.md]
        [--out-json results/_r5f_summary.json] [--r5-prefix R5_f] [--r8-prefix R8_f] [--norms paper train] [--draws 10] [--fig PNG]

--root is a directory that holds `results/<exp>/<label>/seed_<k>/` (or the exp folders directly). Layout read:
    <r5-prefix>_<norm>/{HH,EH,EE}/seed_k       (THINK, TConv+DHHAN, Euclid-Euclid)
    <r8-prefix>_<norm>/{RSR_I,STHGCN}/seed_k
Only seeds with metrics.json count. Pairs use the seeds BOTH arms have, so partial runs are fine.
Logic follows scripts/r5_g2_analysis.py and r8_analysis.py (those scripts run at import time with hard-coded paths, so they are
not importable; the per-run diagnostics are re-implemented here parametrized by path) and reuses scripts/tiebreak_report.py.one
for the random tie-break column. Stats: hypershift.eval.stats (paired Wilcoxon, Holm, stationary block bootstrap, verdict words);
INSUFFICIENT SEEDS follows scripts/aggregate.py (_holm): Holm p unavailable, or the smallest attainable p (m * 2^(1-n)) >= 0.01."""
import argparse, glob, json, math, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from hypershift.eval.metrics import evaluate_all, ndcg_at_k, sharpe, topk_daily_returns
from hypershift.eval.stats import holm, sharpe_diff_ci, verdict, wilcoxon_paired
import tiebreak_report

SQ = math.sqrt(252)
LAB5 = {"HH": "HH (THINK)", "EH": "EH (TConv+DHHAN)", "EE": "EE"}
LAB8 = {"RSR_I": "RSR-I", "STHGCN": "STHGCN"}
LABEL = {**LAB5, **LAB8}
# Old Phase 1 numbers (leak-free, mean over seeds), from docs/phase1/R5_full_nyse_g2.md and R8_baselines_results.md (old optimizer, wd 5e-4, level inputs)
OLD = {"HH": 0.089, "EH": 0.383, "RSR_I": 0.879, "STHGCN": 0.649}
OLD_N = {"HH": 25, "EH": 25, "RSR_I": 10, "STHGCN": 10}
OLD_HOLD = 1.531
# Paper Table II, p. 852 (docs/paper/icdm22-think.pdf), NYSE columns SR / NDCG@5
PAPER = {"HH": (1.18, 0.86), "EH": (1.14, 0.81), "STHGCN": (1.10, 0.78), "RSR_I": (1.05, 0.75)}


def mean_sd(x, f=3):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "n/a"
    return f"{np.nanmean(x):.{f}f} ± {np.nanstd(x):.{f}f}"


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


class Source:
    def __init__(self, root):
        root = Path(root)
        self.base = root / "results" if (root / "results").is_dir() else root

    def runs(self, exp, label):
        out = {}
        for d in sorted(glob.glob(str(self.base / exp / label / "seed_*"))):
            if os.path.exists(d + "/metrics.json") and os.path.exists(d + "/test_pred.npy"):
                out[int(d.rsplit("_", 1)[1])] = d
        return out

    def partial(self, exp, label):
        return len(glob.glob(str(self.base / exp / label / "seed_*")))


def run_diag(d):
    m = json.load(open(d + "/metrics.json"))
    h = {r["epoch"]: r for r in jl(d + "/history.jsonl")}
    p, g, k = (np.load(f"{d}/test_{n}.npy") for n in ("pred", "gt", "mask"))
    k1 = k > 0.5
    ics = [np.corrcoef(rankdata(p[k1[:, t], t]), rankdata(g[k1[:, t], t]))[0, 1] for t in range(p.shape[1]) if k1[:, t].sum() > 2]
    sp = np.mean([p[k1[:, t], t].std() / g[k1[:, t], t].std() for t in range(p.shape[1]) if k1[:, t].sum() > 2])
    o = h.get(m["test_oracle_epoch"], {}).get("test", {})
    return dict(sr_v=m["test"]["sr"], sr_o=m["test_oracle_sr"], val_sr=m["val"]["sr"], nd_v=m["test"]["ndcg5"], nd_o=o.get("ndcg5", float("nan")),
                irr=m["test"]["irr"], ic=float(np.nanmean(ics)) if np.any(np.isfinite(ics)) else float("nan"), spread=float(sp),
                best_ep=m["best_epoch"], orc_ep=m["test_oracle_epoch"], epochs_run=m["epochs_run"])


def paired(a, b):
    """p-value paired Wilcoxon on the common seeds; nan with fewer than 2 pairs."""
    return wilcoxon_paired(a, b) if len(a) >= 2 else float("nan")


def family(rows):
    """rows: dicts with p and n. Adds p_holm, p_floor, verdict (aggregate.py rule)."""
    ps = {i: r["p"] for i, r in enumerate(rows) if not np.isnan(r["p"])}
    adj = holm(ps) if ps else {}
    m = max(len(ps), 1)
    for i, r in enumerate(rows):
        r["p_holm"] = adj.get(i, float("nan"))
        r["p_floor"] = min(1.0, m * 2.0 ** (1 - r["n"])) if r["n"] >= 1 else 1.0
        if i not in adj or (r["p_floor"] >= 0.01 and r["p_holm"] >= 0.01):
            r["verdict"] = "INSUFFICIENT SEEDS"
        else:
            r["verdict"] = verdict(r["p_holm"], r["ci_lo"], r["ci_hi"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-md", default="docs/phase1_5/F_r5f_results.md")
    ap.add_argument("--out-json", default="results/_r5f_summary.json")
    ap.add_argument("--r5-prefix", default="R5_f")
    ap.add_argument("--r8-prefix", default="R8_f")
    ap.add_argument("--norms", nargs="+", default=["paper", "train"])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    S = Source(a.root)
    L, J = [], {"root": str(a.root), "prefixes": [a.r5_prefix, a.r8_prefix]}
    w = L.append

    w("# Phase 1.5 F: corrected full-NYSE reruns (R5_f and R8_f)\n")
    w(f"Generated by `scripts/r5f_analysis.py` (root `{a.root}`; exps `{a.r5_prefix}_<norm>`, `{a.r8_prefix}_<norm>`). CPU only, recomputed from `metrics.json`, `history.jsonl`, `test_*.npy`.")
    w("Config of these runs (kaggle/run_kaggle.py presets 10 and 11): `input_mode=relative weight_decay=0 log_ic=true epochs=100 patience=1000`; R8 also `batch_days=8`. "
      "Sharpe = ours (top-5 daily, mean/std x sqrt(252), no risk-free rate, no costs); the paper's formula differs (PA4 in `docs/PHASE1_TRACKER.md`), so absolute levels are not directly comparable to the paper. "
      "Unannualized columns are ours / sqrt(252). Leak-free = epoch chosen on validation Sharpe; best-test = max over epochs of test Sharpe (diagnostic only, selects on the test year).\n")
    w("Paper reference, Table II p. 852 of `docs/paper/icdm22-think.pdf` (NYSE, mean of 25 runs, SR / NDCG@5): "
      + "; ".join(f"{LABEL[k]} {v[0]:.2f} / {v[1]:.2f}" for k, v in PAPER.items()) + " (the paper's EH row is the TConv+DHHAN row).\n")

    per_norm = {}
    for norm in a.norms:
        e5, e8 = f"{a.r5_prefix}_{norm}", f"{a.r8_prefix}_{norm}"
        R = {l: S.runs(e5, l) for l in LAB5}
        R.update({l: S.runs(e8, l) for l in LAB8})
        npart = {l: S.partial(e5 if l in LAB5 else e8, l) for l in LABEL}
        present = [l for l in LABEL if R[l]]
        w(f"\n---\n\n## norm = {norm}\n")
        w("### Seeds finished\n| arm | exp | finished (metrics.json) | seeds | dirs on disk |\n|---|---|---|---|---|")
        for l in LABEL:
            w(f"| {LABEL[l]} | {e5 if l in LAB5 else e8} | {len(R[l])} | {sorted(R[l]) or '-'} | {npart[l]} |")
        N = {"norm": norm, "n": {l: len(R[l]) for l in LABEL}}
        per_norm[norm] = N
        if not present:
            w("\nNo finished runs for this norm.")
            continue
        D = {l: [run_diag(R[l][s]) for s in sorted(R[l])] for l in present}
        col = lambda l, k: np.array([r[k] for r in D[l]], float)
        # reference mask / returns
        ref = R[present[0]][min(R[present[0]])]
        G, K = np.load(ref + "/test_gt.npy"), np.load(ref + "/test_mask.npy")
        const = evaluate_all(np.zeros_like(G), G, K)["sr"]
        market = sharpe((G * K).sum(0) / np.maximum(K.sum(0), 1.0))
        rng = np.random.default_rng(1)
        elig = np.nonzero(K.min(axis=1) > 0.5)[0]
        fr = np.array([sharpe(G[rng.choice(elig, 5, replace=False)].mean(axis=0)) for _ in range(500)])
        rnd = float(np.mean([ndcg_at_k(np.random.default_rng(i).standard_normal(G.shape), G, K) for i in range(20)]))
        N.update(hold_all=market, const=const, fixed_random5=float(fr.mean()), random_ndcg=rnd)
        w(f"\nBaselines on the {G.shape[1]} test days (2017) and this mask: hold-all {market:.3f} (unannualized {market/SQ:.4f}); fixed random 5-stock set {fr.mean():.3f} ± {fr.std():.3f} (500 draws); "
          f"constant-prediction Sharpe {const:.3f}; random NDCG@5 {rnd:.4f}.")
        w("\n### Table 1: Sharpe\n| arm | n | leak-free | best-test | val Sharpe at selected epoch | leak-free unann. | best-test unann. | leak-free > hold-all | median selected epoch | median best-test epoch |\n|---|---|---|---|---|---|---|---|---|---|")
        for l in present:
            n = len(D[l])
            w(f"| {LABEL[l]} | {n} | {mean_sd(col(l,'sr_v'))} | {mean_sd(col(l,'sr_o'))} | {mean_sd(col(l,'val_sr'))} | {mean_sd(col(l,'sr_v')/SQ,4)} | {mean_sd(col(l,'sr_o')/SQ,4)} | "
              f"{int((col(l,'sr_v')>market).sum())}/{n} | {int(np.median(col(l,'best_ep')))} | {int(np.median(col(l,'orc_ep')))} |")
            N[l] = {k: [float(np.mean(col(l, k))), float(np.std(col(l, k)))] for k in ("sr_v", "sr_o", "val_sr", "nd_v", "nd_o", "irr", "ic", "spread")}
            N[l]["n"] = n
        w("\n### Table 2: ranking diagnostics at the leak-free epoch (mean ± std over seeds)\n| arm | NDCG@5 | NDCG@5 at best-test epoch | IRR | IC (daily rank corr.) | pred spread / actual | best-test epoch == last epoch |\n|---|---|---|---|---|---|---|")
        for l in present:
            last = int(np.sum(col(l, "orc_ep") >= col(l, "epochs_run") - 1))
            w(f"| {LABEL[l]} | {mean_sd(col(l,'nd_v'),4)} | {mean_sd(col(l,'nd_o'),4)} | {mean_sd(col(l,'irr'),3)} | {mean_sd(col(l,'ic'),4)} | {mean_sd(col(l,'spread'),3)} | {last}/{len(D[l])} |")
        # tie-break
        w(f"\n### Table 3: random tie-break (scripts/tiebreak_report.py, {a.draws} draws; default evaluator = lowest-index tie-break)\n| arm | n | tie days at top-5 boundary | SR default | SR random tie-break | delta | excess SR (top-5 minus hold) default | excess SR random | NDCG@5 default | NDCG@5 random |\n|---|---|---|---|---|---|---|---|---|---|")
        for l in present:
            T = [tiebreak_report.one(R[l][s], a.draws) for s in sorted(R[l])]
            f = lambda k: np.array([t[k] for t in T])
            w(f"| {LABEL[l]} | {len(T)} | {100*f('ties').mean():.0f}% | {mean_sd(f('sr'))} | {mean_sd(f('sr_rt'))} | {(f('sr_rt')-f('sr')).mean():+.3f} | {mean_sd(f('xsr'))} | {mean_sd(f('xsr_rt'))} | {mean_sd(f('nd'),4)} | {mean_sd(f('nd_rt'),4)} |")
            N[l]["tiebreak"] = {k: float(f(k).mean()) for k in ("sr", "sr_rt", "xsr", "xsr_rt", "nd", "nd_rt", "ties")}

        def compare(pairs, title):
            rows = []
            for x, y in pairs:
                if not (R.get(x) and R.get(y)):
                    continue
                common = sorted(set(R[x]) & set(R[y]))
                if not common:
                    continue
                sx = np.array([json.load(open(R[x][s] + "/metrics.json"))["test"]["sr"] for s in common])
                sy = np.array([json.load(open(R[y][s] + "/metrics.json"))["test"]["sr"] for s in common])
                bx = np.array([json.load(open(R[x][s] + "/metrics.json"))["test_oracle_sr"] for s in common])
                by = np.array([json.load(open(R[y][s] + "/metrics.json"))["test_oracle_sr"] for s in common])
                ser = lambda l: np.mean([np.load(R[l][s] + "/test_daily.npy") for s in common], axis=0)
                ci = sharpe_diff_ci(ser(x), ser(y), n_boot=a.boot)
                rows.append(dict(q=f"{LABEL[x]} vs {LABEL[y]}", n=len(common), seeds=common, d=float((sx - sy).mean()), p=paired(sx, sy), wins=int((sx > sy).sum()),
                                 ci_lo=ci["lo"], ci_hi=ci["hi"], bt_d=float((bx - by).mean()), bt_p=paired(bx, by), bt_wins=int((bx > by).sum()),
                                 ens_x=float(sharpe(ser(x))), ens_y=float(sharpe(ser(y)))))
            family(rows)
            w(f"\n### {title}\nPaired by seed (only seeds both arms finished), paired Wilcoxon two-sided, Holm over this family of {len(rows)}, 95% stationary block bootstrap CI of the difference of the seed-averaged daily Sharpes. "
              "`p floor` = smallest Holm p attainable with n pairs.\n| comparison | n pairs | diff (a-b) leak-free | Wilcoxon p | Holm p | p floor | block-bootstrap CI | a wins | verdict | seed-ensemble Sharpe a / b | best-test diff | best-test p (raw) | best-test a wins |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                w(f"| {r['q']} | {r['n']} | {r['d']:+.3f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['p_floor']:.4f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['wins']}/{r['n']} | **{r['verdict']}** | "
                  f"{r['ens_x']:.3f} / {r['ens_y']:.3f} | {r['bt_d']:+.3f} | {r['bt_p']:.4f} | {r['bt_wins']}/{r['n']} |")
            return rows

        N["cmp_think"] = compare([("HH", "EH"), ("HH", "EE")], "Table 4: THINK (HH) vs EH and vs EE (leak-free)")
        N["cmp_baselines"] = compare([("RSR_I", "HH"), ("STHGCN", "HH"), ("RSR_I", "EH"), ("STHGCN", "EH")], "Table 5: baselines under the fix (RSR-I, STHGCN) vs THINK and EH (leak-free)")
        # old vs new
        w("\n### Table 6: old Phase 1 (wd 5e-4, level inputs, old optimizer) vs this rerun, leak-free Sharpe\n"
          f"Old numbers from `docs/phase1/R5_full_nyse_g2.md` and `R8_baselines_results.md`; old hold-all {OLD_HOLD:.3f}; hold-all recomputed here {market:.3f}. Paper column: Table II p. 852 (different Sharpe formula, indicative only).\n"
          "| arm | old n | old leak-free | new n | new leak-free | new minus old | paper SR p. 852 |\n|---|---|---|---|---|---|---|")
        for l in ("HH", "EH", "STHGCN", "RSR_I"):
            new = col(l, "sr_v").mean() if l in D else float("nan")
            w(f"| {LABEL[l]} | {OLD_N[l]} | {OLD[l]:.3f} | {len(D.get(l, []))} | {new:.3f} | {new-OLD[l]:+.3f} | {PAPER[l][0]:.2f} |")
        if "HH" in D and "EH" in D:
            w(f"\nOld THINK minus EH = {OLD['HH']-OLD['EH']:+.3f}; new = {col('HH','sr_v').mean()-col('EH','sr_v').mean():+.3f} (all seeds each; Table 4 gives the paired value). Paper: {PAPER['HH'][0]-PAPER['EH'][0]:+.2f}.")
    # cross-norm
    if len(a.norms) == 2 and all(per_norm[n].get("HH") for n in a.norms):
        w("\n---\n\n## norm = paper vs norm = train (THINK, leak-free Sharpe)\n| norm | n | leak-free | best-test |\n|---|---|---|---|")
        for n in a.norms:
            hh = per_norm[n]["HH"]
            w(f"| {n} | {hh['n']} | {hh['sr_v'][0]:.3f} ± {hh['sr_v'][1]:.3f} | {hh['sr_o'][0]:.3f} ± {hh['sr_o'][1]:.3f} |")
    w("\nVerdict words (`hypershift.eval.stats.verdict`, `scripts/aggregate.py`): STRONG = Holm p < 0.01 and bootstrap CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0; "
      "NO EVIDENCE = Holm p >= 0.01; INSUFFICIENT SEEDS = too few paired seeds for p < 0.01 to be attainable.")
    J["norms"] = per_norm
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(J, open(a.out_json, "w"), indent=1, default=float)
    print(f"wrote {a.out_md} and {a.out_json}")
    if a.fig:
        figure(per_norm, a.norms, a.fig)


def figure(per_norm, norms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(norms), figsize=(6.5 * len(norms), 4.6), squeeze=False)
    for ax, norm in zip(axs[0], norms):
        N = per_norm[norm]
        arms = [l for l in LABEL if N.get(l)]
        x = np.arange(len(arms))
        for j, (k, c, lab) in enumerate((("sr_o", "#2a78d6", "best-test epoch"), ("sr_v", "#eb6834", "validation-selected (leak-free)"))):
            ax.bar(x + (j - 0.5) * 0.38, [N[l][k][0] for l in arms], 0.35, yerr=[N[l][k][1] for l in arms], color=c, label=lab, capsize=2)
        if "hold_all" in N:
            ax.axhline(N["hold_all"], color="#1a1a19", ls="--", lw=1)
        ax.set_xticks(x, [f"{LABEL[l]}\nn={N[l]['n']}" for l in arms], fontsize=8)
        ax.set_title(f"norm={norm}", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0][0].set_ylabel("2017 test Sharpe (ours, annualized)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
