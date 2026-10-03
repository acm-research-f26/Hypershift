"""Phase 1.5 F analysis of the small-scale rerun (Kaggle preset `p1f`): 309-stock g2 arms, R8 small baselines and R7 NASDAQ F1 with the F fix. CPU only.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/p1f_analysis.py [--root DIR] [--suffix _f] [--summarize] [--out-md docs/phase1_5/F_p1f_results.md]

--root holds `results/<exp>/<label>/seed_<k>/` (cwd must also hold data/raw/rsr/data when --summarize is given). Exps read (suffix = `_f`):
    POC_sectors_rel_g2<suffix>/{HH,EE,EH}_{hyper,clique,none}/seed_k      (poc_sectors.py run --input-mode relative --set weight_decay=0)
    POC_sectors_R8_{rsr_i,sthgcn}_g2<suffix>/{rsr_i,sthgcn}/seed_k
    E11_clf_g2<suffix>/{HH,EH,EE}/seed_k                                   (run_clf.py --set input_mode=relative weight_decay=0)
Missing arms or seeds are skipped (smoke runs have 1 seed). `--suffix ""` reads the Phase 1 exps (read-only self-check; do not combine with --summarize,
which rewrites results/<exp>/summary.md). Stats follow scripts/r5f_analysis.py (paired Wilcoxon by seed, Holm per family, stationary block bootstrap, verdict words)."""
import argparse, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from hypershift.eval.metrics import sharpe
from hypershift.eval.stats import sharpe_diff_ci, verdict
import r5f_analysis as R5

SQ = math.sqrt(252)
ARMS = ["HH_hyper", "HH_clique", "HH_none", "EH_hyper", "EH_clique", "EE_hyper", "EE_clique", "EE_none"]
# Phase 1 numbers for comparison, from docs/phase1/g2_small_results.md sec. 1 (rel_g2: relative inputs, wd 5e-4; 10 seeds; leak-free / best-test, mean, std)
P1_POC = {"HH_hyper": (0.007, 0.781, 1.924, 0.000), "HH_clique": (0.463, 0.458, 1.924, 0.000), "HH_none": (0.484, 0.370, 0.948, 0.296),
          "EH_hyper": (-0.297, 0.249, 1.629, 0.230), "EH_clique": (0.023, 0.478, 1.064, 0.585), "EE_hyper": (-0.431, 0.280, 1.709, 0.313),
          "EE_clique": (0.009, 0.513, 0.988, 0.257), "EE_none": (0.061, 0.288, 1.032, 0.263)}
# Phase 1 R8 small (POC_sectors_R8_*_g2, LEVEL inputs, wd 5e-4, 10 seeds; recomputed from metrics.json 2026-10-02): leak-free mean, std, best-test mean, std
P1_R8 = {"rsr_i": (0.130, 0.646, 2.096, 0.494), "sthgcn": (-0.586, 0.770, 1.997, 0.532)}
# Phase 1 R7 (E11_clf_g2, LEVEL inputs, wd 5e-4, 25 seeds; docs/phase1/R7_nasdaq_clf.md sec. 1): test macro mean, std, test micro mean, std
P1_CLF = {"HH": (0.2897, 0.0108, 0.3809, 0.0008), "EH": (0.2819, 0.0169, 0.3795, 0.0043), "EE": (0.2946, 0.0120, 0.3824, 0.0005)}
PAPER_CLF = {"HH": 0.49, "EH": 0.44}     # paper Table II p. 852 "Clf" column (F1; averaging unstated)
PAIRS = [("HH_hyper", "EH_hyper", "THINK vs TConv+DHHAN (paper ablation), hyperedges"),
         ("HH_clique", "EH_clique", "THINK vs TConv+DHHAN, pairwise edges"),
         ("HH_hyper", "EE_hyper", "hyperbolic vs Euclidean, hyperedges"),
         ("HH_hyper", "HH_clique", "hyperedges vs pairwise (hyperbolic)"),
         ("HH_hyper", "HH_none", "relations vs none (hyperbolic)"),
         ("EH_hyper", "EH_clique", "hyperedges vs pairwise (EH)"),
         ("EE_hyper", "EE_clique", "hyperedges vs pairwise (Euclidean)"),
         ("EE_hyper", "EE_none", "relations vs none (Euclidean)"),
         ("EH_hyper", "EE_hyper", "hyperbolic vs Euclidean attention (Euclidean temporal conv)")]


def ms(x, f=3):
    return R5.mean_sd(x, f)


def load_runs(base, exp, label):
    out = {}
    for d in sorted((base / exp / label).glob("seed_*")):
        if (d / "metrics.json").exists():
            out[int(d.name.split("_")[1])] = d
    return out


def jm(d):
    return json.load(open(d / "metrics.json"))


def plain_verdict(rows):
    """Phase 1 small-scale docs use hypershift.eval.stats.verdict directly (poc_sectors.py summarize), not the INSUFFICIENT SEEDS rule of r5f_analysis.family()."""
    for r in rows:
        r["verdict"] = verdict(r["p_holm"], r["ci_lo"], r["ci_hi"]) if not np.isnan(r["p_holm"]) else "n/a (fewer than 2 pairs)"
    return rows


def boot_ci(diff, n=5000, seed=0):
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(diff, (n, len(diff))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--suffix", default="_f")
    ap.add_argument("--summarize", action="store_true", help="also run poc_sectors.py summarize for the rel_g2 exp (cwd = --root) and append its table")
    ap.add_argument("--out-md", default="docs/phase1_5/F_p1f_results.md")
    ap.add_argument("--boot", type=int, default=3000)
    a = ap.parse_args()
    root = Path(a.root)
    base = root / "results" if (root / "results").is_dir() else root
    sx = a.suffix
    poc_exp, clf_exp = f"POC_sectors_rel_g2{sx}", f"E11_clf_g2{sx}"
    L = []
    w = L.append
    w(f"# Phase 1.5 F: small-scale rerun with the F fix (preset p1f), exps `{poc_exp}`, `POC_sectors_R8_*_g2{sx}`, `{clf_exp}`\n")
    w("Generated by `scripts/p1f_analysis.py`; CPU only, recomputed from `metrics.json`, `history.jsonl`, `test_*.npy`. Sharpe = ours (top-5 daily, mean/std x sqrt(252), no risk-free rate, "
      "no costs); the paper's Sharpe formula differs (PA4 in `docs/PHASE1_TRACKER.md`), so levels are not comparable to the paper. Leak-free = epoch chosen on validation Sharpe; best-test = max over epochs of test Sharpe (diagnostic, selects on the test year).\n")
    w("**Settings, labelled.** 309 stocks (Energy/Utilities + Finance), corrected hypergraph (cache v2, App. B p. 854), 30 epochs, patience 10, batch_days 8, lr 1e-3, alpha 1, seeds 0-9: all as Phase 1 `rel_g2` (INFERRED hyperparameters: the paper states none, audit P44). "
      "`input_mode=relative` was already in Phase 1 `rel_g2` (DEPARTURE from the paper's raw price inputs; level inputs are not rerun). **The one change is `weight_decay` 5e-4 -> 0** (DEPARTURE from the repo default, itself INFERRED; "
      "the paper states no weight decay), plus `log_ic=true` (diagnostic only). R8 small baselines and R7 NASDAQ F1 use `input_mode=relative weight_decay=0` against Phase 1 runs that used level inputs and wd 5e-4 (two factors change). "
      "The R7 label definition (tertiles of pooled training returns) and the F1 averaging are INFERRED (paper does not state them; p. 852).\n")

    # ---------------- POC arms
    R = {arm: load_runs(base, poc_exp, arm) for arm in ARMS}
    present = [x for x in ARMS if R[x]]
    if not present:
        w(f"No finished runs under `{poc_exp}` yet.\n")
    else:
        D = {x: {s: R5.run_diag(str(d)) for s, d in R[x].items()} for x in present}
        ref = next(iter(R[present[0]].values()))
        G, K = np.load(ref / "test_gt.npy"), np.load(ref / "test_mask.npy")
        hold = float(sharpe((G * K).sum(0) / np.maximum(K.sum(0), 1.0)))
        w(f"## 1. 309-stock arms (`{poc_exp}`)\n\nTest days {G.shape[1]}; hold-all-309 Sharpe {hold:.3f} (Phase 1: 0.752, random 5 stocks 0.338, 5-day momentum -0.276). "
          "IC = mean daily rank correlation of predictions with realised returns; spread = within-day std of predictions / std of returns (about 0 = collapsed to a constant). "
          "Phase 1 columns are `rel_g2` (wd 5e-4) from `docs/phase1/g2_small_results.md` sec. 1.\n")
        w("| arm | n | leak-free (F) | Phase 1 leak-free | F minus P1 | best-test (F) | Phase 1 best-test | IC | NDCG@5 | spread | median epochs run | leak-free > hold-all |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in present:
            rows = [D[x][s] for s in sorted(D[x])]
            col = lambda k: np.array([r[k] for r in rows], float)
            p1 = P1_POC.get(x)
            p1v = f"{p1[0]:.3f} ± {p1[1]:.3f}" if p1 else "n/a"
            p1b = f"{p1[2]:.3f} ± {p1[3]:.3f}" if p1 else "n/a"
            dl = f"{col('sr_v').mean() - p1[0]:+.3f}" if p1 else "n/a"
            w(f"| {x} | {len(rows)} | {ms(col('sr_v'))} | {p1v} | {dl} | {ms(col('sr_o'))} | {p1b} | {ms(col('ic'), 4)} | {ms(col('nd_v'), 4)} | {ms(col('spread'), 3)} | "
              f"{int(np.median(col('epochs_run')))} | {int((col('sr_v') > hold).sum())}/{len(rows)} |")
        rows, ps = [], {}
        for x, y, q in PAIRS:
            common = sorted(set(R.get(x, {})) & set(R.get(y, {})))
            if not common:
                continue
            sxv = np.array([jm(R[x][s])["test"]["sr"] for s in common])
            syv = np.array([jm(R[y][s])["test"]["sr"] for s in common])
            bxv = np.array([jm(R[x][s])["test_oracle_sr"] for s in common])
            byv = np.array([jm(R[y][s])["test_oracle_sr"] for s in common])
            ser = lambda l: np.mean([np.load(R[l][s] / "test_daily.npy") for s in common], axis=0)
            ci = sharpe_diff_ci(ser(x), ser(y), n_boot=a.boot)
            rows.append(dict(q=f"{q} ({x} - {y})", n=len(common), d=float((sxv - syv).mean()), p=R5.paired(sxv, syv), wins=int((sxv > syv).sum()),
                             ci_lo=ci["lo"], ci_hi=ci["hi"], bt_d=float((bxv - byv).mean()), bt_p=R5.paired(bxv, byv)))
        plain_verdict(R5.family(rows))
        w(f"\n### Paired comparisons (leak-free; only seeds both arms finished; Wilcoxon two-sided, Holm over this family of {len(rows)}, 95% block-bootstrap CI of the seed-ensemble Sharpe difference)\n")
        w("| comparison | n pairs | diff leak-free | Wilcoxon p | Holm p | block-bootstrap CI | a wins | verdict | best-test diff | best-test p (raw) |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            w(f"| {r['q']} | {r['n']} | {r['d']:+.3f} | {r['p']:.4f} | {r['p_holm']:.4f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['wins']}/{r['n']} | **{r['verdict']}** | {r['bt_d']:+.3f} | {r['bt_p']:.4f} |")
        w("\nPaper reference: Table II p. 852 (NYSE, full universe): THINK 1.18 vs TCONV+DHHAN 1.14 Sharpe. These are 309 stocks and one test year, so the paper's 0.04 gap is far below what this test resolves (Phase 1 sec. 2a).\n")

    # ---------------- R8 small
    w(f"## 2. R8 baselines, 309 stocks (`POC_sectors_R8_<model>_g2{sx}`)\n")
    w("| model | n | leak-free (F) | Phase 1 leak-free (level, wd 5e-4) | best-test (F) | Phase 1 best-test | IC | spread |\n|---|---|---|---|---|---|---|---|")
    anyr8 = False
    R8 = {}
    for m in ("rsr_i", "sthgcn"):
        R8[m] = load_runs(base, f"POC_sectors_R8_{m}_g2{sx}", m)
        if not R8[m]:
            continue
        anyr8 = True
        rows = [R5.run_diag(str(d)) for d in R8[m].values()]
        col = lambda k: np.array([r[k] for r in rows], float)
        p1 = P1_R8[m]
        w(f"| {m} | {len(rows)} | {ms(col('sr_v'))} | {p1[0]:.3f} ± {p1[1]:.3f} | {ms(col('sr_o'))} | {p1[2]:.3f} ± {p1[3]:.3f} | {ms(col('ic'), 4)} | {ms(col('spread'), 3)} |")
    if not anyr8:
        w("| (no finished runs) | | | | | | | |")
    for m in ("rsr_i", "sthgcn"):
        for y in ("HH_hyper", "EH_hyper"):
            common = sorted(set(R8[m]) & set(R.get(y, {}))) if present else []
            if len(common) >= 1:
                dv = np.array([jm(R8[m][s])["test"]["sr"] - jm(R[y][s])["test"]["sr"] for s in common])
                w(f"\n{m} minus {y} (leak-free, {len(common)} paired seeds): {dv.mean():+.3f}, Wilcoxon p {R5.paired(np.array([jm(R8[m][s])['test']['sr'] for s in common]), np.array([jm(R[y][s])['test']['sr'] for s in common])):.4f}, raw, not Holm.")

    # ---------------- R7 clf
    w(f"\n## 3. R7 NASDAQ 3-class movement F1 (`{clf_exp}`)\n")
    w("Test macro- and micro-F1 (micro = accuracy) on 2017, epoch chosen on validation macro-F1 (patience 20, 60 epochs max). Chance: macro-F1 about 0.333 and micro about 0.333 for random labels, "
      "always-neutral 0.180 / 0.370 (`docs/phase1/R7_nasdaq_clf.md` sec. 2). Paper Table II p. 852 'Clf' F1: THINK 0.49, TCONV+DHHAN 0.44 (averaging not stated).\n")
    C = {g: load_runs(base, clf_exp, g) for g in ("HH", "EH", "EE")}
    w("| arm | n | test macro-F1 (F) | Phase 1 macro | test micro-F1 (F) | Phase 1 micro | val macro-F1 | median best epoch | paper |\n|---|---|---|---|---|---|---|---|---|")
    M = {}
    for g in ("HH", "EH", "EE"):
        if not C[g]:
            continue
        ss = sorted(C[g])
        M[g] = {s: jm(C[g][s]) for s in ss}
        v = lambda k: np.array([M[g][s].get(k, float("nan")) for s in ss], float)
        p1 = P1_CLF[g]
        w(f"| {g} | {len(ss)} | {ms(v('test_f1'), 4)} | {p1[0]:.4f} ± {p1[1]:.4f} | {ms(v('test_micro_f1'), 4)} | {p1[2]:.4f} ± {p1[3]:.4f} | {ms(v('val_f1'), 4)} | "
          f"{int(np.median(v('best_epoch')))} | {PAPER_CLF.get(g, '-')} |")
    if not M:
        w("| (no finished runs) | | | | | | | | |")
    crow = []
    for key, name in (("test_f1", "macro"), ("test_micro_f1", "micro")):
        for x, y in (("HH", "EH"), ("HH", "EE"), ("EH", "EE")):
            common = sorted(set(M.get(x, {})) & set(M.get(y, {})))
            if not common:
                continue
            dv = np.array([M[x][s][key] - M[y][s][key] for s in common])
            lo, hi = boot_ci(dv, a.boot)
            xa = np.array([M[x][s][key] for s in common]); ya = np.array([M[y][s][key] for s in common])
            crow.append(dict(q=f"{x} - {y} test {name}-F1", n=len(common), d=float(dv.mean()), p=R5.paired(xa, ya), wins=int((dv > 0).sum()), ci_lo=lo, ci_hi=hi))
    if crow:
        plain_verdict(R5.family(crow))
        w(f"\n### Paired by seed (Wilcoxon two-sided, Holm over {len(crow)} comparisons, 95% paired bootstrap CI of the mean difference)\n")
        w("| contrast | n pairs | mean diff | Wilcoxon p | Holm p | CI | a wins | verdict |\n|---|---|---|---|---|---|---|---|")
        for r in crow:
            w(f"| {r['q']} | {r['n']} | {r['d']:+.4f} | {r['p']:.4f} | {r['p_holm']:.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wins']}/{r['n']} | **{r['verdict']}** |")
    w("\nVerdict words (`hypershift.eval.stats.verdict`): STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0; NO EVIDENCE = Holm p >= 0.01; "
      "INSUFFICIENT SEEDS = too few paired seeds for p < 0.01 to be attainable.\n")
    w("**Not rerun in preset p1f (dropped for the ~6 GPU-h budget, see `docs/phase1_5/F_learnability.md`):** level-input g2 (`POC_sectors_g2`; the F stage-2 level runs `f_lvl_wd0*` already showed level inputs are the worse setting), "
      "tuned `rel_tuned_g2` (about 9 h of laptop time, and the tuned lr/alpha were chosen under the collapsed setup), A10/C1/G2/G12 ablations, R1-R3 (`run_pygt.py` already uses `--wd 0`, so it never had the decay collapse).")

    if a.summarize:
        try:
            r = subprocess.run([sys.executable, str(HERE / "poc_sectors.py"), "summarize", "--variant", f"rel_g2{sx}"], cwd=root, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH=str(HERE.parent / "src"), CUDA_VISIBLE_DEVICES="-1"), timeout=600)
            summ = root / "results" / poc_exp / "summary.md"
            if r.returncode == 0 and summ.exists():
                w(f"\n---\n\n## Appendix: `poc_sectors.py summarize --variant rel_g2{sx}` (Phase 1 format, its own Holm family of up to 10)\n")
                w(summ.read_text(encoding="utf-8"))
            else:
                w(f"\n(poc_sectors summarize failed rc={r.returncode}: {r.stderr[-400:]})")
        except Exception as e:
            w(f"\n(poc_sectors summarize not available: {e!r})")
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out_md}")


if __name__ == "__main__":
    main()
