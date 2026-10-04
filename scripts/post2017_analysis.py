"""Phase 1.5b Task 5: persistence and ranking evidence on the locked 2018-2023 outputs (CPU only).

Reads results/post2017_frozen/{HH,EH}/seed_k/ (date-keyed) and writes docs/phase1_5b/post2017_results.json + post2017_tables.md.
Predeclared (see REPORT_POST2017.md, Predeclaration): B_NULL=2000, B_PERM=500, N_BOOT=5000, mean block 10 days, k=5,
costs 0/5/10/25 bp per side, rng seeds from hypershift.eval.forensics.rng_for. All comparisons are same-day, same-mask with
explicit date joins; pooled Sharpe is computed from the concatenated daily series, never as a mean of annual Sharpes.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import math
import sys
from pathlib import Path

import numpy as np

import hypershift.eval.forensics as F
from hypershift.eval.metrics import daily_ic, sharpe
from hypershift.eval.stats import holm, stationary_bootstrap_indices, wilcoxon_paired

B_NULL, B_PERM, N_BOOT, BLOCK, K = 2000, 500, 5000, 10, 5
COSTS = (0, 5, 10, 25)
SEEDS = range(5)
ARMS = ("HH", "EH")
ROOT = Path("results/post2017_frozen")
DOCS = Path("docs/phase1_5b")


# ---------- pure helpers (unit-tested) ----------
def date_join(da, ra, db, rb):
    """Explicit inner join on dates. Returns (dates, ra, rb, n_dropped_a, n_dropped_b). Never aligns tails."""
    da, db = np.asarray(da), np.asarray(db)
    if len(set(da.tolist())) != len(da) or len(set(db.tolist())) != len(db):
        raise ValueError("duplicate dates")
    common, ia, ib = np.intersect1d(da, db, return_indices=True)
    return common, np.asarray(ra)[ia], np.asarray(rb)[ib], len(da) - len(common), len(db) - len(common)


def pooled_sharpe(r) -> float:
    r = np.asarray(r, float)
    sd = r.std()
    return 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(252))


def annual(dates, r):
    yrs = np.asarray(dates).astype("datetime64[Y]").astype(int) + 1970
    return {int(y): {"sr": pooled_sharpe(r[yrs == y]), "mean": float(r[yrs == y].mean()), "n": int((yrs == y).sum())} for y in np.unique(yrs)}


def leave_one_year_out(dates, r):
    yrs = np.asarray(dates).astype("datetime64[Y]").astype(int) + 1970
    return {int(y): pooled_sharpe(r[yrs != y]) for y in np.unique(yrs)}


def boot_indices(n, B, block, rng):
    return np.stack([stationary_bootstrap_indices(n, block, rng) for _ in range(B)])


def sr_rows(R):
    m, s = R.mean(1), R.std(1)
    return np.where(s > 0, m / np.where(s > 0, s, 1) * math.sqrt(252), 0.0)


def boot_ci(r, idx):
    s = sr_rows(r[idx])
    return [float(x) for x in np.percentile(s, [2.5, 97.5])]


def boot_contrast(ra, rb, idx):
    d = sr_rows(ra[idx]) - sr_rows(rb[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"est": pooled_sharpe(ra) - pooled_sharpe(rb), "lo": float(lo), "hi": float(hi), "p_boot": float(p)}


def holm_family(p_by_name):
    return holm(p_by_name)


# ---------- run ----------
def load(arm, s):
    d = ROOT / arm / f"seed_{s}"
    pred, gt, mask = (np.load(d / f"test_{n}.npy").astype(np.float64) for n in ("pred", "gt", "mask"))
    return pred, gt, mask > 0.5, np.load(d / "test_daily.npy"), np.load(d / "eligible_count.npy"), np.load(d / "tie_flag.npy")


def main(quick=False):
    bn, bp, nb = (50, 20, 200) if quick else (B_NULL, B_PERM, N_BOOT)
    from hypershift.data.rsr import load_rsr, read_ticker_file
    root = Path("data/raw/rsr/data")
    market = load_rsr(root, "NYSE", norm="train")
    tickers = read_ticker_file(root / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    dates = np.load(ROOT / "dates.npy")
    D = len(dates)
    beta = F.train_beta(market.gt, market.mask, market.valid_index)       # RSR training period only (2013-2015)
    qs = np.nanquantile(beta, [0.2, 0.4, 0.6, 0.8])
    s_beta = np.where(np.isnan(beta), 5, np.digitize(np.nan_to_num(beta), qs))
    ind = F.industry_of(tickers, root / "relation" / "sector_industry" / "NYSE_industry_ticker.json")
    codes = {n: i for i, n in enumerate(sorted(set(ind)))}
    s_ind = np.array([codes[x] for x in ind])
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    gt, mask = runs[("HH", 0)][1], runs[("HH", 0)][2]
    for k, v in runs.items():
        assert (v[2] == mask).all() and np.array_equal(v[1], gt), "mask/gt must be identical across runs"
    ha = F.hold_all(gt, mask)
    rng_b = F.rng_for("boot", 77)
    idx = boot_indices(D, nb, BLOCK, rng_b)
    res = {"declared": {"B_NULL": bn, "B_PERM": bp, "N_BOOT": nb, "block": BLOCK, "k": K, "quick": quick},
           "n_days": D, "dates": [str(dates[0]), str(dates[-1])], "beta_edges": [float(x) for x in qs],
           "n_beta_nan": int(np.isnan(beta).sum()), "n_industries": len(codes),
           "hold_all": {"gross": F.perf(ha), "annual": annual(dates, ha), "loyo": leave_one_year_out(dates, ha),
                        "ci95": boot_ci(ha, idx)}}
    yrs = dates.astype("datetime64[Y]").astype(int) + 1970
    res["eligible_by_year"] = {int(y): {"mean": float(mask[:, yrs == y].sum(0).mean()), "min": int(mask[:, yrs == y].sum(0).min()),
                                         "max": int(mask[:, yrs == y].sum(0).max())} for y in np.unique(yrs)}
    print("null random top-5 ...", flush=True)
    R_rand = F.null_random_topk(gt, mask, K, bn, F.rng_for("null_random", 77))
    srn = sr_rows(R_rand)
    res["null_random_sr_p5_50_95"] = [float(x) for x in np.percentile(srn, [5, 50, 95])]
    daily = {}
    per_seed_p = {f: {} for f in ("F1", "F2", "F3", "F4")}
    res["runs"] = {}
    for arm in ARMS:
        for s in SEEDS:
            pred, _, _, saved, elig, tie = runs[(arm, s)]
            r, base = F.portfolio(pred, gt, mask, K)
            assert np.abs(r - saved).max() < 1e-6, "stable-tie portfolio must reproduce the saved daily returns"
            daily[(arm, s)] = r
            to = F.turnover(base)
            ent = {"gross": F.perf(r), "ci95_sharpe": boot_ci(r, idx), "annual": annual(dates, r), "loyo": leave_one_year_out(dates, r),
                   "excess_over_hold_all": {"mean_daily": float((r - ha).mean()), "sr": pooled_sharpe(r - ha), "ci95_sr": boot_ci(r - ha, idx),
                                            "mean_ci95": [float(x) for x in np.percentile((r - ha)[idx].mean(1), [2.5, 97.5])]},
                   "net_sharpe": {str(c): pooled_sharpe(F.net(r, to, c)) for c in COSTS}, "turnover_mean": float(to.mean()),
                   "ic": float(daily_ic(pred, gt, mask)), "ic_by_year": {int(y): float(daily_ic(pred[:, yrs == y], gt[:, yrs == y], mask[:, yrs == y])) for y in np.unique(yrs)},
                   "tie_days": int(tie.sum())}
            diag = F.topk_diag(pred, gt, mask, K)
            ent["diag"] = {k: float(v) for k, v in diag.items()}
            bb = [np.nanmean(beta[b]) if len(b) else np.nan for b in base]
            ent["basket_beta_mean"] = float(np.nanmean(bb))
            ent["universe_beta_mean"] = float(np.nanmean([np.nanmean(beta[mask[:, d]]) for d in range(D)]))
            # nulls
            def summ(R):
                sr_ = sr_rows(R)
                return {"p": F.empirical_p(sr_, ent["gross"]["sr"]), "null_p5_50_95": [float(x) for x in np.percentile(sr_, [5, 50, 95])]}
            ent["F1_random_top5"] = summ(R_rand)
            ent["F2_beta_matched"] = summ(F.null_matched(base, gt, mask, s_beta, bp, F.rng_for("null_matched", 100 + s + (10 if arm == "EH" else 0))))
            ent["F3_industry_matched"] = summ(F.null_matched(base, gt, mask, s_ind, bp, F.rng_for("null_matched", s + (10 if arm == "EH" else 0))))
            if arm == "HH":
                print("label perm seed", s, flush=True)
                ent["F4_label_perm"] = summ(F.null_label_perm(pred, gt, mask, K, bp, F.rng_for("null_perm", s)))
                for f, k in (("F1", "F1_random_top5"), ("F2", "F2_beta_matched"), ("F3", "F3_industry_matched"), ("F4", "F4_label_perm")):
                    per_seed_p[f][s] = ent[k]["p"]
            res["runs"][f"{arm}/{s}"] = ent
            print(arm, s, round(ent["gross"]["sr"], 3), {k: round(ent[k]["p"], 4) for k in ent if k.startswith("F")}, flush=True)
    # seed-averaged series
    for arm in ARMS:
        avg = np.mean([daily[(arm, s)] for s in SEEDS], axis=0)
        daily[(arm, "avg")] = avg
        res[f"{arm}_seedavg"] = {"gross": F.perf(avg), "ci95_sharpe": boot_ci(avg, idx), "annual": annual(dates, avg),
                                 "loyo": leave_one_year_out(dates, avg), "excess_over_hold_all_sr": pooled_sharpe(avg - ha)}
    # F5 HH vs EH (date-joined; identical dates here, join is explicit)
    sr_hh = np.array([res["runs"][f"HH/{s}"]["gross"]["sr"] for s in SEEDS])
    sr_eh = np.array([res["runs"][f"EH/{s}"]["gross"]["sr"] for s in SEEDS])
    _, a, b, da_, db_ = date_join(dates, daily[("HH", "avg")], dates, daily[("EH", "avg")])
    assert da_ == db_ == 0
    f5 = {"per_seed_sr_hh": sr_hh.tolist(), "per_seed_sr_eh": sr_eh.tolist(), "per_seed_diff": (sr_hh - sr_eh).tolist(),
          "wilcoxon_p": wilcoxon_paired(sr_hh, sr_eh), "min_attainable_wilcoxon_p": 2.0 / 2 ** len(SEEDS),
          "seedavg_boot_contrast": boot_contrast(a, b, idx),
          "per_seed_boot_contrast": {str(s): boot_contrast(daily[("HH", s)], daily[("EH", s)], idx) for s in SEEDS}}
    res["F5_HH_vs_EH"] = f5
    fam = {k: max(v.values()) for k, v in per_seed_p.items()}
    fam["F5"] = f5["wilcoxon_p"]
    res["family"] = {"per_seed_p": {k: {str(s): p for s, p in v.items()} for k, v in per_seed_p.items()},
                     "iut_p": fam, "holm": holm_family(fam),
                     "rule": "F1-F4: HH vs null, intersection-union over seeds (max per-seed p); F5: HH vs EH Wilcoxon on per-seed pooled Sharpe; Holm over F1-F5"}
    min_p = len(fam) * 2.0 ** (1 - len(SEEDS))
    res["F5_verdict"] = ("INSUFFICIENT SEEDS" if min_p >= 0.01 else
                         ("NO EVIDENCE" if res["family"]["holm"]["F5"] >= 0.01 else "SEED-ROBUST ONLY/STRONG"))
    res["F5_verdict_note"] = f"repo rule (aggregate.py/_holm): INSUFFICIENT SEEDS when m*2^(1-n)={min_p:.3f} >= 0.01 with n={len(SEEDS)} seeds, m={len(fam)}"
    out = DOCS / ("post2017_results_quick.json" if quick else "post2017_results.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    tables(res, dates, quick)
    print("family", fam, "holm", res["family"]["holm"], res["F5_verdict"])


def tables(res, dates, quick):
    f = lambda x, n=3: f"{x:.{n}f}"
    L = []
    ha = res["hold_all"]
    L += ["### T1. Pooled 2018-2023 gross Sharpe (concatenated daily returns, top-5, stable ties)", "",
          f"Days: {res['n_days']} ({res['dates'][0]} to {res['dates'][1]}). Hold-all (equal weight, same mask): Sharpe {f(ha['gross']['sr'])} (95% block-bootstrap CI {f(ha['ci95'][0])} to {f(ha['ci95'][1])}), mean daily {ha['gross']['mean']*1e4:.2f} bp.", "",
          "| arm | seed | Sharpe | 95% CI (block 10) | excess-over-hold-all mean (bp/day) [95% CI] | excess SR | net 5bp | net 10bp | net 25bp | turnover | IC | NDCG@5 | tie days |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for s in SEEDS:
            e = res["runs"][f"{arm}/{s}"]
            ex = e["excess_over_hold_all"]
            L.append(f"| {arm} | {s} | {f(e['gross']['sr'])} | [{f(e['ci95_sharpe'][0])}, {f(e['ci95_sharpe'][1])}] | {ex['mean_daily']*1e4:.2f} [{ex['mean_ci95'][0]*1e4:.2f}, {ex['mean_ci95'][1]*1e4:.2f}] | {f(ex['sr'])} | {f(e['net_sharpe']['5'])} | {f(e['net_sharpe']['10'])} | {f(e['net_sharpe']['25'])} | {f(e['turnover_mean'],2)} | {f(e['ic'],4)} | {f(e['diag']['ndcg_k'])} | {e['tie_days']} |")
        sa = res[f"{arm}_seedavg"]
        L.append(f"| {arm} | avg of 5 seeds | {f(sa['gross']['sr'])} | [{f(sa['ci95_sharpe'][0])}, {f(sa['ci95_sharpe'][1])}] | | {f(sa['excess_over_hold_all_sr'])} | | | | | | | |")
    yrs = sorted(ha["annual"], key=int)
    L += ["", "### T2. Annual Sharpe (days) and leave-one-year-out pooled Sharpe", "", "| series | " + " | ".join(str(y) for y in yrs) + " | LOYO: " + " / ".join(f"-{y}" for y in yrs) + " |", "|---|" + "---|" * (len(yrs) + 1)]
    def row(name, a, lo):
        return f"| {name} | " + " | ".join(f"{f(a[str(y)]['sr'] if str(y) in a else a[y]['sr'],2)} ({(a[str(y)] if str(y) in a else a[y])['n']})" for y in yrs) + " | " + " / ".join(f(lo[str(y)] if str(y) in lo else lo[y], 2) for y in yrs) + " |"
    L.append(row("hold-all", ha["annual"], ha["loyo"]))
    for arm in ARMS:
        for s in SEEDS:
            e = res["runs"][f"{arm}/{s}"]
            L.append(row(f"{arm} seed {s}", e["annual"], e["loyo"]))
        sa = res[f"{arm}_seedavg"]
        L.append(row(f"{arm} seed-avg", sa["annual"], sa["loyo"]))
    L += ["", "### T3. Eligible names per year (mask used for every series)", "", "| year | mean | min | max |", "|---|---|---|---|"]
    for y, e in res["eligible_by_year"].items():
        L.append(f"| {y} | {e['mean']:.0f} | {e['min']} | {e['max']} |")
    nul = res["null_random_sr_p5_50_95"]
    L += ["", f"### T4. Ranking-skill nulls (one-sided empirical p on pooled Sharpe; B_NULL={res['declared']['B_NULL']}, B_PERM={res['declared']['B_PERM']})", "",
          f"Random daily top-5 null Sharpe 5/50/95 pct: {f(nul[0])} / {f(nul[1])} / {f(nul[2])}.", "",
          "| arm | seed | F1 random top-5 | F2 beta-quintile matched | F3 industry matched | F4 label permutation |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for s in SEEDS:
            e = res["runs"][f"{arm}/{s}"]
            L.append(f"| {arm} | {s} | {f(e['F1_random_top5']['p'])} | {f(e['F2_beta_matched']['p'])} | {f(e['F3_industry_matched']['p'])} | " + (f(e['F4_label_perm']['p']) if 'F4_label_perm' in e else "n/a (HH only)") + " |")
    fam = res["family"]
    f5 = res["F5_HH_vs_EH"]
    L += ["", "### T5. Primary formal family (Holm over 5 tests)", "", "| test | p (IUT over seeds for F1-F4) | Holm-adjusted |", "|---|---|---|"]
    names = {"F1": "F1 HH vs random daily top-5", "F2": "F2 HH vs beta-matched", "F3": "F3 HH vs industry-matched", "F4": "F4 HH vs label permutation", "F5": "F5 HH vs EH (Wilcoxon, per-seed pooled Sharpe)"}
    for k in ("F1", "F2", "F3", "F4", "F5"):
        L.append(f"| {names[k]} | {f(fam['iut_p'][k],4)} | {f(fam['holm'][k],4)} |")
    sb = f5["seedavg_boot_contrast"]
    L += ["", f"HH vs EH: per-seed pooled Sharpe diff (HH-EH) {[round(x,3) for x in f5['per_seed_diff']]}; seed-averaged daily-series contrast {f(sb['est'])} (95% block-bootstrap CI {f(sb['lo'])} to {f(sb['hi'])}, p_boot {f(sb['p_boot'])}). "
          f"Smallest attainable two-sided Wilcoxon p with 5 seeds is {f5['min_attainable_wilcoxon_p']:.4f}. **Verdict word (repo rule): {res['F5_verdict']}.** {res['F5_verdict_note']}.", "",
          "### T6. Diagnostics (top-5 baskets)", "", "| arm | seed | hit top-10% | hit top-20% | miss bottom-10% | precision@5 | basket beta | universe beta |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for s in SEEDS:
            e = res["runs"][f"{arm}/{s}"]
            d = e["diag"]
            L.append(f"| {arm} | {s} | {f(d['hit_top10'])} | {f(d['hit_top20'])} | {f(d['miss_bottom10'])} | {f(d['prec_at_k'],4)} | {f(e['basket_beta_mean'],2)} | {f(e['universe_beta_mean'],2)} |")
    (DOCS / ("post2017_tables_quick.md" if quick else "post2017_tables.md")).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
