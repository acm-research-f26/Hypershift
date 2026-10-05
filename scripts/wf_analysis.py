"""Phase 1.5c: walk-forward (annual retraining) analysis, 2019-2023 (CPU only). Spec: docs/phase1_5c/SPEC.md (binding).

Reads results/WF_<year>_alpha0/<arm>/seed_k (test_pred/gt/mask/daily, config.json, metrics.json) and the frozen 1.5b outputs
results/post2017_frozen/HH (date-keyed). Writes docs/phase1_5c/wf_results.json + wf_tables.md. Reuses post2017_analysis helpers.
Per seed: the five yearly daily-return series are concatenated by explicit date join (never a mean of annual Sharpes).
Nulls are drawn per yearly window (each window is a different model, train-period beta per window) and concatenated by date.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypershift.eval.forensics as F
from hypershift.data.alpaca_wf import PANEL_NAME, split_indices
from hypershift.eval.metrics import daily_ic
from hypershift.eval.stats import wilcoxon_paired
from post2017_analysis import annual, boot_ci, boot_contrast, boot_indices, date_join, holm_family, pooled_sharpe, sr_rows

B_NULL, B_PERM, N_BOOT, BLOCK, K = 2000, 500, 5000, 10, 5
COSTS = (0, 5, 10, 25)
SEEDS = range(5)
YEARS = (2019, 2020, 2021, 2022, 2023)
RESULTS = Path("results")
DOCS = Path("docs/phase1_5c")
RSR = Path("data/raw/rsr/data")


def year_slices(dates, years=YEARS):
    """Column slice of the panel calendar for each test year: exactly the trading days of that calendar year."""
    out = {}
    for y in years:
        _, ti, end = split_indices(dates, y)
        out[y] = (ti, end)
        yy = np.asarray(dates[ti:end]).astype("datetime64[Y]").astype(int) + 1970
        assert (yy == y).all() and len(yy) > 240, f"test slice for {y} is not exactly that year"
    return out


def concat_by_date(parts):
    """parts: list of (dates, array[..., D]) with disjoint, increasing dates. Returns (dates, array) concatenated on the last axis."""
    dts = np.concatenate([np.asarray(d) for d, _ in parts])
    if len(set(dts.tolist())) != len(dts) or (np.diff(dts.astype("datetime64[D]").astype(int)) <= 0).any():
        raise ValueError("overlapping or non-increasing dates")
    return dts, np.concatenate([np.asarray(a) for _, a in parts], axis=-1)


def verdict_word(p_holm, lo, hi, n_seeds, m_tests):
    """CLAUDE.md verdict words. INSUFFICIENT SEEDS when the smallest attainable Holm p (m*2^(1-n)) cannot reach 0.01 (as in aggregate.py/_holm)."""
    if m_tests * 2.0 ** (1 - n_seeds) >= 0.01:
        return "INSUFFICIENT SEEDS"
    if p_holm < 0.01:
        return "STRONG" if (lo > 0 or hi < 0) else "SEED-ROBUST ONLY"
    return "NO EVIDENCE"


def concat_nulls(parts):
    return np.concatenate(parts, axis=1)


def main(quick=False):
    bn, bp, nb = (50, 20, 200) if quick else (B_NULL, B_PERM, N_BOOT)
    z = np.load(RSR / PANEL_NAME, allow_pickle=False)
    pdates, gt_full, mask_full = z["dates"], z["gt"].astype(np.float64), z["mask"] > 0
    tickers = [str(t) for t in z["tickers"]]
    sl = year_slices(pdates)
    ind = F.industry_of(tickers, RSR / "relation" / "sector_industry" / "NYSE_industry_ticker.json")
    codes = {n: i for i, n in enumerate(sorted(set(ind)))}
    s_ind = np.array([codes[x] for x in ind])
    # per-window beta from its own training period (targets < valid_index) -> quintile strata
    beta_y, s_beta_y, checks = {}, {}, {}
    for y, (ti, end) in sl.items():
        vi, ti2, _ = split_indices(pdates, y)
        assert ti2 == ti
        b = F.train_beta(gt_full[:, :end], mask_full[:, :end].astype(float), vi)
        qs = np.nanquantile(b, [0.2, 0.4, 0.6, 0.8])
        beta_y[y], s_beta_y[y] = b, np.where(np.isnan(b), 5, np.digitize(np.nan_to_num(b), qs))
        checks[y] = {"valid_index": vi, "test_index": ti, "end": end, "val_dates": [str(pdates[vi]), str(pdates[ti - 1])],
                     "test_dates": [str(pdates[ti]), str(pdates[end - 1])], "n_test": end - ti, "n_beta_nan": int(np.isnan(b).sum())}
    # ---- Task 1: verify runs / dates / splits
    def load_runs(arm):
        runs = {}
        for y in YEARS:
            ti, end = sl[y]
            for s in SEEDS:
                d = RESULTS / f"WF_{y}_alpha0" / arm / f"seed_{s}"
                assert (d / "metrics.json").exists() and not (d / "failed.json").exists() and (d / "best_state.pt").exists(), d
                cfg = json.loads((d / "config.json").read_text())
                assert cfg["wf_test_year"] == y and cfg["seed"] == s and cfg["alpha"] == 0 and cfg["weight_decay"] == 0, cfg
                pred, gt, mask = (np.load(d / f"test_{n}.npy").astype(np.float64) for n in ("pred", "gt", "mask"))
                mask = mask > 0.5
                assert pred.shape[1] == end - ti and np.array_equal(gt, gt_full[:, ti:end]) and (mask == mask_full[:, ti:end]).all(), (arm, y, s)
                m = json.loads((d / "metrics.json").read_text())
                runs[(y, s)] = (pred, np.load(d / "test_daily.npy"), m)
        return runs

    runs_by_arm = {"HH": load_runs("HH"), "EH": load_runs("EH")}
    for (y, s_) in [(y, s_) for y in YEARS for s_ in SEEDS]:
        for arm, tmp in (("HH", "hyp"), ("EH", "euc")):
            c_ = json.loads((RESULTS / f"WF_{y}_alpha0" / arm / f"seed_{s_}" / "config.json").read_text())
            assert (c_["temporal"], c_["spatial"], c_["structure"], c_["model"]) == (tmp, "hyp", "hyper", "think"), (arm, y, s_)
    dts, gt_c = concat_by_date([(pdates[sl[y][0]:sl[y][1]], gt_full[:, sl[y][0]:sl[y][1]]) for y in YEARS])
    mask_c = np.concatenate([mask_full[:, sl[y][0]:sl[y][1]] for y in YEARS], axis=1)
    D = len(dts)
    fz_dates = np.load("results/post2017_frozen/dates.npy")
    assert set(dts.tolist()) <= set(fz_dates.tolist()), "WF calendar must be a subset of the 1.5b calendar"
    yrs = dts.astype("datetime64[Y]").astype(int) + 1970
    # ---- series
    ha = F.hold_all(gt_c, mask_c)
    idx = boot_indices(D, nb, BLOCK, F.rng_for("boot", 177))
    res = {"declared": {"B_NULL": bn, "B_PERM": bp, "N_BOOT": nb, "block": BLOCK, "k": K, "quick": quick}, "n_days": D,
           "dates": [str(dts[0]), str(dts[-1])], "window_checks": checks, "n_industries": len(codes),
           "hold_all": {"gross": F.perf(ha), "annual": annual(dts, ha), "ci95": boot_ci(ha, idx)}}
    res["eligible_by_year"] = {y: {"mean": float(mask_c[:, yrs == y].sum(0).mean()), "min": int(mask_c[:, yrs == y].sum(0).min())} for y in YEARS}
    print("null random top-5 ...", flush=True)
    R_rand = F.null_random_topk(gt_c, mask_c, K, bn, F.rng_for("null_random", 177))
    res["null_random_sr_p5_50_95"] = [float(x) for x in np.percentile(sr_rows(R_rand), [5, 50, 95])]
    def arm_loop(arm, runs):
        daily, per_seed_p, out_runs = {}, {f: {} for f in ("F1", "F2", "F3", "F4")}, {}
        for s in SEEDS:
            parts_r, parts_b, parts_p, parts_bb, parts_ub = [], [], [], [], []
            n2, n3, n4 = [], [], []
            for y in YEARS:
                ti, end = sl[y]
                pred, saved, m = runs[(y, s)]
                gt, mask = gt_full[:, ti:end], mask_full[:, ti:end]
                r, base = F.portfolio(pred, gt, mask, K)
                assert np.abs(r - saved).max() < 1e-6, "stable-tie portfolio must reproduce the saved daily returns"
                parts_r.append(r); parts_b += list(base); parts_p.append(pred)
                parts_bb += [np.nanmean(beta_y[y][b]) if len(b) else np.nan for b in base]
                parts_ub += [np.nanmean(beta_y[y][mask[:, d]]) for d in range(mask.shape[1])]
                n2.append(F.null_matched(base, gt, mask, s_beta_y[y], bp, F.rng_for("null_matched", 100 + s + 10 * (y - 2018))))
                n3.append(F.null_matched(base, gt, mask, s_ind, bp, F.rng_for("null_matched", s + 10 * (y - 2018))))
                n4.append(F.null_label_perm(pred, gt, mask, K, bp, F.rng_for("null_perm", s + 10 * (y - 2018))))
            r = np.concatenate(parts_r)
            pred = np.concatenate(parts_p, axis=1)
            daily[s] = r
            to = F.turnover(parts_b)
            ent = {"gross": F.perf(r), "ci95_sharpe": boot_ci(r, idx), "annual": annual(dts, r),
                   "excess_over_hold_all": {"mean_daily": float((r - ha).mean()), "sr": pooled_sharpe(r - ha),
                                            "mean_ci95": [float(x) for x in np.percentile((r - ha)[idx].mean(1), [2.5, 97.5])]},
                   "net_sharpe": {str(c): pooled_sharpe(F.net(r, to, c)) for c in COSTS}, "turnover_mean": float(to.mean()),
                   "ic": float(daily_ic(pred, gt_c, mask_c)),
                   "ic_by_year": {y: float(daily_ic(pred[:, yrs == y], gt_c[:, yrs == y], mask_c[:, yrs == y])) for y in YEARS},
                   "basket_beta_mean": float(np.nanmean(parts_bb)), "universe_beta_mean": float(np.nanmean(parts_ub)),
                   "selected_epoch": {y: runs[(y, s)][2]["best_epoch"] for y in YEARS},
                   "val_sr": {y: runs[(y, s)][2]["val"]["sr"] for y in YEARS},
                   "test_sr_year": {y: runs[(y, s)][2]["test"]["sr"] for y in YEARS}}
            ent["diag"] = {k: float(v) for k, v in F.topk_diag(pred, gt_c, mask_c, K).items()}

            def summ(R, ent=ent):
                sr_ = sr_rows(R)
                return {"p": F.empirical_p(sr_, ent["gross"]["sr"]), "null_p5_50_95": [float(x) for x in np.percentile(sr_, [5, 50, 95])]}
            ent["F1_random_top5"], ent["F2_beta_matched"] = summ(R_rand), summ(concat_nulls(n2))
            ent["F3_industry_matched"], ent["F4_label_perm"] = summ(concat_nulls(n3)), summ(concat_nulls(n4))
            for f, k in (("F1", "F1_random_top5"), ("F2", "F2_beta_matched"), ("F3", "F3_industry_matched"), ("F4", "F4_label_perm")):
                per_seed_p[f][s] = ent[k]["p"]
            out_runs[str(s)] = ent
            print(s, round(ent["gross"]["sr"], 3), {k: round(ent[k]["p"], 4) for k in ent if k.startswith("F")}, flush=True)
        avg = np.mean([daily[s] for s in SEEDS], axis=0)
        sa = {"gross": F.perf(avg), "ci95_sharpe": boot_ci(avg, idx), "annual": annual(dts, avg),
              "excess_over_hold_all_sr": pooled_sharpe(avg - ha)}
        return daily, per_seed_p, out_runs, avg, sa

    daily, per_seed_p, res["runs"], avg, res["seedavg"] = arm_loop("HH", runs_by_arm["HH"])
    daily_eh, per_seed_p_eh, res["runs_EH"], avg_eh, res["seedavg_EH"] = arm_loop("EH", runs_by_arm["EH"])
    # ---- family: F1-F4 (HH vs nulls, IUT over seeds) + F5 (HH vs EH, Wilcoxon on per-seed pooled Sharpes)
    iut = {k: max(v.values()) for k, v in per_seed_p.items()}
    sr_hh = np.array([res["runs"][str(s)]["gross"]["sr"] for s in SEEDS])
    sr_eh = np.array([res["runs_EH"][str(s)]["gross"]["sr"] for s in SEEDS])
    _, a_, b_, da_, db_ = date_join(dts, avg, dts, avg_eh)
    assert da_ == db_ == 0
    f5 = {"per_seed_sr_hh": sr_hh.tolist(), "per_seed_sr_eh": sr_eh.tolist(), "per_seed_diff_hh_minus_eh": (sr_hh - sr_eh).tolist(),
          "wilcoxon_p": wilcoxon_paired(sr_hh, sr_eh), "min_attainable_wilcoxon_p": 2.0 / 2 ** len(SEEDS),
          "seedavg_boot_contrast": boot_contrast(a_, b_, idx),
          "per_seed_boot_contrast": {str(s): boot_contrast(daily[s], daily_eh[s], idx) for s in SEEDS}}
    res["F5_HH_vs_EH"] = f5
    fam = {**iut, "F5": f5["wilcoxon_p"]}
    holm5 = holm_family(fam)
    min_p = len(fam) * 2.0 ** (1 - len(SEEDS))
    res["family"] = {"per_seed_p": {k: {str(s): p for s, p in v.items()} for k, v in per_seed_p.items()}, "iut_p": fam, "holm": holm5,
                     "rule": "F1-F4: HH vs null, intersection-union over seeds (max per-seed p); F5: HH vs EH Wilcoxon on per-seed pooled Sharpe; Holm over F1-F5"}
    res["F5_verdict"] = verdict_word(holm5["F5"], f5["seedavg_boot_contrast"]["lo"], f5["seedavg_boot_contrast"]["hi"], len(SEEDS), len(fam))
    res["F5_verdict_note"] = f"repo rule (aggregate.py/_holm): INSUFFICIENT SEEDS when m*2^(1-n)={min_p:.3f} >= 0.01 with n={len(SEEDS)} seeds, m={len(fam)}"
    # ---- walk-forward vs frozen 1.5b on the same 2019-2023 dates (per arm)
    pos = {d: i for i, d in enumerate(fz_dates.tolist())}
    cols = np.array([pos[d] for d in dts.tolist()])

    def vs_frozen(arm, dly, av):
        fz = {}
        for s in SEEDS:
            d = Path("results/post2017_frozen") / arm / f"seed_{s}"
            pf, gf, mf, sd = (np.load(d / f"test_{n}.npy") for n in ("pred", "gt", "mask", "daily"))
            pf, gf, mf, sd = pf[:, cols].astype(np.float64), gf[:, cols].astype(np.float64), mf[:, cols] > 0.5, sd[cols]
            assert (mf == mask_c).all() and np.array_equal(gf, gt_c), "frozen and WF mask/gt must match on shared days"
            r, _ = F.portfolio(pf, gt_c, mask_c, K)
            assert np.abs(r - sd).max() < 1e-6
            fz[s] = (r, float(daily_ic(pf, gt_c, mask_c)))
        fz_avg = np.mean([fz[s][0] for s in SEEDS], axis=0)
        wf_ic = [res["runs" if arm == "HH" else "runs_EH"][str(s)]["ic"] for s in SEEDS]
        return {
            "frozen_pooled_sr": {str(s): pooled_sharpe(fz[s][0]) for s in SEEDS}, "frozen_ic": {str(s): fz[s][1] for s in SEEDS},
            "frozen_seedavg_sr": pooled_sharpe(fz_avg), "wf_seedavg_sr": pooled_sharpe(av), "frozen_annual_seedavg": annual(dts, fz_avg),
            "seedavg_contrast_wf_minus_frozen": boot_contrast(av, fz_avg, idx),
            "per_seed_contrast_wf_minus_frozen": {str(s): boot_contrast(dly[s], fz[s][0], idx) for s in SEEDS},
            "mean_daily_diff_bp_seedavg": float((av - fz_avg).mean() * 1e4),
            "wf_ic_mean": float(np.mean(wf_ic)), "frozen_ic_mean": float(np.mean([fz[s][1] for s in SEEDS]))}
    res["wf_vs_frozen"] = vs_frozen("HH", daily, avg)
    res["wf_vs_frozen_EH"] = vs_frozen("EH", daily_eh, avg_eh)
    out = DOCS / ("wf_results_quick.json" if quick else "wf_results.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    tables(res, quick)
    print("iut", fam, "holm5", holm5, res["F5_verdict"])


def tables(res, quick):
    f = lambda x, n=3: f"{x:.{n}f}"
    ha, L = res["hold_all"], []
    ARMS = (("HH", "runs", "seedavg", "wf_vs_frozen"), ("EH", "runs_EH", "seedavg_EH", "wf_vs_frozen_EH"))
    yrs = sorted(ha["annual"], key=int)
    for arm, rk, sk, _ in ARMS:
        sa = res[sk]
        L += [f"### T1{arm}. Pooled 2019-2023 gross Sharpe, walk-forward {arm} (concatenated daily top-5 returns, stable ties)", "",
              f"Days: {res['n_days']} ({res['dates'][0]} to {res['dates'][1]}). Hold-all (same mask): Sharpe {f(ha['gross']['sr'])} (95% CI {f(ha['ci95'][0])} to {f(ha['ci95'][1])}), mean daily {ha['gross']['mean']*1e4:.2f} bp.", "",
              "| seed | Sharpe | 95% CI (block 10) | excess-over-hold-all mean bp/day [95% CI] | excess SR | net 5bp | net 10bp | net 25bp | turnover | IC | NDCG@5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in range(5):
            e = res[rk][str(s)]; ex = e["excess_over_hold_all"]
            L.append(f"| {s} | {f(e['gross']['sr'])} | [{f(e['ci95_sharpe'][0])}, {f(e['ci95_sharpe'][1])}] | {ex['mean_daily']*1e4:.2f} [{ex['mean_ci95'][0]*1e4:.2f}, {ex['mean_ci95'][1]*1e4:.2f}] | {f(ex['sr'])} | {f(e['net_sharpe']['5'])} | {f(e['net_sharpe']['10'])} | {f(e['net_sharpe']['25'])} | {f(e['turnover_mean'],2)} | {f(e['ic'],4)} | {f(e['diag']['ndcg_k'])} |")
        mean_sr = np.mean([res[rk][str(s)]["gross"]["sr"] for s in range(5)])
        L.append(f"| mean of 5 seeds | {f(mean_sr)} | | | | | | | | | |")
        L.append(f"| seed-avg series | {f(sa['gross']['sr'])} | [{f(sa['ci95_sharpe'][0])}, {f(sa['ci95_sharpe'][1])}] | | {f(sa['excess_over_hold_all_sr'])} | | | | | | |")
        L.append("")
    L += ["### T2. Annual Sharpe (days); walk-forward HH and EH, hold-all, frozen 1.5b seed-avg", "", "| series | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
    row = lambda n, a: f"| {n} | " + " | ".join(f"{f(a[y]['sr'],2)} ({a[y]['n']})" for y in yrs) + " |"
    L.append(row("hold-all", ha["annual"]))
    for arm, rk, sk, fk in ARMS:
        for s in range(5):
            L.append(row(f"WF {arm} seed {s}", res[rk][str(s)]["annual"]))
        L.append(row(f"WF {arm} seed-avg", res[sk]["annual"]))
        L.append(row(f"frozen 1.5b {arm} seed-avg", res[fk]["frozen_annual_seedavg"]))
    for arm, rk, _, _ in ARMS:
        L += ["", f"### T3{arm}. {arm}: selected epoch / validation Sharpe / that window's test Sharpe, per seed and test year", "", "| seed | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
        for s in range(5):
            e = res[rk][str(s)]
            L.append(f"| {s} | " + " | ".join(f"ep {e['selected_epoch'][y]} / {f(e['val_sr'][y],2)} / {f(e['test_sr_year'][y],2)}" for y in yrs) + " |")
    L += ["", "### T4. Eligible names per year", "", "| year | mean | min |", "|---|---|---|"] + [f"| {y} | {e['mean']:.0f} | {e['min']} |" for y, e in res["eligible_by_year"].items()]
    nul = res["null_random_sr_p5_50_95"]
    L += ["", f"### T5. Ranking-skill nulls (one-sided empirical p on pooled Sharpe; B_NULL={res['declared']['B_NULL']}, B_PERM={res['declared']['B_PERM']}); EH rows descriptive only (not in the family)", "",
          f"Random daily top-5 null Sharpe 5/50/95 pct: {f(nul[0])} / {f(nul[1])} / {f(nul[2])}.", "",
          "| arm | seed | F1 random top-5 | F2 beta-quintile matched (per-window beta) | F3 industry matched | F4 label permutation |", "|---|---|---|---|---|---|"]
    for arm, rk, _, _ in ARMS:
        for s in range(5):
            e = res[rk][str(s)]
            L.append(f"| {arm} | {s} | {f(e['F1_random_top5']['p'])} | {f(e['F2_beta_matched']['p'])} | {f(e['F3_industry_matched']['p'])} | {f(e['F4_label_perm']['p'])} |")
    fam, f5 = res["family"], res["F5_HH_vs_EH"]
    L += ["", "### T6. Primary formal family (F1-F4 IUT over seeds; F5 Wilcoxon), Holm over 5", "", "| test | p | Holm over 5 |", "|---|---|---|"]
    nm = {"F1": "F1 HH vs random daily top-5", "F2": "F2 HH vs beta-matched", "F3": "F3 HH vs industry-matched", "F4": "F4 HH vs label permutation", "F5": "F5 HH vs EH (Wilcoxon, per-seed pooled Sharpe)"}
    for k in nm:
        L.append(f"| {nm[k]} | {f(fam['iut_p'][k],4)} | {f(fam['holm'][k],4)} |")
    sc5 = f5["seedavg_boot_contrast"]
    L += ["", "### T9. F5 HH vs EH", "",
          f"Per-seed pooled Sharpe HH {[round(x,3) for x in f5['per_seed_sr_hh']]}; EH {[round(x,3) for x in f5['per_seed_sr_eh']]}; diff HH-EH {[round(x,3) for x in f5['per_seed_diff_hh_minus_eh']]}.", "",
          f"Wilcoxon two-sided p = {f(f5['wilcoxon_p'],4)} (smallest attainable with 5 seeds: {f(f5['min_attainable_wilcoxon_p'],4)}). Seed-averaged daily contrast HH-EH: {f(sc5['est'])} [{f(sc5['lo'])}, {f(sc5['hi'])}], p_boot {f(sc5['p_boot'])} (block 10, N_BOOT {res['declared']['N_BOOT']}). Holm-adjusted p {f(fam['holm']['F5'],4)}.", "",
          f"**Verdict word: {res['F5_verdict']}.** {res['F5_verdict_note']}.", "",
          "| seed | HH-EH Sharpe diff [95% block-bootstrap CI] |", "|---|---|"]
    for s in range(5):
        pc = f5["per_seed_boot_contrast"][str(s)]
        L.append(f"| {s} | {f(pc['est'])} [{f(pc['lo'])}, {f(pc['hi'])}] |")
    for arm, rk, _, fk in ARMS:
        c = res[fk]; sc = c["seedavg_contrast_wf_minus_frozen"]
        L += ["", f"### T7{arm}. Walk-forward vs frozen 1.5b {arm}, same 2019-2023 days (date-joined; mask and gt identical)", "",
              "| seed | WF Sharpe | frozen Sharpe | diff (WF-frozen) [95% block-bootstrap CI] | WF IC | frozen IC |", "|---|---|---|---|---|---|"]
        for s in range(5):
            pc = c["per_seed_contrast_wf_minus_frozen"][str(s)]; e = res[rk][str(s)]
            L.append(f"| {s} | {f(e['gross']['sr'])} | {f(c['frozen_pooled_sr'][str(s)])} | {f(pc['est'])} [{f(pc['lo'])}, {f(pc['hi'])}] | {f(e['ic'],4)} | {f(c['frozen_ic'][str(s)],4)} |")
        L.append(f"| seed-avg series | {f(c['wf_seedavg_sr'])} | {f(c['frozen_seedavg_sr'])} | {f(sc['est'])} [{f(sc['lo'])}, {f(sc['hi'])}], p_boot {f(sc['p_boot'])} | {f(c['wf_ic_mean'],4)} (mean) | {f(c['frozen_ic_mean'],4)} (mean) |")
    for arm, rk, _, _ in ARMS:
        L += ["", f"### T8{arm}. Diagnostics (top-5 baskets), {arm}", "", "| seed | hit top-10% | hit top-20% | miss bottom-10% | precision@5 | basket beta | universe beta |", "|---|---|---|---|---|---|---|"]
        for s in range(5):
            e = res[rk][str(s)]; d = e["diag"]
            L.append(f"| {s} | {f(d['hit_top10'])} | {f(d['hit_top20'])} | {f(d['miss_bottom10'])} | {f(d['prec_at_k'],4)} | {f(e['basket_beta_mean'],2)} | {f(e['universe_beta_mean'],2)} |")
    (DOCS / ("wf_tables_quick.md" if quick else "wf_tables.md")).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
