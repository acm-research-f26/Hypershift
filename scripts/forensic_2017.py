"""Phase 1.5a forensic pipeline (CPU only). Run from repo root:
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage inventory
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hypershift.eval import forensics as F  # noqa: E402

PRIMARY = ("R5_f2_alpha0_train", "HH")
REFERENCES = (("R5_f_train", "HH"), ("R5_f_train", "EH"), ("R5_f2_alpha0_train", "EH"), ("R5_f_paper", "HH"))
SEEDS = (0, 1, 2, 3, 4)
DOCS = ROOT / "docs" / "phase1_5a"
FIGS = ROOT / "docs" / "figures"
DATA_ROOT = "data/raw/rsr/data"
EVIDENCE_KEYS = ("weight_decay", "input_mode", "alpha", "norm", "spatial_residual", "topk", "seq", "seed")


def stage_inventory(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    dates = [ln.strip()[:10] for ln in open(ROOT / DATA_ROOT / "NYSE_aver_line_dates.csv") if ln.strip()]
    test_dates = [dates[29 + 1008 + j] for j in range(237)]
    assert test_dates[0] == "2017-01-03" and test_dates[-1] == "2017-12-08", (test_dates[0], test_dates[-1])
    from hypershift.data.rsr import read_ticker_file
    tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    assert len(tickers) == 1737
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    inv = {"git_head": head, "test_dates": [test_dates[0], test_dates[-1]], "runs": {}, "n_verified": 0}
    for exp, label in (PRIMARY,) + REFERENCES:
        rows = []
        for s in SEEDS:
            run = f"{exp}/{label}"
            ar = F.load_run(exp, label, s, ROOT / "results")
            r, baskets = F.portfolio(ar.pred, ar.gt, ar.mask)
            diff = float(np.abs(r - ar.daily).max())
            sr = F.perf(r)["sr"]
            be = ar.metrics["best_epoch"]
            if diff > 1e-7 or abs(sr - ar.metrics["test"]["sr"]) > 1e-5 or abs(sr - ar.history[be]["test"]["sr"]) > 1e-5:
                raise SystemExit(f"{run} seed {s}: max|diff|={diff} sr={sr} metrics={ar.metrics['test']['sr']}")
            inv["n_verified"] += 1
            files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ar.path.iterdir())}
            dtype = str(np.load(ar.path / "test_pred.npy", mmap_mode="r").dtype)
            inv["runs"][f"{run}/seed_{s}"] = {
                "sha256": files, "pred_dtype": dtype, "best_epoch": be, "test_oracle_epoch": ar.metrics.get("test_oracle_epoch"),
                "epochs_run": ar.metrics.get("epochs_run"), "config": {k: ar.config.get(k) for k in EVIDENCE_KEYS},
                "sr_recomputed": sr, "max_abs_daily_diff": diff}
            # tidy exports
            b = F.boundary_stats(ar.pred, ar.mask)
            ha = F.hold_all(ar.gt, ar.mask)
            to = F.turnover(baskets)
            recs = []
            for d in range(237):
                idx = np.nonzero(ar.mask[:, d])[0]
                o = idx[np.argsort(-ar.pred[idx, d], kind="stable")]
                rank = np.empty(len(o), int)
                rank[:] = np.arange(1, len(o) + 1)
                top5 = set(baskets[d].tolist())
                recs.append(pd.DataFrame({
                    "date": test_dates[d], "day": d, "ticker": [tickers[i] for i in o], "index": o,
                    "score": ar.pred[o, d], "ret_next": ar.gt[o, d], "rank": rank,
                    "in_top5": [int(i in top5) for i in o], "s5": b["s_k"][d], "s6": b["s_k1"][d],
                    "margin5": b["margin"][d], "exact_tie5": bool(b["exact_tie"][d]), "seed": s}))
            pd.concat(recs).to_csv(out / f"stockday_{exp}_{label}_s{s}.csv.gz", index=False)
            rows.append(pd.DataFrame({"date": test_dates, "day": np.arange(237), "seed": s, "ret_gross": r,
                                      "ret_hold_all": ha, "turnover": to, "exact_tie5": b["exact_tie"],
                                      "margin5": b["margin"], "score_sd": b["sd"], "n_valid": b["n_valid"]}))
        pd.concat(rows).to_csv(out / f"portday_{exp}_{label}.csv", index=False)
    inv["not_recoverable"] = [
        "model weights / checkpoints (none saved; loop.py has no torch.save)",
        "per-stock predictions for non-selected epochs (overwritten; only the last val-improving epoch is on disk)",
        "through-model index permutation of the trained model",
        "frozen-model evaluation outside 2017"]
    (DOCS / "inventory.json").write_text(json.dumps(inv, indent=1))
    print(f"inventory: {inv['n_verified']} run-seeds verified")



# ---------------------------------------------------------------- shared helpers
def _jacc(b1, b2):
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / max(len(set(x.tolist()) | set(y.tolist())), 1)
                          for x, y in zip(b1, b2)]))


def _dist(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std()), "p5": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))}


def _market():
    from hypershift.data.rsr import load_rsr
    return load_rsr(DATA_ROOT, "NYSE", norm="train")


def _edges(cfg_json):
    import dataclasses
    from hypershift.config import RunConfig
    from hypershift.train.loop import base_hypergraph
    names = {f.name for f in dataclasses.fields(RunConfig)}
    kw = {k: v for k, v in cfg_json.items() if k in names}
    kw["sources"] = tuple(kw.get("sources", ()))
    cfg = RunConfig(**kw)
    data = _market()
    return base_hypergraph(cfg, data).edges


def _first_k_perm_rule(gt, mask, k, perms, rng):
    """Constant scores on a randomly relabelled universe: pick the first k valid stocks in permuted order."""
    out = np.zeros((perms, gt.shape[1]))
    for b in range(perms):
        perm = rng.permutation(gt.shape[0])
        m = mask[perm]
        g = gt[perm]
        order = np.argsort(~m, axis=0, kind="stable")[:k]            # first k valid rows per day (valid first, by index)
        valid_cnt = np.minimum(m.sum(0), k)
        vals = np.take_along_axis(g, order, axis=0)
        sel = np.arange(k)[:, None] < valid_cnt[None, :]
        out[b] = (vals * sel).sum(0) / np.maximum(valid_cnt, 1)
    return out


def stage_mechanism(a):
    R_T, R_P, R_FIX = (20, 20, 50) if a.quick else (F.R_TIE, 200, 1000)
    data = _market()
    tdays = data.test_index + np.arange(237)
    close = data.features[:, :, 4]
    runs = [PRIMARY, ("R5_f_train", "HH"), ("R5_f_train", "EH")]
    res = {"quick": bool(a.quick), "runs": {}}
    # relabelled-universe rule is independent of the model (depends only on gt/mask): compute once
    ar0 = F.load_run(*PRIMARY, 0, ROOT / "results")
    fixperm = F.sr_rows(_first_k_perm_rule(ar0.gt, ar0.mask, 5, R_FIX, F.rng_for("index_perm", 77)))
    res["random_fixed_index_basket_rule"] = _dist(fixperm)
    edges_cache = {}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, ROOT / "results")
            ha = F.hold_all(ar.gt, ar.mask)
            r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
            ent = {"hold_all_sr": F.perf(ha)["sr"], "stable": F.perf(r), "turnover_mean": float(F.turnover(base).mean())}
            # random exact-tie order
            dr = F.eps_draws(ar.pred, ar.gt, ar.mask, 5, R_T, F.rng_for("tie", s), eps_abs=0.0)
            srs = F.sr_rows(dr["returns"])
            ent["random_tie"] = {**_dist(srs), "mean_jaccard_vs_stable": float(dr["jaccard"].mean()),
                                 "amb_frac": dr["amb_frac"], "p_sr_ge_stable": float((srs >= ent["stable"]["sr"]).mean()),
                                 "group_size_max": int(dr["group_size"].max())}
            # reverse
            rr, brev = F.portfolio(ar.pred, ar.gt, ar.mask, order="reverse")
            ent["reverse"] = {**F.perf(rr), "jaccard_vs_stable": _jacc(base, brev)}
            # evaluator index permutations (stable tie rule on relabelled universe)
            rng = F.rng_for("index_perm", s)
            ip_sr, ip_j = [], []
            for _ in range(R_P):
                perm = rng.permutation(ar.pred.shape[0])
                p2, g2, m2 = F.evaluator_permutation(ar.pred, ar.gt, ar.mask, perm)
                r2, b2 = F.portfolio(p2, g2, m2)
                ip_sr.append(F.perf(r2)["sr"])
                ip_j.append(_jacc(base, [perm[x] for x in b2]))
            ent["index_perm"] = {**_dist(ip_sr), "mean_jaccard_vs_stable": float(np.mean(ip_j)),
                                 "note": "consistency check vs random_tie, not independent evidence"}
            # constant / first / last 5
            rc, bc = F.portfolio(ar.pred * 0, ar.gt, ar.mask)
            rl, _ = F.portfolio(ar.pred * 0, ar.gt, ar.mask, order="reverse")
            ent["constant_first5"] = {**F.perf(rc), "jaccard_vs_stable": _jacc(base, bc)}
            ent["constant_last5"] = F.perf(rl)
            # tie vs non-tie days
            b = F.boundary_stats(ar.pred, ar.mask)
            tie = b["exact_tie"]
            def _split(m):
                x = r[m]
                if len(x) == 0:
                    return {"n_days": 0}
                return {"n_days": int(m.sum()), "mean": float(x.mean()), "vol_d": float(x.std()), "sum_ret": float(x.sum()),
                        "hit_rate": float((x > 0).mean()), "sr": F.perf(x)["sr"],
                        "hold_all_mean": float(ha[m].mean()), "excess_mean": float((x - ha[m]).mean())}
            ent["tie_days"], ent["nontie_days"] = _split(tie), _split(~tie)
            ent["zero_spread_days"] = int((b["sd"] == 0).sum())
            # tie group composition
            if key not in edges_cache:
                edges_cache[key] = None
            if edges_cache[key] is None:
                edges_cache[key] = _edges(ar.config)
            deg = F.graph_degree(edges_cache[key], ar.pred.shape[0]) 
            grp_sizes, iso, stale, part, vals = [], [], [], [], []
            for d in np.nonzero(tie)[0]:
                idx = np.nonzero(ar.mask[:, d])[0]
                grp = idx[ar.pred[idx, d] == b["s_k"][d]]
                grp_sizes.append(len(grp))
                vals.append(float(b["s_k"][d]))
                t = tdays[d]
                if deg is not None:
                    iso.append(float((deg[grp] == 0).mean()))
                win = close[grp, t - 16:t]
                stale.append(float((np.ptp(win, axis=1) == 0).mean()))
                part.append(float((data.mask[grp, t - 16:t + 1].min(axis=1) < 1).mean()))
            ent["tie_group"] = {
                "n_tie_days": int(tie.sum()), "size_mean": float(np.mean(grp_sizes)) if grp_sizes else 0,
                "size_median": float(np.median(grp_sizes)) if grp_sizes else 0, "size_max": int(max(grp_sizes)) if grp_sizes else 0,
                "frac_isolated": float(np.mean(iso)) if iso else None, "frac_stale_window": float(np.mean(stale)) if stale else None,
                "frac_partly_masked": float(np.mean(part)) if part else None,
                "n_distinct_tied_values": len(set(vals)),
                "top_tied_values": [[v, int(c)] for v, c in __import__("collections").Counter(vals).most_common(5)]}
            res["runs"][key][f"seed_{s}"] = ent
            print(key, s, "stable", round(ent["stable"]["sr"], 3), "rand-tie med", round(ent["random_tie"]["p50"], 3),
                  "rev", round(ent["reverse"]["sr"], 3), "hold", round(ent["hold_all_sr"], 3),
                  "tie days", ent["tie_days"]["n_days"], flush=True)
            res["runs"][key][f"seed_{s}"]["_srs"] = [float(x) for x in srs]
            res["runs"][key][f"seed_{s}"]["_ip"] = [float(x) for x in ip_sr]
    fp = DOCS / "mechanism.json"
    keep = json.loads(json.dumps(res))
    for k in keep["runs"].values():
        for v in k.values():
            v.pop("_srs", None); v.pop("_ip", None)
    fp.write_text(json.dumps(keep, indent=1))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
    for ax, (exp, label) in zip(axs, runs):
        key = f"{exp}/{label}"
        data_v = [res["runs"][key][f"seed_{s}"]["_srs"] for s in SEEDS]
        ax.violinplot(data_v, positions=range(5), showmedians=True)
        ax.scatter(range(5), [res["runs"][key][f"seed_{s}"]["stable"]["sr"] for s in SEEDS], c="r", zorder=3, label="stable (saved)")
        ax.scatter(range(5), [res["runs"][key][f"seed_{s}"]["hold_all_sr"] for s in SEEDS], c="k", marker="_", s=200, zorder=3, label="hold-all")
        ax.set_title(key, fontsize=9); ax.set_xlabel("seed")
    axs[0].set_ylabel("2017 test Sharpe (random exact-tie order)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); FIGS.mkdir(exist_ok=True); fig.savefig(FIGS / "phase1_5a_mechanism.png", dpi=110)
    print("mechanism written")



def _proxy_inputs():
    data = _market()
    tdays = data.test_index + np.arange(237)
    ar0 = F.load_run(*PRIMARY, 0, ROOT / "results")
    edges = _edges(ar0.config)
    feats = F.proxy_features(data.features, data.gt, data.mask, data.valid_index, tdays, edges)
    # features that look back beyond the 16-day input window must not use fill values: NaN them where any day is masked
    t = tdays - 1
    for name, lb in (("ret20", 20), ("vol20", 20), ("ret5", 5), ("ret1", 1)):
        bad = np.stack([(data.mask[:, tt - lb:tt + 1].min(axis=1) < 1) for tt in t], axis=1)
        feats[name] = np.where(bad, np.nan, feats[name])
    return data, tdays, feats


def _r2_per_day(pred, resid, mask):
    out = np.full(pred.shape[1], np.nan)
    for d in range(pred.shape[1]):
        i = mask[:, d] & np.isfinite(resid[:, d])
        if i.sum() > 5 and pred[i, d].var() > 0:
            out[d] = 1 - resid[i, d].var() / pred[i, d].var()
    return out


def stage_proxy(a):
    n_boot = 200 if a.quick else F.N_BOOT
    from hypershift.eval.stats import stationary_bootstrap_indices
    data, tdays, feats = _proxy_inputs()
    runs = [PRIMARY, ("R5_f_train", "HH"), ("R5_f_train", "EH")]
    res = {"quick": bool(a.quick), "features": sorted(feats), "runs": {}}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, ROOT / "results")
            r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
            ha = F.perf(F.hold_all(ar.gt, ar.mask))["sr"]
            rows = {}
            for name, f in feats.items():
                sp = F.daily_spearman(ar.pred, f, ar.mask)
                ok = np.isfinite(sp)
                m = float(np.nanmean(sp)) if ok.any() else float("nan")
                rng = F.rng_for("boot", s)
                bm = []
                spv = sp[ok]
                if len(spv) > 10:
                    for _ in range(n_boot):
                        bm.append(spv[stationary_bootstrap_indices(len(spv), F.BLOCK, rng)].mean())
                rows[name] = {"mean_spearman": m, "n_days": int(ok.sum()),
                              "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))] if bm else None}
            ports = {}
            for name, f in feats.items():
                sg = np.sign(rows[name]["mean_spearman"]) if np.isfinite(rows[name]["mean_spearman"]) else 0
                if sg == 0:
                    continue
                sc = np.where(np.isfinite(f), sg * f, 0.0)
                mk = ar.mask & np.isfinite(f)
                rp, bp = F.portfolio(sc, ar.gt, mk)
                ports[name] = {**F.perf(rp), "sign": float(sg), "jaccard_vs_model": _jacc(base, bp)}
            fitted, resid = F.project_scores(ar.pred, feats, ar.mask)
            r2 = _r2_per_day(ar.pred, resid, ar.mask)
            for nm, sc in (("fitted", fitted), ("resid", resid)):
                mk = ar.mask & np.isfinite(sc)
                rp, bp = F.portfolio(np.where(np.isfinite(sc), sc, 0.0), ar.gt, mk)
                ports[nm] = {**F.perf(rp), "jaccard_vs_model": _jacc(base, bp)}
            ranked = sorted(rows.items(), key=lambda kv: -abs(kv[1]["mean_spearman"]) if np.isfinite(kv[1]["mean_spearman"]) else 0)
            res["runs"][key][f"seed_{s}"] = {
                "model_sr": F.perf(r)["sr"], "hold_all_sr": ha, "spearman_ranked": [[k, v] for k, v in ranked],
                "r2_mean": float(np.nanmean(r2)), "r2_median": float(np.nanmedian(r2)), "portfolios": ports}
            top3 = ", ".join(f"{k}={v['mean_spearman']:.2f}" for k, v in ranked[:3])
            print(key, s, "model", round(F.perf(r)["sr"], 2), "R2", round(float(np.nanmean(r2)), 2), "top:", top3,
                  "| fitted", round(ports["fitted"]["sr"], 2), "resid", round(ports["resid"]["sr"], 2), flush=True)
    (DOCS / "proxy.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prim = res["runs"]["%s/%s" % PRIMARY]
    names = res["features"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.16
    for j, s in enumerate(SEEDS):
        d = {k: v["mean_spearman"] for k, v in prim[f"seed_{s}"]["spearman_ranked"]}
        axs[0].bar(np.arange(len(names)) + (j - 2) * w, [d[n] for n in names], w, label=f"seed {s}")
    axs[0].set_xticks(range(len(names))); axs[0].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    axs[0].set_ylabel("mean daily Spearman(score, feature)"); axs[0].legend(fontsize=7)
    cats = ["model", "hold_all", "fitted", "resid"]
    for j, s in enumerate(SEEDS):
        e = prim[f"seed_{s}"]
        axs[1].bar(np.arange(4) + (j - 2) * w, [e["model_sr"], e["hold_all_sr"], e["portfolios"]["fitted"]["sr"], e["portfolios"]["resid"]["sr"]], w)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(cats); axs[1].set_ylabel("2017 top-5 Sharpe")
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_proxy.png", dpi=110)
    print("proxy written")



def _null_summary(R, ha_mean, obs_r, obs_ha):
    srs = F.sr_rows(R)
    obs = F.perf(obs_r)
    exc = obs["mean"] - ha_mean
    return {"sharpe": {"obs": obs["sr"], "pct": float((srs < obs["sr"]).mean() * 100), "p": F.empirical_p(srs, obs["sr"]),
                       "null_p5_50_95": [float(x) for x in np.percentile(srs, [5, 50, 95])]},
            "mean": {"obs": obs["mean"], "p": F.empirical_p(R.mean(1), obs["mean"])},
            "excess_mean": {"obs": exc, "p": F.empirical_p(R.mean(1) - ha_mean, exc)}}, srs


def stage_nulls(a):
    B_N, B_P, N_BT = (500, 100, 200) if a.quick else (F.B_NULL, F.B_PERM, F.N_BOOT)
    from hypershift.eval.stats import holm, stationary_bootstrap_indices
    from hypershift.data.rsr import read_ticker_file
    data = _market()
    tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    ar0 = F.load_run(*PRIMARY, 0, ROOT / "results")
    gt, mask = ar0.gt, ar0.mask
    ha = F.hold_all(gt, mask)
    res = {"quick": bool(a.quick), "B_NULL": B_N, "B_PERM": B_P, "N_BOOT": N_BT,
           "note": "seeds are repeated runs on the same 237 days; they are not independent samples. "
                   "random_topk is identical to the within-day score permutation null for every seed/arm.",
           "hold_all": F.perf(ha), "seeds": {}}
    R_rand = F.null_random_topk(gt, mask, 5, B_N, F.rng_for("null_random"))
    R_fix = F.null_fixed(gt, mask, 5, B_N, F.rng_for("null_fixed"))
    sr_rand, sr_fix = F.sr_rows(R_rand), F.sr_rows(R_fix)
    res["null_random_topk"] = {"sr_p5_50_95": [float(x) for x in np.percentile(sr_rand, [5, 50, 95])]}
    res["null_fixed"] = {"sr_p5_50_95": [float(x) for x in np.percentile(sr_fix, [5, 50, 95])]}
    ind = F.industry_of(tickers, ROOT / DATA_ROOT / "relation" / "sector_industry" / "NYSE_industry_ticker.json")
    codes = {n: i for i, n in enumerate(sorted(set(ind)))}
    s_ind = np.array([codes[x] for x in ind])
    beta = F.train_beta(data.gt, data.mask, data.valid_index)
    qs = np.nanquantile(beta, [0.2, 0.4, 0.6, 0.8])
    s_beta = np.where(np.isnan(beta), 5, np.digitize(np.nan_to_num(beta), qs))
    res["strata"] = {"n_industries": len(codes), "beta_edges": [float(x) for x in qs], "n_beta_nan": int(np.isnan(beta).sum())}
    # bootstrap indices shared across seeds/stats (paired days)
    boot = {}
    for blk in (F.BLOCK,) + tuple(F.BLOCK_SENS):
        rng = F.rng_for("boot", blk)
        boot[blk] = np.stack([stationary_bootstrap_indices(237, blk, rng) for _ in range(N_BT)])
    pooled = {"label_perm": [], "industry": [], "beta": []}
    per_seed_p = {f: {} for f in ("F1", "F2", "F3", "F4")}
    for s in SEEDS:
        ar = F.load_run(*PRIMARY, s, ROOT / "results")
        r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
        to = F.turnover(base)
        ent = {"observed": F.perf(r), "turnover_mean": float(to.mean()),
               "hold_all_sr": F.perf(ha)["sr"], "excess_over_hold_all_sr": F.perf(r - ha)["sr"]}
        ent["random_topk"], _ = _null_summary(R_rand, ha.mean(), r, ha)
        ent["fixed_basket"], _ = _null_summary(R_fix, ha.mean(), r, ha)
        ent["random_topk"]["mdd_null_median"] = float(np.median([F.perf(x)["mdd"] for x in R_rand[:500]]))
        R_lp = F.null_label_perm(ar.pred, ar.gt, ar.mask, 5, B_P, F.rng_for("null_perm", s))
        ent["label_perm"], sp = _null_summary(R_lp, ha.mean(), r, ha); pooled["label_perm"].append(sp)
        R_i = F.null_matched(base, ar.gt, ar.mask, s_ind, B_P, F.rng_for("null_matched", s))
        ent["industry_matched"], sp = _null_summary(R_i, ha.mean(), r, ha); pooled["industry"].append(sp)
        R_b = F.null_matched(base, ar.gt, ar.mask, s_beta, B_P, F.rng_for("null_matched", 100 + s))
        ent["beta_matched"], sp = _null_summary(R_b, ha.mean(), r, ha); pooled["beta"].append(sp)
        per_seed_p["F1"][s] = ent["random_topk"]["sharpe"]["p"]
        per_seed_p["F2"][s] = ent["beta_matched"]["sharpe"]["p"]
        per_seed_p["F3"][s] = ent["industry_matched"]["sharpe"]["p"]
        per_seed_p["F4"][s] = ent["label_perm"]["sharpe"]["p"]
        # block bootstrap
        bt = {}
        for blk, idx in boot.items():
            sr_b = F.sr_rows(r[idx]); ex_b = F.sr_rows((r - ha)[idx])
            bt[f"block_{blk}"] = {"sharpe_ci95": [float(x) for x in np.percentile(sr_b, [2.5, 97.5])],
                                  "excess_sharpe_ci95": [float(x) for x in np.percentile(ex_b, [2.5, 97.5])]}
        ent["bootstrap"] = bt
        res["seeds"][f"seed_{s}"] = ent
        print("seed", s, "SR", round(ent["observed"]["sr"], 3), "p:", {k: round(v[s], 4) for k, v in per_seed_p.items()}, flush=True)
    fam = {k: max(v.values()) for k, v in per_seed_p.items()}
    res["family"] = {"per_seed_p": {k: {str(s): p for s, p in v.items()} for k, v in per_seed_p.items()},
                     "iut_family_p_max_over_seeds": fam, "holm_adjusted": holm(fam),
                     "tests": {"F1": "vs random daily top-5 (== within-day permutation)", "F2": "vs beta-quintile-matched",
                               "F3": "vs industry-matched", "F4": "vs common label permutation"},
                     "rule": "intersection-union over seeds (max per-seed p), Holm over F1-F4; 2017 is exploratory"}
    print("family p", fam, "holm", res["family"]["holm_adjusted"])
    (DOCS / "nulls.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("random daily top-5 (= within-day perm.)", sr_rand), ("fixed random basket", sr_fix),
              ("common label permutation", np.concatenate(pooled["label_perm"])),
              ("industry-matched", np.concatenate(pooled["industry"])), ("beta-quintile-matched", np.concatenate(pooled["beta"]))]
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.6), sharex=True)
    for ax, (t, x) in zip(axs, panels):
        ax.hist(x, bins=60, color="0.75")
        for s in SEEDS:
            ax.axvline(res["seeds"][f"seed_{s}"]["observed"]["sr"], color="r", lw=1)
        ax.axvline(F.perf(ha)["sr"], color="k", ls="--", lw=1)
        ax.set_title(t, fontsize=8); ax.set_xlabel("Sharpe")
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_nulls.png", dpi=110)
    print("nulls written")



def stage_report(a):
    """Gate A tables + predeclared outcome checks, built only from the committed json files."""
    mech = json.loads((DOCS / "mechanism.json").read_text())
    prox = json.loads((DOCS / "proxy.json").read_text())
    nl = json.loads((DOCS / "nulls.json").read_text())
    pk = "%s/%s" % PRIMARY
    ECON = {"ma5_rel", "ma10_rel", "ma20_rel", "ma30_rel", "ret1", "ret5", "ret20", "vol20", "beta", "close_level"}
    rows1, rows3, o1, o2, o3, o3_econ = [], [], [], [], [], []
    for s in SEEDS:
        m = mech["runs"][pk][f"seed_{s}"]
        ha = m["hold_all_sr"]
        c1 = m["stable"]["sr"] > ha and m["random_tie"]["p50"] <= ha and m["reverse"]["sr"] <= ha
        o1.append(c1)
        n = nl["seeds"][f"seed_{s}"]
        c2 = n["beta_matched"]["sharpe"]["p"] > 0.05 or n["industry_matched"]["sharpe"]["p"] > 0.05
        o2.append(c2)
        p = prox["runs"][pk][f"seed_{s}"]
        port = p["portfolios"]
        thr = 0.8 * p["model_sr"]
        qual = [k for k, v in port.items() if k != "resid" and v["sr"] >= thr]
        qual_econ = [k for k in qual if k in ECON or k == "fitted"]
        rnd_lo, _, rnd_hi = nl["null_random_topk"]["sr_p5_50_95"]
        resid_in = rnd_lo <= port["resid"]["sr"] <= rnd_hi
        o3.append(bool(qual) and resid_in)
        o3_econ.append(bool(qual_econ) and resid_in)
        top = [(k, round(v["mean_spearman"], 2)) for k, v in p["spearman_ranked"][:3]]
        rows1.append(f"| {s} | {m['stable']['sr']:.2f} | {m['random_tie']['p5']:.2f} / {m['random_tie']['p50']:.2f} / {m['random_tie']['p95']:.2f} | "
                     f"{m['reverse']['sr']:.2f} | {ha:.2f} | {m['constant_first5']['sr']:.2f} | {m['tie_days']['n_days']} | {c1} |")
        rows3.append(f"| {s} | {p['model_sr']:.2f} | {top} | {p['r2_mean']:.2f} | {port['fitted']['sr']:.2f} | {port['resid']['sr']:.2f} | "
                     f"{qual} | {resid_in} |")
    rows2 = []
    for s in SEEDS:
        n = nl["seeds"][f"seed_{s}"]
        rows2.append(f"| {s} | {n['observed']['sr']:.2f} | " + " | ".join(
            f"{n[k]['sharpe']['p']:.3f}" for k in ("random_topk", "beta_matched", "industry_matched", "label_perm", "fixed_basket")) +
            f" | {n['excess_over_hold_all_sr']:.2f} | {n['bootstrap']['block_10']['sharpe_ci95'][0]:.2f} to {n['bootstrap']['block_10']['sharpe_ci95'][1]:.2f} |")
    fam = nl["family"]
    out = {"outcome_1_tie_index_decisive": all(o1), "outcome_2_exposure_sufficient": all(o2),
           "outcome_3_factor_tilt_sufficient_literal": all(o3), "outcome_3_economic_proxies_only": all(o3_econ),
           "per_seed": {"o1": o1, "o2": o2, "o3": o3, "o3_econ": o3_econ}}
    (DOCS / "gate_a.json").write_text(json.dumps(out, indent=1))
    md = ["# Phase 1.5a report: 2017 Sharpe near 2 (R5_f2 alpha=0 THINK, HH)", "",
          "Primary: `R5_f2_alpha0_train/HH` seeds 0-4, validation-selected epoch, top-5 equal weight, 237 test days "
          "(2017-01-03 to 2017-12-08). Seeds are repeated runs on the same days, not independent samples. 2017 is exploratory.",
          "", "## Gate A", "",
          f"Hold-all (equal-weight market of valid stocks) Sharpe on the same days: {nl['hold_all']['sr']:.2f} "
          f"(cap-weighted market: UNKNOWN, not in data).", "",
          "### Ties and index order (Task 2)", "",
          "| seed | stable (saved) | random exact-tie order p5 / p50 / p95 | reverse index | hold-all | constant-score first-5 | tie days (of 237) | outcome-1 |",
          "|---|---|---|---|---|---|---|---|", *rows1, "",
          f"Random fixed-index basket rule (1000 relabellings, constant scores): Sharpe mean {mech['random_fixed_index_basket_rule']['mean']:.2f}, "
          f"p5/p50/p95 {mech['random_fixed_index_basket_rule']['p5']:.2f} / {mech['random_fixed_index_basket_rule']['p50']:.2f} / "
          f"{mech['random_fixed_index_basket_rule']['p95']:.2f}.", "",
          "### Nulls (Task 5): per-seed upper-tail empirical p of the Sharpe", "",
          f"B = {nl['B_NULL']} (random top-5, fixed basket), {nl['B_PERM']} (label permutation, matched baskets).", "",
          "| seed | Sharpe | random top-5 (F1) | beta-matched (F2) | industry-matched (F3) | label perm (F4) | fixed basket (not in family) | Sharpe of excess over hold-all | block-10 bootstrap 95% CI |",
          "|---|---|---|---|---|---|---|---|---|", *rows2, "",
          f"Formal family (intersection-union = max per-seed p, then Holm over F1-F4): family p {fam['iut_family_p_max_over_seeds']}, "
          f"Holm {fam['holm_adjusted']}.", "",
          "### Factor proxy (Task 2B)", "",
          "| seed | model Sharpe | top-3 daily Spearman(score, feature) | mean per-day R2 of all-feature fit | fitted portfolio Sharpe | residual portfolio Sharpe | proxies with Sharpe >= 0.8 x model | resid inside central 90% of random null |",
          "|---|---|---|---|---|---|---|---|", *rows3, "",
          "### Predeclared outcomes", "",
          f"1. Tie/index decisive: **{all(o1)}** (per seed {o1}).",
          f"2. Exposure (B) sufficient: **{all(o2)}** (per seed {o2}).",
          f"3. Factor tilt sufficient (literal rule, any proxy incl. fixed-basket 'index' and 'degree'): **{all(o3)}** (per seed {o3}); "
          f"restricted to economic proxies and fitted: **{all(o3_econ)}** (per seed {o3_econ}).",
          "4. Not decisive: " + str(not (all(o1) or all(o2) or all(o3))) + ".", ""]
    (DOCS / "REPORT_2017.md").write_text("\n".join(md))
    print(json.dumps(out))

def stage_integrity(a):
    """Backtest integrity for the uncovered risks only (Task 3). Existing Phase 1.5 A/C tests are re-run, not redone."""
    from hypershift.data.rsr import read_ticker_file
    res = {"primary": "%s/%s" % PRIMARY}
    # 1. re-run existing integrity tests
    pr = subprocess.run([sys.executable, "-m", "pytest", "-m", "data", "tests/test_phase15_eval.py", "tests/test_phase15_data.py", "-q"],
                        capture_output=True, text=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    res["existing_tests"] = {"cmd": "pytest -m data tests/test_phase15_eval.py tests/test_phase15_data.py -q",
                             "returncode": pr.returncode, "tail": pr.stdout.strip().splitlines()[-3:]}
    data = _market()
    tdays = data.test_index + np.arange(237)
    close = data.features[:, :, 4]
    dates = [ln.strip()[:10] for ln in open(ROOT / DATA_ROOT / "NYSE_aver_line_dates.csv") if ln.strip()]
    test_dates = [dates[29 + 1008 + j] for j in range(237)]
    tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    stale = F.stale_runs(close, 3)
    # 3. duplicate series
    dup = F.duplicate_rows(data.features)
    res["duplicate_series"] = {"pairs": [[tickers[i], tickers[j]] for i, j in dup], "n_pairs": len(dup)}
    res["stale_definition"] = "close unchanged for >= 3 consecutive days ending at the last input day (t-1)"
    ext, per_seed = {}, {}
    for s in SEEDS:
        ar = F.load_run(*PRIMARY, s, ROOT / "results")
        r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
        tot = float(r.sum())
        n_sel = n_stale = n_zero = n_fillwin = 0
        stale_pl = zero_pl = 0.0
        for d, b in enumerate(base):
            t = tdays[d]
            for i in b:
                n_sel += 1
                c = ar.gt[i, d] / len(b)
                if stale[i, t - 1]:
                    n_stale += 1; stale_pl += c
                if ar.gt[i, d] == 0.0:
                    n_zero += 1; zero_pl += c
                if data.mask[i, t - 16:t + 1].min() < 1:
                    n_fillwin += 1
                if abs(ar.gt[i, d]) > 0.2:
                    k = (tickers[i], test_dates[d])
                    e = ext.setdefault(k, {"ticker": tickers[i], "date": test_dates[d], "ret": float(ar.gt[i, d]), "seeds": 0, "pl_share_by_seed": {}})
                    e["seeds"] += 1
                    e["pl_share_by_seed"][str(s)] = float(c / tot) if tot else None
        ext_days = sorted({d for d, b in enumerate(base) if any(abs(ar.gt[i, d]) > 0.2 for i in b)})
        keep = np.setdiff1d(np.arange(len(r)), ext_days)
        assert n_fillwin == 0, f"seed {s}: {n_fillwin} selected stock-days with a fill value in their window"
        per_seed[f"seed_{s}"] = {"selected_stock_days": n_sel, "stale_in_basket": n_stale, "stale_pl_share": stale_pl / tot if tot else None,
                                 "exact_zero_return_selected": n_zero, "zero_return_pl_share": zero_pl / tot if tot else None,
                                 "selected_with_fill_in_window": n_fillwin, "total_return_sum": tot,
                                 "retrospective_sharpe_without_days_with_abs_gt_0.2": F.perf(r[keep])["sr"],
                                 "retrospective_hold_all_sharpe_same_days": F.perf(F.hold_all(ar.gt, ar.mask)[keep])["sr"],
                                 "n_days_removed": len(ext_days)}
    # neighbouring-day reversal check for extremes (candidate split/adjustment errors; listed, never deleted)
    for k, e in ext.items():
        i = tickers.index(e["ticker"]); d = test_dates.index(e["date"]); t = tdays[d]
        nxt = data.gt[i, t + 1:t + 4] if t + 1 < data.gt.shape[1] else np.array([])
        prv = data.gt[i, max(t - 3, 0):t]
        e["next3_returns"] = [float(x) for x in nxt]
        e["reversal_gt_50pct_within_3d"] = bool(len(nxt) and (np.cumprod(1 + nxt) - 1).min() * np.sign(e["ret"]) * -1 > 0.5 * abs(e["ret"]) if e["ret"] > 0
                                                else len(nxt) and (np.cumprod(1 + nxt) - 1).max() > 0.5 * abs(e["ret"]))
        e["prev3_returns"] = [float(x) for x in prv]
    res["per_seed"] = per_seed
    res["extreme_selected_stock_days_abs_gt_0.2"] = sorted(ext.values(), key=lambda e: (-e["seeds"], e["date"]))
    res["n_extreme_unique"] = len(ext)
    res["n_extreme_reversal_candidates"] = sum(e["reversal_gt_50pct_within_3d"] for e in ext.values())
    res["metrics_sharpe_convention"] = "metrics.py:50-52 sharpe = mean / np.std(ddof=0) * sqrt(252), no risk-free rate; pinned by tests/test_phase15_eval.py::test_sharpe_matches_authors_up_to_annualisation_constant"
    (DOCS / "integrity.json").write_text(json.dumps(res, indent=1))
    print("integrity:", res["existing_tests"]["tail"][-1] if res["existing_tests"]["tail"] else "", "| dup pairs", len(dup),
          "| extreme", len(ext), "reversal cands", res["n_extreme_reversal_candidates"],
          "| stale in basket", [v["stale_in_basket"] for v in per_seed.values()])



def stage_decompose(a):
    """Task 4, trimmed by the sequential stopping rule (Gate A outcome 2): items 1 persistence, 2 exposure, 5 seed
    concentration, plus gross vs net for the model and hold-all. Items 3 (contribution/exclusion/drop-days) and 4
    (costs/benchmarks beyond model and hold-all) are skipped."""
    from hypershift.data.rsr import read_ticker_file
    data = _market()
    tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    n = len(tickers)
    ind = np.array(F.industry_of(tickers, ROOT / DATA_ROOT / "relation" / "sector_industry" / "NYSE_industry_ticker.json"))
    beta = F.train_beta(data.gt, data.mask, data.valid_index)
    skip = "skipped by the predeclared sequential stopping rule (Gate A outcome 2)"
    res = {"skipped": {"contribution_exclusion_dropdays (T4 item 3)": skip,
                       "costs_benchmarks beyond model and hold-all (T4 item 4)": skip},
           "runs": {}}
    for exp, label in (PRIMARY, ("R5_f_train", "HH"), ("R5_f_train", "EH")):
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        allb = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, ROOT / "results")
            r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
            allb[s] = (ar, r, base)
            fr = F.selection_freq(base, n)
            order = np.argsort(-fr)[:30]
            jac, dur = F.jaccard_series(base), F.durations(base)
            ent = {"persistence": {"top30": [[tickers[i], int(fr[i])] for i in order],
                                   "top_share": {str(t): F.top_share(fr, t) for t in F.TOP_FREQ},
                                   "jaccard_mean": float(jac.mean()), "jaccard_median": float(np.median(jac)),
                                   "duration_mean": float(dur.mean()), "duration_median": float(np.median(dur)), "duration_max": int(dur.max()),
                                   "distinct_stocks_selected": int((fr > 0).sum())}}
            ever = ar.mask.any(axis=1)
            slots = fr / fr.sum()
            rows = []
            for g in np.unique(ind):
                gi = ind == g
                u = gi[ever].sum() / ever.sum()
                rows.append((g, float(slots[gi].sum()), float(u)))
            rows.sort(key=lambda x: -(x[1] - x[2]))
            okb = ~np.isnan(beta)
            bsel = float((fr[okb] * beta[okb]).sum() / max(fr[okb].sum(), 1))
            ent["exposure"] = {"industry_top_over_weights": [{"industry": g, "slot_share": a_, "universe_share": u} for g, a_, u in rows[:5]],
                               "industry_top_under_weights": [{"industry": g, "slot_share": a_, "universe_share": u} for g, a_, u in rows[-3:]],
                               "n_industries_selected": int(sum(1 for g in np.unique(ind) if slots[ind == g].sum() > 0)),
                               "basket_mean_beta": bsel, "universe_mean_beta": float(np.nanmean(beta[ever])),
                               "slot_share_na_industry": float(slots[ind == "n/a"].sum()),
                               "universe_share_na_industry": float((ind == "n/a")[ever].sum() / ever.sum())}
            if (exp, label) == PRIMARY:
                ha = F.hold_all(ar.gt, ar.mask)
                to = F.turnover(base)
                ent["gross_net"] = {"turnover_mean": float(to.mean()),
                                    "hold_all_turnover_convention": "0 (equal-weight hold-all, no rebalancing cost modelled)",
                                    "model": {str(b): F.perf(F.net(r, to, b)) for b in F.COST_BPS},
                                    "hold_all": {str(b): F.perf(F.net(ha, np.zeros_like(ha), b)) for b in F.COST_BPS}}
            res["runs"][key][f"seed_{s}"] = ent
            print(key, s, "jaccard", round(ent["persistence"]["jaccard_mean"], 3), "dur med", ent["persistence"]["duration_median"],
                  "top5 share", round(ent["persistence"]["top_share"]["5"], 3), "distinct", ent["persistence"]["distinct_stocks_selected"],
                  "beta", round(ent["exposure"]["basket_mean_beta"], 2), "vs", round(ent["exposure"]["universe_mean_beta"], 2), flush=True)
        pair, corr = [], []
        top10 = [set(np.argsort(-F.selection_freq(allb[s][2], n))[:10].tolist()) for s in SEEDS]
        for i in range(5):
            for j in range(i + 1, 5):
                pair.append(_jacc(allb[SEEDS[i]][2], allb[SEEDS[j]][2]))
                corr.append(float(np.corrcoef(allb[SEEDS[i]][1], allb[SEEDS[j]][1])[0, 1]))
        cnt = {}
        for t in top10:
            for x in t:
                cnt[x] = cnt.get(x, 0) + 1
        res["runs"][key]["seed_concentration"] = {
            "pairwise_daily_basket_jaccard_mean": float(np.mean(pair)), "pairwise_jaccard_min_max": [float(min(pair)), float(max(pair))],
            "pairwise_daily_return_corr_mean": float(np.mean(corr)), "pairwise_corr_min_max": [float(min(corr)), float(max(corr))],
            "stocks_in_top10_freq_of_ge3_seeds": sorted([tickers[i] for i, c in cnt.items() if c >= 3]),
            "note": "agreement across seeds under the same data/ordering/backtester is not independent evidence"}
        sc = res["runs"][key]["seed_concentration"]
        print(key, "seed jaccard", round(sc["pairwise_daily_basket_jaccard_mean"], 3), "ret corr", round(sc["pairwise_daily_return_corr_mean"], 3),
              "shared top10:", sc["stocks_in_top10_freq_of_ge3_seeds"], flush=True)
    (DOCS / "decompose.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pk = "%s/%s" % PRIMARY
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for s in SEEDS:
        axs[0].plot(np.arange(1, 31), [c for _, c in res["runs"][pk][f"seed_{s}"]["persistence"]["top30"]], label=f"seed {s}")
    axs[0].set_xlabel("stock rank by selection count"); axs[0].set_ylabel("days selected (of 237)"); axs[0].legend(fontsize=7)
    for s in SEEDS:
        axs[1].plot(F.COST_BPS, [res["runs"][pk][f"seed_{s}"]["gross_net"]["model"][str(b)]["sr"] for b in F.COST_BPS], marker="o", label=f"seed {s}")
    axs[1].plot(F.COST_BPS, [res["runs"][pk]["seed_0"]["gross_net"]["hold_all"][str(b)]["sr"] for b in F.COST_BPS], "k--", label="hold-all")
    axs[1].set_xlabel("cost, bp per side"); axs[1].set_ylabel("Sharpe"); axs[1].legend(fontsize=7)
    axs[2].bar(range(5), [res["runs"][pk][f"seed_{s}"]["persistence"]["jaccard_mean"] for s in SEEDS])
    axs[2].set_xlabel("seed"); axs[2].set_ylabel("mean day-to-day Jaccard")
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_persistence.png", dpi=110)
    print("decompose written")


#@@STAGES@@
STAGES = {"inventory": stage_inventory, "mechanism": stage_mechanism, "proxy": stage_proxy, "nulls": stage_nulls, "integrity": stage_integrity, "decompose": stage_decompose, "report": stage_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--out", default="results/forensics_1_5a")
    ap.add_argument("--quick", action="store_true", help="debug only: tiny B/R")
    a = ap.parse_args()
    os.chdir(ROOT)
    for name in (["inventory", "mechanism", "proxy", "nulls", "report"] if a.stage == "all" else [a.stage]):
        STAGES[name](a)


if __name__ == "__main__":
    main()
