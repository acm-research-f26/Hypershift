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
FIXTURE = False
REFERENCES = (("R5_f_train", "HH"), ("R5_f_train", "EH"), ("R5_f2_alpha0_train", "EH"), ("R5_f_paper", "HH"))
SEEDS = (0, 1, 2, 3, 4)
DOCS = ROOT / "docs" / "phase1_5a"
FIGS = ROOT / "docs" / "figures"
DATA_ROOT = "data/raw/rsr/data"
RESULTS = ROOT / "results"
REF_RUNS = (("R5_f_train", "HH"), ("R5_f_train", "EH"))
EVIDENCE_KEYS = ("weight_decay", "input_mode", "alpha", "norm", "spatial_residual", "topk", "seq", "seed")


def stage_inventory(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if FIXTURE:
        ar_ = F.load_run(*PRIMARY, SEEDS[0], RESULTS)
        test_dates = [f"fixture_day_{j}" for j in range(ar_.pred.shape[1])]
        tickers = [f"S{i}" for i in range(ar_.pred.shape[0])]
    else:
        dates = [ln.strip()[:10] for ln in open(ROOT / DATA_ROOT / "NYSE_aver_line_dates.csv") if ln.strip()]
        test_dates = [dates[29 + 1008 + j] for j in range(237)]
        assert test_dates[0] == "2017-01-03" and test_dates[-1] == "2017-12-08", (test_dates[0], test_dates[-1])
        from hypershift.data.rsr import read_ticker_file
        tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
        assert len(tickers) == 1737
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    inv = {"git_head": head, "test_dates": [test_dates[0], test_dates[-1]], "runs": {}, "n_verified": 0, "fixture": FIXTURE}
    for exp, label in (PRIMARY,) + REFERENCES:
        rows = []
        for s in SEEDS:
            run = f"{exp}/{label}"
            ar = F.load_run(exp, label, s, RESULTS)
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
            for d in range(len(test_dates)):
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
            rows.append(pd.DataFrame({"date": test_dates, "day": np.arange(len(test_dates)), "seed": s, "ret_gross": r,
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
    if not FIXTURE:
        data = _market()
        tdays = data.test_index + np.arange(237)
        close = data.features[:, :, 4]
    runs = [PRIMARY] + list(REF_RUNS)
    res = {"fixture": FIXTURE, "quick": bool(a.quick), "runs": {}}
    # relabelled-universe rule is independent of the model (depends only on gt/mask): compute once
    ar0 = F.load_run(*PRIMARY, 0, RESULTS)
    fixperm = F.sr_rows(_first_k_perm_rule(ar0.gt, ar0.mask, 5, R_FIX, F.rng_for("index_perm", 77)))
    res["random_fixed_index_basket_rule"] = _dist(fixperm)
    edges_cache = {}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, RESULTS)
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
            if FIXTURE:
                ent["tie_group"] = {"fixture": True, "n_tie_days": int(tie.sum())}
                res["runs"][key][f"seed_{s}"] = ent
                ent["_srs"] = [float(x) for x in srs]
                ent["_ip"] = [float(x) for x in ip_sr]
                continue
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
    fig, axs = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, (exp, label) in zip(axs, runs):
        key = f"{exp}/{label}"
        data_v = [res["runs"][key][f"seed_{s}"]["_srs"] for s in SEEDS]
        ax.violinplot(data_v, positions=range(len(SEEDS)), showmedians=True)
        ax.scatter(range(len(SEEDS)), [res["runs"][key][f"seed_{s}"]["stable"]["sr"] for s in SEEDS], c="r", zorder=3, label="stable (saved)")
        ax.scatter(range(len(SEEDS)), [res["runs"][key][f"seed_{s}"]["hold_all_sr"] for s in SEEDS], c="k", marker="_", s=200, zorder=3, label="hold-all")
        ax.set_title(key, fontsize=9); ax.set_xlabel("seed")
    axs[0].set_ylabel("2017 test Sharpe (random exact-tie order)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); FIGS.mkdir(exist_ok=True); fig.savefig(FIGS / "phase1_5a_mechanism.png", dpi=110)
    print("mechanism written")



def _proxy_inputs():
    data = _market()
    tdays = data.test_index + np.arange(237)
    ar0 = F.load_run(*PRIMARY, 0, RESULTS)
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
    if FIXTURE:
        (DOCS / "proxy.json").write_text(json.dumps({"skipped": "fixture"}))
        print("proxy skipped (fixture)")
        return
    n_boot = 200 if a.quick else F.N_BOOT
    from hypershift.eval.stats import stationary_bootstrap_indices
    data, tdays, feats = _proxy_inputs()
    runs = [PRIMARY] + list(REF_RUNS)
    res = {"quick": bool(a.quick), "features": sorted(feats), "runs": {}}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, RESULTS)
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
    ar0 = F.load_run(*PRIMARY, SEEDS[0], RESULTS)
    gt, mask = ar0.gt, ar0.mask
    if not FIXTURE:
        from hypershift.data.rsr import read_ticker_file
        data = _market()
        tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    ha = F.hold_all(gt, mask)
    res = {"fixture": FIXTURE, "quick": bool(a.quick), "B_NULL": B_N, "B_PERM": B_P, "N_BOOT": N_BT,
           "note": "seeds are repeated runs on the same 237 days; they are not independent samples. "
                   "random_topk is identical to the within-day score permutation null for every seed/arm.",
           "hold_all": F.perf(ha), "seeds": {}}
    R_rand = F.null_random_topk(gt, mask, 5, B_N, F.rng_for("null_random"))
    R_fix = F.null_fixed(gt, mask, 5, B_N, F.rng_for("null_fixed"))
    sr_rand, sr_fix = F.sr_rows(R_rand), F.sr_rows(R_fix)
    res["null_random_topk"] = {"sr_p5_50_95": [float(x) for x in np.percentile(sr_rand, [5, 50, 95])]}
    res["null_fixed"] = {"sr_p5_50_95": [float(x) for x in np.percentile(sr_fix, [5, 50, 95])]}
    if FIXTURE:
        s_ind = np.arange(gt.shape[0]) % 3
        s_beta = (np.arange(gt.shape[0]) + 1) % 3
        res["strata"] = {"fixture": "index-based strata (index % 3), not industry/beta"}
    else:
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
        boot[blk] = np.stack([stationary_bootstrap_indices(gt.shape[1], blk, rng) for _ in range(N_BT)])
    pooled = {"label_perm": [], "industry": [], "beta": []}
    per_seed_p = {f: {} for f in ("F1", "F2", "F3", "F4")}
    for s in SEEDS:
        ar = F.load_run(*PRIMARY, s, RESULTS)
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



def _gate_a_section():
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
    md = ["## Gate A (Tasks 1, 2, 2B, 5)", "",
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
    return md, out, o2


def stage_integrity(a):
    """Backtest integrity for the uncovered risks only (Task 3). Existing Phase 1.5 A/C tests are re-run, not redone."""
    if FIXTURE:
        (DOCS / "integrity.json").write_text(json.dumps({"skipped": "fixture"}))
        print("integrity skipped (fixture)")
        return
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
        ar = F.load_run(*PRIMARY, s, RESULTS)
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
    if FIXTURE:
        n = F.load_run(*PRIMARY, SEEDS[0], RESULTS).pred.shape[0]
        tickers = [f"S{i}" for i in range(n)]
        ind = np.array([f"ind{i % 3}" for i in range(n)])
        beta = np.ones(n)                                    # fixture placeholder
    else:
        from hypershift.data.rsr import read_ticker_file
        data = _market()
        tickers = read_ticker_file(ROOT / DATA_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
        n = len(tickers)
        ind = np.array(F.industry_of(tickers, ROOT / DATA_ROOT / "relation" / "sector_industry" / "NYSE_industry_ticker.json"))
        beta = F.train_beta(data.gt, data.mask, data.valid_index)
    skip = "skipped by the predeclared sequential stopping rule (Gate A outcome 2)"
    res = {"fixture": FIXTURE, "contribution": {"skipped": "gate A outcome 2"}, "costs_benchmarks": {"skipped": "gate A outcome 2"},
           "skipped": {"contribution_exclusion_dropdays (T4 item 3)": skip,
                       "costs_benchmarks beyond model and hold-all (T4 item 4)": skip},
           "runs": {}}
    for exp, label in [PRIMARY] + list(REF_RUNS):
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        allb = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, RESULTS)
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
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
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
        axs[0].plot(np.arange(1, len(res["runs"][pk][f"seed_{s}"]["persistence"]["top30"]) + 1), [c for _, c in res["runs"][pk][f"seed_{s}"]["persistence"]["top30"]], label=f"seed {s}")
    axs[0].set_xlabel("stock rank by selection count"); axs[0].set_ylabel("days selected (of 237)"); axs[0].legend(fontsize=7)
    for s in SEEDS:
        axs[1].plot(F.COST_BPS, [res["runs"][pk][f"seed_{s}"]["gross_net"]["model"][str(b)]["sr"] for b in F.COST_BPS], marker="o", label=f"seed {s}")
    axs[1].plot(F.COST_BPS, [res["runs"][pk]["seed_0"]["gross_net"]["hold_all"][str(b)]["sr"] for b in F.COST_BPS], "k--", label="hold-all")
    axs[1].set_xlabel("cost, bp per side"); axs[1].set_ylabel("Sharpe"); axs[1].legend(fontsize=7)
    axs[2].bar(range(len(SEEDS)), [res["runs"][pk][f"seed_{s}"]["persistence"]["jaccard_mean"] for s in SEEDS])
    axs[2].set_xlabel("seed"); axs[2].set_ylabel("mean day-to-day Jaccard")
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_persistence.png", dpi=110)
    print("decompose written")



def _boot_mean_ci(x, rng, n_boot, block=None):
    from hypershift.eval.stats import stationary_bootstrap_indices
    x = np.asarray(x, float)
    if len(x) < 3:
        return [float("nan"), float("nan")]
    block = block or F.BLOCK
    bm = [x[stationary_bootstrap_indices(len(x), block, rng)].mean() for _ in range(n_boot)]
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def stage_gaps(a):
    """Task 6, trimmed by the sequential stopping rule (Gate A outcome 2): items 1 (margin/spread), 4 (margin buckets),
    5 (top-k), 6 (global/local IC, calibration), 7 (reconciliation). Items 2-3 (epsilon and jitter curves) are skipped."""
    from hypershift.eval.metrics import daily_ic, ndcg_at_k
    from scipy.stats import spearmanr
    n_boot = 200 if a.quick else F.N_BOOT
    B_K = 20 if a.quick else F.B_NULL // 10
    skip = "skipped by the predeclared sequential stopping rule (Gate A outcome 2)"
    res = {"quick": bool(a.quick), "skipped": {"epsilon_curves (T6 item 2)": skip, "jitter_curves (T6 item 3)": skip},
           "epsilon": {"skipped": "gate A outcome 2"}, "jitter": {"skipped": "gate A outcome 2"}, "runs": {}}
    runs = [PRIMARY] + list(REF_RUNS)
    ar0 = F.load_run(*PRIMARY, SEEDS[0], RESULTS)
    gt, mask = ar0.gt, ar0.mask
    # k-null (same for every seed and arm: depends only on gt, mask)
    knull = {}
    for k in F.K_GRID:
        R = F.null_random_topk(gt, mask, k, B_K, F.rng_for("null_random", 50 + k))
        knull[k] = {"mean_of_means": float(R.mean()), "sr_mean": float(F.sr_rows(R).mean()),
                    "sr_p5_50_95": [float(x) for x in np.percentile(F.sr_rows(R), [5, 50, 95])]}
    res["random_k_null"] = {str(k): v for k, v in knull.items()}
    rng_rand_ndcg = np.random.default_rng(0)
    nd_rand = float(np.mean([ndcg_at_k(rng_rand_ndcg.standard_normal(gt.shape), gt, mask.astype(float), 5) for _ in range(20 if a.quick else 100)]))
    res["random_ndcg5"] = nd_rand
    pooled = {}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, RESULTS)
            assert (ar.mask == mask).all() and np.allclose(ar.gt, gt), "reference run is not on the same days/universe"
            r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
            b = F.boundary_stats(ar.pred, ar.mask)
            ent = {}
            # 1. margin / spread distributions
            nz = b["margin"] > 0
            ent["spread"] = {"margin5": _dist(b["margin"]), "sd": _dist(b["sd"]), "ptp": _dist(b["ptp"]),
                             "margin5_over_sd_nonzero": _dist((b["margin"] / np.where(b["sd"] > 0, b["sd"], 1))[nz]) if nz.any() else None,
                             "zero_spread_days": int((b["sd"] == 0).sum()), "exact_tie_days": int(b["exact_tie"].sum())}
            # 4. margin buckets
            mb = F.margin_buckets(b["margin"], r)
            nzi = np.nonzero(nz)[0]
            rng = F.rng_for("boot", 400 + s)
            lab_edges = [x for x in mb if x["bucket"] != "exact_tie"]
            edges = np.array([lab_edges[0]["lo"]] + [x["hi"] for x in lab_edges])
            labs = np.full(len(r), -1)
            labs[nzi] = np.clip(np.searchsorted(edges, b["margin"][nzi], side="right") - 1, 0, len(lab_edges) - 1)
            for row in mb:
                days = np.nonzero(b["exact_tie"] if row["bucket"] == "exact_tie" else labs == int(row["bucket"][1:]) - 1)[0]
                row["days"] = len(days)
                if len(days) >= 3:
                    row["mean_ci95"] = _boot_mean_ci(r[days], rng, n_boot)
                    sd_ = r[days].std()
                    row["sr_annualised_on_bucket_days"] = float(r[days].mean() / sd_ * np.sqrt(252)) if sd_ > 0 else 0.0
                    td = F.topk_diag(ar.pred[:, days], ar.gt[:, days], ar.mask[:, days], 5)
                    row["hit_top10"], row["hit_top20"] = td["hit_top10"], td["hit_top20"]
                    row["miss_bottom10"] = td["miss_bottom10"]
                else:
                    row["mean_ci95"] = None
            ent["margin_buckets"] = mb
            if len(nzi) > 10:
                rho = spearmanr(b["margin"][nzi], r[nzi]).correlation
                brho = []
                from hypershift.eval.stats import stationary_bootstrap_indices
                for _ in range(n_boot):
                    ii = stationary_bootstrap_indices(len(nzi), F.BLOCK, rng)
                    brho.append(spearmanr(b["margin"][nzi][ii], r[nzi][ii]).correlation)
                ent["margin_vs_return_spearman_nontie_days"] = {"rho": float(rho), "ci95": [float(np.percentile(brho, 2.5)), float(np.percentile(brho, 97.5))],
                                                                  "n": int(len(nzi))}
            pooled.setdefault(key, []).append((b["margin"], r, b["exact_tie"]))
            # 5. top-k
            tk = {}
            for k in F.K_GRID:
                rk, bk = F.portfolio(ar.pred, ar.gt, ar.mask, k)
                td = F.topk_diag(ar.pred, ar.gt, ar.mask, k)
                tk[str(k)] = {**F.perf(rk), "turnover_mean": float(F.turnover(bk).mean()), "jaccard_mean": float(F.jaccard_series(bk).mean()),
                              **td, "excess_mean_over_random_k_null": float(rk.mean() - knull[k]["mean_of_means"]),
                              "sr_minus_random_k_null_mean_sr": float(F.perf(rk)["sr"] - knull[k]["sr_mean"]),
                              "random_k_null_sr_p95": knull[k]["sr_p5_50_95"][2], "primary": k == F.K_PRIMARY}
            ent["topk"] = tk
            # 6. global / local IC and calibration
            ent["global_ic"] = daily_ic(ar.pred, ar.gt, ar.mask.astype(float))
            ent["local_ic"] = {str(q): F.local_ic(ar.pred, ar.gt, ar.mask, q) for q in F.LOCAL_IC_Q}
            cal_days = np.stack([F.calibration(ar.pred[:, [d]], ar.gt[:, [d]], ar.mask[:, [d]]) for d in range(len(r))])
            valid_days = np.array([ar.mask[:, d].sum() >= F.CALIB_BINS for d in range(len(r))])
            cm = cal_days[valid_days].mean(0)
            cse = cal_days[valid_days].std(0, ddof=1) / np.sqrt(valid_days.sum())
            ent["calibration"] = {"bin_mean_ret": [float(x) for x in cm], "se_day_clustered": [float(x) for x in cse],
                                  "ci95_halfwidth": [float(1.96 * x) for x in cse],
                                  "overall_mean_ret": float(np.mean([ar.gt[ar.mask[:, d], d].mean() for d in range(len(r))])),
                                  "top_decile_minus_bottom_decile": float(cm[-1] - cm[0]), "top_decile_minus_overall": float(cm[-1] - np.mean(cm))}
            ent["ndcg5"] = ndcg_at_k(ar.pred, ar.gt, ar.mask.astype(float), 5)
            res["runs"][key][f"seed_{s}"] = ent
            t5, t10 = tk["5"], tk["50"]
            print(key, s, "IC", round(ent["global_ic"], 4), "localIC.05/.1/.2", [round(v, 3) for v in ent["local_ic"].values()],
                  "hit10", round(t5["hit_top10"], 3), "miss10", round(t5["miss_bottom10"], 3), "prec5", round(t5["prec_at_k"], 4), "ndcg5", round(ent["ndcg5"], 4),
                  "exc k5", round(t5["excess_mean_over_random_k_null"] * 1e4, 1), "bp/d | topdec-botdec", round(ent["calibration"]["top_decile_minus_bottom_decile"] * 1e4, 1), "bp", flush=True)
    # pooled-over-seeds margin buckets for the primary run
    pk = "%s/%s" % PRIMARY
    m = np.concatenate([x[0] for x in pooled[pk]]); rr = np.concatenate([x[1] for x in pooled[pk]])
    res["primary_pooled_margin_buckets"] = {"rows": F.margin_buckets(m, rr),
                                            "note": "days pooled across seeds (same 237 days, correlated); descriptive only"}
    (DOCS / "gaps.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [x["bucket"] for x in res["runs"][pk][f"seed_{SEEDS[0]}"]["margin_buckets"]]
    for s in SEEDS:
        ax.plot(range(len(names)), [x["mean"] * 1e4 for x in res["runs"][pk][f"seed_{s}"]["margin_buckets"]], marker="o", label=f"seed {s}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("mean next-day top-5 return (bp)")
    ax.set_xlabel("5th-6th score margin bucket (exact tie, then quintiles)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_margin.png", dpi=110)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in SEEDS:
        c = res["runs"][pk][f"seed_{s}"]["calibration"]
        ax.errorbar(np.arange(1, len(c["bin_mean_ret"]) + 1) + (s - 2) * 0.07, np.array(c["bin_mean_ret"]) * 1e4, yerr=np.array(c["ci95_halfwidth"]) * 1e4,
                    fmt="o-", ms=3, lw=0.8, label=f"seed {s}")
    ax.axhline(res["runs"][pk][f"seed_{SEEDS[0]}"]["calibration"]["overall_mean_ret"] * 1e4, color="k", ls="--", lw=0.8)
    ax.set_xlabel("predicted score decile (1 = lowest)"); ax.set_ylabel("mean realised next-day return (bp), +/-1.96 SE"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_calib.png", dpi=110)
    print("gaps written")



NOT_ANSWERABLE = [
    "basket identities, margins and tie rates at non-selected epochs",
    "overlap of epoch-0 vs selected baskets",
    "through-model index permutation",
    "frozen-model 2018+ evaluation"]


def stage_trajectory(a):
    """Task 7: history-only epoch trajectory, cross-seed early-vs-late selected epochs, wd-control statement."""
    runs = [PRIMARY] + list(REF_RUNS)
    res = {"epoch0_note": "epoch 0 = after the first training pass (loop.py:211-243), not an untrained model",
           "not_answerable_from_artifacts": NOT_ANSWERABLE, "runs": {}}
    series = {}
    for exp, label in runs:
        key = f"{exp}/{label}"
        res["runs"][key] = {}
        for s in SEEDS:
            ar = F.load_run(exp, label, s, RESULTS)
            h = ar.history
            tsr = np.array([x["test"]["sr"] for x in h]); vsr = np.array([x["val"]["sr"] for x in h])
            ic = np.array([x["test_ic"] for x in h]); sd = np.array([x["test_pred_sd"] for x in h])
            nd = np.array([x["test"]["ndcg5"] for x in h]); tl = np.array([x["train_loss"] for x in h])
            be = int(ar.metrics["best_epoch"]); oe = ar.metrics.get("test_oracle_epoch")
            series[(key, s)] = dict(tsr=tsr, vsr=vsr, ic=ic, sd=sd, nd=nd, tl=tl, be=be, oe=oe)

            def _c(x, y):
                return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
            res["runs"][key][f"seed_{s}"] = {
                "epochs": len(h), "selected_epoch": be, "test_oracle_epoch_diagnostic": oe,
                "test_sr_epoch0": float(tsr[0]), "test_sr_selected": float(tsr[be]), "test_sr_last": float(tsr[-1]),
                "test_sr_oracle_diagnostic": float(tsr.max()), "val_sr_selected": float(vsr[be]),
                "corr_over_epochs_test_sr_vs_pred_sd": _c(tsr, sd), "corr_over_epochs_test_sr_vs_test_ic": _c(tsr, ic),
                "test_sr_mean_over_epochs": float(tsr.mean()), "test_sr_sd_over_epochs": float(tsr.std())}
    # early vs late selected epochs, cross-seed only, primary run
    pk = "%s/%s" % PRIMARY
    sel = {s: res["runs"][pk][f"seed_{s}"]["selected_epoch"] for s in SEEDS}
    early = [s for s in SEEDS if sel[s] <= 2]; late = [s for s in SEEDS if sel[s] >= 8]
    prox = json.loads((DOCS / "proxy.json").read_text()) if (DOCS / "proxy.json").exists() else None
    dec = json.loads((DOCS / "decompose.json").read_text()) if (DOCS / "decompose.json").exists() else None
    per, baskets = {}, {}
    for s in SEEDS:
        ar = F.load_run(*PRIMARY, s, RESULTS)
        r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
        b = F.boundary_stats(ar.pred, ar.mask)
        baskets[s] = base
        per[s] = {"selected_epoch": sel[s], "sharpe": F.perf(r)["sr"], "tie_rate": float(b["exact_tie"].mean()),
                  "score_sd_median": float(np.median(b["sd"])), "margin5_median": float(np.median(b["margin"])),
                  "top5_stock_share": F.top_share(F.selection_freq(base, ar.pred.shape[0]), 5)}
        if prox and isinstance(prox.get("runs"), dict) and pk in prox["runs"]:
            d = {k: v["mean_spearman"] for k, v in prox["runs"][pk][f"seed_{s}"]["spearman_ranked"]}
            per[s]["spearman_ret20"], per[s]["spearman_ma30_rel"] = d.get("ret20"), d.get("ma30_rel")
    def _grp(g):
        return {k: float(np.mean([per[s][k] for s in g])) for k in ("sharpe", "tie_rate", "score_sd_median", "margin5_median", "top5_stock_share")} if g else None
    within = lambda g: float(np.mean([_jacc(baskets[i], baskets[j]) for ii, i in enumerate(g) for j in g[ii + 1:]])) if len(g) > 1 else None
    between = float(np.mean([_jacc(baskets[i], baskets[j]) for i in early for j in late])) if early and late else None
    res["early_vs_late_primary"] = {"early_seeds(selected epoch<=2)": early, "late_seeds(selected epoch>=8)": late, "per_seed": {str(s): v for s, v in per.items()},
                                    "early_mean": _grp(early), "late_mean": _grp(late), "mean_daily_jaccard_within_early": within(early),
                                    "mean_daily_jaccard_within_late": within(late), "mean_daily_jaccard_between": between,
                                    "caveat": "between-seed comparison: confounded with seed; not a within-run trajectory"}
    res["seed_concentration_from_decompose"] = dec["runs"][pk]["seed_concentration"] if dec else None
    res["seed_note"] = "agreement across seeds under the same data, ordering and backtester is not independent evidence"
    res["wd_control"] = {
        "source": "docs/phase1_5/F_learnability.md lines 7 and 11-12",
        "quote": ("Root cause (confirmed): coupled L2 weight decay 5e-4 in Adam, with lr 1e-3, destroys the model because the loss gradient is far smaller "
                  "than the decay gradient. Setting weight_decay=0 alone raises THINK (HH_hyper) from 26% to 63-84% of the oracle IC and EH from 32% to 72-77%. "
                  "[...] weight decay is 10-40x larger [than the loss gradient]; PoincareLinear output is proportional to |z|, so three stacked layers shrink multiplicatively."),
        "design": "one-factor control: same planted signal, data, code path and seeds; only weight decay changed",
        "narrow_conclusion": ("wd 5e-4 impaired planted-signal learnability; this does not attribute the real-data Sharpe change to wd "
                              "(input mode, log_ic and seed count also changed)"),
        "recommendation": "no new run"}
    (DOCS / "trajectory.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("tsr", "test Sharpe"), ("vsr", "val Sharpe"), ("ic", "test IC"), ("nd", "test NDCG@5"), ("sd", "test pred SD"), ("tl", "train loss")]
    fig, axs = plt.subplots(len(runs), len(metrics), figsize=(3.2 * len(metrics), 2.6 * len(runs)), squeeze=False)
    for i, (exp, label) in enumerate(runs):
        key = f"{exp}/{label}"
        for j, (m, title) in enumerate(metrics):
            ax = axs[i][j]
            for s in SEEDS:
                sr = series[(key, s)]
                l, = ax.plot(sr[m], lw=0.7)
                ax.axvline(sr["be"], color=l.get_color(), ls=":", lw=0.8)
                if sr["oe"] is not None and m == "tsr":
                    ax.plot([sr["oe"]], [sr["tsr"][sr["oe"]]], marker="x", color=l.get_color(), ms=4)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(key, fontsize=6)
            ax.tick_params(labelsize=6)
    fig.suptitle("epoch 0 = after the first training pass (loop.py:211-243), not an untrained model; dotted = validation-selected epoch, x = best-test epoch (diagnostic)", fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "phase1_5a_trajectory.png", dpi=110)
    e = res["early_vs_late_primary"]
    print("trajectory: selected", sel, "early", early, "late", late, "| tie rate early/late",
          e["early_mean"] and round(e["early_mean"]["tie_rate"], 3), e["late_mean"] and round(e["late_mean"]["tie_rate"], 3),
          "| jaccard within E/L, between", e["mean_daily_jaccard_within_early"], e["mean_daily_jaccard_within_late"], e["mean_daily_jaccard_between"])
    for s in SEEDS:
        x = res["runs"][pk][f"seed_{s}"]
        print(" seed", s, "ep0/sel/last/oracle SR", round(x["test_sr_epoch0"], 2), round(x["test_sr_selected"], 2), round(x["test_sr_last"], 2),
              round(x["test_sr_oracle_diagnostic"], 2), "corr(SR,sd)", x["corr_over_epochs_test_sr_vs_pred_sd"] and round(x["corr_over_epochs_test_sr_vs_pred_sd"], 2),
              "corr(SR,IC)", x["corr_over_epochs_test_sr_vs_test_ic"] and round(x["corr_over_epochs_test_sr_vs_test_ic"], 2))





def _rng(vals, f="{:.2f}"):
    vals = [v for v in vals if v is not None]
    lo, hi = min(vals), max(vals)
    return f.format(lo) if f.format(lo) == f.format(hi) else f"{f.format(lo)} to {f.format(hi)}"


GATE_A_INTERPRETATION = """### Interpretation (Gate A, exploratory; written after batch 1)

1. Outcome 2 (exposure B sufficient) holds: in every primary seed the Sharpe sits inside the central 90% of the beta-matched or industry-matched null (industry p 0.145 to 0.641; beta p 0.035 to 0.081, so seed 4 is below 0.05 on beta alone). Outcomes 1 and 3 do not hold, so Gate A is not "not decisive" but it is a single-mechanism result, not a full explanation. Outcome 2 is also implied by the non-rejection of F1 (random daily top-5): if the Sharpe is already typical of random 5-stock baskets in this year, market exposure plus chance suffices to account for it.
2. The 2017 equal-weight market (hold-all) already has Sharpe 1.53. The 5 seeds' 1.88 to 2.14 are 0.35 to 0.61 above it; the Sharpe of the daily excess over hold-all is 1.5 to 1.9 but its block-bootstrap CI is wide (lowest lower bound 0.03, highest upper bound 3.9).
3. Against a uniform random daily top-5 (F1, same days, mask, k) the per-seed one-sided p is 0.067 to 0.120. Family p (max over seeds) F1 0.120, F2 0.081, F3 0.641, F4 0.142; Holm-adjusted 0.32 to 0.64. No test rejects. This is "no evidence against the null" on 237 days, not proof of no skill; power is low (see the CI width).
4. Tie/index path (A) is not decisive: with random exact-tie order the median Sharpe stays above hold-all in every seed (1.78, 1.79, 2.07, 2.14, 2.17 vs 1.53); the reverse-index Sharpe is 1.60 to 2.42. For seeds 0 and 1 the saved stable value (1.88, 1.93) sits inside the random-tie 5-95% band (1.40 to 2.15), so the stable tie-break neither inflates nor deflates materially. 112 and 109 of 237 days have an exact tie at the 5th/6th boundary in seeds 0 and 1 (24, 16, 43 in seeds 2 to 4). Constant-score first-5 gives 0.37 and a random fixed-index basket rule has median 0.95 (p95 2.52), so an index-order basket alone does not reproduce 2 as a typical outcome. Not an artifact claim from tie rate alone.
5. Tie composition in the primary run: boundary tie groups are small (mean size 3.4 to 6.5 in seeds 0, 2, 3, 4; seed 1 has a few large groups, max 1296), every day has a different tied value (no repeated saturated value except 4 days in seed 1), 0% graph-isolated, 0% stale window, 0% partly masked. So the ties are not explained by identical inputs under these three tests (mechanism cause UNKNOWN). In the R5_f_train/HH reference, ties are mass ties of about 1700 valid stocks sharing one constant output (collapsed days), a different phenomenon.
6. Model-level equivariance (random-init THINK, 4 temporal/spatial combos, node permutation of inputs and graph) holds to rtol 1e-5, so the model has no index-dependent step; the index only enters through the evaluator's stable tie-break. The trained-model version is not recoverable (no weights saved).
7. Factor proxy (descriptive, not causal): the score is positively associated with ma30_rel and ma20_rel and negatively with ret20 (daily Spearman 0.2 to 0.4 in magnitude): a short-horizon mean-reversion / oversold tilt, shared across seeds. An all-feature linear fit explains a mean per-day R2 of 0.09 to 0.21. The fitted portfolio gets 0.33 to 1.76 (below the model in every seed), while the residual (score minus fit) portfolio keeps 1.83 to 2.29. So the simple proxies do not carry the return; most of it stays in the unexplained part. Literal outcome 3 is not met (seed 3 residual 2.29 lies above the random null p95; also 'index' and 'degree' qualify as "proxies" only as arbitrary fixed baskets, which the economic-proxy reading excludes).
"""


def stage_report(a):
    """Full report built only from the stage jsons (and test_results.txt if present)."""
    if FIXTURE:
        (DOCS / "REPORT_2017.md").write_text("fixture run: report not generated\n")
        (DOCS / "gate_a.json").write_text(json.dumps({"skipped": "fixture"}))
        print("report skipped (fixture)")
        return
    J = lambda n: json.loads((DOCS / f"{n}.json").read_text())
    mech, prox, nl, integ, dec, gaps, traj, inv = (J(n) for n in ("mechanism", "proxy", "nulls", "integrity", "decompose", "gaps", "trajectory", "inventory"))
    gate_md, gate_out, o2 = _gate_a_section()
    (DOCS / "gate_a.json").write_text(json.dumps(gate_out, indent=1))
    pk = "%s/%s" % PRIMARY
    S = [f"seed_{s}" for s in SEEDS]
    ha = nl["hold_all"]["sr"]
    # per-seed reproduction
    rep_rows = []
    for s in SEEDS:
        ar = F.load_run(*PRIMARY, s, RESULTS)
        r, base = F.portfolio(ar.pred, ar.gt, ar.mask)
        b = F.boundary_stats(ar.pred, ar.mask)
        real_sd = float(np.median([ar.gt[ar.mask[:, d], d].std() for d in range(ar.gt.shape[1])]))
        p = F.perf(r)
        g = gaps["runs"][pk][f"seed_{s}"]
        rep_rows.append(f"| {s} | {ar.metrics['best_epoch']} | {p['sr']:.2f} | {p['mean'] * 1e4:.1f} | {p['vol_d'] * 1e2:.2f} | {p['mdd']:.3f} | {g['global_ic']:.4f} | "
                        f"{g['ndcg5']:.4f} | {np.median(b['sd']):.2e} | {np.median(b['sd']) / real_sd:.3f} | {b['exact_tie'].mean():.2f} | "
                        f"{F.turnover(base).mean():.2f} | {ha:.2f} |")
    pct = [nl["seeds"][k]["random_topk"]["sharpe"]["pct"] for k in S]
    p1 = [nl["seeds"][k]["random_topk"]["sharpe"]["p"] for k in S]
    ci = [nl["seeds"][k]["bootstrap"]["block_10"]["sharpe_ci95"] for k in S]
    fam = nl["family"]
    nrp = nl["null_random_topk"]["sr_p5_50_95"]
    sharpes = [nl["seeds"][k]["observed"]["sr"] for k in S]
    de = dec["runs"][pk]
    tk = {k: gaps["runs"][pk][f"seed_{s}"]["topk"] for k, s in zip(S, SEEDS)}
    gp = gaps["runs"][pk]
    hit10 = [gp[k]["topk"]["5"]["hit_top10"] for k in S]; miss10 = [gp[k]["topk"]["5"]["miss_bottom10"] for k in S]
    ic = [gp[k]["global_ic"] for k in S]; nd = [gp[k]["ndcg5"] for k in S]
    tbd = [gp[k]["calibration"]["top_decile_minus_bottom_decile"] * 1e4 for k in S]
    cal_hw = [gp[k]["calibration"]["ci95_halfwidth"][-1] * 1e4 for k in S]
    exc5 = [gp[k]["topk"]["5"]["excess_mean_over_random_k_null"] * 1e4 for k in S]
    rho = [gp[k]["margin_vs_return_spearman_nontie_days"] for k in S]
    ig = integ["per_seed"]
    ext = integ["extreme_selected_stock_days_abs_gt_0.2"]
    syx = [e for e in ext if e["ticker"] == "SYX"][0]
    syx_sh = [syx["pl_share_by_seed"][str(s)] for s in SEEDS]
    net10 = [de[k]["gross_net"]["model"]["10"]["sr"] for k in S]; net5 = [de[k]["gross_net"]["model"]["5"]["sr"] for k in S]
    net25 = [de[k]["gross_net"]["model"]["25"]["sr"] for k in S]
    T = traj["runs"][pk]
    e0 = [T[k]["test_sr_epoch0"] for k in S]; mo = [T[k]["test_sr_mean_over_epochs"] for k in S]
    evl = traj["early_vs_late_primary"]

    md = ["# Phase 1.5a report: 2017 Sharpe near 2 (R5_f2 alpha=0 THINK, HH)", "",
          "Reproduce: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage all` (CPU, a few minutes plus the nulls). "
          "Primary: `R5_f2_alpha0_train/HH` seeds 0-4, validation-selected epoch, top-5 equal weight, daily rebalance, 237 test days (2017-01-03 to 2017-12-08). "
          "References: `R5_f_train/{HH,EH}` seeds 0-4. Seeds are repeated runs on the same days, not independent samples. 2017 is an exploratory year. "
          "Market = equal-weight hold-all of valid stocks (cap-weighted market: UNKNOWN, not in the data). "
          "Batch 2 was trimmed by the predeclared sequential stopping rule (Gate A outcome 2); skipped parts are listed in the section \"Skipped by the predeclared sequential stopping rule\".", "",
          "## Executive summary", "",
          f"1. **Why Sharpe near 2:** a concentrated daily top-5 portfolio in a strong market year. Equal-weight hold-all already has Sharpe {ha:.2f} on these 237 days, the basket carries a high-beta tilt "
          f"(mean basket beta {_rng([de[k]['exposure']['basket_mean_beta'] for k in S])} vs universe {de[S[0]]['exposure']['universe_mean_beta']:.2f}), and a random daily 5-stock basket has a wide Sharpe distribution "
          f"(5th-95th percentile {nrp[0]:.2f} to {nrp[2]:.2f}).",
          f"2. **The model is not distinguishable from random selection.** The five Sharpes ({_rng(sharpes)}) sit at the {_rng(pct, '{:.0f}').replace(' to ', '-')} percentile (in percent of random baskets beaten) of random daily top-5 baskets (F1 per-seed p {_rng(p1, '{:.3f}')}, family p {fam['iut_family_p_max_over_seeds']['F1']:.3f}, Holm {fam['holm_adjusted']['F1']:.2f}). "
          f"None of F1-F4 rejects (Holm {min(fam['holm_adjusted'].values()):.2f} to {max(fam['holm_adjusted'].values()):.2f}). Gate A outcome 2 (exposure sufficient) is implied by the non-rejection of F1: a Sharpe typical of random 5-stock baskets in this year needs no skill to explain it.",
          f"3. **Power is low.** Seed-0 block-bootstrap Sharpe 95% CI is {ci[0][0]:.2f} to {ci[0][1]:.2f}; over seeds the CI bounds range {min(c[0] for c in ci):.2f} to {max(c[1] for c in ci):.2f}. "
          "\"Not distinguishable from random\" is not \"no signal\": hypothesis D (a genuine top-tail signal) is not excluded, only unsupported.",
          f"4. **Weak positive signs, no calibrated ranking.** Global IC {_rng(ic, '{:.4f}')}, NDCG@5 {_rng(nd, '{:.4f}')} vs random {gaps['random_ndcg5']:.4f}; the top-5 are in the realised top decile {_rng(hit10, '{:.3f}')} of the time (random 0.10) "
          f"but also in the realised bottom decile {_rng(miss10, '{:.3f}')}, so most of that is a volatility effect; top minus bottom predicted-decile return {_rng(tbd, '{:.1f}')} bp/day with a top-decile 95% half-width of about {_rng(cal_hw, '{:.0f}')} bp. Margin buckets show no monotone pattern.",
          f"5. **Mechanisms ruled down:** ties/index order (A) are not the main mechanism (random exact-tie median Sharpe stays above hold-all in every seed); backtest integrity (C) is clean except one stock-day (SYX 2017-03-27, +{syx['ret'] * 100:.1f}%, selected by all five seeds) that is about "
          f"{_rng([x * 100 for x in syx_sh], '{:.0f}')}% of each seed's total P&L. Retrospective Sharpe without the day(s) with an abs(return) > 0.2 stock in the basket, seeds 0-4: "
          + ", ".join(f"{ig[k]['retrospective_sharpe_without_days_with_abs_gt_0.2']:.2f}" for k in S) + " (seeds 3-4 also lose the WAIR 2017-08-09 loss day, so they rise).",
          f"6. **Costs and training:** at 10 bp per side the net Sharpe is {_rng(net10)} and at 25 bp {_rng(net25)} (hold-all has no cost modelled, {ha:.2f}). Test Sharpe after the first training pass is already {_rng(e0)} and its mean over all 100 epochs is {_rng(mo)}: the level is not something training clearly added.",
          "7. **Verdict:** Sharpe near 2 in this run is **not** evidence of learned stock ranking (NO EVIDENCE; exploratory, 2017 only, seeds not independent). See the decision table and the verdict section.", "",
          "## Reproduction table by seed (primary run)", "",
          "Spread = median daily cross-sectional SD of the scores; spread/realised = that divided by the median daily cross-sectional SD of realised next-day returns. Tie rate = share of days with an exact tie at the 5th/6th score. Turnover = mean daily share of names replaced.", "",
          "| seed | selected epoch | Sharpe | mean (bp/day) | vol (%/day) | max drawdown | IC | NDCG@5 | spread | spread / realised | tie rate | turnover | hold-all Sharpe |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|", *rep_rows, "",
          "All 25 run-seeds (primary + four references) recompute from the saved arrays: daily returns to atol 1e-7 and Sharpe to 1e-5 against `metrics.json` and `history.jsonl` (`inventory.json`).", ""]
    md += gate_md
    md += GATE_A_INTERPRETATION.split("\n")
    md += ["**What Gate A shows:** F1-F4 do not reject; the tie path is not decisive; the score has a mean-reversion tilt that does not carry the return. **What it does not show:** that the model has no skill (power is low), or anything about years other than 2017.", ""]
    # integrity
    md += ["## Integrity (Task 3, uncovered risks only)", "",
           f"Existing Phase 1.5 A/C checks re-run: `{integ['existing_tests']['cmd']}` -> {integ['existing_tests']['tail'][-1]} (verbatim last line). Target timing, split dates, mask-to-raw-close trace and Sharpe convention (`metrics.py:50-52`: mean / np.std ddof 0 x sqrt(252), no risk-free rate; pinned by `test_sharpe_matches_authors_up_to_annualisation_constant`) are therefore not redone.", "",
           f"- Duplicate stock series: {integ['duplicate_series']['n_pairs']} pairs.",
           f"- Selected stock-days whose 17-day window touches a fill value: {_rng([ig[k]['selected_with_fill_in_window'] for k in S], '{:.0f}')} (asserted 0; mask = min over the window).",
           f"- Stale close (unchanged for 3+ days ending at t-1) inside selected baskets: {_rng([ig[k]['stale_in_basket'] for k in S], '{:.0f}')} of {ig[S[0]]['selected_stock_days']} stock-days per seed, P&L share {_rng([ig[k]['stale_pl_share'] * 100 for k in S], '{:.2f}')}%. Exactly-zero realised return on {_rng([ig[k]['exact_zero_return_selected'] for k in S], '{:.0f}')} selected stock-days (P&L share 0 by construction).",
           f"- Selected stock-days with |return| > 0.2: {integ['n_extreme_unique']} unique ({', '.join(e['ticker'] + ' ' + e['date'] + ' ' + format(e['ret'] * 100, '+.1f') + '% x' + str(e['seeds']) + ' seeds' for e in ext)}). "
           f"Neither is followed by a reversal above 50% of its size within 3 days ({integ['n_extreme_reversal_candidates']} candidates), so there is no sign of a split/adjustment error; whether SYX really rose that day is not verified against an external source (UNKNOWN). None is deleted.",
           f"- SYX 2017-03-27 is selected by every seed and is {_rng([x * 100 for x in syx_sh], '{:.0f}')}% of each seed's total P&L (one stock-day, weight 1/5). **Retrospective** Sharpe without the day(s) with abs(ret) > 0.2 in the basket: "
           + ", ".join(f"seed {s} {ig[k]['retrospective_sharpe_without_days_with_abs_gt_0.2']:.2f} (hold-all same days {ig[k]['retrospective_hold_all_sharpe_same_days']:.2f}, {ig[k]['n_days_removed']} day(s) removed)" for s, k in zip(SEEDS, S)) + ".", "",
           "**What this shows:** no data or backtest defect changes the returns. **What it does not show:** that the result is robust to a single large winner; one stock-day carries a fifth of the P&L, which is itself an example of the concentration in explanation B.", ""]
    # decomposition
    pe = de[S[0]]["persistence"]
    md += ["## Decomposition (Task 4, trimmed: persistence, exposure, seed concentration, gross/net)", "",
           f"- Persistence: mean day-to-day basket Jaccard {_rng([de[k]['persistence']['jaccard_mean'] for k in S], '{:.2f}')}, median holding spell {_rng([de[k]['persistence']['duration_median'] for k in S], '{:.0f}')} days, "
           f"{_rng([de[k]['persistence']['distinct_stocks_selected'] for k in S], '{:.0f}')} distinct stocks ever selected (of 1737). Top-5 most-selected stocks take {_rng([de[k]['persistence']['top_share']['5'] * 100 for k in S], '{:.0f}')}% of the 1185 selection slots, top-20 take {_rng([de[k]['persistence']['top_share']['20'] * 100 for k in S], '{:.0f}')}%.",
           f"- Exposure: mean training-period beta of the selected stocks {_rng([de[k]['exposure']['basket_mean_beta'] for k in S])} vs {de[S[0]]['exposure']['universe_mean_beta']:.2f} for the universe (beta from training days only, against the equal-weight hold-all). "
           f"The model never selects a stock with no industry label ({_rng([de[k]['exposure']['slot_share_na_industry'] * 100 for k in S], '{:.0f}')}% of slots vs {de[S[0]]['exposure']['universe_share_na_industry'] * 100:.0f}% of the universe). "
           "Over-weighted industries are tiny ones (e.g. Wholesale Distributors: 7-11% of slots, 0.06% of the universe, one stock), which also means the industry-matched null F3 replaces those stocks with themselves and is a weak test (conservative toward not rejecting).",
           f"- Gross vs net (cost = 2 x bps x share of names replaced, day 1 full; mean turnover {_rng([de[k]['gross_net']['turnover_mean'] for k in S], '{:.2f}')}); hold-all is charged no cost (equal-weight hold, turnover convention 0):", "",
           "| seed | gross | 5 bp | 10 bp | 25 bp | hold-all (any cost) |", "|---|---|---|---|---|---|"]
    for s, k in zip(SEEDS, S):
        gn = de[k]["gross_net"]
        md.append(f"| {s} | {gn['model']['0']['sr']:.2f} | {gn['model']['5']['sr']:.2f} | {gn['model']['10']['sr']:.2f} | {gn['model']['25']['sr']:.2f} | {gn['hold_all']['0']['sr']:.2f} |")
    sc = de["seed_concentration"]
    md += ["", f"- Seed concentration: pairwise daily basket Jaccard between seeds {sc['pairwise_daily_basket_jaccard_mean']:.2f} (range {sc['pairwise_jaccard_min_max'][0]:.2f} to {sc['pairwise_jaccard_min_max'][1]:.2f}), pairwise correlation of daily returns "
           f"{sc['pairwise_daily_return_corr_mean']:.2f} (range {sc['pairwise_corr_min_max'][0]:.2f} to {sc['pairwise_corr_min_max'][1]:.2f}); stocks in the top-10 selection frequency of at least 3 seeds: {', '.join(sc['stocks_in_top10_freq_of_ge3_seeds'])}. "
           "Agreement across seeds under the same data, ordering and backtester is not independent evidence.", "",
           "Skipped by the predeclared sequential stopping rule (Gate A outcome 2): per-stock contribution, retrospective exclude-and-reselect and best-day removal (T4 item 3); costs/benchmarks beyond model and hold-all, momentum and daily excess-series CIs (T4 item 4).", "",
           "Figure: `docs/figures/phase1_5a_persistence.png`.", "",
           "**What this shows:** a rotating, moderately persistent basket tilted to high-beta names, with a net Sharpe below hold-all once cost is charged. **What it does not show:** which stocks drive the P&L beyond the single SYX day (contribution analysis skipped).", ""]
    # gaps
    bk = gp[S[0]]["margin_buckets"]
    md += ["## Tiny score gaps (Task 6, trimmed: margin buckets, top-k, local/global IC, calibration)", "",
           f"Spread: median 5th-minus-6th margin {_rng([gp[k]['spread']['margin5']['p50'] for k in S], '{:.1e}')}, median daily score SD {_rng([gp[k]['spread']['sd']['p50'] for k in S], '{:.1e}')}, zero-spread days {_rng([gp[k]['spread']['zero_spread_days'] for k in S], '{:.0f}')}, exact-tie days {_rng([gp[k]['spread']['exact_tie_days'] for k in S], '{:.0f}')}.", "",
           "Margin buckets (mean next-day top-5 return in bp, block-bootstrap 95% CI; exact tie first, then margin quintiles q1 smallest to q5 largest):", "",
           "| seed | " + " | ".join(b["bucket"] for b in bk) + " | Spearman(margin, return), non-tie days (95% CI) |", "|---|" + "---|" * (len(bk) + 1)]
    for s, k in zip(SEEDS, S):
        cells = []
        for b in gp[k]["margin_buckets"]:
            c = b.get("mean_ci95")
            cells.append(f"{b['mean'] * 1e4:.0f} ({c[0] * 1e4:.0f},{c[1] * 1e4:.0f}) n={b['n']}" if c else f"n={b['n']}")
        r_ = gp[k]["margin_vs_return_spearman_nontie_days"]
        md.append(f"| {s} | " + " | ".join(cells) + f" | {r_['rho']:.2f} ({r_['ci95'][0]:.2f},{r_['ci95'][1]:.2f}) |")
    md += ["", "Top-k (primary k = 5; the others are diagnostic). Sharpe / excess mean daily return over the random-k null mean (bp/day) / hit rate of the realised top decile / realised bottom decile:", "",
           "| seed | " + " | ".join(f"k={k_}" for k_ in F.K_GRID) + " |", "|---|" + "---|" * len(F.K_GRID)]
    for s, k in zip(SEEDS, S):
        md.append(f"| {s} | " + " | ".join(f"{tk[k][str(k_)]['sr']:.2f} / {tk[k][str(k_)]['excess_mean_over_random_k_null'] * 1e4:.1f} / {tk[k][str(k_)]['hit_top10']:.2f} / {tk[k][str(k_)]['miss_bottom10']:.2f}" for k_ in F.K_GRID) + " |")
    rk = gaps["random_k_null"]
    md += ["", "Random-k null Sharpe 5th / 50th / 95th percentile: " + "; ".join(f"k={k_}: {rk[str(k_)]['sr_p5_50_95'][0]:.2f} / {rk[str(k_)]['sr_p5_50_95'][1]:.2f} / {rk[str(k_)]['sr_p5_50_95'][2]:.2f}" for k_ in F.K_GRID) + ".", "",
           f"Ranking diagnostics: global IC {_rng(ic, '{:.4f}')}; local IC within the predicted top 5% / 10% / 20%: "
           + "; ".join(f"q={q}: {_rng([gp[k]['local_ic'][str(q)] for k in S], '{:.3f}')}" for q in F.LOCAL_IC_Q)
           + f". Calibration (mean realised next-day return per predicted decile, bp/day; overall mean {gp[S[0]]['calibration']['overall_mean_ret'] * 1e4:.1f}):", "",
           "| seed | " + " | ".join(f"d{i + 1}" for i in range(F.CALIB_BINS)) + " | top - bottom |", "|---|" + "---|" * (F.CALIB_BINS + 1)]
    for s, k in zip(SEEDS, S):
        c = gp[k]["calibration"]
        md.append(f"| {s} | " + " | ".join(f"{x * 1e4:.1f}" for x in c["bin_mean_ret"]) + f" | {c['top_decile_minus_bottom_decile'] * 1e4:.1f} |")
    md += ["", f"Day-clustered 95% half-width of a single decile mean is about {_rng(cal_hw, '{:.0f}')} bp/day, so the decile profiles are not distinguishable from flat. Figures: `docs/figures/phase1_5a_margin.png`, `docs/figures/phase1_5a_calib.png`.", "",
           "**Reconciliation (T6 item 7).** NDCG@5 is "
           f"{_rng(nd, '{:.4f}')} against {gaps['random_ndcg5']:.4f} for random scores (100 draws, same function, days and mask), a gap of {_rng([x - gaps['random_ndcg5'] for x in nd], '{:+.4f}')}: the shifted-relevance NDCG has a high floor, so a gap of half a percent is tiny. "
           f"The hit rate of the realised top decile ({_rng(hit10, '{:.3f}')}) is above the random 0.10, but the realised bottom decile is hit {_rng(miss10, '{:.3f}')}: picking volatile stocks puts more mass in both tails. "
           f"The directional part is hit minus miss = {_rng([h - m for h, m in zip(hit10, miss10)], '{:+.3f}')} (about 1.5 to 3 percentage points, no confidence interval computed), consistent with the small positive IC and NDCG gap but not with a "
           f"calibrated top-tail effect: the top predicted decile is not reliably above the bottom ({_rng(tbd, '{:.1f}')} bp/day, CI about +/-{_rng(cal_hw, '{:.0f}')} bp). "
           "So top-tail skill (hit_top10) and NDCG@5 near random are compatible: both reflect a small directional lean on top of a large volatility/beta tilt. This is a weak positive sign for D, not support.", "",
           "Skipped by the predeclared sequential stopping rule (Gate A outcome 2): epsilon tie-group curves and score-jitter curves (T6 items 2-3); the json carries `{\"skipped\": \"gate A outcome 2\"}`.", "",
           "**What this shows:** tiny score gaps carry no consistent information about the next-day return (no monotone margin pattern; Spearman 0.03 to 0.10 with intervals including 0); the exact-tie bucket is positive in some seeds and negative in others. **What it does not show:** that no signal exists; each bucket holds 16 to 112 days.", ""]
    # trajectory
    md += ["## Epochs, seeds, controls (Task 7, history only)", "",
           "Epoch 0 means after the first training pass (about 93 Adam steps, `loop.py:211-243`), not an untrained model.", "",
           "| seed | selected epoch | test Sharpe epoch 0 | selected | last (epoch 99) | best-test (diagnostic) | mean over 100 epochs | corr over epochs of test Sharpe with pred SD | with test IC |", "|---|---|---|---|---|---|---|---|---|"]
    for s, k in zip(SEEDS, S):
        x = T[k]
        md.append(f"| {s} | {x['selected_epoch']} | {x['test_sr_epoch0']:.2f} | {x['test_sr_selected']:.2f} | {x['test_sr_last']:.2f} | {x['test_sr_oracle_diagnostic']:.2f} | {x['test_sr_mean_over_epochs']:.2f} | "
                  f"{x['corr_over_epochs_test_sr_vs_pred_sd']:.2f} | {x['corr_over_epochs_test_sr_vs_test_ic']:.2f} |")
    em, lm = evl["early_mean"], evl["late_mean"]
    md += ["", f"Between-seed comparison (confounded with seed, not a within-run trajectory): seeds selected at epoch <= 2 ({evl['early_seeds(selected epoch<=2)']}) vs >= 8 ({evl['late_seeds(selected epoch>=8)']}): "
           f"tie rate {em['tie_rate']:.2f} vs {lm['tie_rate']:.2f}, median score SD {em['score_sd_median']:.1e} vs {lm['score_sd_median']:.1e}, median margin {em['margin5_median']:.1e} vs {lm['margin5_median']:.1e}, "
           f"top-5 stock share {em['top5_stock_share']:.2f} vs {lm['top5_stock_share']:.2f}, Sharpe {em['sharpe']:.2f} vs {lm['sharpe']:.2f}; mean daily basket Jaccard within early {evl['mean_daily_jaccard_within_early']:.2f}, within late {evl['mean_daily_jaccard_within_late']:.2f}, between {evl['mean_daily_jaccard_between']:.2f}. "
           "Seeds selected late have many more exact ties; with three versus two seeds this is a pattern, not a test.", "",
           "Not answerable from the artifacts (no weights saved; predictions kept only for the last validation-improving epoch):"]
    md += [f"- {x}" for x in traj["not_answerable_from_artifacts"]]
    md += ["", f"Seeds: {traj['seed_note']}.", "",
           "Weight-decay control (`docs/phase1_5/F_learnability.md` lines 7 and 11-12): same planted signal, data, code path and seeds, only weight decay changed; THINK (HH_hyper) IC/oracle 26% -> 63-84% with wd 0, decay gradient 10-40x the loss gradient, `|z|` shrinks multiplicatively over three stacked layers. "
           f"Narrow conclusion: {traj['wd_control']['narrow_conclusion']}. No new run is recommended. Figure: `docs/figures/phase1_5a_trajectory.png`.", "",
           "**What this shows:** a test Sharpe of 1.8 to 2.5 is already present after the first training pass in every seed and averages 1.6 to 2.0 across all 100 epochs, so it is not the product of a specific selected epoch. **What it does not show:** anything within-run about baskets or ties.", ""]
    # decision table
    nb = [nl["seeds"][k]["beta_matched"]["sharpe"]["p"] for k in S]
    md += ["## Decision table", "",
           "| Hypothesis | Evidence for | Evidence against | Unresolved |", "|---|---|---|---|",
           f"| **A. Tie / index-order artifact** | Exact 5th/6th ties on {_rng([mech['runs'][pk][k]['tie_days']['n_days'] for k in S], '{:.0f}')} of 237 days (112 and 109 in seeds 0, 1); stable tie-break gives the lowest-index name. | Random exact-tie order keeps the median Sharpe above hold-all in every seed ({_rng([mech['runs'][pk][k]['random_tie']['p50'] for k in S])} vs {ha:.2f}); the saved stable value lies inside the random-tie 5-95% band for seeds 0-1; model-level equivariance holds; tie groups are small, not stale, isolated or masked; a constant-score first-5 basket gets {_rng([mech['runs'][pk][k]['constant_first5']['sr'] for k in S])}. | Why ties occur at all (cause UNKNOWN); epoch-wise tie behaviour and the trained-model permutation test (not recoverable). |",
           f"| **B. Luck / concentration / market exposure** | Hold-all Sharpe {ha:.2f}; basket beta {_rng([de[k]['exposure']['basket_mean_beta'] for k in S])} vs 1.00; Sharpe at about the {_rng(pct, '{:.0f}').replace(' to ', '-')} percentile of random daily top-5, F1 p {_rng(p1, '{:.3f}')}, Holm {fam['holm_adjusted']['F1']:.2f}; F2-F4 also not rejected; 56-133 distinct stocks and one stock-day (SYX) about {_rng([x * 100 for x in syx_sh], '{:.0f}')}% of P&L; test Sharpe {_rng(e0)} already after the first training pass; net of 10 bp the Sharpe is {_rng(net10)}, below hold-all. | Beta-matched p is {_rng(nb, '{:.3f}')} (seed 4 {min(nb):.3f} < 0.05 before correction, Holm family {fam['holm_adjusted']['F2']:.2f}); the industry-matched null is weak because several over-weighted industries hold one stock. | Power (CI {min(c[0] for c in ci):.2f} to {max(c[1] for c in ci):.2f}); other years; cap-weighted market UNKNOWN; whether a better exposure control would reject. |",
           f"| **C. Data / backtest defect** | One stock-day (SYX 2017-03-27, +{syx['ret'] * 100:.1f}%) is selected by all seeds and is about {_rng([x * 100 for x in syx_sh], '{:.0f}')}% of P&L (not verified externally); {_rng([ig[k]['stale_in_basket'] for k in S], '{:.0f}')} stale-close stock-days per seed in baskets. | Existing 15 data tests pass; 25/25 run-seeds recompute; 0 duplicate series; 0 selected stock-days with a fill value; no reversal after either abs(ret) > 0.2 day; stale share of P&L about -0.3%; no look-ahead in `norm=train`. | SYX event authenticity (UNKNOWN); the reference `R5_f_paper` arm uses the full-series max (look-ahead) and is secondary only. |",
           f"| **D. Genuine top-tail skill** | IC {_rng(ic, '{:.4f}')} > 0 in all five seeds; NDCG@5 {_rng(nd, '{:.4f}')} vs random {gaps['random_ndcg5']:.4f}; realised-top-decile hit {_rng(hit10, '{:.3f}')} vs bottom-decile {_rng(miss10, '{:.3f}')} (directional gap about 1.5 to 3 points); top-5 beats the random-5 mean by {_rng(exc5, '{:.1f}')} bp/day. | Top-minus-bottom predicted decile {_rng(tbd, '{:.1f}')} bp/day with CI about +/-{_rng(cal_hw, '{:.0f}')} bp; no monotone margin bucket pattern, Spearman {_rng([r_['rho'] for r_ in rho], '{:.2f}')} with CIs including 0; local IC about 0; F1-F4 not rejected; excess over random-5 is of the size of its own sampling noise. | Not excluded, only unsupported; a small real signal cannot be detected with 237 days. Needs more years or independent seeds (see `PROPOSAL_followups.md`). |", "",
           "## Verdict", "",
           f"**Sharpe near 2 in the R5_f2 alpha=0 THINK 2017 run is not evidence of learned stock ranking (NO EVIDENCE).** The five per-seed Sharpes ({_rng(sharpes)}) are at about the {_rng(pct, '{:.0f}').replace(' to ', '-')} percentile of random daily top-5 baskets in a year where equal-weight hold-all already gets {ha:.2f}; "
           f"F1-F4 are not rejected (family p F1 {fam['iut_family_p_max_over_seeds']['F1']:.3f}, F2 {fam['iut_family_p_max_over_seeds']['F2']:.3f}, F3 {fam['iut_family_p_max_over_seeds']['F3']:.3f}, F4 {fam['iut_family_p_max_over_seeds']['F4']:.3f}; Holm {_rng(list(fam['holm_adjusted'].values()))}); "
           f"the block-bootstrap Sharpe 95% intervals are wide ({ci[0][0]:.2f} to {ci[0][1]:.2f} in seed 0). A weak genuine signal (hypothesis D) is neither supported nor excluded. This is a statement about this exploratory 2017 sample and these seeds (not independent), not about the authors' model.", "",
           "## Skipped by the predeclared sequential stopping rule (Gate A outcome 2)", "",
           "- T4 items 3-4: per-stock contribution, exclude-and-reselect, best-day removal, momentum benchmark, daily excess-series CIs.",
           "- T6 items 2-3: epsilon tie-group curves and score-jitter curves.",
           "Each is written `skipped by the predeclared sequential stopping rule (Gate A outcome 2)` in `decompose.json` / `gaps.json`; the full script run writes `{\"skipped\": \"gate A outcome 2\"}` for the T6 curves and for the T4 `contribution` and `costs_benchmarks` keys.", "",
           "## Limitations", "",
           "- Not recoverable (`inventory.json`): " + "; ".join(inv["not_recoverable"]) + ".",
           "- 2017 is an exploratory year; the five seeds share the days, the ordering and the evaluator, so they are repeated runs, not independent samples. Formal family: F1-F4 only (intersection-union over seeds, Holm); everything else is exploratory.",
           "- Market = equal-weight hold-all of valid stocks; cap-weighted market UNKNOWN (not in the data).",
           "- Matched nulls resample within a stratum with replacement across draws; beta uses training-period data only; industry strata include singleton industries and a large `n/a` bucket.",
           "- Hit-rate and decile gaps have no formal interval except where stated; the margin-bucket intervals use a stationary block bootstrap over the bucket's days in time order, which only approximates dependence.",
           "- Epoch 0 means after the first training pass (about 93 Adam steps), never untrained.", ""]
    # evidence index
    md += ["## Evidence file index", "", "Large per-stock-day exports are in `results/forensics_1_5a/` (git-ignored). Committed files (sha256, first 12 hex):", "", "| file | sha256 |", "|---|---|"]
    for pth in sorted(DOCS.glob("*.json")) + sorted(FIGS.glob("phase1_5a_*.png")):
        md.append(f"| `{pth.relative_to(ROOT).as_posix()}` | {hashlib.sha256(pth.read_bytes()).hexdigest()[:12]} |")
    md += ["", "Figures: persistence, margin, calib, trajectory, mechanism, proxy, nulls (`docs/figures/phase1_5a_*.png`).", "", "## Test results", ""]
    tr = DOCS / "test_results.txt"
    md += [tr.read_text().rstrip(), ""] if tr.exists() else ["(pending: run the three commands in the plan, Task 8 Step 3)", ""]
    (DOCS / "REPORT_2017.md").write_text("\n".join(md), encoding="utf-8")
    print("report written:", len(md), "lines; gate A:", json.dumps({k: v for k, v in gate_out.items() if k != "per_seed"}))


STAGES = {"inventory": stage_inventory, "mechanism": stage_mechanism, "proxy": stage_proxy, "nulls": stage_nulls, "integrity": stage_integrity,
          "decompose": stage_decompose, "gaps": stage_gaps, "trajectory": stage_trajectory, "report": stage_report}


def main():
    global PRIMARY, REFERENCES, REF_RUNS, SEEDS, RESULTS, DOCS, FIGS, FIXTURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--out", default="results/forensics_1_5a")
    ap.add_argument("--quick", action="store_true", help="debug only: tiny B/R")
    ap.add_argument("--root", default=None, help="results root (default: results/)")
    ap.add_argument("--runs", default=None, help="exp/label of the primary run, e.g. FIX/HH (replaces primary and references)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--fixture", action="store_true", help="synthetic fixture: skip steps that need real NYSE data")
    ap.add_argument("--docs", default=None, help="redirect doc outputs (json, md, figures)")
    a = ap.parse_args()
    os.chdir(ROOT)
    lo, _, hi = a.seeds.partition("-")
    SEEDS = tuple(range(int(lo), int(hi) + 1)) if hi else (int(lo),)
    if a.root:
        RESULTS = Path(a.root)
    if a.runs:
        exp, label = a.runs.split("/")
        PRIMARY, REFERENCES, REF_RUNS = (exp, label), (), ()
    FIXTURE = bool(a.fixture)
    if a.docs:
        DOCS = Path(a.docs)
        FIGS = DOCS / "figures"
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    order = ["inventory", "mechanism", "proxy", "nulls", "integrity", "decompose", "gaps", "trajectory", "report"]
    for name in (order if a.stage == "all" else [a.stage]):
        STAGES[name](a)


if __name__ == "__main__":
    main()
