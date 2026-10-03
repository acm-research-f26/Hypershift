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


#@@STAGES@@
STAGES = {"inventory": stage_inventory, "mechanism": stage_mechanism, "proxy": stage_proxy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--out", default="results/forensics_1_5a")
    ap.add_argument("--quick", action="store_true", help="debug only: tiny B/R")
    a = ap.parse_args()
    os.chdir(ROOT)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        STAGES[name](a)


if __name__ == "__main__":
    main()
