"""Phase 2 W3: protocol matrix. Rescore SAVED predictions (no training) under every documented reading of the paper's protocol.

Axes
  selection : "validation" = the run's own saved epoch (max validation Sharpe, ours, k=5; leak-free)
              "oracle"     = epoch maximising the TEST value of the very metric reported (selection on the test year: LEAKY)
  norm      : train (leak-free) | paper (full-series max, look-ahead: LEAKY); only where both runs exist
  Sharpe    : ours      = mean/std * sqrt(252), rf 0           (src/hypershift/eval/metrics.py)
              unann     = mean/std, rf 0                        (paper p.852 Sec. IV-B prints E[Ra-Rf]/std[Ra-Rf], no annualisation)
              unann_rf  = (mean-rf)/std, rf = 2017 3-month T-bill daily (INFERRED: the paper does not state Rf; value = FRED DTB3
                          2017 mean 0.93412 % p.a. over 250 obs / 252)
  k         : 1, 5, 10 (top-k equal weight, daily buy-hold)
  NDCG      : "correct" = NDCG@5 averaged over all days; "buggy" = STHAN-SR last-day ticker-index evaluator (ndcg_sthan)
Where only the validation-selected epoch's predictions are saved (everything except R5_f3), oracle cells come from
history.jsonl (per-epoch test sr, ann_vol, ndcg5, ndcg_sthan): Sharpe variants are recomputed exactly for k=5
(mean = sr*ann_vol/252, std = ann_vol/sqrt252); k != 5 oracle is NOT recoverable and is reported as NA.
Usage: python scripts/protocol_matrix.py [--out docs/phase2/protocol_matrix] [--root results]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from hypershift.eval.metrics import ndcg_at_k, ndcg_sthan_compat

KS = (1, 5, 10)
SQ = math.sqrt(252)
RF_ANNUAL_PCT = 0.93412          # FRED DTB3, mean of 250 daily observations in 2017 (INFERRED as the paper's Rf)
RF_DAILY = RF_ANNUAL_PCT / 100 / 252
VARIANTS = ("ours", "unann", "unann_rf")
PAPER = {  # Table II p.852, NYSE (mean of 25 runs, +- SD as printed)
    "THINK": {"sr": (1.18, 4e-3), "ndcg": (0.86, 9e-4)},
    "TCONV+DHHAN": {"sr": (1.14, 7e-3), "ndcg": (0.81, 1e-3)},
    "STHGCN": {"sr": (1.10, 3e-3), "ndcg": (0.78, 2e-3)},
    "RSR_I": {"sr": (1.05, 1e-3), "ndcg": (0.75, 6e-3)},
}
MODEL = {"HH": "THINK", "EH": "TCONV+DHHAN", "EE": "EE", "RSR_I": "RSR_I", "STHGCN": "STHGCN"}


# ---------------------------------------------------------------- core scoring
def topk_returns(pred, gt, mask, ks=KS):
    """{k: daily equal-weight top-k return}. Ties by lowest index (stable sort), same as hypershift.eval.metrics."""
    out = {k: np.zeros(pred.shape[1]) for k in ks}
    kmax = max(ks)
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-pred[idx, d], kind="stable")[:kmax]]
        for k in ks:
            out[k][d] = gt[top[:k], d].mean()
    return out


def sharpe_variants(r, rf=RF_DAILY):
    r = np.asarray(r, dtype=np.float64)
    mu, sd = float(r.mean()), float(r.std())
    if sd == 0:
        return {"ours": 0.0, "unann": 0.0, "unann_rf": 0.0}
    return {"ours": mu / sd * SQ, "unann": mu / sd, "unann_rf": (mu - rf) / sd}


def variants_from_hist(sr, ann_vol, rf=RF_DAILY):
    """Recover the three Sharpe variants (k=5) from history.jsonl's sr and ann_vol."""
    if ann_vol == 0:
        return {"ours": 0.0, "unann": 0.0, "unann_rf": 0.0}
    mu, sd = sr * ann_vol / 252, ann_vol / SQ
    return {"ours": sr, "unann": mu / sd, "unann_rf": (mu - rf) / sd}


def val_epoch(history):
    """Epoch the training loop saves: first max of validation Sharpe (strict >)."""
    best, be = None, 0
    for e, h in enumerate(history):
        if best is None or h["val"]["sr"] > best:
            best, be = h["val"]["sr"], e
    return be


def oracle_pick(values):
    """Index of the maximal test value (first max)."""
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------- per-run extraction
def extract_run(run_dir, rf=RF_DAILY):
    run_dir = Path(run_dir)
    hist = [json.loads(l) for l in open(run_dir / "history.jsonl") if l.strip()]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    be = metrics["best_epoch"]
    assert be == val_epoch(hist), (run_dir, be, val_epoch(hist))
    pred, gt, mask = (np.load(run_dir / f) for f in ("test_pred.npy", "test_gt.npy", "test_mask.npy"))
    rk = topk_returns(pred, gt, mask)
    val = {"epoch": be}
    for k in KS:
        v = sharpe_variants(rk[k], rf)
        val[f"sharpe_{k}"] = v
        val[f"irr_{k}"] = float(rk[k].sum())
    chk = abs(val["sharpe_5"]["ours"] - hist[be]["test"]["sr"])
    assert chk < 1e-6, (run_dir, chk)
    val["ndcg_correct"] = hist[be]["test"]["ndcg5"]
    val["ndcg_buggy"] = hist[be]["test"]["ndcg_sthan"]

    nd_c = [h["test"]["ndcg5"] for h in hist]
    nd_b = [h["test"]["ndcg_sthan"] for h in hist]
    exact = (run_dir / "epoch_preds").is_dir()
    orc = {}
    if exact:
        per = {k: [] for k in KS}      # [epoch] -> variants dict
        for e in range(len(hist)):
            ep = np.load(run_dir / "epoch_preds" / f"test_e{e:03d}.npy")
            r = topk_returns(ep, gt, mask)
            for k in KS:
                per[k].append(sharpe_variants(r[k], rf))
        assert abs(per[5][be]["ours"] - hist[be]["test"]["sr"]) < 1e-5
        for k in KS:
            for var in VARIANTS:
                arr = [p[var] for p in per[k]]
                e = oracle_pick(arr)
                orc[f"sharpe_{k}_{var}"] = {"value": arr[e], "epoch": e}
    else:
        per5 = [variants_from_hist(h["test"]["sr"], h["test"]["ann_vol"], rf) for h in hist]
        for var in VARIANTS:
            arr = [p[var] for p in per5]
            e = oracle_pick(arr)
            orc[f"sharpe_5_{var}"] = {"value": arr[e], "epoch": e}
        for k in (1, 10):
            for var in VARIANTS:
                orc[f"sharpe_{k}_{var}"] = None      # NA: needs per-epoch predictions
    for name, arr in (("ndcg_correct", nd_c), ("ndcg_buggy", nd_b)):
        e = oracle_pick(arr)
        orc[name] = {"value": arr[e], "epoch": e}
    return {"validation": val, "oracle": orc, "exact_oracle": exact, "n_epochs": len(hist)}


def cells_of(run, selection):
    """Flat {metric_name: value or None} for one run under one selection."""
    out = {}
    if selection == "validation":
        v = run["validation"]
        for k in KS:
            for var in VARIANTS:
                out[f"sharpe_{k}_{var}"] = v[f"sharpe_{k}"][var]
        out["ndcg_correct"], out["ndcg_buggy"] = v["ndcg_correct"], v["ndcg_buggy"]
    else:
        for name, o in run["oracle"].items():
            out[name] = None if o is None else o["value"]
    return out


def aggregate(runs_cells):
    """runs_cells: list of {metric: value|None} -> {metric: {mean, sd, n, per_seed} or None}."""
    out = {}
    for m in runs_cells[0]:
        vals = [c[m] for c in runs_cells]
        if any(v is None for v in vals):
            out[m] = None
            continue
        a = np.array(vals, dtype=np.float64)
        out[m] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a),
                  "per_seed": a.tolist()}
    return out


# ---------------------------------------------------------------- ordering and closeness
def paper_for(metric, model):
    if model not in PAPER:
        return None
    return PAPER[model]["ndcg" if metric.startswith("ndcg") else "sr"]


def compare(cells):
    """cells: {model: aggregated cell or None}. Ordering THINK > TCONV+DHHAN > STHGCN and closeness to Table II."""
    T, E, S = (cells.get(m) for m in ("THINK", "TCONV+DHHAN", "STHGCN"))
    res = {}
    res["think_gt_eh"] = None if not (T and E) else T["mean"] > E["mean"]
    res["eh_gt_sthgcn"] = None if not (E and S) else E["mean"] > S["mean"]
    res["full_order"] = None if None in (res["think_gt_eh"], res["eh_gt_sthgcn"]) else \
        bool(res["think_gt_eh"] and res["eh_gt_sthgcn"])
    res["z_think_eh"] = _z(T, E)
    res["z_eh_sthgcn"] = _z(E, S)
    # "sig": each adjacent gap is > 2 standard errors in the paper's direction (unpaired, crude; seeds are not paired across models)
    res["full_order_sig"] = None if res["full_order"] is None else bool(res["z_think_eh"] > 2 and res["z_eh_sthgcn"] > 2)
    return res


def _z(a, b):
    """(mean_a - mean_b)/SE, unpaired; None when a cell is missing."""
    if not (a and b):
        return None
    se = math.sqrt(a["sd"] ** 2 / max(a["n"], 1) + b["sd"] ** 2 / max(b["n"], 1))
    return float("inf") if se == 0 else (a["mean"] - b["mean"]) / se


def closeness(metric, cells):
    """Per model: |mean - paper| and 'value match' (<= max(paper SD, our seed SD), the W1-style rule)."""
    out = {}
    for model, c in cells.items():
        p = paper_for(metric, model)
        if c is None or p is None:
            continue
        diff = abs(c["mean"] - p[0])
        out[model] = {"mean": c["mean"], "paper": p[0], "abs_diff": diff, "tol": max(p[1], c["sd"]),
                      "match": bool(diff <= max(p[1], c["sd"])), "noise_dominated": bool(c["sd"] > 0.25 * p[0])}
    return out


# ---------------------------------------------------------------- driver
FAMILIES = {   # family -> (norm, {label: (exp, label)}, leaky_norm)
    "R5_f_paper": ("paper", {"THINK": ("R5_f_paper", "HH"), "TCONV+DHHAN": ("R5_f_paper", "EH"), "EE": ("R5_f_paper", "EE"),
                             "STHGCN": ("R8_f_paper", "STHGCN"), "RSR_I": ("R8_f_paper", "RSR_I")}),
    "R5_f_train": ("train", {"THINK": ("R5_f_train", "HH"), "TCONV+DHHAN": ("R5_f_train", "EH"), "EE": ("R5_f_train", "EE"),
                             "STHGCN": ("R8_f_train", "STHGCN"), "RSR_I": ("R8_f_train", "RSR_I")}),
    "R5_f2_alpha0_train": ("train", {"THINK": ("R5_f2_alpha0_train", "HH"), "TCONV+DHHAN": ("R5_f2_alpha0_train", "EH"),
                                     "STHGCN": ("R8_f_train", "STHGCN"), "RSR_I": ("R8_f_train", "RSR_I")}),
    "R5_f3_alpha0_train": ("train", {"THINK": ("R5_f3_alpha0_train", "HH"), "TCONV+DHHAN": ("R5_f3_alpha0_train", "EH"),
                                     "STHGCN": ("R8_f_train", "STHGCN"), "RSR_I": ("R8_f_train", "RSR_I")}),
}
METRICS = [f"sharpe_{k}_{v}" for k in KS for v in VARIANTS] + ["ndcg_correct", "ndcg_buggy"]


def load_group(root, exp, label, cache, rf):
    key = (exp, label)
    if key not in cache:
        runs = []
        for sd in sorted((Path(root) / exp / label).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
            if (sd / "metrics.json").exists():
                runs.append(extract_run(sd, rf))
        cache[key] = runs
    return cache[key]


def rsr_orig(path="docs/phase1_5/E_rsr_original_scores.json", rf=RF_DAILY):
    """The authors' RSR-I code: predictions are not saved locally (kaggle/build/out_rsr_full absent), so use the E json.
    Only k=5 and the k=5 Sharpe variants (from sr + ann_vol); selection: their rule / our rule (both leak-free) / oracle (leaky)."""
    j = json.load(open(path))
    out = {}
    for sel, key in (("their_rule", "their_rule"), ("validation", "our_rule"), ("oracle", "oracle")):
        cs = []
        for s in sorted(j):
            r = j[s][key]
            v = variants_from_hist(r["sr"], r["ann_vol"], rf)
            c = {f"sharpe_5_{var}": v[var] for var in VARIANTS}
            c["ndcg_correct"], c["ndcg_buggy"] = r["ndcg5"], r["ndcg_sthan"]
            cs.append(c)
        out[sel] = aggregate(cs)
    return out


def build(root="results"):
    cache, fams = {}, {}
    for fam, (norm, members) in FAMILIES.items():
        fams[fam] = {"norm": norm, "members": {m: list(v) for m, v in members.items()}, "selections": {}}
        for sel in ("validation", "oracle"):
            cells = {}
            for model, (exp, label) in members.items():
                runs = load_group(root, exp, label, cache, RF_DAILY)
                cells[model] = aggregate([cells_of(r, sel) for r in runs]) if runs else None
            n = {m: (len(cache[members[m]]) if members[m] in cache else 0) for m in members}
            rows = {}
            for metric in METRICS:
                mc = {m: (cells[m][metric] if cells[m] else None) for m in members}
                rows[metric] = {"cells": mc, "ordering": compare(mc), "closeness": closeness(metric, mc)}
            leaky = sel == "oracle" or norm == "paper"
            fams[fam]["selections"][sel] = {"leaky": leaky, "n_seeds": n, "rows": rows}
    exact = {f"{e}/{l}": all(r["exact_oracle"] for r in v) for (e, l), v in cache.items()}
    return {"rf_annual_pct": RF_ANNUAL_PCT, "rf_daily": RF_DAILY, "paper": PAPER, "families": fams,
            "exact_oracle_groups": exact, "rsr_original": rsr_orig()}


def fmt(c):
    return "NA" if c is None else f"{c['mean']:.3f}±{c['sd']:.3f}"


def to_md(res):
    L = ["# Phase 2 W3: protocol matrix (saved predictions rescored; no training)", "",
         "Generated by `scripts/protocol_matrix.py`. Paper values: Table II p.852 NYSE (THINK SR 1.18±4e-3 / NDCG 0.86±9e-4; "
         "TCONV+DHHAN 1.14±7e-3 / 0.81±1e-3; STHGCN 1.10±3e-3 / 0.78±2e-3; RSR-I 1.05±1e-3 / 0.75±6e-3; mean of 25 runs).", "",
         "## Definitions", "",
         "- **validation** selection = the run's saved epoch (max validation Sharpe, ours, k=5); leak-free. **oracle** = epoch maximising the TEST value of the reported cell itself (selection on the test year): **LEAKY**. `norm=paper` (full-series max) is look-ahead: **LEAKY**. A table row is leaky if either holds.",
         "- Sharpe `ours` = mean/std·√252, rf 0. `unann` = mean/std (paper p.852 Sec. IV-B prints E[Ra−Rf]/std[Ra−Rf] with no annualisation; whether it annualised is UNKNOWN). "
         f"`unann_rf` = (mean−rf)/std with rf = {RF_ANNUAL_PCT}%/252 per day = {RF_DAILY:.3e}: FRED DTB3 (3-month T-bill secondary market) mean over 250 daily 2017 observations; the paper does not state Rf, so this is INFERRED.",
         "- k = number of stocks held (equal weight, daily). NDCG `correct` = NDCG@5 over all test days; `buggy` = STHAN-SR last-day ticker-index evaluator (reference reading; the paper's 0.75-0.86 band matches the buggy one for no-skill models, Phase 1 E3).",
         "- Oracle cells for groups without saved per-epoch predictions use `history.jsonl`: Sharpe variants for **k=5 only** (exact, from sr and ann_vol); **k=1 and k=10 oracle = NA** for those groups. Exact per-epoch oracle (all k) only for R5_f3 HH/EH.",
         "- Ordering: `HH>EH` = mean THINK > mean TCONV+DHHAN; `EH>ST` = TCONV+DHHAN > STHGCN. STHGCN comes from R8_f_{norm} (our reimplementation); for R5_f2/R5_f3 families it is the R8_f_train STHGCN (cross-experiment, different training config). "
         "`match` = |our mean − paper| ≤ max(paper SD, our seed SD) for ALL of THINK, TCONV+DHHAN, STHGCN (the paper SDs are 1e-3..7e-3, so the tolerance is our seed SD: a wide-SD cell matches trivially; see the 'tight' count). Ordering is by means; `sig` additionally needs both adjacent gaps > 2 SE.", ""]
    s = res["summary"]
    L += ["## Summary (all combos listed below; counts only)", ""]
    for line in s["lines"]:
        L.append(f"- {line}")
    L.append("")
    for fam, f in res["families"].items():
        for sel, blk in f["selections"].items():
            n = blk["n_seeds"]
            L += [f"## {fam} | selection={sel} | norm={f['norm']} | {'LEAKY' if blk['leaky'] else 'leak-free'}",
                  f"seeds: " + ", ".join(f"{m} {n[m]}" for m in n), "",
                  "| metric | THINK | TCONV+DHHAN | EE | STHGCN | RSR_I | HH>EH | EH>ST | order | value match (T/E/S) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for metric, r in blk["rows"].items():
                c, o, cl = r["cells"], r["ordering"], r["closeness"]
                flag = lambda x: "NA" if x is None else ("Y" if x else "n")
                vm = "/".join("Y" if cl.get(m, {}).get("match") else ("n" if m in cl else "NA") for m in ("THINK", "TCONV+DHHAN", "STHGCN"))
                L.append(f"| {metric} | {fmt(c.get('THINK'))} | {fmt(c.get('TCONV+DHHAN'))} | {fmt(c.get('EE'))} | {fmt(c.get('STHGCN'))} | "
                         f"{fmt(c.get('RSR_I'))} | {flag(o['think_gt_eh'])} | {flag(o['eh_gt_sthgcn'])} | {flag(o['full_order'])} | {vm} |")
            L.append("")
    L += ["## RSR original code (authors' RSR-I, Phase 1.5 E; k=5 only; predictions not saved locally, numbers from E json)", "",
          "| selection | leaky | " + " | ".join(METRICS_RSR) + " |", "|---|---|" + "---|" * len(METRICS_RSR)]
    for sel, cells in res["rsr_original"].items():
        L.append(f"| {sel} | {'yes' if sel == 'oracle' else 'no'} | " + " | ".join(fmt(cells[m]) for m in METRICS_RSR) + " |")
    L.append("")
    return "\n".join(L)


METRICS_RSR = [f"sharpe_5_{v}" for v in VARIANTS] + ["ndcg_correct", "ndcg_buggy"]


def summarize(res):
    lines = []
    tot = {"value_all": [], "order_full": [], "order_sig": [], "order_any_eh": []}
    for fam, f in res["families"].items():
        for sel, blk in f["selections"].items():
            for metric, r in blk["rows"].items():
                cl = r["closeness"]
                allm = all(cl.get(m, {}).get("match") for m in ("THINK", "TCONV+DHHAN", "STHGCN"))
                if allm and all(not cl[m]["noise_dominated"] for m in ("THINK", "TCONV+DHHAN", "STHGCN")):
                    tot.setdefault("value_all_tight", []).append((fam, sel, metric, blk["leaky"]))
                anym = any(v["match"] for v in cl.values())
                tag = (fam, sel, metric, blk["leaky"])
                if allm:
                    tot["value_all"].append(tag)
                if r["ordering"]["full_order"]:
                    tot["order_full"].append(tag)
                if r["ordering"]["full_order_sig"]:
                    tot["order_sig"].append(tag)
                if r["ordering"]["think_gt_eh"] and anym:
                    tot["order_any_eh"].append(tag)
    tot.setdefault("value_all_tight", [])
    ncomb = sum(len(b["rows"]) for f in res["families"].values() for b in f["selections"].values())
    lines.append(f"Combos evaluated: {ncomb} (family x selection x metric cell).")
    for key, desc in (("value_all", "value match for THINK, TCONV+DHHAN and STHGCN simultaneously"),
                      ("value_all_tight", "  ... of which every model's seed SD <= 25% of the paper value (i.e. not just a wide tolerance)"),
                      ("order_full", "full ordering THINK > TCONV+DHHAN > STHGCN holds (means only)"),
                      ("order_sig", "full ordering holds AND both gaps > 2 SE (crude, unpaired)")):
        tl = tot[key]
        lines.append(f"{desc}: {len(tl)} combos ({sum(1 for t in tl if not t[3])} leak-free, {sum(1 for t in tl if t[3])} leaky).")
        for fam, sel, metric, leaky in tl:
            lines.append(f"  - {fam} | {sel} | {metric} | {'LEAKY' if leaky else 'leak-free'}")
    res["summary_lists"] = {k: [list(t) for t in v] for k, v in tot.items()}
    return {"lines": lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="docs/phase2/protocol_matrix")
    a = ap.parse_args()
    res = build(a.root)
    res["summary"] = summarize(res)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(a.out + ".md").write_text(to_md(res), encoding="utf-8")
    print("\n".join(res["summary"]["lines"]))


if __name__ == "__main__":
    main()
