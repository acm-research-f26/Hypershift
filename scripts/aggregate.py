"""python scripts/aggregate.py [--select-tuning] [--select-attn] [--costs]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypershift.eval.metrics import sharpe, topk_daily_returns_net
from hypershift.eval.stats import (
    holm, sharpe_contrast_ci, sharpe_diff_ci, verdict, wilcoxon_one_sample, wilcoxon_paired,
)
from hypershift.experiments.grid import FAMILIES, INTERACTIONS

R = Path("results")
T = R / "tables"
# NYSE SR from Table II. 1.18 sits only on the paper-protocol row, so it is never read against the honest THINK row.
PAPER = {"E1_main/THINK_paperProtocol": 1.18, "E2_geometry/EH": 1.14}


def load_runs() -> pd.DataFrame:
    rows = []
    for mf in R.glob("*/*/seed_*/metrics.json"):
        m = json.loads(mf.read_text())
        if "val" not in m:                     # e.g. Task 18 classification runs: not a ranking run
            continue
        exp, label, seed = mf.parts[-4], mf.parts[-3], int(mf.parts[-2].split("_")[1])
        rows.append({"exp": exp, "label": label, "key": f"{exp}/{label}", "seed": seed,
                     "val_sr": m["val"]["sr"], "test_sr": m["test"]["sr"], "test_ndcg5": m["test"]["ndcg5"],
                     "test_irr": m["test"]["irr"], "test_mse": m["test"]["mse"], "test_mdd": m["test"]["mdd"],
                     "test_ndcg_sthan": m["test"]["ndcg_sthan"], "test_oracle_sr": m["test_oracle_sr"],
                     "epochs_run": m["epochs_run"], "num_edges": m.get("num_edges", np.nan),
                     "covered_frac": m.get("covered_frac", np.nan), "dir": str(mf.parent)})
    return pd.DataFrame(rows)


def n_failed(key: str) -> int:
    exp, label = key.split("/")
    return len(list((R / exp / label).glob("seed_*/failed.json")))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["test_sr", "test_ndcg5", "test_irr", "test_mse", "test_mdd", "val_sr", "test_oracle_sr"]
    g = df.groupby("key")[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = df.groupby("key").size()
    g["n_failed"] = [n_failed(k) for k in g.index]
    return g.reset_index()


def seed_series(df, key, seeds=None):
    """Per-seed test SR, and the daily top-k return averaged over `seeds` (default: all seeds of `key`).

    Always pass the seeds common to every compared arm: an ensemble over more seeds is smoother, so its
    Sharpe would be mechanically higher and bias the bootstrap CI.
    """
    sub = df[df.key == key].sort_values("seed")
    if seeds is not None:
        sub = sub[sub.seed.isin(list(seeds))]
    daily = np.mean([np.load(Path(d) / "test_daily.npy") for d in sub.dir], axis=0)
    return sub.set_index("seed")["test_sr"], daily


def _common_seeds(df, keys):
    common = None
    for k in keys:
        s = set(df[df.key == k].seed)
        common = s if common is None else common & s
    return sorted(common)


def compare_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, pairs in FAMILIES.items():
        fam_rows = []
        for a, b in pairs:
            if a not in set(df.key) or b not in set(df.key):
                continue
            common = _common_seeds(df, (a, b))
            sa, da = seed_series(df, a, common)
            sb, db = seed_series(df, b, common)
            ci = sharpe_diff_ci(da, db)
            fam_rows.append({"family": fam, "A": a, "B": b, "n": len(common), "meanA": sa.mean(), "meanB": sb.mean(),
                             "diff": sa.mean() - sb.mean(),
                             "p": wilcoxon_paired(sa[common], sb[common]) if len(common) >= 6 else np.nan,
                             "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        rows += _holm(fam_rows)
    for fam, quads in INTERACTIONS.items():
        fam_rows = []
        for q in quads:
            if not all(k in set(df.key) for k in q):
                continue
            common = _common_seeds(df, q)
            s = [seed_series(df, k, common) for k in q]
            inter = (s[0][0][common] - s[1][0][common]) - (s[2][0][common] - s[3][0][common])
            ci = sharpe_contrast_ci([x[1] for x in s], [1, -1, -1, 1])
            fam_rows.append({"family": fam, "A": f"({q[0]} - {q[1]})", "B": f"({q[2]} - {q[3]})", "n": len(common),
                             "meanA": np.nan, "meanB": np.nan, "diff": inter.mean(),
                             "p": wilcoxon_one_sample(inter) if len(common) >= 6 else np.nan,
                             "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        rows += _holm(fam_rows)
    return pd.DataFrame(rows)


def _holm(fam_rows):
    ps = {i: r["p"] for i, r in enumerate(fam_rows) if not np.isnan(r["p"])}
    adj = holm(ps) if ps else {}
    m = max(len(ps), 1)
    for i, r in enumerate(fam_rows):
        r["p_holm"] = adj.get(i, np.nan)
        r["p_floor"] = min(1.0, m * 2.0 ** (1 - r["n"]))       # smallest Holm p attainable with n seeds
        if i not in adj or (r["p_floor"] >= 0.01 and r["p_holm"] >= 0.01):
            r["verdict"] = "INSUFFICIENT SEEDS"
        else:
            r["verdict"] = verdict(r["p_holm"], r["ci_lo"], r["ci_hi"])
    return fam_rows


def select_tuning(df):
    """Best mean validation SR per geometry over E_tune labels '<G>_lr<lr>_a<alpha>'."""
    tune = df[df.exp == "E_tune"].groupby("label")["val_sr"].mean()
    out = {}
    for g in ("HH", "HE", "EH", "EE"):
        cand = tune[tune.index.str.startswith(g + "_")]
        if cand.empty:
            raise SystemExit(f"no E_tune runs for {g}; run `python scripts/run_grid.py E_tune` first")
        _, lr, a = cand.idxmax().split("_")
        out[g] = {"lr": float(lr[2:]), "alpha": float(a[1:])}
    (R / "tuned.json").write_text(json.dumps(out, indent=2))
    print(tune.to_string(), "\ntuned:", out)


def select_attn(df):
    att = df[df.exp == "E_attn"].groupby("label")["val_sr"].agg(["mean", "std"])
    att = att[~att.index.str.endswith("_off")].sort_values("mean", ascending=False)
    top = att.index[0]
    choice = top
    if "mobius_mult" in att.index and att.loc[top, "mean"] - att.loc["mobius_mult", "mean"] < att.loc[top, "std"]:
        choice = "mobius_mult"                    # decision node D8: keep the paper's literal form unless clearly beaten
    sc, di = choice.split("_")
    Path("configs/chosen.yaml").write_text(yaml.safe_dump({"attn_score": sc, "attn_dist": di}))
    print(att, "\nchosen:", choice)


def costs_table(df):
    lines = ["| arm | cost bps | SR mean | SR std |", "|---|---|---|---|"]
    for key in sorted(k for k in df.key.unique() if k.startswith("E10_hourly/")):
        ppy = 1764 if "/hourly_" in key else 252
        for bps in (0, 5, 10):
            srs = []
            for d in df[df.key == key].dir:
                p, g, m = (np.load(Path(d) / f) for f in ("test_pred.npy", "test_gt.npy", "test_mask.npy"))
                srs.append(sharpe(topk_daily_returns_net(p, g, m, 5, bps), ppy))
            lines.append(f"| {key} | {bps} | {np.mean(srs):.3f} | {np.std(srs):.3f} |")
    (T / "costs.md").write_text("\n".join(lines) + "\n")


def table2_like(summ):
    """'our test SR' = validation-selected epoch (honest). 'best-test-epoch SR' = the paper/STHAN-SR protocol
    (optimistic; comparable to the paper's 1.18 only on the THINK_paperProtocol row)."""
    lines = ["| model | our test SR (val-selected) | std | best-test-epoch SR | our NDCG@5 | paper SR |",
             "|---|---|---|---|---|---|"]
    paper = PAPER
    for _, r in summ.iterrows():
        if r.key.startswith(("E1_main/", "E2_geometry/", "E3_hhn/", "E5_structure/")):
            lines.append(f"| {r.key} | {r.test_sr_mean:.3f} | {r.test_sr_std:.3f} | {r.test_oracle_sr_mean:.3f} | "
                         f"{r.test_ndcg5_mean:.3f} | {paper.get(r.key, '')} |")
    for f in sorted((R / "baselines").glob("NYSE.json")):
        for k, v in json.loads(f.read_text()).items():
            lines.append(f"| baseline {k} (NYSE) | {v['sr']:.3f} | | | {v.get('ndcg5', float('nan')):.3f} | |")
    (T / "table2_like.md").write_text("\n".join(lines) + "\n")


def universe_table(df):
    """Q5. One row per (N, universe) cell plus the full-universe point; Spearman trends over the cells."""
    from scipy.stats import spearmanr

    hyp_file = R / "hyperbolicity.json"
    hyp = json.loads(hyp_file.read_text()) if hyp_file.exists() else {}
    cells = [(n, u, f"E8_universe/HH_N{n}_u{u}", f"E8_universe/EE_N{n}_u{u}", f"NYSE_N{n}_u{u}")
             for n in (50, 100, 250, 500, 1000) for u in (0, 1, 2)]
    cells.append((1737, 0, "E1_main/THINK", "E2_geometry/EE", "NYSE"))
    rows = []
    for n, u, kh, ke, bname in cells:
        if kh not in set(df.key) or ke not in set(df.key):
            continue
        common = _common_seeds(df, (kh, ke))
        sh, _ = seed_series(df, kh, common)
        se, _ = seed_series(df, ke, common)
        bfile = R / "baselines" / f"{bname}.json"
        rnd = json.loads(bfile.read_text())["random"]["sr"] if bfile.exists() else np.nan
        hrow = df[df.key == kh]
        hkey = "NYSE_full" if n == 1737 else f"NYSE_N{n}_u{u}"      # keys written by scripts/hyperbolicity.py
        rows.append({"N": n, "u": u, "n_seeds": len(common), "HH_sr": sh.mean(), "EE_sr": se.mean(),
                     "gap_HH_minus_EE": (sh - se).mean(), "random5_sr": rnd,
                     "HH_excess": sh.mean() - rnd, "EE_excess": se.mean() - rnd,
                     "num_edges": hrow.num_edges.mean(), "covered_frac": hrow.covered_frac.mean(),
                     "delta_hg": hyp.get(hkey, {}).get("hg", {}).get("delta_max", np.nan)})
    if not rows:
        return
    U = pd.DataFrame(rows)
    trend = []
    for x, y in (("N", "gap_HH_minus_EE"), ("N", "HH_excess"), ("N", "EE_excess"), ("delta_hg", "gap_HH_minus_EE")):
        ok = U[[x, y]].dropna()
        if len(ok) >= 5 and ok[x].nunique() > 1:
            rho, p = spearmanr(ok[x], ok[y])
            trend.append({"x": x, "y": y, "cells": len(ok), "spearman_rho": rho, "p": p,
                          "verdict": "STRONG trend" if p < 0.01 else "NO EVIDENCE of a trend"})
    text = "## Cells (E8: 5 seeds each; N=1737 row = E1/E2 arms)\n\n" + U.to_markdown(index=False, floatfmt=".3f")
    text += "\n\n## Trends\n\n" + (pd.DataFrame(trend).to_markdown(index=False, floatfmt=".4f") if trend else "none")
    text += ("\n\nCaveat: random subsets thin the hypergraph (see num_edges, covered_frac), so N and "
             "hypergraph density are confounded.\n")
    (T / "universe.md").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-tuning", action="store_true")
    ap.add_argument("--select-attn", action="store_true")
    ap.add_argument("--costs", action="store_true")
    args = ap.parse_args()
    T.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    if args.select_tuning:
        select_tuning(df)
    if args.select_attn:
        select_attn(df)
    summ = summary(df)
    summ.to_csv(T / "summary.csv", index=False)
    comp = compare_all(df)
    comp.to_csv(T / "comparisons.csv", index=False)
    (T / "comparisons.md").write_text(comp.to_markdown(index=False, floatfmt=".4f") if len(comp) else "none\n")
    table2_like(summ)
    universe_table(df)
    if args.costs:
        costs_table(df)
    print(summ.to_string())
    print(comp.to_string())
