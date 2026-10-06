"""W5 analysis: read results/w5_delta/*.jsonl -> docs/phase2/W5_DELTA_RESULTS.{md,json} (rule in W5_DELTA_SPEC.md)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/w5_delta")
TARGET = {"NYSE": {"hg": 0.5, "rel": 0.087}, "NASDAQ": {"hg": 1.0, "rel": 0.107}}
TOL = {"hg": 0.25, "rel": 0.01}
MK = ("NYSE", "NASDAQ")


def load(name):
    p = OUT / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def hg_cells(rows):
    """-> {(market, setting): (value, share_hit, n)} ; setting = (graph, s, scheme, k, rule)"""
    out = {}
    for r in rows:
        m, g, s = r["market"], r["graph"], r["s"]
        t = TARGET[m]["hg"]
        for c in r["cells"]:
            for rule, vals in (("single", c["single"]), ("allbases", c["all"])):
                if vals:
                    v = np.array(vals)
                    out[(m, (g, s, c["scheme"], c["k"], rule))] = (float(np.median(v)), float(np.mean(abs(v - t) <= TOL["hg"])), len(v))
        f = np.array(r["full_single_bases"])
        out[(m, (g, s, "full", "LCC", "single-median"))] = (float(np.median(f)), float(np.mean(abs(f - t) <= TOL["hg"])), len(f))
        out[(m, (g, s, "full", "LCC", "allbases-LB24"))] = (float(f.max()), float(abs(f.max() - t) <= TOL["hg"]), len(f))
    return out


def rel_cells(rows):
    """-> {(market, setting): (mean, std)}; setting = (feature, norm, nominal_m, replace)"""
    out = {}
    for r in rows:
        m = r["market"]
        cells = {(c["m"], c["replace"]): c for c in r["cells"]}
        N = max(c["m"] for c in r["cells"])
        for nom in (500, 1000, 1500, 2000, "all"):
            mm = N if nom == "all" else min(nom, N)
            for repl in (True, False):
                c = cells.get((mm, repl))
                if c:
                    out[(m, (r["feature"], r["norm"], str(nom), repl))] = (c["mean"], c["std"])
    return out


def matches(cells, kind):
    """settings matching per market and jointly"""
    per = {m: {s for (mm, s), v in cells.items() if mm == m and abs(v[0] - TARGET[m][kind]) <= TOL[kind]} for m in MK}
    allset = {m: {s for (mm, s) in cells if mm == m} for m in MK}
    return per, allset, per["NYSE"] & per["NASDAQ"]


def main():
    hg, rel = hg_cells(load("hg")), rel_cells(load("rel"))
    res = {"tol": TOL, "target": TARGET}
    L = ["# W5 results: delta_hg / delta_rel grid vs THINK Table I", "",
         "Spec (predeclared, committed before computing): `docs/phase2/W5_DELTA_SPEC.md`. Raw: `results/w5_delta/*.jsonl` (git-ignored); summary json `W5_DELTA_RESULTS.json`.",
         f"Targets (paper p.850 Table I): NYSE delta_hg 0.5, delta_rel 0.087; NASDAQ delta_hg 1.0, delta_rel 0.107. Tolerances: delta_hg +-0.25, delta_rel +-0.01.", ""]
    for kind, cells in (("hg", hg), ("rel", rel)):
        per, allset, joint = matches(cells, kind)
        res[kind] = {"total_cells": {m: len(allset[m]) for m in MK}, "match": {m: len(per[m]) for m in MK},
                     "joint_match": len(joint), "joint_settings": sorted(map(list, joint))[:200]}
        L += [f"## Match counts, delta_{kind}", "",
              f"| | NYSE | NASDAQ | both (same setting) |", "|---|---|---|---|",
              f"| cells scanned | {len(allset['NYSE'])} | {len(allset['NASDAQ'])} | {len(allset['NYSE'] & allset['NASDAQ'])} |",
              f"| cells within tolerance | {len(per['NYSE'])} | {len(per['NASDAQ'])} | {len(joint)} |", ""]
    # ---- delta_hg tables
    L += ["## delta_hg tables (median over draws; share of draws within tolerance in parentheses)", ""]
    for m in MK:
        L += [f"### {m}, target {TARGET[m]['hg']}", ""]
        meta = {(r["graph"], r["s"]): r for r in load("hg") if r["market"] == m}
        L += ["Graph / s: edges, LCC nodes, diameter of LCC", "", "| graph | s | edges | LCC | diam |", "|---|---|---|---|---|"]
        for (g, s), r in sorted(meta.items()):
            L.append(f"| {g} | {s} | {r['n_edges']} | {r['lcc']} | {r['diam']:.0f} |")
        L.append("")
        for scheme, rule in (("dist", "single"), ("dist", "allbases"), ("induced", "single"), ("induced", "allbases")):
            L += [f"**scheme={scheme}, base rule={rule}** (rows graph,s; columns k)", "",
                  "| graph | s | " + " | ".join(str(k) for k in (10, 20, 30, 50, 100, 200, 500)) + " |", "|---|---|" + "---|" * 7]
            for (g, s) in sorted(meta):
                row = []
                for k in (10, 20, 30, 50, 100, 200, 500):
                    v = hg.get((m, (g, s, scheme, k, rule)))
                    row.append("-" if v is None else f"{v[0]:.1f} ({v[1]:.2f})")
                L.append(f"| {g} | {s} | " + " | ".join(row) + " |")
            L.append("")
        L += ["**full LCC** (24 random base points: median / min / max single-base delta; max = lower bound on exact)", "",
              "| graph | s | median | min | max | share of bases within tol |", "|---|---|---|---|---|---|"]
        for (g, s), r in sorted(meta.items()):
            f = np.array(r["full_single_bases"])
            L.append(f"| {g} | {s} | {np.median(f):.1f} | {f.min():.1f} | {f.max():.1f} | {np.mean(abs(f - TARGET[m]['hg']) <= TOL['hg']):.2f} |")
        L.append("")
    ex = defaultdict(list)
    for r in load("exact"):
        ex[r["market"]].append(r["delta"])
    if ex:
        L += ["### Exact all-bases delta_hg, v2 graph, s=1 (every LCC node as base point)", ""]
        for m, v in ex.items():
            v = np.array(v)
            L.append(f"- {m}: bases {len(v)}, exact delta = {v.max():.1f}, single-base min {v.min():.1f}, median {np.median(v):.1f}")
            res.setdefault("exact", {})[m] = {"n": len(v), "max": float(v.max()), "min": float(v.min()), "median": float(np.median(v))}
        L.append("")
    # ---- closest settings
    L += ["## Closest settings per target", ""]
    for kind, cells in (("hg", hg), ("rel", rel)):
        for m in MK:
            t = TARGET[m][kind]
            items = sorted(((abs(v[0] - t), v[0], s) for (mm, s), v in cells.items() if mm == m), key=lambda x: x[0])
            L += [f"delta_{kind} {m} (target {t}): {sum(1 for i in items if i[0] <= TOL[kind])} of {len(items)} cells within tol; closest 6:", ""]
            for d, v, s in items[:6]:
                L.append(f"- {v:.4f} at {s}")
            L.append("")
    # ---- delta_rel feature summary
    L += ["## delta_rel by feature (Khrulkov convention: with replacement, base 0, 20 subsets, m = all)", "",
          "| feature | norm | NYSE | NASDAQ |", "|---|---|---|---|"]
    keys = sorted({s[:2] for (m, s) in rel})
    for f, n in keys:
        a = rel.get(("NYSE", (f, n, "all", True)))
        b = rel.get(("NASDAQ", (f, n, "all", True)))
        L.append(f"| {f} | {n} | {a[0]:.4f} | {b[0]:.4f} |" if a and b else f"| {f} | {n} | - | - |")
    L.append("")
    for m in MK:
        rng = [v[0] for (mm, s), v in rel.items() if mm == m and s[0] not in ()]
        L.append(f"- {m}: delta_rel over all cells ranges {min(rng):.4f} to {max(rng):.4f}; target {TARGET[m]['rel']}")
    L.append("")
    Path("docs/phase2/W5_DELTA_RESULTS.json").write_text(json.dumps(res, indent=1))
    Path("docs/phase2/W5_DELTA_RESULTS_tables.md").write_text("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
