"""Phase 2 W1: authors' RSR-I run (Phase 1.5 E) vs the RSR paper's NYSE numbers.

Paper: Feng et al. 2019, TOIS, arXiv 1809.09441 (copy used: 27-page arXiv PDF). Table 6 (industry relations only,
arXiv-PDF p.15) is the like-for-like table for the README NYSE command (-rn default = sector_industry, -ip 0).
Predeclared rule (docs/phase2/SPEC.md + task): reproduces iff |our 5-seed mean - paper value| <= max(paper SD, our seed SD).
IRR = btl - 1 (their evaluator prints btl = 1 + sum of daily top-1 return ratios; E_rsr_original.md sec. 2).
"""
import json
import numpy as np

PAPER = {  # (mean, sd) ; Table 6 / Table 7 / Table 5, NYSE
    "Table6_industry": {"Rank_LSTM": {"mse": (2.28e-4, 1.16e-6), "mrr": (3.79e-2, 8.82e-3), "irr": (0.56, 0.68)},
                        "RSR_E": {"mse": (2.29e-4, 2.77e-6), "mrr": (4.28e-2, 6.18e-3), "irr": (1.00, 0.58)},
                        "RSR_I": {"mse": (2.26e-4, 5.30e-7), "mrr": (4.51e-2, 2.41e-3), "irr": (1.06, 0.27)}},
    "Table7_wiki": {"RSR_E": {"mse": (2.29e-4, 2.77e-6), "mrr": (4.28e-2, 6.18e-3), "irr": (0.96, 0.47)},
                    "RSR_I": {"mse": (2.26e-4, 1.37e-6), "mrr": (4.58e-2, 5.55e-3), "irr": (0.79, 0.34)}},
}
j = json.load(open("docs/phase1_5/E_rsr_original_scores.json"))
seeds = sorted(j)
out = {"seeds": seeds, "rows": {}}
lines = ["| selection | metric | ours mean | ours SD (ddof1) | paper RSR_I (T6 industry) | tolerance | reproduces |", "|---|---|---|---|---|---|---|"]
for sel, key in (("their rule (min val loss)", "their_rule"), ("our rule (max val Sharpe)", "our_rule"), ("oracle (max test Sharpe)", "oracle")):
    vals = {"mse": [], "mrr": [], "irr": []}
    for s in seeds:
        p = j[s][key]["their_printed"]
        vals["mse"].append(p["mse"]); vals["mrr"].append(p["mrrt"]); vals["irr"].append(p["btl"] - 1.0)
    out["rows"][key] = {}
    for m in ("mse", "mrr", "irr"):
        v = np.array(vals[m]); mu, sd = v.mean(), v.std(ddof=1)
        pm, ps = PAPER["Table6_industry"]["RSR_I"][m]
        tol = max(ps, sd); ok = abs(mu - pm) <= tol
        out["rows"][key][m] = dict(per_seed=v.tolist(), mean=mu, sd=sd, paper=pm, paper_sd=ps, tol=tol, reproduces=bool(ok))
        f = (lambda x: f"{x:.3e}") if m == "mse" else (lambda x: f"{x:.3f}")
        lines.append(f"| {sel} | {m.upper()} | {f(mu)} | {f(sd)} | {f(pm)} ± {f(ps)} | {f(tol)} | {'YES' if ok else 'NO'} |")
out["paper"] = PAPER
json.dump(out, open("docs/phase2/rsr_paper_compare.json", "w"), indent=1)
open("docs/phase2/rsr_paper_compare_table.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
for key in out["rows"]: print(key, [round(x, 3) for x in out["rows"][key]["irr"]["per_seed"]])
