"""Figures analogous to paper Fig 2, Fig 3a, Fig 3b, plus geometry bars and the universe-size sweep."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = pd.read_csv("results/tables/summary.csv").set_index("key")
F = Path("results/figures")
F.mkdir(parents=True, exist_ok=True)
COL = {"HH": "#1f77b4", "EE": "#d62728"}
FULL = {"HH": "E1_main/THINK", "EE": "E2_geometry/EE"}   # undecomposed, all hyperedges kept


def bar(keys, names, fname, title):
    keys = [k for k in keys if k in S.index]
    plt.figure(figsize=(6, 3.5))
    plt.bar(range(len(keys)), S.loc[keys, "test_sr_mean"], yerr=S.loc[keys, "test_sr_std"], capsize=4, color="#888")
    plt.xticks(range(len(keys)), [names[k] for k in keys], rotation=20)
    plt.ylabel("Test Sharpe (mean ± std over seeds)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(F / fname, dpi=150)
    plt.close()


bar(["E1_main/THINK", "E2_geometry/HE", "E2_geometry/EH", "E2_geometry/EE"],
    {"E1_main/THINK": "HH (THINK)", "E2_geometry/HE": "hyp-T, euc-S", "E2_geometry/EH": "euc-T, hyp-S",
     "E2_geometry/EE": "EE"}, "geometry.png", "Q2: temporal x spatial geometry (NYSE)")
bar(["E1_main/THINK", "E3_hhn/HHN"], {"E1_main/THINK": "THINK", "E3_hhn/HHN": "HHN (no distance attn)"},
    "attention.png", "Fig 2 analogue (NYSE)")


def curve(pattern, xs_label, fname, title, extra_x=None):
    plt.figure(figsize=(6, 3.5))
    for g in ("HH", "EE"):
        pts = []
        for k in S.index:
            m = re.fullmatch(pattern.format(g=g), k)
            if m:
                pts.append((int(m.group(1)), S.loc[k, "test_sr_mean"], S.loc[k, "test_sr_std"]))
        if extra_x is not None and FULL[g] in S.index:
            pts.append((extra_x, S.loc[FULL[g], "test_sr_mean"], S.loc[FULL[g], "test_sr_std"]))
        if not pts:
            continue
        pts.sort(key=lambda t: -t[0])
        x = np.arange(len(pts))
        plt.errorbar(x, [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", color=COL[g], label=g, capsize=3)
        plt.xticks(x, [str(p[0]) for p in pts])
    plt.xlabel(xs_label)
    plt.ylabel("Test Sharpe")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(F / fname, dpi=150)
    plt.close()


curve(r"E6_decompose/{g}_large_(\d+)", "hyperedges larger than x decomposed into pairs", "fig3a_decompose.png",
      "Fig 3a analogue (NYSE)", extra_x=500)
curve(r"E7_hubs/{g}_hub(\d+)", "hyperedges of nodes with degree >= x removed", "fig3b_hubs.png",
      "Fig 3b analogue (NYSE)")

rows = []
for k in S.index:
    m = re.fullmatch(r"E8_universe/(HH|EE)_N(\d+)_u(\d+)", k)
    if m:
        base = json.loads(Path(f"results/baselines/NYSE_N{m.group(2)}_u{m.group(3)}.json").read_text())
        rows.append({"g": m.group(1), "N": int(m.group(2)), "sr": S.loc[k, "test_sr_mean"],
                     "excess": S.loc[k, "test_sr_mean"] - base["random"]["sr"]})
if rows:
    U = pd.DataFrame(rows).groupby(["g", "N"]).agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for g in ("HH", "EE"):
        u = U[U.g == g]
        ax[0].errorbar(u.N, u[("sr", "mean")], yerr=u[("sr", "std")], marker="o", color=COL[g], label=g, capsize=3)
        ax[1].errorbar(u.N, u[("excess", "mean")], yerr=u[("excess", "std")], marker="o", color=COL[g], label=g, capsize=3)
    for a, t in zip(ax, ["Test Sharpe", "Sharpe minus Random-5 (same universe)"]):
        a.set_xscale("log")
        a.set_xlabel("number of stocks N")
        a.set_ylabel(t)
        a.legend()
    plt.tight_layout()
    plt.savefig(F / "universe.png", dpi=150)
    plt.close()
print("figures in", F)
