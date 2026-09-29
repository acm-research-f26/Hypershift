"""Figures for the small-scale (309-stock) THINK proof of concept. Run from repo root -> docs/figures/."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5e0"
ARMS = [("HH_hyper", "THINK\n(hyp + hyperedges)"), ("HH_clique", "hyp +\npairwise"), ("HH_none", "hyp,\nno relations"),
        ("EE_hyper", "Euclid +\nhyperedges"), ("EE_clique", "Euclid +\npairwise"), ("EE_none", "Euclid,\nno relations")]
MARKET = 0.752

def load(exp, arm):
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"results/{exp}/{arm}/seed_*/metrics.json"))]
    return np.array([m["test"]["sr"] for m in ms]), np.array([m["test_oracle_sr"] for m in ms])

def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)

def paired(exp_pairs, names, fname, title):
    fig, ax = plt.subplots(figsize=(10, 5.2)); x = np.arange(len(ARMS)); w = 0.38
    for k, ((exp, which), name, col) in enumerate(zip(exp_pairs, names, (BLUE, ORANGE))):
        vals = [load(exp, a)[which] for a, _ in ARMS]
        ax.bar(x + (k - 0.5) * w, [v.mean() for v in vals], w - 0.03, yerr=[v.std() for v in vals],
               color=col, label=name, capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
    ax.axhline(MARKET, color=INK, lw=1.2, ls="--"); ax.text(-0.45, MARKET + 0.06, "hold all 309 stocks (0.75)", ha="left", fontsize=9, color=INK)
    ax.set_xticks(x, [n for _, n in ARMS]); ax.set_ylabel("2017 test Sharpe ratio (mean ± std, 10 seeds)", color=MUTED, fontsize=9)
    style(ax, title); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(); fig.savefig(f"docs/figures/{fname}", dpi=160); plt.close(fig)

paired([("POC_sectors", 1), ("POC_sectors", 0)], ["best test epoch (upper bound)", "epoch chosen on validation (leak-free)"],
       "fig1_protocol_gap.png", "Fig 1. Same models, two ways of picking the epoch (faithful THINK, v1)")
paired([("POC_sectors", 0), ("POC_sectors_rel", 0)], ["v1: faithful inputs (price levels)", "v2: relative inputs (fix)"],
       "fig2_fix_effect.png", "Fig 2. Leak-free Sharpe before and after the relative-input fix")

# Fig 3: per-epoch validation vs test Sharpe for THINK (v1), mean over seeds
H = [[json.loads(l) for l in open(f)] for f in glob.glob("results/POC_sectors/HH_hyper/seed_*/history.jsonl")]
L = min(len(h) for h in H); ep = np.arange(1, L + 1)
v = np.array([[r["val"]["sr"] for r in h[:L]] for h in H]).mean(0); t = np.array([[r["test"]["sr"] for r in h[:L]] for h in H]).mean(0)
fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(ep, v, color=BLUE, lw=2, marker="o", ms=5, label="validation year 2016")
ax.plot(ep, t, color=ORANGE, lw=2, marker="o", ms=5, label="test year 2017")
ax.text(ep[-1] + 0.2, v[-1], "2016 (val)", color=INK, fontsize=9, va="center"); ax.text(ep[-1] + 0.2, t[-1], "2017 (test)", color=INK, fontsize=9, va="center")
ax.set_xlabel("training epoch", color=MUTED, fontsize=9); ax.set_ylabel("Sharpe (mean of 10 seeds)", color=MUTED, fontsize=9)
style(ax, f"Fig 3. THINK: 2016 and 2017 Sharpe move in opposite directions (first {L} epochs)")
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig("docs/figures/fig3_val_vs_test.png", dpi=160); plt.close(fig)
print("wrote docs/figures/fig1..3")
