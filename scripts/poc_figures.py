"""Figures for the small-scale (309-stock) THINK proof of concept. Run from repo root -> docs/figures/."""
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "src")
from hypershift.eval.metrics import ndcg_at_k
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


# ---------------- Fig 1: SR / MSE / NDCG / leak-free error diagnostics ----------------
def diag(exp, arm):
    """Per-seed metrics at the best-test epoch and at the validation-chosen (leak-free) epoch, plus leak-free error diagnostics."""
    out = {k: [] for k in ("sr_o", "sr_v", "mse_o", "mse_v", "nd_o", "nd_v", "ic", "hit", "spread")}
    for d in sorted(glob.glob(f"results/{exp}/{arm}/seed_*")):
        m = json.load(open(f"{d}/metrics.json")); byep = {r["epoch"]: r for r in map(json.loads, open(f"{d}/history.jsonl"))}
        o, v = byep[m["test_oracle_epoch"]]["test"], byep[m["best_epoch"]]["test"]
        p, g, k = (np.load(f"{d}/test_{n}.npy") for n in ("pred", "gt", "mask")); k1 = k > 0.5
        zero = float(np.sum((g * k) ** 2) / max(np.sum(k), 1.0))            # masked_mse of predicting 0
        for tag, r in (("o", o), ("v", v)):
            out["sr_" + tag].append(r["sr"]); out["mse_" + tag].append(r["mse"] / zero); out["nd_" + tag].append(r["ndcg5"])
        ics = [np.corrcoef(rankdata(p[k1[:, t], t]), rankdata(g[k1[:, t], t]))[0, 1] for t in range(p.shape[1]) if k1[:, t].sum() > 2]
        out["ic"].append(np.nanmean(ics)); nz = k1 & (g != 0)   # exact-zero returns (~2%) have no sign; excluded from hit rate
        out["hit"].append(100 * np.mean(np.sign(p[nz]) == np.sign(g[nz])) - 50)
        out["spread"].append(np.mean([p[k1[:, t], t].std() / g[k1[:, t], t].std() for t in range(p.shape[1]) if k1[:, t].sum() > 2]))   # within-day, across stocks
    return {k: np.array(v) for k, v in out.items()}

def random_ndcg(exp="POC_sectors", arm="HH_hyper", draws=int(os.environ.get("FIG1_DRAWS", 200))):
    d = f"results/{exp}/{arm}/seed_0"; g, k = np.load(f"{d}/test_gt.npy"), np.load(f"{d}/test_mask.npy"); rng = np.random.default_rng(0)
    return float(np.mean([ndcg_at_k(rng.standard_normal(g.shape), g, k) for _ in range(draws)]))

def fig1():
    D = {a: diag("POC_sectors", a) for a, _ in ARMS}; rnd = random_ndcg()
    fig = plt.figure(figsize=(14, 10.5)); gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
    x = np.arange(len(ARMS)); w = 0.38; names = ["best test epoch (upper bound)", "epoch chosen on validation (leak-free)"]
    def panel(ax, key, title, ylabel, ref=None, reftxt=None):
        for j, (suf, col) in enumerate((("o", BLUE), ("v", ORANGE))):
            vals = [D[a][f"{key}_{suf}"] for a, _ in ARMS]
            ax.bar(x + (j - 0.5) * w, [v.mean() for v in vals], w - 0.03, yerr=[v.std() for v in vals], color=col,
                   label=names[j], capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
        if key == "mse":
            for xi, (a_, _) in zip(x, ARMS):
                for j_, suf in enumerate("ov"):
                    ax.text(xi + (j_ - 0.5) * w, 0.04, f"{D[a_][f'mse_{suf}'].mean():.3f}", rotation=90, ha="center", va="bottom", fontsize=8, color="white")
        if ref is not None:
            ln = ax.axhline(ref, color=INK, lw=1.2, ls="--"); ax.legend([ln], [reftxt], frameon=False, fontsize=8.5, loc="upper right")
        ax.set_xticks(x, [n.replace("(hyp + hyperedges)", "(hyp +\nhyperedges)") for _, n in ARMS], fontsize=7.5); ax.set_ylabel(ylabel, color=MUTED, fontsize=9); style(ax, title)
    a = fig.add_subplot(gs[0, 0]); panel(a, "sr", "(a) Sharpe ratio", "2017 test Sharpe (mean ± std, 10 seeds)", MARKET, "hold all 309 stocks (0.75)")
    b = fig.add_subplot(gs[0, 1]); panel(b, "mse", "(b) Squared error vs. predicting 0 (lower is better)", "test MSE ÷ MSE of predicting 0", 1.0, "dashed line: predict 0 for every stock (= 1.0)")
    b.set_ylim(0, max(b.get_ylim()[1], 2.7))
    c = fig.add_subplot(gs[1, 0]); panel(c, "nd", "(c) NDCG@5 (higher is better)", "2017 test NDCG@5", rnd, f"random ranking ({rnd:.3f})")
    lo = min(v.mean() for a_ in D.values() for v in (a_["nd_o"], a_["nd_v"])); c.set_ylim(min(0.5, lo - 0.01), None)
    sub = gs[1, 1].subgridspec(1, 3, wspace=0.45); cols = [("ic", "Rank correlation (IC)\nbetween pred. and actual", 0.0),
        ("hit", "Sign hit rate\n(percentage points vs 50%)", 0.0), ("spread", "Prediction spread\n÷ actual spread", None)]
    y = np.arange(len(ARMS))[::-1]
    for i, (key, ttl, ref) in enumerate(cols):
        ax = fig.add_subplot(sub[0, i]); vals = [D[a][key] for a, _ in ARMS]
        ax.barh(y, [v.mean() for v in vals], 0.62, xerr=[v.std() for v in vals], color=ORANGE, capsize=2, error_kw=dict(ecolor=MUTED, lw=1))
        ax.set_title(ttl, fontsize=8.5, color=INK, pad=6); ax.tick_params(colors=MUTED, labelsize=8)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_yticks(y, [n.replace("\n", " ") if i == 0 else "" for _, n in ARMS], fontsize=7.5)
        if ref is not None: ax.axvline(ref, color=INK, lw=1.2, ls="--")
        ext = max(abs(v.mean()) + v.std() for v in vals) * 1.15
        if key == "spread": ax.axvline(1.0, color=INK, lw=1.2, ls="--"); ax.set_xlim(0, max(1.05, ext))
        else: ax.set_xlim(-ext, ext)
        for yi, v in zip(y, vals): ax.text(1.03, yi, f"{v.mean():.3f}" if key != "hit" else f"{v.mean():+.2f}", ha="left", va="center", fontsize=8, color=INK, transform=ax.get_yaxis_transform(), clip_on=False)
    fig.text(0.575, 0.475, "(d) Leak-free epoch only (dashed: no skill / equal spread)", fontsize=12, color=INK, ha="left")
    h = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in (BLUE, ORANGE)]; fig.legend(h, names, frameon=False, fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fig 1. Same models, two ways of picking the epoch, and why neither is accurate (faithful THINK, v1)", x=0.06, ha="left", fontsize=14, color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.06, right=0.965); fig.savefig("docs/figures/fig1_protocol_gap.png", dpi=150); plt.close(fig)
    return D, rnd

D1, RND = fig1()
if "--table" in sys.argv:
        print(f"random NDCG@5 {RND:.4f}")
        for a, _ in ARMS:
            d = D1[a]; f = lambda k: f"{d[k].mean():.3f}"
            print(a, "MSEr o/v", f("mse_o"), f("mse_v"), "| NDCG o/v", f("nd_o"), f("nd_v"), "| IC", f("ic"), "| hit-50", f("hit"), "| spread", f("spread"), "| SR o/v", f("sr_o"), f("sr_v"))
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
