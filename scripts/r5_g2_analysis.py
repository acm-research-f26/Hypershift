"""R5_g2 analysis (full NYSE, 1737 stocks, corrected graph, paper protocol). CPU only. Run from the repo root:
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/r5_g2_analysis.py [--fig]
Recomputes everything from results/R5_g2/*/seed_*/{metrics.json,history.jsonl,test_*.npy}. Prints markdown tables.
Diagnostics reuse the logic of scripts/poc_figures.py (diag, random_ndcg), re-implemented so that script is not imported
(it writes figures on import)."""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr, mannwhitneyu

sys.path.insert(0, "src")
from hypershift.eval.metrics import evaluate_all, ndcg_at_k, sharpe, topk_daily_returns
from hypershift.eval.stats import holm, sharpe_diff_ci, verdict, wilcoxon_paired

EXP = "R5_g2"
ARMS = [("THINK_paperProtocol", "HH (THINK)"), ("EH", "EH (TConv+DHHAN)"), ("EE", "EE"), ("HE", "HE")]
SRC = {"THINK_paperProtocol": "laptop RTX 3050", "EH": "Kaggle T4", "EE": "Kaggle T4", "HE": "Kaggle T4"}
PAPER = {"HH": 1.18, "EH": 1.14}
OUT = {}


def runs(arm, exp=EXP):
    return {int(d.rsplit("_", 1)[1]): d for d in sorted(glob.glob(f"results/{exp}/{arm}/seed_*")) if os.path.exists(d + "/metrics.json")}


def metrics(d):
    return json.load(open(d + "/metrics.json"))


def hist(d):
    return [json.loads(l) for l in open(d + "/history.jsonl")]


R = {a: runs(a) for a, _ in ARMS}
ref = "results/R5_g2/THINK_paperProtocol/seed_0"
G, K = np.load(ref + "/test_gt.npy"), np.load(ref + "/test_mask.npy")


def const_sharpe():
    return evaluate_all(np.zeros_like(G), G, K)["sr"]


CONST = const_sharpe()


def diag(arm):
    out = {k: [] for k in ("sr_v", "sr_o", "nd_v", "nd_o", "ndsthan_v", "mse_v", "ic", "hit", "spread", "val_sr_at_sel", "best_ep", "orc_ep", "orc_const", "spread_const")}
    for s, d in sorted(R[arm].items()):
        m = metrics(d)
        byep = {r["epoch"]: r for r in hist(d)}
        o, v = byep[m["test_oracle_epoch"]]["test"], byep[m["best_epoch"]]["test"]
        p, g, k = (np.load(f"{d}/test_{n}.npy") for n in ("pred", "gt", "mask"))
        k1 = k > 0.5
        zero = float(np.sum((g * k) ** 2) / max(np.sum(k), 1.0))
        out["sr_v"].append(v["sr"]); out["sr_o"].append(m["test_oracle_sr"]); out["nd_v"].append(v["ndcg5"]); out["nd_o"].append(o["ndcg5"])
        out["ndsthan_v"].append(v["ndcg_sthan"]); out["mse_v"].append(v["mse"] / zero)
        out["val_sr_at_sel"].append(m["val"]["sr"]); out["best_ep"].append(m["best_epoch"]); out["orc_ep"].append(m["test_oracle_epoch"])
        out["orc_const"].append(abs(m["test_oracle_sr"] - CONST) < 1e-6)
        ics = [np.corrcoef(rankdata(p[k1[:, t], t]), rankdata(g[k1[:, t], t]))[0, 1] for t in range(p.shape[1]) if k1[:, t].sum() > 2]
        out["ic"].append(np.nanmean(ics) if np.any(np.isfinite(ics)) else np.nan)
        nz = k1 & (g != 0)
        out["hit"].append(100 * np.mean(np.sign(p[nz]) == np.sign(g[nz])) - 50)
        sp = np.mean([p[k1[:, t], t].std() / g[k1[:, t], t].std() for t in range(p.shape[1]) if k1[:, t].sum() > 2])
        out["spread"].append(sp); out["spread_const"].append(sp < 1e-5)
    return {k: np.array(v) for k, v in out.items()}


def random_ndcg(draws=100):
    rng = np.random.default_rng(0)
    return float(np.mean([ndcg_at_k(rng.standard_normal(G.shape), G, K) for _ in range(draws)]))


def ms(x, f=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{f}f} ± {np.nanstd(x):.{f}f}"


def fixed_random5(draws=2000):
    rng = np.random.default_rng(1)
    srs = []
    elig = np.nonzero(K.min(axis=1) > 0.5)[0]   # stocks present all test days
    for _ in range(draws):
        pick = rng.choice(elig, 5, replace=False)
        srs.append(sharpe(G[pick].mean(axis=0)))
    return np.array(srs)


def per_epoch_rankcorr(arm):
    rs = []
    curves_v, curves_t = [], []
    for s, d in sorted(R[arm].items()):
        h = hist(d)
        v = np.array([r["val"]["sr"] for r in h]); t = np.array([r["test"]["sr"] for r in h])
        rs.append(spearmanr(v, t).correlation); curves_v.append(v); curves_t.append(t)
    L = min(len(c) for c in curves_v)
    mv, mt = np.mean([c[:L] for c in curves_v], 0), np.mean([c[:L] for c in curves_t], 0)
    return np.array(rs), spearmanr(mv, mt).correlation, mv, mt


def series(arm, seeds):
    return np.mean([np.load(R[arm][s] + "/test_daily.npy") for s in seeds], axis=0)


def vec(arm, seeds, key):
    out = []
    for s in seeds:
        m = metrics(R[arm][s])
        out.append(m["test"]["sr"] if key == "lf" else m["test_oracle_sr"])
    return np.array(out)


def main():
    print(f"# R5_g2 analysis\nconstant-prediction Sharpe on the test mask: {CONST:.4f}")
    # sanity: configs and masks
    for a, _ in ARMS:
        cs = {(metrics(d)["config"]["attn_score"], metrics(d)["config"]["norm"], metrics(d)["epochs_run"], metrics(d)["num_nodes"], metrics(d)["num_edges"]) for d in R[a].values()}
        masks_equal = all(np.array_equal(np.load(d + "/test_mask.npy"), K) for d in R[a].values())
        print(f"{a}: n={len(R[a])} configs(eq14,norm,epochs_run,nodes,edges)={cs} identical-test-mask={masks_equal} seeds={sorted(R[a])[0]}..{sorted(R[a])[-1]}")
    D = {a: diag(a) for a, _ in ARMS}
    rnd = random_ndcg()
    # baselines
    mr = (G * K).sum(0) / np.maximum(K.sum(0), 1.0)
    market = sharpe(mr)
    rr = []
    for sd in range(100):
        p = np.random.default_rng(sd).standard_normal(G.shape)
        rr.append(sharpe(topk_daily_returns(p, G, K, 5)))
    fr = fixed_random5()
    print(f"\nBaselines (test 2017, {G.shape[1]} days, {int(K.any(1).sum())} stocks with data): hold-all {market:.3f} (unannualized {market/math.sqrt(252):.4f}); random-5 re-drawn daily {np.mean(rr):.3f} ± {np.std(rr):.3f} (100 draws); fixed random 5-stock set {fr.mean():.3f} ± {fr.std():.3f} (2000 draws, median {np.median(fr):.3f}, 5-95% [{np.quantile(fr,.05):.2f}, {np.quantile(fr,.95):.2f}]); random NDCG@5 {rnd:.4f}")
    OUT.update(market=market, rand_daily=float(np.mean(rr)), rand_daily_sd=float(np.std(rr)), rand_fixed=float(fr.mean()), rand_fixed_sd=float(fr.std()), rnd_ndcg=rnd, const=CONST)
    # table 1
    print("\n## Table 1: Sharpe\n| arm | n | source | leak-free | best-test | val Sharpe at selected epoch | leak-free unannualized | best-test unannualized |\n|---|---|---|---|---|---|---|---|")
    for a, n in ARMS:
        d = D[a]
        print(f"| {n} | {len(R[a])} | {SRC[a]} | {ms(d['sr_v'])} | {ms(d['sr_o'])} | {ms(d['val_sr_at_sel'])} | {ms(d['sr_v']/math.sqrt(252),4)} | {ms(d['sr_o']/math.sqrt(252),4)} |")
        OUT[a] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items() if v.dtype != bool}
    print("\n## Table 2: diagnostics at the leak-free epoch (mean ± std over seeds)\n| arm | NDCG@5 leak-free | NDCG@5 at best-test epoch | NDCG_sthan (authors' evaluator) leak-free | MSE / zero-MSE | IC | hit-rate pp over 50 | pred spread / actual | runs spread<1e-5 | best-test == constant Sharpe | median selected epoch | median best-test epoch |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, n in ARMS:
        d = D[a]
        print(f"| {n} | {ms(d['nd_v'],4)} | {ms(d['nd_o'],4)} | {ms(d['ndsthan_v'],3)} | {ms(d['mse_v'],3)} | {ms(d['ic'],4)} | {ms(d['hit'],2)} | {ms(d['spread'],3)} | {int(d['spread_const'].sum())}/{len(d['spread_const'])} | {int(d['orc_const'].sum())}/{len(d['orc_const'])} | {int(np.median(d['best_ep']))} | {int(np.median(d['orc_ep']))} |")
    # val vs test rank corr
    print("\n## Per-epoch validation vs test Sharpe, Spearman over the 100 epochs\n| arm | mean per-seed rho ± std | min..max | rho of the seed-mean curves |\n|---|---|---|---|")
    for a, n in ARMS:
        rs, rm, mv, mt = per_epoch_rankcorr(a)
        print(f"| {n} | {ms(rs,3)} | {rs.min():.2f}..{rs.max():.2f} | {rm:.3f} |  (mean val {mv.mean():.2f}, mean test {mt.mean():.2f})")
        OUT[a + "_rho"] = (float(rs.mean()), float(rs.std()), float(rm))
    # old graph
    print("\n## Old-graph full NYSE (5 seeds, pre-eq.14)")
    for exp, arm in (("E1_main", "THINK_paperProtocol"), ("R_paperProtocol", "EH"), ("R_paperProtocol", "EE")):
        rr_ = runs(arm, exp)
        if rr_:
            lf = np.array([metrics(d)["test"]["sr"] for d in rr_.values()]); bt = np.array([metrics(d)["test_oracle_sr"] for d in rr_.values()])
            rho = np.array([spearmanr([r["val"]["sr"] for r in hist(d)], [r["test"]["sr"] for r in hist(d)]).correlation for d in rr_.values()])
            print(f"{exp}/{arm}: n={len(rr_)} leak-free {ms(lf)} best-test {ms(bt)} per-seed val-test rho {ms(rho,3)} (epochs {len(hist(list(rr_.values())[0]))})")
            OUT[f"old_{arm}"] = dict(n=len(rr_), lf=(float(lf.mean()), float(lf.std())), bt=(float(bt.mean()), float(bt.std())), rho=float(rho.mean()))
    # new vs old THINK
    new_lf = vec("THINK_paperProtocol", sorted(R["THINK_paperProtocol"]), "lf"); new_bt = vec("THINK_paperProtocol", sorted(R["THINK_paperProtocol"]), "bt")
    old = runs("THINK_paperProtocol", "E1_main")
    ol = np.array([metrics(d)["test"]["sr"] for d in old.values()]); ob = np.array([metrics(d)["test_oracle_sr"] for d in old.values()])
    print(f"THINK new(25) vs old(5): leak-free {new_lf.mean():.3f} vs {ol.mean():.3f}, Mann-Whitney p {mannwhitneyu(new_lf, ol).pvalue:.3f}; best-test {new_bt.mean():.3f} vs {ob.mean():.3f}, p {mannwhitneyu(new_bt, ob).pvalue:.3f}")
    # comparisons
    pairs = [("THINK_paperProtocol", "EH", "HH vs EH (the paper's comparison)"), ("THINK_paperProtocol", "EE", "HH vs EE"),
             ("THINK_paperProtocol", "HE", "HH vs HE"), ("EH", "EE", "EH vs EE"), ("HE", "EE", "HE vs EE")]
    rows, ps = [], {}
    for a, b, q in pairs:
        common = sorted(set(R[a]) & set(R[b]))
        sa, sb = vec(a, common, "lf"), vec(b, common, "lf")
        ci = sharpe_diff_ci(series(a, common), series(b, common))
        p = wilcoxon_paired(sa, sb)
        ps[q] = p
        ba, bb = vec(a, common, "bt"), vec(b, common, "bt")
        rows.append((q, len(common), sa.mean() - sb.mean(), p, ci, int((sa > sb).sum()), ba.mean() - bb.mean(), wilcoxon_paired(ba, bb), int((ba > bb).sum()),
                     mannwhitneyu(sa, sb).pvalue, sharpe(series(a, common)), sharpe(series(b, common))))
    adj = holm(ps)
    print("\n## Table 3: leak-free comparisons (family of 5, Holm)\n| comparison | n seeds | diff | Wilcoxon p | Holm p | 95% block-bootstrap CI | a wins | Mann-Whitney p | verdict | seed-ensemble Sharpe a / b |\n|---|---|---|---|---|---|---|---|---|---|")
    for q, n, d, p, ci, w, bd, bp, bw, mw, ea, eb in rows:
        print(f"| {q} | {n} | {d:+.3f} | {p:.4f} | {adj[q]:.4f} | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {w}/{n} | {mw:.3f} | {verdict(adj[q], ci['lo'], ci['hi'])} | {ea:.3f} / {eb:.3f} |")
    print("\n## Table 4: best-test comparisons (raw p, diagnostic; no CI because test_daily is the leak-free epoch only)\n| comparison | n | diff | Wilcoxon p | a wins |\n|---|---|---|---|---|")
    for q, n, d, p, ci, w, bd, bp, bw, mw, ea, eb in rows:
        print(f"| {q} | {n} | {bd:+.3f} | {bp:.4f} | {bw}/{n} |")
    OUT["cmp"] = [dict(q=q, n=n, d=d, p=p, holm=adj[q], lo=ci["lo"], hi=ci["hi"], wins=w, bt_d=bd, bt_p=bp, bt_w=bw) for q, n, d, p, ci, w, bd, bp, bw, mw, ea, eb in rows]
    # HH vs EH on seeds 0-9 only (comparable to EE/HE sample)
    s10 = list(range(10))
    print(f"\nHH vs EH on seeds 0-9 only: leak-free {vec('THINK_paperProtocol', s10,'lf').mean():.3f} vs {vec('EH', s10,'lf').mean():.3f}; best-test {vec('THINK_paperProtocol', s10,'bt').mean():.3f} vs {vec('EH', s10,'bt').mean():.3f}")
    # does HH beat market / random leak-free
    for a, n in ARMS:
        lf = D[a]["sr_v"]
        print(f"{n}: leak-free > hold-all in {int((lf > market).sum())}/{len(lf)} seeds; > fixed-random-5 mean in {int((lf > fr.mean()).sum())}/{len(lf)}; best-test > hold-all in {int((D[a]['sr_o'] > market).sum())}/{len(lf)}")
    json.dump(OUT, open(os.environ.get("R5_OUT", "results/_r5_g2_summary.json"), "w"), indent=1, default=float)
    if "--fig" in sys.argv:
        fig(D, market, fr, rnd)


def fig(D, market, fr, rnd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5e0"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[2.0, 1]))
    x = np.arange(len(ARMS)); w = 0.38
    for j, (key, col, lab) in enumerate((("sr_o", BLUE, "best test epoch (selects on the test year)"), ("sr_v", ORANGE, "epoch chosen on validation (leak-free)"))):
        vals = [D[a][key] for a, _ in ARMS]
        ax.bar(x + (j - 0.5) * w, [v.mean() for v in vals], w - 0.03, yerr=[v.std() for v in vals], color=col, label=lab, capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * w, (v.mean() + v.std() + 0.05) if v.mean() >= 0 else 0.05, f"{v.mean():.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(market, color=INK, lw=1.2, ls="--"); ax.text(3.5, market + 0.05, f"hold all 1737 stocks ({market:.2f})", ha="left", fontsize=8, color=INK); ax.set_xlim(-0.6, 4.9)
    ax.axhline(fr.mean(), color=MUTED, lw=1.0, ls=":"); ax.text(3.5, fr.mean() - 0.45, f"fixed random 5 stocks ({fr.mean():.2f})", ha="left", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"{n}\n{len(R[a])} seeds, {SRC[a].split()[0]}" for a, n in ARMS], fontsize=8.5)
    ax.set_ylabel("2017 test Sharpe (ours: top-5, x sqrt(252)); mean ± std over seeds", color=MUTED, fontsize=9)
    ax.set_title("R5_g2, full NYSE (1737 stocks), corrected graph, paper protocol", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylim(-0.9, 4.0)
    # panel 2: per-epoch val vs test, THINK
    _, _, mv, mt = per_epoch_rankcorr("THINK_paperProtocol")
    ep = np.arange(1, len(mv) + 1)
    ax2.plot(ep, mv, color=BLUE, lw=1.8, label="validation 2016"); ax2.plot(ep, mt, color=ORANGE, lw=1.8, label="test 2017")
    ax2.axhline(market, color=INK, lw=1.0, ls="--"); ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel("epoch", color=MUTED, fontsize=9); ax2.set_ylabel("Sharpe, mean of 25 THINK seeds", color=MUTED, fontsize=9)
    ax2.set_title("THINK: validation vs test Sharpe per epoch", loc="left", fontsize=11, color=INK); ax2.legend(frameon=False, fontsize=8.5)
    for a_ in (ax, ax2):
        for s in ("top", "right"): a_.spines[s].set_visible(False)
        for s in ("left", "bottom"): a_.spines[s].set_color(GRID)
        a_.tick_params(colors=MUTED, labelsize=9); a_.yaxis.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("docs/figures/r5_g2.png", dpi=150); plt.close(fig)
    print("wrote docs/figures/r5_g2.png")


if __name__ == "__main__":
    main()
