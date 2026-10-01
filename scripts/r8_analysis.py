"""R8 analysis: paper baselines RSR-I and STHGCN vs THINK (HH) and EH. CPU only. From the repo root:
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/r8_analysis.py [--fig]
Recomputes everything from metrics.json / history.jsonl / test_*.npy. Prints markdown tables.
Full NYSE: results/R8_baselines_g2/{RSR_I,STHGCN} vs results/R5_g2/{THINK_paperProtocol,EH} (norm paper, 100 epochs).
Small scale (309 stocks, level inputs, 30 epochs): results/POC_sectors_R8_{rsr_i,sthgcn}_g2 vs results/POC_sectors_g2.
Diagnostics reuse the logic of scripts/r5_g2_analysis.py."""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import rankdata, mannwhitneyu

sys.path.insert(0, "src")
from hypershift.eval.metrics import evaluate_all, ndcg_at_k, sharpe, topk_daily_returns
from hypershift.eval.stats import holm, sharpe_diff_ci, verdict, wilcoxon_paired

FULL = [("HH", "THINK (HH)", "R5_g2", "THINK_paperProtocol"), ("EH", "TConv+DHHAN (EH)", "R5_g2", "EH"),
        ("STHGCN", "STHGCN", "R8_baselines_g2", "STHGCN"), ("RSR_I", "RSR-I", "R8_baselines_g2", "RSR_I")]
SMALL = [("HH", "HH_hyper (THINK)", "POC_sectors_g2", "HH_hyper"), ("EH", "EH_hyper", "POC_sectors_g2", "EH_hyper"),
         ("EE", "EE_hyper", "POC_sectors_g2", "EE_hyper"), ("HHn", "HH_none", "POC_sectors_g2", "HH_none"),
         ("EEn", "EE_none", "POC_sectors_g2", "EE_none"),
         ("STHGCN", "STHGCN", "POC_sectors_R8_sthgcn_g2", "sthgcn"), ("RSR_I", "RSR-I", "POC_sectors_R8_rsr_i_g2", "rsr_i")]
OUT = {}


def runs(exp, arm):
    return {int(d.rsplit("_", 1)[1]): d for d in sorted(glob.glob(f"results/{exp}/{arm}/seed_*")) if os.path.exists(d + "/metrics.json")}


def metrics(d):
    return json.load(open(d + "/metrics.json"))


def hist(d):
    return [json.loads(l) for l in open(d + "/history.jsonl")]


class Scale:
    def __init__(self, name, spec):
        self.name = name
        self.arms = {s[0]: s for s in spec}
        self.R = {s[0]: runs(s[2], s[3]) for s in spec}
        ref = self.R["HH"][min(self.R["HH"])]
        self.G, self.K = np.load(ref + "/test_gt.npy"), np.load(ref + "/test_mask.npy")
        self.const = evaluate_all(np.zeros_like(self.G), self.G, self.K)["sr"]

    def label(self, a):
        return self.arms[a][1]

    def configs(self):
        for a, R in self.R.items():
            cs = set()
            for m in (metrics(d) for d in R.values()):
                c = m["config"]
                cs.add((c["model"], c["norm"], c["input_mode"], c["attn_score"], c["lr"], c["alpha"], c["epochs"], m["num_nodes"], m["num_edges"]))
            masks = all(np.array_equal(np.load(d + "/test_mask.npy"), self.K) for d in R.values())
            er = [metrics(d)["epochs_run"] for d in R.values()]
            print(f"- {self.label(a)}: n={len(R)} seeds={sorted(R)} epochs_run min/median/max={min(er)}/{int(np.median(er))}/{max(er)} identical-test-mask={masks}\n  config(model,norm,input_mode,attn_score,lr,alpha,epochs,nodes,edges)={sorted(cs, key=str)}")

    def diag(self, a):
        out = {k: [] for k in ("sr_v", "sr_o", "nd_v", "nd_o", "ndsthan_v", "mse_v", "ic", "hit", "spread", "val_sr", "best_ep", "orc_ep", "orc_const", "spread_const")}
        for s, d in sorted(self.R[a].items()):
            m = metrics(d)
            byep = {r["epoch"]: r for r in hist(d)}
            o, v = byep[m["test_oracle_epoch"]]["test"], byep[m["best_epoch"]]["test"]
            p, g, k = (np.load(f"{d}/test_{n}.npy") for n in ("pred", "gt", "mask"))
            k1 = k > 0.5
            zero = float(np.sum((g * k) ** 2) / max(np.sum(k), 1.0))
            out["sr_v"].append(v["sr"]); out["sr_o"].append(m["test_oracle_sr"]); out["nd_v"].append(v["ndcg5"]); out["nd_o"].append(o["ndcg5"])
            out["ndsthan_v"].append(v["ndcg_sthan"]); out["mse_v"].append(v["mse"] / zero)
            out["val_sr"].append(m["val"]["sr"]); out["best_ep"].append(m["best_epoch"]); out["orc_ep"].append(m["test_oracle_epoch"])
            out["orc_const"].append(abs(m["test_oracle_sr"] - self.const) < 1e-6)
            ics = [np.corrcoef(rankdata(p[k1[:, t], t]), rankdata(g[k1[:, t], t]))[0, 1] for t in range(p.shape[1]) if k1[:, t].sum() > 2]
            out["ic"].append(np.nanmean(ics) if np.any(np.isfinite(ics)) else np.nan)
            nz = k1 & (g != 0)
            out["hit"].append(100 * np.mean(np.sign(p[nz]) == np.sign(g[nz])) - 50)
            sp = np.mean([p[k1[:, t], t].std() / g[k1[:, t], t].std() for t in range(p.shape[1]) if k1[:, t].sum() > 2])
            out["spread"].append(sp); out["spread_const"].append(sp < 1e-5)
        return {k: np.array(v) for k, v in out.items()}

    def baselines(self):
        G, K = self.G, self.K
        mr = (G * K).sum(0) / np.maximum(K.sum(0), 1.0)
        market = sharpe(mr)
        rr = [sharpe(topk_daily_returns(np.random.default_rng(sd).standard_normal(G.shape), G, K, 5)) for sd in range(100)]
        rng = np.random.default_rng(1)
        elig = np.nonzero(K.min(axis=1) > 0.5)[0]
        fr = np.array([sharpe(G[rng.choice(elig, 5, replace=False)].mean(axis=0)) for _ in range(2000)])
        rng0 = np.random.default_rng(0)
        rnd = float(np.mean([ndcg_at_k(rng0.standard_normal(G.shape), G, K) for _ in range(100)]))
        return market, np.array(rr), fr, rnd

    def vec(self, a, seeds, key):
        return np.array([metrics(self.R[a][s])["test"]["sr"] if key == "lf" else metrics(self.R[a][s])["test_oracle_sr"] for s in seeds])

    def series(self, a, seeds):
        return np.mean([np.load(self.R[a][s] + "/test_daily.npy") for s in seeds], axis=0)


def ms(x, f=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{f}f} ± {np.nanstd(x):.{f}f}"


def report(S, pairs, order):
    print(f"\n## {S.name}: configuration check")
    S.configs()
    D = {a: S.diag(a) for a in S.R}
    market, rr, fr, rnd = S.baselines()
    G = S.G
    print(f"\nBaselines (test 2017, {G.shape[1]} days, {int(S.K.any(1).sum())} stocks): hold-all {market:.3f}; random-5 re-drawn daily {rr.mean():.3f} ± {rr.std():.3f}; fixed random 5-stock set {fr.mean():.3f} ± {fr.std():.3f} (median {np.median(fr):.3f}, 5-95% [{np.quantile(fr,.05):.2f}, {np.quantile(fr,.95):.2f}]); constant-prediction Sharpe {S.const:.3f}; random NDCG@5 {rnd:.4f}")
    OUT[S.name] = dict(market=market, rand_daily=float(rr.mean()), rand_fixed=float(fr.mean()), rnd_ndcg=rnd, const=S.const)
    print(f"\n### Table 1 ({S.name}): Sharpe\n| arm | n | leak-free | best-test | val Sharpe at selected epoch | leak-free > hold-all | leak-free > fixed-random mean | best-test > hold-all | leak-free unann. | best-test unann. |\n|---|---|---|---|---|---|---|---|---|---|")
    for a in order:
        d = D[a]; n = len(d["sr_v"])
        print(f"| {S.label(a)} | {n} | {ms(d['sr_v'])} | {ms(d['sr_o'])} | {ms(d['val_sr'])} | {int((d['sr_v']>market).sum())}/{n} | {int((d['sr_v']>fr.mean()).sum())}/{n} | {int((d['sr_o']>market).sum())}/{n} | {ms(d['sr_v']/math.sqrt(252),4)} | {ms(d['sr_o']/math.sqrt(252),4)} |")
        OUT[S.name + "_" + a] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items() if v.dtype != bool}
        OUT[S.name + "_" + a]["n"] = n
    print(f"\n### Table 2 ({S.name}): diagnostics at the leak-free epoch\n| arm | NDCG@5 | NDCG@5 at best-test epoch | NDCG_sthan | MSE/zero-MSE | IC | hit pp | spread | spread<1e-5 | best-test==const | median sel. epoch | median best-test epoch |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in order:
        d = D[a]
        print(f"| {S.label(a)} | {ms(d['nd_v'],4)} | {ms(d['nd_o'],4)} | {ms(d['ndsthan_v'],3)} | {ms(d['mse_v'],3)} | {ms(d['ic'],4)} | {ms(d['hit'],2)} | {ms(d['spread'],3)} | {int(d['spread_const'].sum())}/{len(d['spread_const'])} | {int(d['orc_const'].sum())}/{len(d['orc_const'])} | {int(np.median(d['best_ep']))} | {int(np.median(d['orc_ep']))} |")
    rows, ps, pb = [], {}, {}
    small = S.name == "small"
    for a, b in pairs:
        qa = f"{S.label(a)} vs {S.label(b)}"
        sa, sb = S.vec(a, sorted(S.R[a]), "lf"), S.vec(b, sorted(S.R[b]), "lf")
        ba, bb = S.vec(a, sorted(S.R[a]), "bt"), S.vec(b, sorted(S.R[b]), "bt")
        common = sorted(set(S.R[a]) & set(S.R[b]))
        ci = sharpe_diff_ci(S.series(a, sorted(S.R[a])), S.series(b, sorted(S.R[b])))
        mw, mwb = mannwhitneyu(sa, sb).pvalue, mannwhitneyu(ba, bb).pvalue
        wp = wilcoxon_paired(S.vec(a, common, "lf"), S.vec(b, common, "lf")) if small else float("nan")
        ps[qa] = mw; pb[qa] = mwb
        rows.append((qa, len(sa), len(sb), sa.mean() - sb.mean(), mw, ci, ba.mean() - bb.mean(), mwb, wp, float((sa[:, None] > sb[None, :]).mean())))
    adj, adjb = holm(ps), holm(pb)
    hdr = "| comparison | n a / n b | diff | MW p | Holm p | bootstrap CI | P(a>b) | verdict |" + (" paired Wilcoxon p |" if small else "")
    print(f"\n### Table 3 ({S.name}): leak-free, Mann-Whitney (unpaired), Holm over {len(pairs)}, block-bootstrap CI of the Sharpe difference of the seed-averaged daily series\n{hdr}\n|" + "---|" * (9 if small else 8))
    for qa, na, nb, d, mw, ci, bd, mwb, wp, pab in rows:
        print(f"| {qa} | {na} / {nb} | {d:+.3f} | {mw:.4f} | {adj[qa]:.4f} | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {pab:.2f} | {verdict(adj[qa], ci['lo'], ci['hi'])} |" + (f" {wp:.4f} |" if small else ""))
    print(f"\n### Table 4 ({S.name}): best-test (diagnostic only; selects on the test year), Mann-Whitney, Holm\n| comparison | diff | MW p | Holm p |\n|---|---|---|---|")
    for qa, na, nb, d, mw, ci, bd, mwb, wp, pab in rows:
        print(f"| {qa} | {bd:+.3f} | {mwb:.4f} | {adjb[qa]:.4f} |")
    OUT[S.name + "_cmp"] = [dict(q=r[0], d=r[3], p=r[4], holm=adj[r[0]], lo=r[5]["lo"], hi=r[5]["hi"], bt_d=r[6], bt_p=r[7], bt_holm=adjb[r[0]]) for r in rows]
    return D, market, fr


def ordering(S, D, keys, names):
    for key, nm in (("sr_v", "leak-free"), ("sr_o", "best-test")):
        means = {k: D[k][key].mean() for k in keys}
        rk = sorted(keys, key=lambda k: -means[k])
        print(f"Ordering by mean {nm} ({S.name}): " + " > ".join(f"{names[k]} {means[k]:.3f}" for k in rk))
        OUT.setdefault("order_" + S.name, {})[nm] = [(names[k], float(means[k])) for k in rk]


def main():
    F = Scale("full", FULL)
    D, mk, fr = report(F, [("RSR_I", "HH"), ("STHGCN", "HH"), ("RSR_I", "EH"), ("STHGCN", "EH"), ("RSR_I", "STHGCN")], ["HH", "EH", "STHGCN", "RSR_I"])
    names = {"HH": "THINK", "EH": "EH", "STHGCN": "STHGCN", "RSR_I": "RSR-I"}
    print("\nPaper ordering: THINK 1.18 > EH 1.14 > STHGCN 1.10 > RSR-I 1.05 (p852 Table II)")
    ordering(F, D, ["HH", "EH", "STHGCN", "RSR_I"], names)
    for a, b in (("HH", "EH"), ("EH", "STHGCN"), ("STHGCN", "RSR_I"), ("HH", "STHGCN"), ("HH", "RSR_I"), ("EH", "RSR_I")):
        for key, nm in (("lf", "leak-free"), ("bt", "best-test")):
            va, vb = F.vec(a, sorted(F.R[a]), key), F.vec(b, sorted(F.R[b]), key)
            print(f"  {names[a]} - {names[b]} {nm}: {va.mean()-vb.mean():+.3f}, two-sided MW p {mannwhitneyu(va, vb).pvalue:.3f}, one-sided (paper direction) p {mannwhitneyu(va, vb, alternative='greater').pvalue:.3f}")
    S = Scale("small", SMALL)
    Ds, mks, frs = report(S, [("RSR_I", "HH"), ("STHGCN", "HH"), ("RSR_I", "EH"), ("STHGCN", "EH"), ("RSR_I", "STHGCN"), ("RSR_I", "HHn"), ("STHGCN", "HHn")],
                          ["HH", "EH", "EE", "HHn", "EEn", "STHGCN", "RSR_I"])
    ordering(S, Ds, ["HH", "EH", "STHGCN", "RSR_I"], names)
    for a, b in (("HH", "EH"), ("EH", "STHGCN"), ("STHGCN", "RSR_I"), ("HH", "STHGCN"), ("HH", "RSR_I")):
        for key, nm in (("lf", "leak-free"), ("bt", "best-test")):
            va, vb = S.vec(a, sorted(S.R[a]), key), S.vec(b, sorted(S.R[b]), key)
            print(f"  small {names[a]} - {names[b]} {nm}: {va.mean()-vb.mean():+.3f}, two-sided MW p {mannwhitneyu(va, vb).pvalue:.3f}, one-sided (paper direction) p {mannwhitneyu(va, vb, alternative='greater').pvalue:.3f}")
    json.dump(OUT, open(os.environ.get("R8_OUT", "results/_r8_summary.json"), "w"), indent=1, default=float)
    if "--fig" in sys.argv:
        fig(F, D, mk, fr, S, Ds, mks, frs)


def fig(F, D, mk, fr, S, Ds, mks, frs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5e0"
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    for ax, Sx, Dx, market, frx, order, title in ((axs[0], F, D, mk, fr, ["HH", "EH", "STHGCN", "RSR_I"], "Full NYSE (1737 stocks), paper protocol"),
                                                    (axs[1], S, Ds, mks, frs, ["HH", "EH", "EE", "HHn", "EEn", "STHGCN", "RSR_I"], "Small scale (309 stocks), level inputs, 30 epochs")):
        x = np.arange(len(order)); w = 0.38
        for j, (key, col, lab) in enumerate((("sr_o", BLUE, "best test epoch (selects on the test year)"), ("sr_v", ORANGE, "epoch chosen on validation (leak-free)"))):
            vals = [Dx[a][key] for a in order]
            ax.bar(x + (j - 0.5) * w, [v.mean() for v in vals], w - 0.03, yerr=[v.std() for v in vals], color=col, label=lab if ax is axs[0] else None, capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
            for xi, v in zip(x, vals):
                ax.text(xi + (j - 0.5) * w, (v.mean() + v.std() + 0.05) if v.mean() >= 0 else 0.05, f"{v.mean():.2f}", ha="center", fontsize=7.5, color=INK)
        ax.axhline(market, color=INK, lw=1.2, ls="--"); ax.axhline(frx.mean(), color=MUTED, lw=1.0, ls=":"); ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(-0.6, len(order) + 0.9); ax.text(len(order) - 0.35, market + 0.06, f"hold all\n({market:.2f})", ha="left", fontsize=8, color=INK)
        ax.text(len(order) - 0.35, frx.mean() - 0.5, f"fixed\nrandom 5\n({frx.mean():.2f})", ha="left", fontsize=8, color=MUTED)
        ax.set_xticks(x, [f"{Sx.label(a)}\nn={len(Sx.R[a])}" for a in order], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK); ax.set_ylim(-1.0, 4.2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].set_ylabel("2017 test Sharpe (ours: top-5, x sqrt(252)); mean ± std over seeds", color=MUTED, fontsize=9)
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("R8: paper baselines (STHGCN, RSR-I) vs THINK and EH", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig("docs/figures/r8_baselines.png", dpi=150); plt.close(fig)
    print("wrote docs/figures/r8_baselines.png")


if __name__ == "__main__":
    main()
