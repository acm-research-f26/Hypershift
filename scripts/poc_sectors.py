"""Small-scale proof of concept: THINK ablations on a few NYSE sectors.

Universe = Energy/Utilities + Finance industries (~300 stocks). The RSR "n/a" industry bucket
(500 stocks with no known industry) is excluded. Arms: {HH, EE} x {hyper, clique, none}.

  python scripts/poc_sectors.py run --seeds 0-4        # one worker
  python scripts/poc_sectors.py run --seeds 5-9        # second worker, in parallel
  python scripts/poc_sectors.py summarize              # -> results/POC_sectors/summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hypershift.config import RunConfig
from hypershift.data.hypergraph import build_rsr_hypergraph, induced_subgraph
from hypershift.data.rsr import load_rsr, read_ticker_file
from hypershift.eval.baselines import evaluate_baselines
from hypershift.eval.stats import holm, sharpe_contrast_ci, sharpe_diff_ci, verdict, wilcoxon_one_sample, wilcoxon_paired
from hypershift.run import parse_seeds
from hypershift.train.loop import train_one_run

ROOT = Path("data/raw/rsr/data")
EXP = "POC_sectors"
INDUSTRIES = {
    "Energy & Utilities": ["Oil & Gas Production", "Integrated oil Companies", "Natural Gas Distribution",
                           "Electric Utilities: Central", "Power Generation"],
    "Finance": ["Major Banks", "Commercial Banks", "Property-Casualty Insurers", "Life Insurance",
                "Investment Managers", "Finance: Consumer Services", "Investment Bankers/Brokers/Service"],
}
GEOMS = {"HH": {"temporal": "hyp", "spatial": "hyp"}, "EE": {"temporal": "euc", "spatial": "euc"}}
STRUCTS = ("hyper", "clique", "none")


def universe():
    tickers = read_ticker_file(ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    pos = {t: i for i, t in enumerate(tickers)}
    groups = json.loads((ROOT / "relation" / "sector_industry" / "NYSE_industry_ticker.json").read_text())
    keep = sorted({pos[t] for names in INDUSTRIES.values() for n in names for t in groups[n] if t in pos})
    data = load_rsr(ROOT, "NYSE", "train")
    hg = build_rsr_hypergraph(ROOT, "NYSE")
    return data.subset(np.array(keep)), induced_subgraph(hg, np.array(keep))


def cfg(label, geo, st, seed, epochs):
    return RunConfig(exp=EXP, label=label, market="NYSE", structure=st, seed=seed, batch_days=8,
                     epochs=epochs, patience=10, **GEOMS[geo])


def run(seeds, epochs):
    data, hg = universe()
    print(f"universe: {data.num_nodes} stocks, {len(hg.edges)} hyperedges, "
          f"{int((hg.node_degree() > 0).sum())} stocks in >=1 hyperedge")
    for s in seeds:
        for geo in GEOMS:
            for st in STRUCTS:
                m = train_one_run(cfg(f"{geo}_{st}", geo, st, s, epochs), data, hg)
                print(f"seed {s} {geo}_{st}: val_sr {m['val']['sr']:.3f} test_sr {m['test']['sr']:.3f}", flush=True)


def load(label):
    out = {}
    for d in sorted(Path("results", EXP, label).glob("seed_*")):
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            out[int(d.name.split("_")[1])] = (m, np.load(d / "test_daily.npy"))
    return out


def summarize():
    data, hg = universe()
    runs = {f"{g}_{s}": load(f"{g}_{s}") for g in GEOMS for s in STRUCTS}
    base = evaluate_baselines(data)
    L = [f"# THINK proof of concept — {data.num_nodes} NYSE stocks (Energy/Utilities + Finance)", "",
         f"{len(hg.edges)} hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).", "",
         "| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 |", "|---|---|---|---|---|"]
    for k, r in runs.items():
        if r:
            t = [m["test"]["sr"] for m, _ in r.values()]
            v = [m["val"]["sr"] for m, _ in r.values()]
            nd = [m["test"]["ndcg5"] for m, _ in r.values()]
            L.append(f"| {k} | {len(r)} | {np.mean(t):.3f} ± {np.std(t):.3f} | {np.mean(v):.3f} | {np.mean(nd):.3f} |")
    for k in ("market", "random", "momentum", "oracle"):
        L.append(f"| baseline: {k} | - | {base[k]['sr']:.3f} | - | - |")

    def series(key, seeds):
        return np.mean([runs[key][s][1] for s in seeds], axis=0), np.array([runs[key][s][0]["test"]["sr"] for s in seeds])

    pairs = [("HH_hyper", "EE_hyper", "hyperbolic vs Euclidean (with hyperedges)"),
             ("HH_hyper", "HH_clique", "hyperedges vs pairwise edges (hyperbolic)"),
             ("HH_hyper", "HH_none", "relations vs none (hyperbolic)"),
             ("EE_hyper", "EE_clique", "hyperedges vs pairwise edges (Euclidean)"),
             ("EE_hyper", "EE_none", "relations vs none (Euclidean)")]
    rows, ps = [], {}
    for a, b, q in pairs:
        common = sorted(set(runs[a]) & set(runs[b]))
        if len(common) < 2:
            continue
        da, sa = series(a, common)
        db, sb = series(b, common)
        ci = sharpe_diff_ci(da, db)
        p = wilcoxon_paired(sa, sb)
        ps[q] = p
        rows.append((q, a, b, len(common), sa.mean() - sb.mean(), p, ci))
    quad = ("HH_hyper", "HH_clique", "EE_hyper", "EE_clique")
    common = sorted(set.intersection(*(set(runs[k]) for k in quad)))
    if len(common) >= 2:
        s = [series(k, common) for k in quad]
        inter = (s[0][1] - s[1][1]) - (s[2][1] - s[3][1])
        ci = sharpe_contrast_ci([x[0] for x in s], [1, -1, -1, 1])
        q = "interaction: (hyperedge gain in hyperbolic) − (in Euclidean)"
        ps[q] = wilcoxon_one_sample(inter)
        rows.append((q, "", "", len(common), inter.mean(), ps[q], ci))
    adj = holm(ps) if ps else {}
    L += ["", "| question | seeds | Sharpe diff | Wilcoxon p | Holm p | 95% bootstrap CI | verdict |", "|---|---|---|---|---|---|---|"]
    for q, a, b, n, d, p, ci in rows:
        L.append(f"| {q} | {n} | {d:+.3f} | {p:.4f} | {adj[q]:.4f} | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {verdict(adj[q], ci['lo'], ci['hi'])} |")
    L += ["", "Verdicts: STRONG = Holm p < 0.01 and CI excludes 0; SEED-ROBUST ONLY = Holm p < 0.01 but CI includes 0 "
          "(consistent across seeds, within market noise); NO EVIDENCE otherwise. With n seeds the smallest possible "
          "Wilcoxon p is 2/2^n."]
    out = Path("results", EXP, "summary.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    run(parse_seeds(a.seeds), a.epochs) if a.cmd == "run" else summarize()
