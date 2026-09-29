"""Small-scale proof of concept: THINK ablations on a few NYSE sectors.

Universe = Energy/Utilities + Finance industries (~300 stocks). The RSR "n/a" industry bucket
(500 stocks with no known industry) is excluded. Arms: {HH, EE} x {hyper, clique, none}; optional EH (Euclidean temporal conv + hyperbolic
hypergraph attention = the paper's only "Euclidean" ablation, p852 Sec V.A) x {hyper, clique}.

  python scripts/poc_sectors.py run --seeds 0-4        # one worker
  python scripts/poc_sectors.py run --seeds 5-9        # second worker, in parallel
  python scripts/poc_sectors.py summarize              # -> results/POC_sectors/summary.md
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from hypershift.config import RunConfig, apply_overrides
from hypershift.data.hypergraph import build_rsr_hypergraph, induced_subgraph
from hypershift.data.rsr import load_rsr, read_ticker_file
from hypershift.eval.baselines import evaluate_baselines
from hypershift.eval.metrics import evaluate_all
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
GEOMS = {"HH": {"temporal": "hyp", "spatial": "hyp"}, "EE": {"temporal": "euc", "spatial": "euc"},
         "EH": {"temporal": "euc", "spatial": "hyp"}}
DEFAULT_GEOMS = ("HH", "EE")     # EH is opt-in (--geoms / --arms) so existing invocations and tuned.json keep working
STRUCTS = ("hyper", "clique", "none")


def structs_for(geo):
    """EH_none would be identical to EE_none (structure none has no spatial layer), so it is not an arm."""
    return tuple(st for st in STRUCTS if not (geo == "EH" and st == "none"))


def pick_geoms(arms=None, geoms=None):
    if geoms:
        return tuple(geoms)
    if arms:
        return tuple(g for g in GEOMS if any(a.startswith(g + "_") for a in arms))
    return DEFAULT_GEOMS


def universe():
    tickers = read_ticker_file(ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    pos = {t: i for i, t in enumerate(tickers)}
    groups = json.loads((ROOT / "relation" / "sector_industry" / "NYSE_industry_ticker.json").read_text())
    keep = sorted({pos[t] for names in INDUSTRIES.values() for n in names for t in groups[n] if t in pos})
    data = load_rsr(ROOT, "NYSE", "train")
    hg = build_rsr_hypergraph(ROOT, "NYSE")
    return data.subset(np.array(keep)), induced_subgraph(hg, np.array(keep))


def dry_train(c, data, hg):
    """--dry-run: print what would be trained (no training, no files written)."""
    print(f"DRY {c.exp}/{c.label}/seed_{c.seed} attn_score={c.attn_score} attn_dist={c.attn_dist} input={c.input_mode} "
          f"lr={c.lr:g} alpha={c.alpha:g} temporal={c.temporal} spatial={c.spatial} structure={c.structure} epochs={c.epochs} "
          f"decompose={c.decompose_mode}/{c.decompose_size} hub={c.drop_hub_degree} shuffle={c.shuffle_train_labels} "
          f"micro={c.micro_batch_days} model={c.model}")
    return {"val": {"sr": 0.0}, "test": {"sr": 0.0}}


TRAIN = train_one_run
GRID_LR = (5e-4, 1e-3, 3e-3)
GRID_ALPHA = (0.1, 1.0, 10.0)   # default kept so old tuned.json / tune-select stay reproducible; override with --grid-alpha


def set_grid(lr=None, alpha=None):
    """Override the tuning grid (tune and tune-select must use the same one)."""
    global GRID_LR, GRID_ALPHA
    if lr:
        GRID_LR = tuple(float(x) for x in lr)
    if alpha:
        GRID_ALPHA = tuple(float(x) for x in alpha)


def exp_name(variant):
    return f"POC_sectors_{variant}" if variant else EXP


def tuned_path(exp):
    return Path("results", exp, "tuned.json")


def cfg(label, geo, st, seed, epochs, exp=EXP, input_mode="level", lr=1e-3, alpha=1.0, tuned=None, overrides=None):
    """overrides: list of "k=v" strings (same parser as hypershift.run --set), applied last, recorded in config.json."""
    if tuned is not None:
        lr, alpha = tuned[geo]["lr"], tuned[geo]["alpha"]
    c = RunConfig(exp=exp, label=label, market="NYSE", structure=st, seed=seed, batch_days=8,
                  epochs=epochs, patience=10, input_mode=input_mode, lr=lr, alpha=alpha, **GEOMS[geo])
    return replace(c, **apply_overrides({}, overrides)) if overrides else c


def run(seeds, epochs, exp=EXP, input_mode="level", lr=1e-3, alpha=1.0, use_tuned=False, arms=None, overrides=None):
    data, hg = universe()
    print(f"universe: {data.num_nodes} stocks, {len(hg.edges)} hyperedges, "
          f"{int((hg.node_degree() > 0).sum())} stocks in >=1 hyperedge")
    tuned = json.loads(tuned_path(exp).read_text()) if use_tuned else None
    if tuned is not None:
        missing = [g for g in pick_geoms(arms) if g not in tuned]
        if missing:
            raise SystemExit(f"{tuned_path(exp)} has no tuned lr/alpha for {missing}; run `tune --geoms {' '.join(missing)}` first")
    base_model = next((o.split("=", 1)[1] for o in (overrides or []) if o.startswith("model=")), "think")
    for s in seeds:
        if base_model != "think":     # R8 baseline: one arm, labelled by the model (graph structure is built in)
            m = TRAIN(cfg(base_model, "HH", "hyper", s, epochs, exp, input_mode, lr, alpha, tuned, overrides), data, hg)
            print(f"seed {s} {base_model}: val_sr {m['val']['sr']:.3f} test_sr {m['test']['sr']:.3f}", flush=True)
            continue
        for geo in pick_geoms(arms):
            for st in structs_for(geo):
                if arms and f"{geo}_{st}" not in arms:
                    continue
                m = TRAIN(cfg(f"{geo}_{st}", geo, st, s, epochs, exp, input_mode, lr, alpha, tuned, overrides), data, hg)
                print(f"seed {s} {geo}_{st}: val_sr {m['val']['sr']:.3f} test_sr {m['test']['sr']:.3f}", flush=True)


def tune_label(geo, lr, alpha):
    return f"tune_{geo}_lr{lr:g}_a{alpha:g}"


def tune(seeds, epochs, exp=EXP, input_mode="level", overrides=None, geoms=None):
    data, hg = universe()
    for geo in pick_geoms(geoms=geoms):
        for lr in GRID_LR:
            for alpha in GRID_ALPHA:
                for s in seeds:
                    m = TRAIN(cfg(tune_label(geo, lr, alpha), geo, "hyper", s, epochs, exp, input_mode, lr, alpha, overrides=overrides),
                                      data, hg)
                    print(f"tune {geo} lr={lr:g} alpha={alpha:g} seed {s}: val_sr {m['val']['sr']:.3f}", flush=True)


def tune_select(exp=EXP, geoms=None):
    """Pick, per geometry, the combo with the highest mean validation Sharpe (never test).
    Geometries already present in tuned.json but not selected here are kept."""
    best = json.loads(tuned_path(exp).read_text()) if tuned_path(exp).exists() and geoms else {}
    for geo in pick_geoms(geoms=geoms):
        scores = []
        for lr in GRID_LR:
            for alpha in GRID_ALPHA:
                r = load(tune_label(geo, lr, alpha), exp)
                if r:
                    scores.append((float(np.mean([m["val"]["sr"] for m, _ in r.values()])), len(r), lr, alpha))
        if not scores:
            raise SystemExit(f"no tuning runs found for {geo} under results/{exp}")
        full = max(n for _, n, _, _ in scores)
        val, n, lr, alpha = max((s for s in scores if s[1] == full), key=lambda s: s[0])
        best[geo] = {"lr": lr, "alpha": alpha}
        print(f"{geo}: lr={lr:g} alpha={alpha:g} mean val Sharpe {val:.3f} over {n} seeds "
              f"({len(scores)} combos, {full} seeds each)")
    tuned_path(exp).write_text(json.dumps(best, indent=2))
    print(f"wrote {tuned_path(exp)}")


def load(label, exp=EXP):
    out = {}
    for d in sorted(Path("results", exp, label).glob("seed_*")):
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            out[int(d.name.split("_")[1])] = (m, np.load(d / "test_daily.npy"))
    return out


def ensemble_metrics(exp, label, seeds):
    """Average test predictions over seeds (files share masks and gt), then evaluate."""
    ps = [np.load(Path("results", exp, label, f"seed_{s}", "test_pred.npy")) for s in seeds]
    d = Path("results", exp, label, f"seed_{seeds[0]}")
    return evaluate_all(np.mean(ps, axis=0), np.load(d / "test_gt.npy"), np.load(d / "test_mask.npy"))


def summarize(exp=EXP):
    data, hg = universe()
    runs = {f"{g}_{s}": load(f"{g}_{s}", exp) for g in GEOMS for s in structs_for(g)}
    base = evaluate_baselines(data)
    L = [f"# THINK proof of concept — {data.num_nodes} NYSE stocks (Energy/Utilities + Finance)", "",
         f"{len(hg.edges)} hyperedges; test period = 2017 (237 days); Sharpe = mean/std of daily top-5 return x sqrt(252).", "",
         "| arm | seeds | test Sharpe (mean ± std) | val Sharpe | NDCG@5 | best-test-epoch Sharpe (paper protocol) "
         "| seed-ensemble Sharpe | seed-ensemble NDCG@5 |", "|---|---|---|---|---|---|---|---|"]
    for k, r in runs.items():
        if r:
            t = [m["test"]["sr"] for m, _ in r.values()]
            v = [m["val"]["sr"] for m, _ in r.values()]
            nd = [m["test"]["ndcg5"] for m, _ in r.values()]
            o = [m["test_oracle_sr"] for m, _ in r.values()]
            e = ensemble_metrics(exp, k, sorted(r))
            L.append(f"| {k} | {len(r)} | {np.mean(t):.3f} ± {np.std(t):.3f} | {np.mean(v):.3f} | {np.mean(nd):.3f} "
                     f"| {np.mean(o):.3f} ± {np.std(o):.3f} | {e['sr']:.3f} | {e['ndcg5']:.3f} |")
    for k in ("market", "random", "momentum", "oracle"):
        L.append(f"| baseline: {k} | - | {base[k]['sr']:.3f} | - | - | - | - | - |")

    def series(key, seeds):
        return np.mean([runs[key][s][1] for s in seeds], axis=0), np.array([runs[key][s][0]["test"]["sr"] for s in seeds])

    pairs = [("HH_hyper", "EE_hyper", "hyperbolic vs Euclidean (with hyperedges)"),
             ("HH_hyper", "HH_clique", "hyperedges vs pairwise edges (hyperbolic)"),
             ("HH_hyper", "HH_none", "relations vs none (hyperbolic)"),
             ("EE_hyper", "EE_clique", "hyperedges vs pairwise edges (Euclidean)"),
             ("EE_hyper", "EE_none", "relations vs none (Euclidean)"),
             # EH = the paper's "Euclidean" ablation (TCONV + DHHAN): only temporal geometry differs from HH
             ("HH_hyper", "EH_hyper", "hyperbolic vs Euclidean temporal conv (paper ablation, hyperedges)"),
             ("HH_clique", "EH_clique", "hyperbolic vs Euclidean temporal conv (paper ablation, pairwise edges)"),
             ("EH_hyper", "EH_clique", "hyperedges vs pairwise edges (Euclidean temporal conv)"),
             ("EH_hyper", "EE_hyper", "hyperbolic vs Euclidean hypergraph attention (Euclidean temporal conv)")]
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
    out = Path("results", exp, "summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize", "tune", "tune-select"])
    ap.add_argument("--seeds", default=None, help="default: 0-9 (run), 0-2 (tune)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--variant", default="")
    ap.add_argument("--input-mode", choices=["level", "relative"], default="level")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--use-tuned", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arms for run, e.g. HH_hyper HH_clique EH_hyper")
    ap.add_argument("--geoms", nargs="*", default=None, help="tune/tune-select geometries (default HH EE; EH is opt-in)")
    ap.add_argument("--grid-lr", nargs="*", type=float, default=None, help="tune/tune-select learning-rate grid (default 5e-4 1e-3 3e-3)")
    ap.add_argument("--grid-alpha", nargs="*", type=float, default=None, help="tune/tune-select alpha grid (default 0.1 1 10)")
    ap.add_argument("--dry-run", action="store_true", help="run/tune: print every config that would be trained, then exit")
    ap.add_argument("--set", nargs="*", default=None, metavar="K=V", dest="overrides",
                    help="RunConfig overrides for run/tune, e.g. shuffle_train_labels=true attn_dist=off decompose_size=10")
    a = ap.parse_args()
    exp = exp_name(a.variant)
    set_grid(a.grid_lr, a.grid_alpha)
    if a.dry_run:
        TRAIN = dry_train
    if a.cmd == "run":
        run(parse_seeds(a.seeds or "0-9"), a.epochs, exp, a.input_mode, a.lr, a.alpha, a.use_tuned, a.arms, a.overrides)
    elif a.cmd == "tune":
        tune(parse_seeds(a.seeds or "0-2"), a.epochs, exp, a.input_mode, a.overrides, a.geoms)
        if not a.dry_run:
            tune_select(exp, a.geoms)   # partial-seed workers: rerun tune-select once all seeds are done
    elif a.cmd == "tune-select":
        tune_select(exp, a.geoms)
    else:
        summarize(exp)
