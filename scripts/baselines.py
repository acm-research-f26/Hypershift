"""Baselines for every universe used in the study. Usage: python scripts/baselines.py [--fresh NAME ...]"""
import argparse
import json
from pathlib import Path

from hypershift.config import RunConfig
from hypershift.data.hypergraph import build_rsr_hypergraph
from hypershift.data.rsr import load_rsr
from hypershift.data.universe import select_universe
from hypershift.eval.baselines import evaluate_baselines

ap = argparse.ArgumentParser()
ap.add_argument("--fresh", nargs="*", default=[])
args = ap.parse_args()
out = Path("results/baselines")
out.mkdir(parents=True, exist_ok=True)
root = RunConfig().data_root
for market in ("NYSE", "NASDAQ"):
    data = load_rsr(root, market, "train")
    (out / f"{market}.json").write_text(json.dumps(evaluate_baselines(data), indent=2))
    if market == "NYSE":
        hg = build_rsr_hypergraph(root, market)
        for n in (50, 100, 250, 500, 1000):
            for u in (0, 1, 2):
                sub, _ = select_universe(data, hg, n, u)
                (out / f"NYSE_N{n}_u{u}.json").write_text(json.dumps(evaluate_baselines(sub), indent=2))
for name in args.fresh:
    from hypershift.data.fresh import load_panel
    data = load_panel(Path("data/fresh") / name)
    ppy = 1764 if name.endswith("_hourly") else 252
    (out / f"{name}.json").write_text(json.dumps(evaluate_baselines(data, periods_per_year=ppy), indent=2))
print("wrote", sorted(p.name for p in out.glob("*.json")))
