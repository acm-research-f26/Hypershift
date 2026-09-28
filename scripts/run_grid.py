"""python scripts/run_grid.py E1_main E2_geometry [--labels THINK EE] [--seeds 0-14] [--dry-run] [--set k=v ...]

--labels keeps only those labels (matched before --set is applied); --seeds replaces each label's seed list.
"""
import argparse
import json
from collections import Counter
from dataclasses import replace

from tqdm import tqdm

from hypershift.config import apply_overrides, config_from_dict
from hypershift.experiments.grid import experiment
from hypershift.run import parse_seeds
from hypershift.train.loop import train_one_run

ap = argparse.ArgumentParser()
ap.add_argument("exps", nargs="+")
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--seeds", default=None, help="e.g. 0-14 or 0,3,7; default = the grid's seeds")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
base = [c for e in args.exps for c in experiment(e) if args.labels is None or c.label in args.labels]
if args.labels:
    unknown = set(args.labels) - {c.label for c in base}
    if unknown:
        raise SystemExit(f"unknown labels for {args.exps}: {sorted(unknown)}")
if args.seeds is not None:
    first = {}
    for c in base:
        first.setdefault((c.exp, c.label), c)
    base = [replace(c, seed=s) for c in first.values() for s in parse_seeds(args.seeds)]
cfgs = [config_from_dict(apply_overrides(c.to_dict(), args.set)) for c in base]
todo = [c for c in cfgs if not (c.run_dir() / "metrics.json").exists()]
print(f"{len(cfgs)} configs, {len(todo)} to run")
if args.dry_run:
    for (exp, label), n in sorted(Counter((c.exp, c.label) for c in cfgs).items()):
        print(f"  {exp}/{label}: {n} seeds")
    raise SystemExit(0)
for cfg in tqdm(todo):
    try:
        train_one_run(cfg)
        (cfg.run_dir() / "failed.json").unlink(missing_ok=True)   # a successful re-run clears an old failure
    except (FloatingPointError, RuntimeError) as err:   # RuntimeError covers CUDA OOM
        cfg.run_dir().mkdir(parents=True, exist_ok=True)
        (cfg.run_dir() / "failed.json").write_text(json.dumps({"error": repr(err)}))
        print("FAILED", cfg.exp, cfg.label, cfg.seed, repr(err))
