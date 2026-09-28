"""Decision node D7. Times HH (3 epochs) and HH_clique (1 epoch) on full NYSE, then estimates total GPU hours."""
import json

from hypershift.experiments.grid import EXPERIMENTS, experiment
from hypershift.train.loop import train_one_run

sec = {}
for cfg in experiment("E0_budget"):
    sec[cfg.label] = train_one_run(cfg)["sec_per_epoch"]
print(json.dumps(sec, indent=2))
t_run_min = sec["HH"] * 60 / 60
print(f"t_run ~ {t_run_min:.1f} min (60 epochs incl. early stopping)")
total = 0.0
for e in EXPERIMENTS:
    if e in ("E0_budget", "E7_hubs", "E9_fresh_daily", "E10_hourly"):
        continue
    for c in experiment(e):
        per_epoch = sec["HH_clique"] if c.structure == "clique" else sec["HH"]
        per_epoch *= (c.universe_size / 1737) if c.universe_size else 1.0
        n_epochs = c.epochs if c.patience >= c.epochs else min(c.epochs, 60)   # no-early-stop arms run all epochs
        total += per_epoch * n_epochs
print(f"estimated total for RSR experiments: {total / 3600:.1f} GPU hours")
print("D7: <=6 min/run keep; 6-15 -> configs/global.yaml batch_days: 4; >15 -> batch_days: 8, epochs: 60")
