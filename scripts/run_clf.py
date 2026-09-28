"""python scripts/run_clf.py  -> 25 seeds x {HH, EE, EH} on NASDAQ, prints mean±std macro-F1."""
import numpy as np

from hypershift.config import RunConfig
from hypershift.experiments.grid import _geo
from hypershift.train.clf import train_clf_run

for g in ("HH", "EH", "EE"):
    f1s = [train_clf_run(RunConfig(exp="E11_clf", label=g, seed=s, **_geo(g, market="NASDAQ", alpha=0.1)))["test_f1"]
           for s in range(25)]
    print(g, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
