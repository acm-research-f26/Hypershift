"""R7 analysis: per-arm F1 table, chance baselines on the test labels, paired tests. CPU only."""
import json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from hypershift.config import RunConfig
from hypershift.experiments.grid import _geo
from hypershift.train.loop import prepare, gather_batch, window_offsets
from hypershift.train.clf import tertile_thresholds

EXP = "results/E11_clf_g2"
arms = ("HH", "EH", "EE")
M = {a: [json.load(open(f"{EXP}/{a}/seed_{s}/metrics.json")) for s in range(25)] for a in arms}
print("n per arm", {a: len(M[a]) for a in arms})
for a in arms:
    print(a, "epochs", sorted(m["best_epoch"] for m in M[a]))
    for k in ("test_f1", "test_micro_f1", "val_f1", "val_micro_f1"):
        v = np.array([m[k] for m in M[a]]); print(f"  {k}: {v.mean():.4f} +- {v.std(ddof=1):.4f} (min {v.min():.3f} max {v.max():.3f})")

def labels(cfg, data, th, split):
    ys = []
    for o in window_offsets(data, cfg.seq, split):
        _, m, _, g = gather_batch(data, [o], cfg.seq); ys.append(np.digitize(g, th)[m > 0.5])
    return np.concatenate(ys)
cfg = RunConfig(exp="x", label="HH", seed=0, **_geo("HH", market="NASDAQ", alpha=0.1))
data, hg = prepare(cfg, None, None)
th = tertile_thresholds(data, cfg.seq)
print("thresholds", th)
ytr = labels(cfg, data, th, "train"); yva = labels(cfg, data, th, "val"); yte = labels(cfg, data, th, "test")
for n, y in (("train", ytr), ("val", yva), ("test", yte)):
    print(n, len(y), np.bincount(y, minlength=3) / len(y))
prior_tr = np.bincount(ytr, minlength=3) / len(ytr)
prior_te = np.bincount(yte, minlength=3) / len(yte)
rng = np.random.default_rng(0)
def both(p, y): return f1_score(y, p, average="macro"), f1_score(y, p, average="micro")
R = 200
def avg(f):
    r = np.array([both(f(), yte) for _ in range(R)]); return r.mean(0)
print("uniform random", avg(lambda: rng.integers(0, 3, len(yte))))
print("class-prior (train prior) sampling", avg(lambda: rng.choice(3, len(yte), p=prior_tr)))
print("class-prior (test prior) sampling [oracle]", avg(lambda: rng.choice(3, len(yte), p=prior_te)))
print("always test-majority", both(np.full(len(yte), prior_te.argmax()), yte), "class", prior_te.argmax())
print("always train-majority", both(np.full(len(yte), prior_tr.argmax()), yte), "class", prior_tr.argmax())
for c in range(3): print("always class", c, both(np.full(len(yte), c), yte))

def paired(a, b, key):
    x = np.array([m[key] for m in M[a]]); y = np.array([m[key] for m in M[b]]); d = x - y
    p = wilcoxon(d).pvalue
    bs = np.random.default_rng(1).choice(d, (20000, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{a}-{b} {key}: mean diff {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] wilcoxon p={p:.4g} wins {int((d>0).sum())}/25")
for b in ("EH", "EE"):
    for k in ("test_f1", "test_micro_f1", "val_f1"): paired("HH", b, k)
paired("EH", "EE", "test_f1")
