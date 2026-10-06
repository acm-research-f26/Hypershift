"""W6: rescoring saved history.jsonl under author-code-style epoch readings (CPU, no training).
Readings: val-selected (ours), LAST epoch (no selection; STHAN-SR/HyperStockGAT code prints every epoch, saves nothing),
test-ORACLE (max over epochs; LEAKY), MEAN over epochs. Metrics: ann Sharpe (k=5, x sqrt252), buggy NDCG (ndcg_sthan), correct NDCG@5.
Reports seed mean, SD, SE (SD/sqrt n) to compare with the paper's +-1e-3 level."""
import json, glob, os, sys, numpy as np
ROOT = "results"
GROUPS = ["R5_f_paper", "R5_f_train", "R5_f2_alpha0_train", "R5_f3_alpha0_train", "R8_f_paper", "R8_f_train"]
def load(d):
    h = [json.loads(l) for l in open(os.path.join(d, "history.jsonl"))]
    m = json.load(open(os.path.join(d, "metrics.json")))
    return h, m
out = {}
for g in GROUPS:
    for lab in sorted(os.listdir(f"{ROOT}/{g}")):
        runs = []
        for sd in sorted(glob.glob(f"{ROOT}/{g}/{lab}/seed_*")):
            if not os.path.exists(f"{sd}/metrics.json"): continue
            runs.append(load(sd))
        if not runs: continue
        row = {"n": len(runs), "epochs": [len(h) for h, _ in runs][:1]}
        for key in ("sr", "ndcg_sthan", "ndcg5"):
            v = {"val": [], "last": [], "oracle": [], "mean": []}
            for h, m in runs:
                t = np.array([e["test"][key] for e in h])
                v["val"].append(m["test"][key]); v["last"].append(t[-1]); v["oracle"].append(t.max()); v["mean"].append(t.mean())
            row[key] = {k: [float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else None,
                            float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None] for k, x in v.items()}
        out[f"{g}/{lab}"] = row
json.dump(out, open("docs/phase2/w6_selection_readings.json", "w"), indent=1)
print("| group/arm | n | ep | metric | val mean(SD) | LAST mean(SD) | ORACLE mean(SD,SE) | epoch-mean mean(SD) |\n|---|---|---|---|---|---|---|---|")
for k, r in out.items():
    for key in ("sr", "ndcg_sthan", "ndcg5"):
        d = r[key]; f = lambda a: f"{a[0]:.3f}({a[1]:.3f})" if a[1] is not None else f"{a[0]:.3f}"
        print(f"| {k} | {r['n']} | {r['epochs'][0]} | {key} | {f(d['val'])} | {f(d['last'])} | {f(d['oracle'])}, SE {d['oracle'][2]:.3f} | {f(d['mean'])} |" if d['oracle'][2] is not None else f"| {k} | {r['n']} | {r['epochs'][0]} | {key} | {f(d['val'])} | {f(d['last'])} | {f(d['oracle'])} | {f(d['mean'])} |")
