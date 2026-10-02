"""Phase 1.5 F: tables from the preset-8 grid (results/f_<name>_<level>_<mode>/<arm>/seed_k/known_signal.json).
Baseline = D grid (results/known_signal_<level>_<mode>, seeds 0-2). IC at the validation-selected epoch / oracle IC.
  CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/f_analysis.py [--seeds 3] [--prefix f_]"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--prefix", default="f_")
a = ap.parse_args()
R = Path("results")
cells = defaultdict(list)           # (config, level, mode, arm) -> list of dicts
for p in R.glob("*/*/seed_*/known_signal.json"):
    exp = p.parents[2].name
    m = re.match(r"(.+)_(none|own|group|low|mid|high)_(level|relative)$", exp)
    if not m: continue
    cfg, level, mode = m.groups()
    if cfg == "known_signal": cfg = "D_baseline"
    elif not cfg.startswith(a.prefix): continue
    seed = int(p.parent.name.split("_")[1])
    if seed >= a.seeds: continue
    r = json.loads(p.read_text())
    h = p.parent / "history.jsonl"
    if h.exists():
        hist = [json.loads(l) for l in h.read_text().splitlines()]
        if "test_ic" in hist[0]:
            r["max_ic"] = max(x["test_ic"] for x in hist); r["last_sd"] = hist[-1]["test_pred_sd"]
    cells[(cfg, level, mode, r["arm"])].append(r)
rows = []
print("| config | mode | level | arm | n | test IC | IC/oracle | oracle IC | max-over-epochs IC/oracle (diag) | pred sd (last) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (cfg, level, mode, arm), rs in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][3], kv[0][0] != "D_baseline", kv[0][0])):
    ic = np.mean([r["test_ic"] for r in rs]); orc = np.mean([r["ref"]["oracle"]["ic"] for r in rs])
    mx = [r["max_ic"] for r in rs if "max_ic" in r]
    sd = [r["last_sd"] for r in rs if "last_sd" in r]
    print(f"| {cfg} | {mode} | {level} | {arm} | {len(rs)} | {ic:+.3f} | {ic/orc:.2f} | {orc:.3f} | "
          f"{(np.mean(mx)/orc if mx else float('nan')):.2f} | {(np.mean(sd) if sd else float('nan')):.1e} |")
