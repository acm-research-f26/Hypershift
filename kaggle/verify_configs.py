"""Print the resolved configs of the Kaggle commands (EE, HE, EH, RSR_I, STHGCN) + hypergraph cache check; run from a repo/bundle root.
  python kaggle/verify_configs.py > a.json   (diff the output of the real repo and of the bundle copy: must be identical)"""
import hashlib
import json
from dataclasses import replace

from hypershift.config import apply_overrides, config_from_dict, load_yaml
from hypershift.data.hypergraph import HYPERGRAPH_CACHE_VERSION, build_rsr_hypergraph
from hypershift.experiments.grid import experiment

P = "exp=R5_g2 norm=paper epochs=100 patience=1000".split()
R8 = "exp=R8_baselines_g2 norm=paper epochs=100 patience=1000 batch_days=8".split()
out = {}
for label in ("EE", "HE", "EH"):
    c = next(c for c in experiment("E2_geometry") if c.label == label)
    d = config_from_dict(apply_overrides(replace(c, seed=3).to_dict(), P)).to_dict()
    out[f"R5_g2/{label}"] = d
for label, model, mb in (("RSR_I", "rsr_i", "2"), ("STHGCN", "sthgcn", "4")):
    d = config_from_dict({**apply_overrides(load_yaml("configs/think_nyse.yaml"), R8 + [f"label={label}", f"model={model}", f"micro_batch_days={mb}"]), "seed": 3}).to_dict()
    out[f"R8/{label}"] = d
for k, d in out.items():
    print(f"# {k}: exp={d['exp']} label={d['label']} model={d['model']} attn_score={d['attn_score']} temporal={d['temporal']} "
          f"spatial={d['spatial']} norm={d['norm']} epochs={d['epochs']} batch_days={d['batch_days']} micro={d['micro_batch_days']} "
          f"lr={d['lr']} alpha={d['alpha']} run_dir=results/{d['exp']}/{d['label']}/seed_{d['seed']}", flush=True)
hg = build_rsr_hypergraph("data/raw/rsr/data", "NYSE")   # cache=True: reads the v2 cache, would REBUILD (needs relation) if stale
print(f"# hypergraph NYSE: cache version {HYPERGRAPH_CACHE_VERSION}, {len(hg.edges)} hyperedges, {hg.num_nodes} nodes")
print(json.dumps(out, sort_keys=True, indent=1))
print("# sha256", hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16])
