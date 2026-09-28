"""Paper Table I analogue. delta_rel features = each stock's training-period daily return series."""
import json
from pathlib import Path

import numpy as np

from hypershift.config import RunConfig
from hypershift.data.hypergraph import build_rsr_hypergraph
from hypershift.data.rsr import load_rsr
from hypershift.data.universe import select_universe
from hypershift.geometry.hyperbolicity import feature_hyperbolicity, hypergraph_hyperbolicity

root = RunConfig().data_root
res = {}
for market in ("NYSE", "NASDAQ"):
    data = load_rsr(root, market, "train")
    hg = build_rsr_hypergraph(root, market)
    universes = [("full", data, hg)]
    if market == "NYSE":
        for n in (50, 100, 250, 500, 1000):
            for u in (0, 1, 2):
                universes.append((f"N{n}_u{u}", *select_universe(data, hg, n, u)))
    for name, d, h in universes:
        X = np.where(d.mask[:, 1:d.valid_index] > 0, d.gt[:, 1:d.valid_index], 0.0)
        res[f"{market}_{name}"] = {"hg": hypergraph_hyperbolicity(h), "rel": feature_hyperbolicity(X),
                                   "num_edges": len(h.edges)}
        print(market, name, res[f"{market}_{name}"]["hg"]["delta_max"], res[f"{market}_{name}"]["rel"]["delta_rel"])
Path("results").mkdir(exist_ok=True)
Path("results/hyperbolicity.json").write_text(json.dumps(res, indent=2))
