"""W6: the authors' own hyperbolicity code (HyperStock-GAT WWW'21, training/utils/hyperbolicity.py L12-35) on our graphs.
hyperbolicity_sample: num_samples=50000 draws of 4 distinct nodes (np.random.choice(G.nodes(),4,replace=False)), unweighted shortest-path
distances, pairs with no path skipped (exception -> continue), per tuple (s[-1]-s[-2])/2 of sorted {d01+d23,d02+d13,d03+d12}, returns MAX.
Here G = clique expansion of the hypergraph (s=1 s-distance), nodes = ALL nodes (incl. isolated, as networkx G.nodes() would if added)
or only non-isolated nodes. CPU only."""
import os, sys, json, numpy as np
sys.path.insert(0, "scripts")
import w5_delta_grid as W
from hypershift.geometry.hyperbolicity import s_distance_matrix
res = {}
for market in ("NYSE", "NASDAQ"):
    for gname in ("v2", "v1_old", "industry", "wiki"):
        hg = W._build_graph(market, gname)
        D = s_distance_matrix(hg, 1)
        n = len(D); deg_nonisol = np.isfinite(D).sum(1) > 1
        for pool_name, pool in (("all_nodes", np.arange(n)), ("non_isolated", np.nonzero(deg_nonisol)[0])):
            outs = []; conn = []
            for seed in range(20):
                rng = np.random.default_rng(seed)
                # vectorised 50000 draws of 4 distinct nodes
                T = np.stack([rng.choice(pool, 4, replace=False) for _ in range(50000)])
                a, b, c, d = T.T
                d01, d23, d02, d13, d03, d12 = D[a, b], D[c, d], D[a, c], D[b, d], D[a, d], D[b, c]
                ok = np.isfinite(d01 + d23 + d02 + d13 + d03 + d12)
                S = np.sort(np.stack([d01 + d23, d02 + d13, d03 + d12], 1)[ok], 1)
                h = (S[:, 2] - S[:, 1]) / 2
                outs.append(float(h.max()) if len(h) else float("nan")); conn.append(float(ok.mean()))
            res[f"{market}/{gname}/{pool_name}"] = {"max_per_seed": outs, "frac_connected": float(np.mean(conn)),
                                                   "values": sorted(set(outs)), "n_nodes": int(n), "n_edges": len(hg.edges)}
            print(market, gname, pool_name, "conn %.3f" % np.mean(conn), "max over 50k tuples, 20 seeds:", sorted(set(outs)),
                  "mean %.2f" % np.mean(outs), flush=True)
json.dump(res, open("docs/phase2/w6_authors_delta.json", "w"), indent=1)
