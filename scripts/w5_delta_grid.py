"""W5: grid of delta_hg / delta_rel settings vs THINK Table I (spec: docs/phase2/W5_DELTA_SPEC.md).
usage: w5_delta_grid.py hg|rel|exact [--procs 14]   -> results/w5_delta/{hg,rel,exact}.jsonl (resumable)"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from hypershift.data.hypergraph import (Hypergraph, build_rsr_hypergraph, canonical, induced_subgraph,
                                        industry_hyperedges, wiki_hyperedges)
from hypershift.data.rsr import load_rsr, read_ticker_file
from hypershift.geometry.hyperbolicity import (gromov_delta, khrulkov_delta_rel, largest_component,
                                               s_distance_matrix)
from hypershift.train.loop import apply_input_mode

R = Path("data/raw/rsr/data")
OUT = Path("results/w5_delta")
MARKETS = ("NYSE", "NASDAQ")
SVALS = (1, 2, 3, 4, 5)
KS = (10, 20, 30, 50, 100, 200, 500)
GRAPHS = ("v2", "industry", "wiki", "v2_no_na", "v1_old")


def graph(market, name):
    cf = OUT / "graphs" / f"{market}_{name}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return Hypergraph(d["n"], tuple(tuple(e) for e in d["edges"]))
    return _build_graph(market, name)


def _build_graph(market, name):
    full = build_rsr_hypergraph(R, market)
    n = full.num_nodes
    if name == "v2":
        return full
    rel = np.load(R / "relation/sector_industry" / f"{market}_industry_relation.npy")
    ind = industry_hyperedges(rel)
    if name == "industry":
        return Hypergraph(n, canonical(ind))
    if name == "wiki":
        return build_rsr_hypergraph(R, market, sources=("wiki",))
    if name == "v2_no_na":
        tick = read_ticker_file(R / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv")
        j = json.load(open(R / "relation/sector_industry" / f"{market}_industry_ticker.json"))
        pos = {t: i for i, t in enumerate(tick)}
        na = {pos[t] for t in j.get("n/a", []) if t in pos}
        return Hypergraph(n, canonical([e for e in full.edges if not (len(e) >= 400 and set(e) <= na)]))
    if name == "v1_old":  # star hyperedge for every wiki channel (pre-de20f8e)
        wr = np.load(R / "relation/wikidata" / f"{market}_wiki_relation.npy")
        return Hypergraph(n, canonical(ind + wiki_hyperedges(wr, np.ones(wr.shape[2] - 1, dtype=bool))))
    raise ValueError(name)


def lcc_D(hg, s):
    D = s_distance_matrix(hg, s)
    c = largest_component(D)
    return D[np.ix_(c, c)], len(c)


def allb(D):
    return max(gromov_delta(D, base=b) for b in range(len(D)))


def hg_task(args):
    market, gname, s = args
    t0 = time.time()
    hg = graph(market, gname)
    D, lcc = lcc_D(hg, s)
    fin = D[np.isfinite(D)]
    res = {"market": market, "graph": gname, "s": s, "n_edges": len(hg.edges), "lcc": int(lcc),
           "diam": float(fin.max()), "cells": []}
    rng = np.random.default_rng(1000 * s + len(gname))
    for scheme in ("dist", "induced"):
        for k in KS:
            draws = 200 if k <= 100 else 50
            single, allv = [], []
            for _ in range(draws):
                if scheme == "dist":
                    idx = rng.choice(lcc, size=min(k, lcc), replace=False)
                    Ds = D[np.ix_(idx, idx)]
                else:
                    idx = rng.choice(hg.num_nodes, size=k, replace=False)
                    Ds, _ = lcc_D(induced_subgraph(hg, idx), s)
                    if len(Ds) < 4:
                        continue
                single.append(gromov_delta(Ds, base=0))
                if k <= 100:
                    allv.append(allb(Ds))
            res["cells"].append({"scheme": scheme, "k": k, "draws": len(single), "single": single, "all": allv})
            print(market, gname, s, scheme, k, round(time.time() - t0), flush=True)
    bases = rng.choice(lcc, size=min(24, lcc), replace=False)
    res["full_single_bases"] = [gromov_delta(D, base=int(b)) for b in bases]
    res["secs"] = time.time() - t0
    return res


def exact_task(args):
    market, gname, s, b = args
    D, _ = lcc_D(graph(market, gname), s)
    return {"market": market, "graph": gname, "s": s, "base": int(b), "delta": gromov_delta(D, base=int(b))}


_DATA = {}
FEATS = (["ret_train", "ret_full", "closeseries_train", "closeseries_full", "feat5series_train", "feat5series_full"]
         + [f"{b}_{t}" for t in ("tr", "fu") for b in ("feat5", "win16_level", "win16_rel")])
# 1-D features (close, ma5..ma30 at one day) lie on a line: delta_rel = 0 analytically (measured 2e-7 float noise) -> not scanned
NORM_FREE = {"ret_train", "ret_full", "win16_rel_tr", "win16_rel_fu"}  # (feat5, level windows, series depend on norm)
SIZES = (500, 1000, 1500, 2000, "all")
TRIES = 20


def feats(market, norm, name):
    if (market, norm) not in _DATA:
        _DATA[(market, norm)] = load_rsr(R, market, norm)
    d = _DATA[(market, norm)]
    vi, T, f = d.valid_index, d.num_steps, d.features
    N = f.shape[0]
    if name == "ret_train":
        return np.where(d.mask[:, 1:vi] > 0, d.gt[:, 1:vi], 0.0)
    if name == "ret_full":
        return np.where(d.mask[:, 1:] > 0, d.gt[:, 1:], 0.0)
    if name == "closeseries_train":
        return f[:, :vi, -1]
    if name == "closeseries_full":
        return f[:, :, -1]
    if name == "feat5series_train":
        return f[:, :vi, :].reshape(N, -1)
    if name == "feat5series_full":
        return f.reshape(N, -1)
    base, tag = name.rsplit("_", 1)
    e = vi - 1 if tag == "tr" else T - 1
    if base == "close":
        return f[:, e, -1:]
    if base == "feat5":
        return f[:, e, :]
    if base in ("ma5", "ma10", "ma20", "ma30"):
        j = ("ma5", "ma10", "ma20", "ma30").index(base)
        return f[:, e, j:j + 1]
    if base in ("win16_level", "win16_rel"):
        x = f[:, e - 15:e + 1, :][None]
        if base == "win16_rel":
            x = apply_input_mode(x, "relative")
        return x[0].reshape(N, -1)
    raise ValueError(name)


def rel_task(args):
    market, norm, name = args
    t0 = time.time()
    X = feats(market, norm, name).astype(np.float64)
    N = len(X)
    out = {"market": market, "norm": norm, "feature": name, "dim": int(X.shape[1]), "cells": []}
    seen = set()
    for m in SIZES:
        mm = N if m == "all" else min(m, N)
        for repl in (True, False):
            if (mm, repl) in seen:
                continue
            seen.add((mm, repl))
            if not repl and mm > 1000:
                continue  # compute budget: without-replacement variant only for m <= 1000, 10 tries
            out["cells"].append({"m": int(mm), "replace": repl,
                                 **khrulkov_delta_rel(X, mm, TRIES if repl else 10, seed=7, replace=repl)})
    out["secs"] = time.time() - t0
    return out


def run_tasks(tasks, fn, name, keyf, procs):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = {tuple(json.loads(l)["_key"]) for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [t for t in tasks if t not in done]
    print(len(todo), "tasks", flush=True)
    with Pool(procs) as p, open(path, "a") as f:
        for r in p.imap_unordered(fn, todo):
            r["_key"] = keyf(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["_key"], round(r["secs"]), flush=True)


def precompute_graphs():
    (OUT / "graphs").mkdir(parents=True, exist_ok=True)
    for m in MARKETS:
        for g in GRAPHS:
            hg = _build_graph(m, g)
            (OUT / "graphs" / f"{m}_{g}.json").write_text(json.dumps({"n": hg.num_nodes, "edges": [list(e) for e in hg.edges]}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "graphs":
        precompute_graphs()
        sys.exit(0)
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 14
    if mode == "hg":
        run_tasks([(m, g, s) for m in MARKETS for g in GRAPHS for s in SVALS], hg_task, "hg",
                  lambda r: (r["market"], r["graph"], r["s"]), procs)
    elif mode == "rel":
        run_tasks([(m, n, f) for m in MARKETS for f in FEATS for n in (("train",) if f in NORM_FREE else ("train", "paper"))],
                  rel_task, "rel", lambda r: (r["market"], r["norm"], r["feature"]), procs)
    elif mode == "exact":  # exact all-bases delta_hg on v2, s=1
        OUT.mkdir(parents=True, exist_ok=True)
        with Pool(procs) as p, open(OUT / "exact.jsonl", "a") as f:
            for m in MARKETS:
                _, lcc = lcc_D(graph(m, "v2"), 1)
                for r in p.imap_unordered(exact_task, [(m, "v2", 1, b) for b in range(lcc)], chunksize=8):
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(m, "done", flush=True)
