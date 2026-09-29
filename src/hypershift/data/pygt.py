"""PyG-Temporal loaders (R1 tennis, R2 chickenpox, R3 windmill). Read the raw JSON; no torch-geometric-temporal.

Chickenpox: the JSON stores `FX` [T=521, N=20] ALREADY standardized over the whole series (global z-score, so
mean ~0 / std 1 including the test period). `edges` are 102 directed pairs including self-loops.
Windmill (large): `block` [T=17472, N=319] RAW capacity factors; the PyG-T loader z-scores per node over the full
series (look-ahead). `edges`/`weights` = complete weighted graph (319^2 incl. self-loops).
Tennis (rg17): 120 snapshots x 1000 nodes, per-snapshot edges; target = log(1+y[min(t+1,T-1)]) as in the loader.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hypershift.data.hypergraph import Hypergraph, canonical

DEFAULT_PATH = Path("data/raw/pygt/chickenpox.json")


@dataclass
class PygtData:
    x: np.ndarray            # [N, T] as stored (float32)
    edges: np.ndarray        # [E, 2] directed, as stored (incl. self-loops)
    num_nodes: int


def load_chickenpox(path: str | Path = DEFAULT_PATH) -> PygtData:
    d = json.loads(Path(path).read_text())
    fx = np.asarray(d["FX"], dtype=np.float32)          # [T, N]
    return PygtData(x=fx.T.copy(), edges=np.asarray(d["edges"], dtype=np.int64), num_nodes=fx.shape[1])


def make_windows(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """x [N,T] -> inputs [W,N,lags,1], targets [W,N]; window w predicts time w+lags (PyG-T convention)."""
    n, t = x.shape
    w = t - lags
    idx = np.arange(w)[:, None] + np.arange(lags)[None]
    inp = x[:, idx].transpose(1, 0, 2)[..., None]       # [W,N,lags,1]
    return inp.astype(np.float32), x[:, lags:].T.astype(np.float32)


def neighbourhood_hypergraph(data: PygtData) -> Hypergraph:
    """One hyperedge per node: {v} U out-neighbours(v); deduplicated, singletons dropped."""
    nb = {v: {v} for v in range(data.num_nodes)}
    for a, b in data.edges:
        nb[int(a)].add(int(b))
    return Hypergraph(data.num_nodes, canonical(nb.values()))


def pairwise_hypergraph(data: PygtData) -> Hypergraph:
    """The original graph as 2-node hyperedges (self-loops dropped, direction merged)."""
    return Hypergraph(data.num_nodes, canonical((int(a), int(b)) for a, b in data.edges if a != b))


# ----------------------------------------------------------------------------- windmill (R3)
DEFAULT_WINDMILL = Path("data/raw/pygt/windmill_output.json")


@dataclass
class WindmillData(PygtData):
    weights: np.ndarray = None   # [E] as stored


def load_windmill(path: str | Path = DEFAULT_WINDMILL) -> WindmillData:
    d = json.loads(Path(path).read_text())
    blk = np.asarray(d["block"], dtype=np.float64)       # [T, N] raw
    return WindmillData(x=blk.T.copy(), edges=np.asarray(d["edges"], dtype=np.int64), num_nodes=blk.shape[1],
                        weights=np.asarray(d["weights"], dtype=np.float64))


def topk_hypergraph(data: WindmillData, k: int = 5) -> Hypergraph:
    """One hyperedge per node: {v} U its k strongest out-neighbours by edge weight (self-loops excluded)."""
    e, w = data.edges, data.weights
    keep = e[:, 0] != e[:, 1]
    e, w = e[keep], w[keep]
    nb = {}
    for v in range(data.num_nodes):
        m = e[:, 0] == v
        order = np.argsort(-w[m], kind="stable")[:k]
        nb[v] = {v} | {int(u) for u in e[m][order, 1]}
    return Hypergraph(data.num_nodes, canonical(nb.values()))


def topk_pairs(data: WindmillData, k: int = 5) -> Hypergraph:
    e, w = data.edges, data.weights
    keep = e[:, 0] != e[:, 1]
    e, w = e[keep], w[keep]
    pairs = []
    for v in range(data.num_nodes):
        m = e[:, 0] == v
        order = np.argsort(-w[m], kind="stable")[:k]
        pairs += [(v, int(u)) for u in e[m][order, 1]]
    return Hypergraph(data.num_nodes, canonical(pairs))


def quantile_hypergraph(data: WindmillData, top_frac: float = 0.10) -> Hypergraph:
    """{v} U {u : w(v,u) >= the (1-top_frac) quantile of all non-self edge weights}."""
    e, w = data.edges, data.weights
    keep = e[:, 0] != e[:, 1]
    e, w = e[keep], w[keep]
    thr = np.quantile(w, 1 - top_frac)
    nb = {v: {v} for v in range(data.num_nodes)}
    for (a, b), ww in zip(e, w):
        if ww >= thr:
            nb[int(a)].add(int(b))
    return Hypergraph(data.num_nodes, canonical(nb.values()))


# ----------------------------------------------------------------------------- tennis (R1)
DEFAULT_TENNIS = Path("data/raw/pygt/twitter_tennis_rg17.json")


@dataclass
class TennisData:
    x: np.ndarray             # [N, T, F] features (encoded one-hot by default)
    y: np.ndarray             # [N, T] target aligned to snapshot t (loader convention: y_json[min(t+1,T-1)])
    edges: list               # per-snapshot [E_t, 2] directed mention edges
    num_nodes: int


def encode_features(xr: np.ndarray, cutoff: int = 4) -> np.ndarray:
    """PyG-T `encode_features`: 5 one-hot bins of ceil(log(deg+1)) (capped) + 11 one-hot bins of floor(10*transitivity)."""
    a = np.minimum(np.ceil(np.log(xr[:, 0] + 1.0)), cutoff).astype(int)
    b = np.floor(xr[:, 1] * 10).astype(int)
    out = np.zeros((len(xr), cutoff + 1 + 11), dtype=np.float32)
    out[np.arange(len(xr)), a] = 1.0
    out[np.arange(len(xr)), cutoff + 1 + np.clip(b, 0, 10)] = 1.0
    return out


def load_tennis(path: str | Path = DEFAULT_TENNIS, feature_mode: str = "encoded", target: str = "log1p") -> TennisData:
    d = json.loads(Path(path).read_text())
    t = int(d["time_periods"])
    feats, ys, edges = [], [], []
    for i in range(t):
        s = d[str(i)]
        xr = np.asarray(s["X"], dtype=np.float64)
        feats.append(encode_features(xr) if feature_mode == "encoded" else xr.astype(np.float32))
        edges.append(np.asarray(s["edges"], dtype=np.int64).reshape(-1, 2))
    for i in range(t):
        y = np.asarray(d[str(min(i + 1, t - 1))]["y"], dtype=np.float64)
        ys.append(np.log1p(y) if target == "log1p" else y)
    x = np.stack(feats, 1)                                # [N, T, F]
    return TennisData(x=x.astype(np.float32), y=np.stack(ys, 1).astype(np.float32), edges=edges, num_nodes=x.shape[0])


def make_windows_tennis(data: TennisData, lags: int):
    """Window w = snapshots w..w+lags-1; target = loader target of the LAST snapshot. Returns inp [W,N,lags,F],
    tgt [W,N], hist [W,N,lags] (loader targets of snapshots e-lags..e-1, e=w+lags-1; NaN before snapshot 0)."""
    n, t, f = data.x.shape
    w = t - lags + 1
    idx = np.arange(w)[:, None] + np.arange(lags)[None]
    inp = data.x[:, idx].transpose(1, 0, 2, 3)            # [W,N,lags,F]
    tgt = data.y[:, lags - 1:].T
    hidx = idx - 1                                        # w+j-1  (= e-lags+j)
    hist = data.y[:, np.clip(hidx, 0, None)].transpose(1, 0, 2).copy()
    hist[:, :, :][np.broadcast_to(hidx[:, None, :] < 0, hist.shape)] = np.nan
    return inp.astype(np.float32), tgt.astype(np.float32), hist.astype(np.float32)


def tennis_union_edges(data: TennisData, end_snapshot: int) -> np.ndarray:
    """Union of directed edges over snapshots [0, end_snapshot) only (training period, no future edges)."""
    e = np.concatenate(data.edges[:end_snapshot], 0)
    return np.unique(e, axis=0)


def tennis_neighbourhood(data: TennisData, end_snapshot: int) -> Hypergraph:
    nb = {v: {v} for v in range(data.num_nodes)}
    for a, b in tennis_union_edges(data, end_snapshot):
        nb[int(a)].add(int(b))
    return Hypergraph(data.num_nodes, canonical(nb.values()))


def tennis_pairs(data: TennisData, end_snapshot: int) -> Hypergraph:
    return Hypergraph(data.num_nodes, canonical((int(a), int(b)) for a, b in tennis_union_edges(data, end_snapshot) if a != b))
