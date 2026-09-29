"""PyG-Temporal Hungary Chickenpox loader (R2). Reads the raw JSON; no torch-geometric-temporal needed.

The JSON stores `FX` [T=521, N=20] ALREADY standardized over the whole series (global z-score, so
mean ~0 / std 1 including the test period). `edges` are 102 directed pairs including self-loops.
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
