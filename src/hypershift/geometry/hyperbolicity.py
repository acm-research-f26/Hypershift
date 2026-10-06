from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from hypershift.data.hypergraph import Hypergraph


def s_distance_matrix(hg: Hypergraph, s: int = 1) -> np.ndarray:
    node, edge = hg.incidence()
    H = csr_matrix((np.ones(len(node)), (node, edge)), shape=(hg.num_nodes, max(len(hg.edges), 1)))
    C = (H @ H.T).tolil()
    C.setdiag(0)
    A = (C.tocsr() >= s).astype(float)
    return shortest_path(A, method="D", unweighted=True, directed=False)


def largest_component(D: np.ndarray) -> np.ndarray:
    finite = np.isfinite(D).sum(axis=1)
    return np.nonzero(np.isfinite(D[int(np.argmax(finite))]))[0]


def gromov_delta(D: np.ndarray, base: int = 0) -> float:
    D = np.asarray(D, dtype=np.float32)
    row = D[base]
    A = 0.5 * (row[:, None] + row[None, :] - D)
    M = np.empty_like(A)
    for i in range(0, len(A), 16):
        M[i:i + 16] = np.minimum(A[i:i + 16, :, None], A[None, :, :]).max(axis=1)
    return float(max((M - A).max(), 0.0))


def sampled_delta(D: np.ndarray, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(D)
    vals = []
    for _ in range(repeats):
        idx = rng.choice(n, size=min(sample, n), replace=False)
        vals.append(gromov_delta(D[np.ix_(idx, idx)], base=0))
    diam = float(D[np.isfinite(D)].max())
    dmax = float(max(vals))
    return {"delta_max": dmax, "delta_mean": float(np.mean(vals)), "diam": diam,
            "delta_rel": 2 * dmax / diam if diam > 0 else 0.0}


def hypergraph_hyperbolicity(hg: Hypergraph, s: int = 1, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    D = s_distance_matrix(hg, s)
    comp = largest_component(D)
    out = sampled_delta(D[np.ix_(comp, comp)], sample, repeats, seed)
    out["lcc_size"] = int(len(comp))
    return out


def feature_hyperbolicity(X: np.ndarray, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    return sampled_delta(cdist(X, X), sample, repeats, seed)


def gromov_delta_all_bases(D: np.ndarray, bases=None) -> float:
    """max over base points (all by default) of the single-base Gromov delta: the exact 4-point delta
    when `bases` is None, a lower bound otherwise."""
    idx = range(len(D)) if bases is None else bases
    return max((gromov_delta(D, base=int(b)) for b in idx), default=0.0)


def khrulkov_delta_rel(X: np.ndarray, subset: int, tries: int, seed: int = 0, replace: bool = True) -> dict:
    """Khrulkov et al. 2020 convention (hyptorch/delta.py): per try draw `subset` rows (with replacement by
    default), Euclidean distance matrix, single base point = first row, delta_rel = 2*delta/diam; mean over tries."""
    rng = np.random.default_rng(seed)
    n = len(X)
    subset = min(subset, n)
    vals = []
    for _ in range(tries):
        idx = rng.choice(n, size=subset, replace=replace)
        D = cdist(X[idx], X[idx])
        diam = D.max()
        vals.append(2 * gromov_delta(D, base=0) / diam if diam > 0 else 0.0)
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": tries}
