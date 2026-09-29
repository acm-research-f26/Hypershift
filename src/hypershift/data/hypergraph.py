"""Stock hypergraphs: construction from RSR relations (paper appendix B) and ablation transforms."""
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Hypergraph:
    num_nodes: int
    edges: tuple[tuple[int, ...], ...]

    def incidence(self) -> tuple[np.ndarray, np.ndarray]:
        node = np.fromiter((v for e in self.edges for v in e), dtype=np.int64)
        edge = np.repeat(np.arange(len(self.edges), dtype=np.int64), [len(e) for e in self.edges])
        return node, edge

    def node_degree(self) -> np.ndarray:
        deg = np.zeros(self.num_nodes, dtype=np.int64)
        node, _ = self.incidence()
        np.add.at(deg, node, 1)
        return deg

    def edge_sizes(self) -> np.ndarray:
        return np.array([len(e) for e in self.edges], dtype=np.int64)

    def to_torch(self, device) -> "TorchHypergraph":
        node, edge = self.incidence()
        return TorchHypergraph(
            node_idx=torch.as_tensor(node, device=device),
            edge_idx=torch.as_tensor(edge, device=device),
            num_nodes=self.num_nodes,
            num_edges=len(self.edges),
            has_edge=torch.as_tensor(self.node_degree() > 0, device=device),
            edge_size=torch.as_tensor(self.edge_sizes(), dtype=torch.float32, device=device),
        )


@dataclass
class TorchHypergraph:
    node_idx: torch.Tensor
    edge_idx: torch.Tensor
    num_nodes: int
    num_edges: int
    has_edge: torch.Tensor
    edge_size: torch.Tensor


def canonical(edges, min_size: int = 2) -> tuple[tuple[int, ...], ...]:
    seen, out = set(), []
    for e in edges:
        t = tuple(sorted({int(v) for v in e}))
        if len(t) >= min_size and t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


def industry_hyperedges(rel: np.ndarray) -> list[tuple[int, ...]]:
    """rel [N,N,R]; last channel is the self-relation and is ignored."""
    out = []
    for k in range(rel.shape[2] - 1):
        members = np.nonzero(rel[:, :, k].sum(axis=1) > 0)[0]
        out.append(tuple(int(v) for v in members))
    return out


# Bump when the construction of any cached hypergraph changes; stale cache files are rebuilt.
# v1: star hyperedge for every wiki channel. v2: first-order channels star, second-order channels pairs (paper App. B).
HYPERGRAPH_CACHE_VERSION = 2


def wiki_first_order_channels(rel: np.ndarray, connections: dict, qids: list[str]) -> np.ndarray:
    """bool [R-1]: is wiki channel k a first-order relation (single Wikidata property, X -R1-> Y)?

    RSR's `<market>_connections.json` maps qid_i -> qid_j -> list of property paths. A path with ONE property is
    first-order; a path with TWO properties (X -R2-> Z <-R3- Y) is second-order. `rel[i, j, k]` carries no path
    label, but every channel is exactly the set of directed pairs sharing one path type, so channel k is first-order
    iff its pair set equals the pair set of some single-property path. (A channel whose pair set equals both a
    first-order and a second-order path is only possible for tiny channels, and there a star equals a pair.)
    """
    idx: dict[str, list[int]] = {}
    for i, q in enumerate(qids):
        idx.setdefault(q, []).append(i)
    first: dict[str, set] = {}
    for qa, d in connections.items():
        for qb, paths in d.items():
            for path in paths:
                if len(path) == 1:
                    first.setdefault(path[0], set()).update(
                        (i, j) for i in idx.get(qa, ()) for j in idx.get(qb, ()) if i != j)
    first_sets = [s for s in first.values() if s]
    out = np.zeros(rel.shape[2] - 1, dtype=bool)
    for k in range(rel.shape[2] - 1):
        ii, jj = np.nonzero(rel[:, :, k])
        pairs = {(int(i), int(j)) for i, j in zip(ii, jj) if i != j}
        out[k] = bool(pairs) and any(pairs == s for s in first_sets)
    return out


def wiki_hyperedges(rel: np.ndarray, first_order) -> list[tuple[int, ...]]:
    """Wiki hyperedges (paper App. B, [A854]).

    first-order channel k: one star hyperedge per source stock, {i} U {j : rel[i,j,k]=1}.
    second-order channel k ("pairwise in nature"): one 2-node hyperedge {i, j} per related pair.
    first_order: bool per wiki channel (length R-1), see `wiki_first_order_channels`. The last channel is the self-relation.
    """
    first_order = np.asarray(first_order, dtype=bool)
    if first_order.shape != (rel.shape[2] - 1,):
        raise ValueError(f"first_order must have length {rel.shape[2] - 1}, got {first_order.shape}")
    out = []
    for k in range(rel.shape[2] - 1):
        a = rel[:, :, k].copy()
        np.fill_diagonal(a, 0)
        if first_order[k]:
            for i in np.nonzero(a.sum(axis=1) > 0)[0]:
                out.append((int(i), *(int(j) for j in np.nonzero(a[i])[0])))
        else:
            out.extend((int(i), int(j)) for i, j in zip(*np.nonzero(a)))
    return out


def load_wiki_hyperedges(root, market: str) -> list[tuple[int, ...]]:
    root = Path(root)
    wdir = root / "relation" / "wikidata"
    rel = np.load(wdir / f"{market}_wiki_relation.npy")
    connections = json.loads((wdir / f"{market}_connections.json").read_text())
    with open(root / f"{market}_wiki.csv", newline="") as f:
        qids = [r[1] for r in csv.reader(f)]
    if len(qids) != rel.shape[0]:
        raise ValueError(f"{market}_wiki.csv has {len(qids)} rows but relation has {rel.shape[0]} nodes")
    return wiki_hyperedges(rel, wiki_first_order_channels(rel, connections, qids))


def build_rsr_hypergraph(root, market: str, sources=("industry", "wiki"), cache: bool = True) -> Hypergraph:
    root = Path(root)
    cache_file = root / "hypergraph_cache" / f"{market}_{'-'.join(sorted(sources))}.json"
    if cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        if d.get("version") == HYPERGRAPH_CACHE_VERSION:
            return Hypergraph(d["num_nodes"], tuple(tuple(e) for e in d["edges"]))
    edges: list[tuple[int, ...]] = []
    n = None
    if "industry" in sources:
        rel = np.load(root / "relation" / "sector_industry" / f"{market}_industry_relation.npy")
        n = rel.shape[0]
        edges += industry_hyperedges(rel)
    if "wiki" in sources:
        rel = np.load(root / "relation" / "wikidata" / f"{market}_wiki_relation.npy")
        n = rel.shape[0]
        edges += load_wiki_hyperedges(root, market)
    if n is None:
        raise ValueError("sources must include industry and/or wiki")
    hg = Hypergraph(n, canonical(edges))
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"version": HYPERGRAPH_CACHE_VERSION, "num_nodes": n,
                                          "edges": [list(e) for e in hg.edges]}))
    return hg


def decompose(hg: Hypergraph, mode: str, size: int) -> Hypergraph:
    """Replace selected hyperedges by all their pairwise edges (paper Fig 3a).

    large_first: split hyperedges with |e| > size (matches the Fig 3a axis 500 -> 3).
    small_first: split hyperedges with |e| <= size (matches the text "in increasing order").
    """
    if mode == "none":
        return hg
    out: list[tuple[int, ...]] = []
    for e in hg.edges:
        split = len(e) > size if mode == "large_first" else len(e) <= size
        if mode not in ("large_first", "small_first"):
            raise ValueError(mode)
        if split and len(e) > 2:
            out.extend(itertools.combinations(e, 2))
        else:
            out.append(e)
    return Hypergraph(hg.num_nodes, canonical(out))


def clique_expand(hg: Hypergraph) -> Hypergraph:
    max_size = int(hg.edge_sizes().max()) if hg.edges else 0
    return decompose(hg, "small_first", max_size)


def drop_hub_edges(hg: Hypergraph, min_degree: int) -> Hypergraph:
    """Remove every hyperedge touching a node whose degree >= min_degree (paper Fig 3b). 0 = keep all."""
    if min_degree <= 0:
        return hg
    hubs = set(np.nonzero(hg.node_degree() >= min_degree)[0].tolist())
    return Hypergraph(hg.num_nodes, tuple(e for e in hg.edges if hubs.isdisjoint(e)))


def hub_schedule(hg: Hypergraph, ranks=(1, 5, 20, 100)) -> list[int]:
    deg = np.sort(hg.node_degree())[::-1]
    th = [int(deg[min(r, len(deg)) - 1]) for r in ranks] + [2]
    out: list[int] = []
    for t in th:
        if t >= 2 and t not in out:
            out.append(t)
    return sorted(out, reverse=True)


def random_like(hg: Hypergraph, seed: int) -> Hypergraph:
    """Same hyperedge sizes; members drawn proportional to original node degree (grouping control)."""
    rng = np.random.default_rng(seed)
    deg = hg.node_degree().astype(float)
    p = deg / deg.sum()
    edges = tuple(
        tuple(sorted(int(v) for v in rng.choice(hg.num_nodes, size=len(e), replace=False, p=p)))
        for e in hg.edges
    )
    return Hypergraph(hg.num_nodes, edges)


def correlation_hyperedges(returns: np.ndarray, mask: np.ndarray, n_clusters: int) -> Hypergraph:
    """Average-linkage clusters on 1 - corr(train returns); each cluster (size >= 2) is a hyperedge."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    r = np.where(mask > 0, returns, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(r))
    d = np.clip(1.0 - c, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(z, n_clusters, criterion="maxclust")
    edges = [tuple(np.nonzero(labels == c)[0].tolist()) for c in np.unique(labels)]
    return Hypergraph(returns.shape[0], canonical(edges))


def induced_subgraph(hg: Hypergraph, keep: np.ndarray) -> Hypergraph:
    remap = {int(old): new for new, old in enumerate(np.asarray(keep).tolist())}
    edges = [tuple(remap[v] for v in e if v in remap) for e in hg.edges]
    return Hypergraph(len(remap), canonical(edges))
