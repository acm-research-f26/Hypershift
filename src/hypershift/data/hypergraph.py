"""Stock hypergraphs: construction from RSR relations (paper appendix B) and ablation transforms."""
from __future__ import annotations

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


def wiki_hyperedges(rel: np.ndarray) -> list[tuple[int, ...]]:
    """Star hyperedge per (source stock, wiki relation channel): {i} U {j : rel[i,j,k]=1}."""
    out = []
    for k in range(rel.shape[2] - 1):
        a = rel[:, :, k].copy()
        np.fill_diagonal(a, 0)
        for i in np.nonzero(a.sum(axis=1) > 0)[0]:
            out.append((int(i), *(int(j) for j in np.nonzero(a[i])[0])))
    return out


def build_rsr_hypergraph(root, market: str, sources=("industry", "wiki"), cache: bool = True) -> Hypergraph:
    root = Path(root)
    cache_file = root / "hypergraph_cache" / f"{market}_{'-'.join(sorted(sources))}.json"
    if cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
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
        edges += wiki_hyperedges(rel)
    if n is None:
        raise ValueError("sources must include industry and/or wiki")
    hg = Hypergraph(n, canonical(edges))
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"num_nodes": n, "edges": [list(e) for e in hg.edges]}))
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
