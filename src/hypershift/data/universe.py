import numpy as np

from hypershift.data.hypergraph import Hypergraph, induced_subgraph
from hypershift.data.rsr import MarketData


def select_universe(data: MarketData, hg: Hypergraph, size: int, seed: int) -> tuple[MarketData, Hypergraph]:
    """Uniform random subset of `size` stocks (sorted indices) + induced sub-hypergraph. size<=0 = all."""
    if size <= 0 or size >= data.num_nodes:
        return data, hg
    idx = np.sort(np.random.default_rng(seed).choice(data.num_nodes, size=size, replace=False))
    return data.subset(idx), induced_subgraph(hg, idx)
