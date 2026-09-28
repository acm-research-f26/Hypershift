from pathlib import Path
import numpy as np
import pytest
from hypershift.data.hypergraph import (
    Hypergraph, canonical, clique_expand, correlation_hyperedges, decompose, drop_hub_edges,
    hub_schedule, induced_subgraph, industry_hyperedges, random_like, wiki_hyperedges,
    build_rsr_hypergraph,
)
from hypershift.data.universe import select_universe

REAL = Path("data/raw/rsr/data")


def _industry_rel():
    N = 5
    rel = np.zeros((N, N, 3), int)
    for k, mem in enumerate([[0, 1, 2], [3, 4]]):
        for i in mem:
            for j in mem:
                rel[i, j, k] = 1
    for i in range(N):
        rel[i, i, -1] = 1
    return rel


def test_industry_hyperedges():
    assert sorted(industry_hyperedges(_industry_rel())) == [(0, 1, 2), (3, 4)]


def test_wiki_star_hyperedges_ignore_self_channel():
    N = 5
    rel = np.zeros((N, N, 3), int)
    rel[0, 1, 0] = rel[0, 2, 0] = 1
    rel[3, 4, 1] = 1
    rel[2, 2, 0] = 1                  # diagonal inside a real channel must be ignored
    for i in range(N):
        rel[i, i, -1] = 1
    assert sorted(canonical(wiki_hyperedges(rel))) == [(0, 1, 2), (3, 4)]


def test_canonical_dedup_and_min_size():
    assert canonical([(2, 1), (1, 2), (3,), (4, 5, 6)]) == ((1, 2), (4, 5, 6))


def test_incidence_and_degree():
    hg = Hypergraph(4, ((0, 1, 2), (2, 3)))
    node, edge = hg.incidence()
    assert node.tolist() == [0, 1, 2, 2, 3] and edge.tolist() == [0, 0, 0, 1, 1]
    assert hg.node_degree().tolist() == [1, 1, 2, 1]
    assert hg.edge_sizes().tolist() == [3, 2]


def test_decompose_modes():
    hg = Hypergraph(6, ((0, 1, 2), (3, 4), (0, 1, 2, 5)))
    lf = decompose(hg, "large_first", 3)          # only size > 3 split
    assert (0, 1, 2) in lf.edges and (0, 5) in lf.edges and (0, 1, 2, 5) not in lf.edges
    sf = decompose(hg, "small_first", 3)          # size <= 3 split
    assert (0, 1) in sf.edges and (0, 1, 2) not in sf.edges and (0, 1, 2, 5) in sf.edges
    assert decompose(hg, "none", 0) == hg
    ce = clique_expand(hg)
    assert all(len(e) == 2 for e in ce.edges)
    assert len(ce.edges) == len({(0, 1), (0, 2), (1, 2), (3, 4), (0, 5), (1, 5), (2, 5)})


def test_drop_hub_edges_and_schedule():
    hg = Hypergraph(5, ((0, 1), (0, 2), (0, 3), (3, 4)))
    assert hg.node_degree()[0] == 3
    out = drop_hub_edges(hg, 3)
    assert out.edges == ((3, 4),)
    assert drop_hub_edges(hg, 0) == hg
    sched = hub_schedule(hg, ranks=(1, 2))
    assert sched[0] == 3 and sched[-1] == 2 and sched == sorted(sched, reverse=True)


def test_random_like_preserves_sizes():
    hg = Hypergraph(20, ((0, 1, 2), (3, 4), (5, 6, 7, 8)))
    r = random_like(hg, seed=0)
    assert sorted(r.edge_sizes().tolist()) == [2, 3, 4]
    assert r != hg


def test_correlation_hyperedges_groups_correlated():
    rng = np.random.default_rng(0)
    f1, f2 = rng.normal(size=200), rng.normal(size=200)
    ret = np.stack([f1 + 0.01 * rng.normal(size=200) for _ in range(3)]
                   + [f2 + 0.01 * rng.normal(size=200) for _ in range(3)])
    hg = correlation_hyperedges(ret, np.ones_like(ret), n_clusters=2)
    assert sorted(hg.edges) == [(0, 1, 2), (3, 4, 5)]


def test_induced_subgraph_and_universe(synthetic_market, synthetic_hypergraph):
    sub = induced_subgraph(Hypergraph(5, ((0, 1, 2), (2, 3, 4))), np.array([1, 2, 4]))
    assert sub.num_nodes == 3 and sub.edges == ((0, 1), (1, 2))
    d, h = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    assert d.num_nodes == 6 and h.num_nodes == 6
    d2, _ = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    assert d.tickers == d2.tickers  # deterministic


def test_build_rsr_hypergraph_cache(tmp_path):
    (tmp_path / "relation" / "sector_industry").mkdir(parents=True)
    (tmp_path / "relation" / "wikidata").mkdir(parents=True)
    np.save(tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy", _industry_rel())
    wiki = np.zeros((5, 5, 2), int)
    wiki[0, 3, 0] = 1
    np.save(tmp_path / "relation" / "wikidata" / "NYSE_wiki_relation.npy", wiki)
    first = build_rsr_hypergraph(tmp_path, "NYSE")
    assert first.edges == ((0, 1, 2), (3, 4), (0, 3))
    assert list((tmp_path / "hypergraph_cache").glob("*.json"))
    (tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy").unlink()
    assert build_rsr_hypergraph(tmp_path, "NYSE") == first          # served from the cache
    assert build_rsr_hypergraph(tmp_path, "NYSE", ("wiki",), cache=False).edges == ((0, 3),)


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_nyse_hypergraph_stats():
    hg = build_rsr_hypergraph(REAL, "NYSE")
    assert hg.num_nodes == 1737
    assert 250 <= len(hg.edges) <= 400             # prototype: 312
    assert hg.edge_sizes().max() == 500            # matches paper Fig 3a axis
    assert 20 <= hg.node_degree().max() <= 60      # prototype: 37
