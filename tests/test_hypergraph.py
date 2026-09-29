import json
from pathlib import Path
import numpy as np
import pytest
from hypershift.data.hypergraph import (
    Hypergraph, canonical, clique_expand, correlation_hyperedges, decompose, drop_hub_edges,
    hub_schedule, induced_subgraph, industry_hyperedges, random_like, wiki_hyperedges, wiki_first_order_channels,
    HYPERGRAPH_CACHE_VERSION,
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


def test_wiki_first_order_star_second_order_pairs_ignore_self_channel():
    N = 5
    rel = np.zeros((N, N, 4), int)
    rel[0, 1, 0] = rel[0, 2, 0] = 1            # channel 0: first-order -> star {0,1,2}
    rel[3, 4, 1] = 1                           # channel 1: first-order, one target -> {3,4}
    rel[2, 2, 0] = 1                           # diagonal inside a real channel must be ignored
    rel[0, 3, 2] = rel[3, 0, 2] = rel[0, 4, 2] = rel[4, 0, 2] = 1   # channel 2: second-order -> pairs, not a star
    for i in range(N):
        rel[i, i, -1] = 1
    fo = np.array([True, True, False])
    assert sorted(canonical(wiki_hyperedges(rel, fo))) == [(0, 1, 2), (0, 3), (0, 4), (3, 4)]
    star_all = sorted(canonical(wiki_hyperedges(rel, np.array([True, True, True]))))
    assert (0, 3, 4) in star_all and (0, 3, 4) not in canonical(wiki_hyperedges(rel, fo))       # what the old (buggy) builder produced
    with pytest.raises(ValueError):
        wiki_hyperedges(rel, [True])


def test_wiki_first_order_channels_detected_from_connections():
    N = 4
    rel = np.zeros((N, N, 4), int)
    rel[0, 1, 0] = rel[0, 2, 0] = 1            # channel 0 == pairs of single-property path P127
    rel[1, 2, 1] = rel[2, 1, 1] = rel[1, 3, 1] = rel[3, 1, 1] = 1   # channel 1 == two-property path (P31, P31)
    for i in range(N):
        rel[i, i, -1] = 1
    q = ["Q0", "Q1", "Q2", "Q3"]
    con = {"Q0": {"Q1": [["P127"], ["P414", "P414"]], "Q2": [["P127"]]},
           "Q1": {"Q2": [["P31", "P31"]], "Q3": [["P31", "P31"]]},
           "Q2": {"Q1": [["P31", "P31"]]}, "Q3": {"Q1": [["P31", "P31"]]}}
    assert wiki_first_order_channels(rel, con, q).tolist() == [True, False, False]


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


def _write_tiny_rsr(tmp_path):
    (tmp_path / "relation" / "sector_industry").mkdir(parents=True)
    (tmp_path / "relation" / "wikidata").mkdir(parents=True)
    np.save(tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy", _industry_rel())
    wiki = np.zeros((5, 5, 2), int)
    wiki[0, 3, 0] = 1
    np.save(tmp_path / "relation" / "wikidata" / "NYSE_wiki_relation.npy", wiki)
    (tmp_path / "relation" / "wikidata" / "NYSE_connections.json").write_text(
        json.dumps({"Q0": {"Q3": [["P31", "P31"]]}}))                       # second-order pair
    (tmp_path / "NYSE_wiki.csv").write_text("\n".join(f"T{i},Q{i}" for i in range(5)))


def test_build_rsr_hypergraph_cache(tmp_path):
    _write_tiny_rsr(tmp_path)
    first = build_rsr_hypergraph(tmp_path, "NYSE")
    assert first.edges == ((0, 1, 2), (3, 4), (0, 3))
    assert list((tmp_path / "hypergraph_cache").glob("*.json"))
    (tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy").unlink()
    assert build_rsr_hypergraph(tmp_path, "NYSE") == first          # served from the cache
    assert build_rsr_hypergraph(tmp_path, "NYSE", ("wiki",), cache=False).edges == ((0, 3),)


def test_stale_cache_version_is_rebuilt(tmp_path):
    _write_tiny_rsr(tmp_path)
    cache = tmp_path / "hypergraph_cache"
    cache.mkdir()
    f = cache / "NYSE_industry-wiki.json"
    f.write_text(json.dumps({"num_nodes": 5, "edges": [[0, 1, 2, 3, 4]]}))          # v1 file: no version key
    assert build_rsr_hypergraph(tmp_path, "NYSE").edges == ((0, 1, 2), (3, 4), (0, 3))
    assert json.loads(f.read_text())["version"] == HYPERGRAPH_CACHE_VERSION


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_nyse_hypergraph_stats():
    hg = build_rsr_hypergraph(REAL, "NYSE", cache=False)
    assert hg.num_nodes == 1737
    # first-order wiki -> stars, second-order wiki -> pairs (paper App. B). Measured: 4350 edges (was 312 with all-star).
    assert len(hg.edges) == 4350
    sizes = hg.edge_sizes()
    assert sizes.max() == 500                      # matches paper Fig 3a axis
    assert int((sizes == 2).sum()) == 4250
    assert hg.node_degree().max() == 114           # was 37 with all-star
    assert int((hg.node_degree() == 0).sum()) == 17
    rel = np.load(REAL / "relation" / "wikidata" / "NYSE_wiki_relation.npy")
    import csv
    con = json.loads((REAL / "relation" / "wikidata" / "NYSE_connections.json").read_text())
    qids = [r[1] for r in csv.reader(open(REAL / "NYSE_wiki.csv"))]
    assert np.nonzero(wiki_first_order_channels(rel, con, qids))[0].tolist() == [0, 1, 23]


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_nasdaq_hypergraph_stats():
    hg = build_rsr_hypergraph(REAL, "NASDAQ", cache=False)
    assert hg.num_nodes == 1026 and len(hg.edges) == 1066 and hg.edge_sizes().max() == 156
    assert hg.node_degree().max() == 55
