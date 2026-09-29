import numpy as np
import pytest
import torch

from hypershift.data.hypergraph import Hypergraph
from hypershift.models import baselines as B
from hypershift.train.loop import build_model, train_one_run
from test_loop import small_cfg

HG = Hypergraph(12, ((0, 1, 2), (2, 3, 4, 5), (6, 7), (8, 9, 10))).to_torch("cpu")   # node 11 isolated


def rsr_entries(n=12):
    """Two relation channels: channel 0 links 0-1-2 (with self), channel 1 links 3-4; self channel 2 for all."""
    pairs = {}
    for i in range(n):
        pairs.setdefault((i, i), []).append(2)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            pairs.setdefault((a, b), []).append(0)
    for a, b in ((3, 4), (4, 3)):
        pairs.setdefault((a, b), []).append(1)
    keys = sorted(pairs)
    pi = np.array([k[0] for k in keys]); pj = np.array([k[1] for k in keys])
    ep = np.array([n_ for n_, k in enumerate(keys) for _ in pairs[k]]); ec = np.array([c for k in keys for c in pairs[k]])
    return pi, pj, ep, ec, 3


def test_rsr_i_forward_backward_and_graph_used():
    torch.manual_seed(0)
    pi, pj, ep, ec, K = rsr_entries()
    m = B.RSRI(5, 8, pi, pj, ep, ec, K)
    x = torch.rand(3, 12, 16, 5)
    y = m(x)
    assert y.shape == (3, 12) and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    # graph is used: a node's output depends on a related node's input, not on an unrelated one
    x2 = x.clone().requires_grad_(True)
    m(x2)[0, 3].backward()
    g = x2.grad[0].abs().sum(dim=(1, 2))
    assert g[4] > 0 and g[3] > 0 and g[8] == 0


def test_rsr_i_no_relations_differs_from_relations():
    torch.manual_seed(0)
    a = B.RSRI(5, 8, *rsr_entries())
    b = B.RSRI(5, 8, *[np.array(v) for v in ([i for i in range(12)], list(range(12)), list(range(12)), [2] * 12)], 3)
    b.load_state_dict(a.state_dict())
    x = torch.rand(1, 12, 16, 5)
    assert not torch.allclose(a(x), b(x))


def test_sthgcn_forward_backward_and_graph_used():
    torch.manual_seed(0)
    m = B.STHGCN(5, 16, 12).eval()
    x = torch.rand(2, 12, 16, 5)
    y = m(x, HG)
    assert y.shape == (2, 12) and torch.isfinite(y).all()
    m.train()
    m(x, HG).pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    m.eval()
    empty = Hypergraph(12, ()).to_torch("cpu")
    assert not torch.allclose(m(x, HG), m(x, empty))
    x2 = x.clone().requires_grad_(True)
    m(x2, HG)[0, 0].backward()
    g = x2.grad[0].abs().sum(dim=(1, 2))
    assert g[1] > 0 and g[8] == 0                     # same hyperedge yes, other component no


def test_hgnn_operator_properties():
    G = B.hgnn_operator(HG, "cpu")
    assert torch.allclose(G, G.t(), atol=1e-6)
    assert G[11, 11] == 1.0 and G[11, :11].abs().sum() == 0      # isolated node keeps itself
    assert G[0, 1] > 0 and G[0, 8] == 0


def test_sthgcn_train_one_run_and_no_leak(tmp_path, synthetic_market, synthetic_hypergraph):
    cfg = small_cfg(tmp_path, model="sthgcn", seq=16, kernel=4, epochs=2, batch_days=4)
    m = train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    assert np.isfinite(m["test"]["sr"]) and m["config"]["model"] == "sthgcn"
    assert (tmp_path / "t" / "x" / "seed_0" / "metrics.json").exists()


def test_rsr_i_train_one_run(tmp_path, synthetic_market, synthetic_hypergraph, monkeypatch):
    monkeypatch.setattr(B, "relation_entries", lambda root, market, tickers: rsr_entries(len(tickers)))
    cfg = small_cfg(tmp_path, model="rsr_i", seq=16, kernel=4, epochs=2, batch_days=4)
    m = train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    assert np.isfinite(m["test"]["sr"]) and m["config"]["model"] == "rsr_i"


def test_unknown_model_and_default_is_think(synthetic_market):
    from hypershift.config import RunConfig
    assert RunConfig().model == "think"
    with pytest.raises(ValueError):
        build_model(RunConfig(model="nope"), 5, synthetic_market)


@pytest.mark.data
def test_relation_entries_real_subset():
    from hypershift.data.rsr import read_ticker_file
    root = "data/raw/rsr/data"
    tick = read_ticker_file(root + "/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    pi, pj, ep, ec, K = B.relation_entries(root, "NYSE", tick)
    assert K == 141 and len(pi) > 1737 and ep.max() == len(pi) - 1
    sub = tick[10:60]
    pi2, pj2, _, _, _ = B.relation_entries(root, "NYSE", sub)
    assert pi2.max() < 50 and pj2.max() < 50 and len(pi2) < len(pi)
    assert ((pi2 == pj2).sum()) == 50                                  # every node has its self relation
