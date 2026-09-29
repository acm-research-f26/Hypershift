"""Correctness controls C2 (future mutation), C4 (node permutation), C5 (batch isolation), C6 (isolated node).

Batching design: days live in a separate leading tensor dimension B of x [B, N, seq, C]; the hypergraph
indices (node_idx / edge_idx) address the node axis (dim -2) only and are shared by all days. No hyperedge
can connect nodes of different days, so C5 is an equivalence test rather than an index-range test.

Isolated-node fallback (models/attention.py): both attention layers end with
`torch.where(hg.has_edge[:, None], out, u)`, so a node in no hyperedge (or, in clique mode, no pair) keeps its
incoming representation u unchanged, then goes through TConv2 like every other node. A size-1 hyperedge is a
valid segment: its gyromidpoint/mean is the member itself and the softmax over one incidence gives weight 1
(if the node has several edges the softmax normalises across them).
"""
import dataclasses
import itertools

import numpy as np
import pytest
import torch

from hypershift.data.hypergraph import Hypergraph, clique_expand
from hypershift.geometry.poincare import expmap0
from hypershift.models.think import THINK
from hypershift.train.loop import apply_input_mode, gather_batch, window_offsets
from tests.conftest import make_synthetic_hypergraph, make_synthetic_market

SEQ, KERNEL = 8, 4
VARIANTS = list(itertools.product(["hyp", "euc"], ["hyp", "euc"], ["hyper", "clique", "none"]))
IDS = [f"{t}-{s}-{st}" for t, s, st in VARIANTS]
GRAPH_VARIANTS = [v for v in VARIANTS if v[2] != "none"]
GRAPH_IDS = [f"{t}-{s}-{st}" for t, s, st in GRAPH_VARIANTS]


def _model(temporal, spatial, structure, seed=0):
    torch.manual_seed(seed)
    return THINK(in_dim=5, hidden=8, seq=SEQ, kernel=KERNEL, temporal=temporal, spatial=spatial,
                 structure=structure).eval()


def _graph(structure, hg):
    return clique_expand(hg) if structure == "clique" else hg


def _predict(model, data, offsets, hg, mode):
    x, _, _, _ = gather_batch(data, np.asarray(offsets), SEQ)
    x = apply_input_mode(x, mode)
    with torch.no_grad():
        return model(torch.as_tensor(x), hg.to_torch("cpu")).numpy()


def _mutate_from(data, day, rng):
    """Replace every array at days >= `day` with garbage (features, mask, gt, base_price)."""
    f, m, g, b = data.features.copy(), data.mask.copy(), data.gt.copy(), data.base_price.copy()
    f[:, day:] = rng.uniform(0.05, 3.0, f[:, day:].shape)
    m[:, day:] = rng.integers(0, 2, m[:, day:].shape)
    g[:, day:] = rng.normal(0, 1, g[:, day:].shape)
    b[:, day:] = rng.uniform(0.05, 3.0, b[:, day:].shape)
    return dataclasses.replace(data, features=f, mask=m, gt=g, base_price=b)


@pytest.mark.parametrize("mode", ["level", "relative"])
@pytest.mark.parametrize("temporal,spatial,structure", VARIANTS, ids=IDS)
def test_c2_future_mutation_does_not_change_prediction(temporal, spatial, structure, mode):
    data, hg = make_synthetic_market(), _graph(structure, make_synthetic_hypergraph())
    model = _model(temporal, spatial, structure)
    offs = window_offsets(data, SEQ, "val")[:3]
    rng = np.random.default_rng(1)
    for o in offs:
        base = _predict(model, data, [o], hg, mode)
        # inputs are days o..o+SEQ-1; day o+SEQ is the target day, so everything from there on is "future"
        fut = _mutate_from(data, o + SEQ, rng)
        np.testing.assert_allclose(_predict(model, fut, [o], hg, mode), base, atol=1e-6, rtol=0)
        x0, _, b0, _ = gather_batch(data, [o], SEQ)
        x1, _, b1, _ = gather_batch(fut, [o], SEQ)
        np.testing.assert_array_equal(x0, x1)
        np.testing.assert_array_equal(b0, b1)
        # sanity (non-vacuous): mutating the last day INSIDE the window changes the prediction
        f = data.features.copy()
        f[:, o + SEQ - 1] *= 1.7
        inside = dataclasses.replace(data, features=f)
        assert not np.allclose(_predict(model, inside, [o], hg, mode), base, atol=1e-4)


@pytest.mark.parametrize("mode", ["level", "relative"])
@pytest.mark.parametrize("temporal,spatial,structure", VARIANTS, ids=IDS)
def test_c4_node_permutation_equivariance(temporal, spatial, structure, mode):
    data = make_synthetic_market()
    hg = _graph(structure, make_synthetic_hypergraph())
    N = data.num_nodes
    p = np.random.default_rng(3).permutation(N)          # new node k = old node p[k]
    inv = np.argsort(p)                                  # old node v -> new index inv[v]
    pdata = dataclasses.replace(data, features=data.features[p], mask=data.mask[p], gt=data.gt[p],
                                base_price=data.base_price[p])
    phg = Hypergraph(N, tuple(tuple(int(inv[v]) for v in e) for e in hg.edges))
    model = _model(temporal, spatial, structure)
    offs = window_offsets(data, SEQ, "val")[:4]
    ref = _predict(model, data, offs, hg, mode)
    got = _predict(model, pdata, offs, phg, mode)
    np.testing.assert_allclose(got, ref[:, p], atol=1e-5, rtol=1e-4)
    # reversing hyperedge order (scatter ordering) must not matter either
    ehg = Hypergraph(N, tuple(reversed(hg.edges)))
    np.testing.assert_allclose(_predict(model, data, offs, ehg, mode), ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["level", "relative"])
@pytest.mark.parametrize("temporal,spatial,structure", VARIANTS, ids=IDS)
def test_c5_batch_isolation(temporal, spatial, structure, mode):
    data, hg = make_synthetic_market(), _graph(structure, make_synthetic_hypergraph())
    tg = hg.to_torch("cpu")
    # no cross-date connectivity: incidence indexes the node axis only; days are the leading tensor dim
    assert int(tg.node_idx.max()) < data.num_nodes and tg.num_nodes == data.num_nodes
    model = _model(temporal, spatial, structure)
    offs = window_offsets(data, SEQ, "val")[[0, 1, 10, 11, 12]]   # first two overlap; the rest do not overlap offs[0]
    full = _predict(model, data, offs, hg, mode)
    assert full.shape == (5, data.num_nodes)
    single = np.concatenate([_predict(model, data, [o], hg, mode) for o in offs])
    np.testing.assert_allclose(full, single, atol=1e-5, rtol=1e-4)
    rev = _predict(model, data, offs[::-1], hg, mode)[::-1]
    np.testing.assert_allclose(full, rev, atol=1e-5, rtol=1e-4)
    split = np.concatenate([_predict(model, data, offs[:2], hg, mode), _predict(model, data, offs[2:], hg, mode)])
    np.testing.assert_allclose(full, split, atol=1e-5, rtol=1e-4)
    # garbage in one day's window leaves the predictions of non-overlapping days untouched
    f = data.features.copy()
    f[:, offs[0]:offs[0] + SEQ] = np.random.default_rng(0).uniform(0.1, 2, f[:, offs[0]:offs[0] + SEQ].shape)
    mutated = dataclasses.replace(data, features=f)
    far = [o for o in offs if o >= offs[0] + SEQ]
    assert far, "test needs non-overlapping windows"
    np.testing.assert_allclose(_predict(model, mutated, far, hg, mode), _predict(model, data, far, hg, mode),
                               atol=1e-6)


@pytest.mark.parametrize("temporal,spatial,structure", VARIANTS, ids=IDS)
def test_c6_isolated_node_finite_and_independent(temporal, spatial, structure):
    data = make_synthetic_market()
    hg = _graph(structure, make_synthetic_hypergraph())
    iso = 11
    assert hg.node_degree()[iso] == 0                     # isolated also after clique expansion
    model = _model(temporal, spatial, structure)
    offs = window_offsets(data, SEQ, "val")[:3]
    base = _predict(model, data, offs, hg, "relative")
    assert np.isfinite(base).all()
    # the isolated node depends only on itself: garbage on all other stocks must not move it
    f = data.features.copy()
    f[:iso] = np.random.default_rng(5).uniform(0.1, 2, f[:iso].shape)
    other = dataclasses.replace(data, features=f)
    np.testing.assert_allclose(_predict(model, other, offs, hg, "relative")[:, iso], base[:, iso], atol=1e-6)
    # fallback = identity: same value as with an empty graph
    empty = Hypergraph(data.num_nodes, ())
    np.testing.assert_allclose(_predict(model, data, offs, empty, "relative")[:, iso], base[:, iso], atol=1e-6)


@pytest.mark.parametrize("spatial", ["hyp", "euc"])
def test_c6_attention_layer_returns_input_for_uncovered_nodes(spatial):
    from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention
    torch.manual_seed(0)
    layer = (HypHypergraphAttention if spatial == "hyp" else EucHypergraphAttention)(8).eval()
    u = torch.rand(2, 5, 8) * 0.3
    if spatial == "hyp":
        u = expmap0(u)
    tg = Hypergraph(5, ((0, 1, 2),)).to_torch("cpu")       # nodes 3, 4 uncovered
    with torch.no_grad():
        out = layer(u, tg)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 3:], u[:, 3:])


@pytest.mark.parametrize("temporal,spatial,structure", GRAPH_VARIANTS, ids=GRAPH_IDS)
def test_c6_single_member_hyperedge(temporal, spatial, structure):
    N = 12
    data = make_synthetic_market(N=N)
    # node 0 only in a singleton edge, node 5 in a singleton plus a real edge; node 11 uncovered
    hg = Hypergraph(N, ((0,), (5,), (1, 2, 3), (4, 5, 6), (7, 8)))
    assert (hg.edge_sizes() == 1).sum() == 2
    model = _model(temporal, spatial, structure)
    offs = window_offsets(data, SEQ, "val")[:3]
    x, _, _, _ = gather_batch(data, offs, SEQ)
    xt = torch.as_tensor(apply_input_mode(x, "relative"))
    with torch.no_grad():
        out = model(xt, hg.to_torch("cpu"))
    assert torch.isfinite(out).all()
    model.train()
    model(xt, hg.to_torch("cpu")).pow(2).mean().backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
