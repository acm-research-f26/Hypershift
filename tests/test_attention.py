import pytest
import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.geometry.poincare import expmap0
from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention, segment_softmax

torch.manual_seed(0)
HG = Hypergraph(6, ((0, 1, 2), (2, 3), (1, 2, 3, 4)))   # node 5 isolated


def test_segment_softmax_sums_to_one():
    s = torch.randn(2, 7)
    idx = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    a = segment_softmax(s, idx, 4)
    sums = torch.zeros(2, 4).index_add(-1, idx, a)
    torch.testing.assert_close(sums[:, [0, 1, 3]], torch.ones(2, 3))


@pytest.mark.parametrize("cls", [HypHypergraphAttention, EucHypergraphAttention])
@pytest.mark.parametrize("score", ["mobius", "concat"])
@pytest.mark.parametrize("dist", ["mult", "neg", "off"])
def test_attention_shapes_and_finite(cls, score, dist):
    layer = cls(4, score=score, dist=dist)
    u = torch.randn(2, 3, 6, 4) * 0.5
    if cls is HypHypergraphAttention:
        u = expmap0(u)
    out = layer(u, HG.to_torch("cpu"))
    assert out.shape == u.shape and torch.isfinite(out).all()
    out.sum().backward()


@pytest.mark.parametrize("cls", [HypHypergraphAttention, EucHypergraphAttention])
def test_isolated_node_passthrough(cls):
    layer = cls(4)
    u = expmap0(torch.randn(1, 1, 6, 4) * 0.5)
    out = layer(u, HG.to_torch("cpu"))
    torch.testing.assert_close(out[..., 5, :], u[..., 5, :])


def test_hyp_attention_permutation_equivariant():
    layer = HypHypergraphAttention(4)
    u = expmap0(torch.randn(1, 2, 6, 4) * 0.5)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])              # new position i holds old node perm[i]
    inv = torch.argsort(perm)
    hg_p = Hypergraph(6, tuple(tuple(int(inv[v]) for v in e) for e in HG.edges))
    out = layer(u, HG.to_torch("cpu"))
    out_p = layer(u[..., perm, :], hg_p.to_torch("cpu"))
    torch.testing.assert_close(out_p, out[..., perm, :], atol=1e-5, rtol=1e-4)
