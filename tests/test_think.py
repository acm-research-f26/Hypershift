import itertools
import pytest
import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.models.think import THINK

HG = Hypergraph(7, ((0, 1, 2), (2, 3), (4, 5))).to_torch("cpu")  # node 6 isolated


@pytest.mark.parametrize("temporal,spatial,structure",
                         list(itertools.product(["hyp", "euc"], ["hyp", "euc"], ["hyper", "none"])))
def test_forward_backward_all_variants(temporal, spatial, structure):
    torch.manual_seed(0)
    m = THINK(in_dim=5, hidden=8, seq=16, kernel=4, temporal=temporal, spatial=spatial, structure=structure)
    x = torch.rand(3, 7, 16, 5)
    y = m(x, HG)
    assert y.shape == (3, 7) and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    for name, p in m.named_parameters():
        if name.endswith("gamma"):          # only used when attn_dist == "neg"
            continue
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_structure_none_ignores_graph():
    torch.manual_seed(0)
    m = THINK(hidden=8, structure="none")
    x = torch.rand(1, 7, 16, 5)
    empty = Hypergraph(7, ()).to_torch("cpu")
    torch.testing.assert_close(m(x, HG), m(x, empty))


def test_graph_changes_output():
    torch.manual_seed(0)
    m = THINK(hidden=8)
    x = torch.rand(1, 7, 16, 5)
    empty = Hypergraph(7, ()).to_torch("cpu")
    assert not torch.allclose(m(x, HG), m(x, empty))


def test_multiclass_head():
    m = THINK(hidden=8, out_dim=3)
    assert m(torch.rand(2, 7, 16, 5), HG).shape == (2, 7, 3)
