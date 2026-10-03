import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.models.think import THINK

EDGES = ((0, 1, 2), (2, 3), (4, 5))


def test_think_is_node_permutation_equivariant_random_init():
    torch.manual_seed(0)
    for temporal in ("hyp", "euc"):
        for spatial in ("hyp", "euc"):
            m = THINK(in_dim=5, hidden=8, seq=16, kernel=4, temporal=temporal, spatial=spatial).eval()
            x = torch.rand(2, 7, 16, 5) * 0.1
            perm = torch.tensor([3, 6, 0, 5, 1, 4, 2])          # new position p holds old node perm[p]
            inv = torch.argsort(perm)
            hg = Hypergraph(7, EDGES).to_torch("cpu")
            hg_p = Hypergraph(7, tuple(tuple(int(inv[v]) for v in e) for e in EDGES)).to_torch("cpu")
            with torch.no_grad():
                y = m(x, hg)
                y_p = m(x[:, perm], hg_p)
            torch.testing.assert_close(y_p, y[:, perm], rtol=1e-5, atol=1e-7)
