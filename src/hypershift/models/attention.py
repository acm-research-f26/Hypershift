"""Distance-aware hypergraph attention (DHHAN, paper eq. 13-15) and its Euclidean mirror."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hypershift.data.hypergraph import TorchHypergraph
from hypershift.geometry.poincare import expmap0, gyromidpoint, logmap0, mobius_add, poincare_dist
from hypershift.models.layers import PoincareLinear


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    idx = index.expand_as(scores)
    shape = (*scores.shape[:-1], num_segments)
    mx = scores.new_full(shape, float("-inf")).scatter_reduce(-1, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx.detach().gather(-1, idx))
    den = scores.new_zeros(shape).index_add(-1, index, ex)
    return ex / den.gather(-1, idx).clamp_min(1e-16)


class _Base(nn.Module):
    def __init__(self, dim: int, score: str, dist: str):
        super().__init__()
        if score not in ("mobius", "concat") or dist not in ("mult", "neg", "off"):
            raise ValueError((score, dist))
        self.score, self.dist = score, dist
        self.a = nn.Parameter(torch.randn(dim if score == "mobius" else 2 * dim) * dim ** -0.5)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _combine(self, base: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.dist == "mult":
            return base * d
        if self.dist == "neg":
            return F.leaky_relu(base, 0.2) - F.softplus(self.gamma) * d
        return F.leaky_relu(base, 0.2)


class HypHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "mobius", dist: str = "mult"):
        super().__init__(dim, score, dist)
        self.fc = PoincareLinear(dim, dim)

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        z = gyromidpoint(u, hg.node_idx, hg.edge_idx, hg.num_edges)          # eq 13
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        if self.score == "mobius":
            base = mobius_add(uj, zi) @ self.a
        else:
            base = torch.cat([logmap0(uj), logmap0(zi)], dim=-1) @ self.a
        alpha = segment_softmax(self._combine(base, poincare_dist(uj, zi)), hg.node_idx, hg.num_nodes)  # eq 14
        msg = logmap0(self.fc(z)).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        agg = u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg)
        out = expmap0(F.relu(agg))                                             # eq 15
        return torch.where(hg.has_edge[:, None], out, u)


class EucHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "mobius", dist: str = "mult"):
        super().__init__(dim, score, dist)
        self.fc = nn.Linear(dim, dim)

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        lead = u.shape[:-2]
        z = u.new_zeros(*lead, hg.num_edges, u.shape[-1]).index_add(-2, hg.edge_idx, u.index_select(-2, hg.node_idx))
        z = z / hg.edge_size[:, None]
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        base = (uj + zi) @ self.a if self.score == "mobius" else torch.cat([uj, zi], dim=-1) @ self.a
        alpha = segment_softmax(self._combine(base, (uj - zi).norm(dim=-1)), hg.node_idx, hg.num_nodes)
        msg = self.fc(z).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        out = F.relu(u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg))
        return torch.where(hg.has_edge[:, None], out, u)
