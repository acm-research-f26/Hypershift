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


def segment_sum_norm(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """alpha = s / sum_segment |s| (|.| keeps the denominator positive; flagged variant, not the paper)."""
    shape = (*scores.shape[:-1], num_segments)
    den = scores.new_zeros(shape).index_add(-1, index, scores.abs())
    return scores / den.gather(-1, index.expand_as(scores)).clamp_min(1e-8)


def odot_mobius(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Eq. 7 read literally, tan/arctan -> tanh/artanh:  x (.) y = tanh((||xy||/||y||) artanh(||y||)) * ||xy||/||y||.

    Both operands are scalars here (x = s = a^T (x) (u (+) z) in (-1,1), y = d_B >= 0).  ||xy|| is read as |x*y|,
    so ||xy||/||y|| = |s|.  artanh needs ||y|| < 1, so y is taken as exp_0(d) = tanh(d) (INFERRED, keeps the domain
    valid and gives artanh(||y||) = d).  Sign of s is restored.  Result: sign(s) |s| tanh(|s| d).
    """
    a = s.abs()
    return torch.sign(s) * a * torch.tanh(a * d)


class _Base(nn.Module):
    def __init__(self, dim: int, score: str, dist: str, odot: str = "product", norm: str = "softmax",
                 residual: bool = False):
        super().__init__()
        self.residual = residual
        if score not in ("mobius", "concat", "eq14") or dist not in ("mult", "neg", "off"):
            raise ValueError((score, dist))
        if odot not in ("product", "mobius") or norm not in ("softmax", "none", "sum"):
            raise ValueError((odot, norm))
        self.score, self.dist, self.odot, self.norm = score, dist, odot, norm
        self.a = nn.Parameter(torch.randn(dim if score in ("mobius", "eq14") else 2 * dim) * dim ** -0.5)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _combine(self, base: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.dist == "mult":
            return odot_mobius(base, d) if self.odot == "mobius" else base * d
        if self.dist == "neg":
            return F.leaky_relu(base, 0.2) - F.softplus(self.gamma) * d
        return F.leaky_relu(base, 0.2)


    def _alpha(self, scores: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        if self.norm == "softmax":
            return segment_softmax(scores, hg.node_idx, hg.num_nodes)
        if self.norm == "sum":
            return segment_sum_norm(scores, hg.node_idx, hg.num_nodes)
        return scores                                   # "none": raw eq. 14 value as the coefficient


class HypHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "eq14", dist: str = "mult", odot: str = "product", norm: str = "softmax",
                 residual: bool = False, init_gain: float = 1.0):
        super().__init__(dim, score, dist, odot, norm, residual)
        self.fc = PoincareLinear(dim, dim, init_gain)

    def _scores(self, uj: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        """Pre-softmax score per incidence."""
        if self.score == "eq14":   # a^T (x) (u_j (+) z_i) = tanh(a . log0(u_j (+) z_i)), then (.) d_B
            base = torch.tanh(logmap0(mobius_add(uj, zi)) @ self.a)
        elif self.score == "mobius":
            base = mobius_add(uj, zi) @ self.a
        else:
            base = torch.cat([logmap0(uj), logmap0(zi)], dim=-1) @ self.a
        return self._combine(base, poincare_dist(uj, zi))

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        z = gyromidpoint(u, hg.node_idx, hg.edge_idx, hg.num_edges)          # eq 13
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        alpha = self._alpha(self._scores(uj, zi), hg)                            # eq 14 (+ softmax)
        msg = logmap0(self.fc(z)).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        agg = u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg)
        if self.residual:                                                      # F: DEPARTURE, self path in tangent space
            out = expmap0(logmap0(u) + F.relu(agg))
        else:
            out = expmap0(F.relu(agg))                                         # eq 15
        return torch.where(hg.has_edge[:, None], out, u)


class EucHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "eq14", dist: str = "mult", odot: str = "product", norm: str = "softmax",
                 residual: bool = False, init_gain: float = 1.0):
        super().__init__(dim, score, dist, odot, norm, residual)
        self.fc = nn.Linear(dim, dim)

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        lead = u.shape[:-2]
        z = u.new_zeros(*lead, hg.num_edges, u.shape[-1]).index_add(-2, hg.edge_idx, u.index_select(-2, hg.node_idx))
        z = z / hg.edge_size[:, None]
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        # eq14 Euclidean analogue: the Mobius matvec becomes the ordinary one, so same as "mobius".
        base = (uj + zi) @ self.a if self.score in ("mobius", "eq14") else torch.cat([uj, zi], dim=-1) @ self.a
        alpha = self._alpha(self._combine(base, (uj - zi).norm(dim=-1)), hg)
        msg = self.fc(z).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        out = F.relu(u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg))
        if self.residual:                                                      # F: DEPARTURE
            out = u + out
        return torch.where(hg.has_edge[:, None], out, u)
