"""Poincare ball (curvature -1) operations. All functions act on the last dimension."""
from __future__ import annotations

import torch

MAX_NORM = 1.0 - 1e-5
_MIN = 1e-15


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(_MIN)


def project(x: torch.Tensor) -> torch.Tensor:
    n = _norm(x)
    return torch.where(n > MAX_NORM, x / n * MAX_NORM, x)


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def expmap0(v: torch.Tensor) -> torch.Tensor:
    n = _norm(v)
    return project(torch.tanh(n) * v / n)


def logmap0(y: torch.Tensor) -> torch.Tensor:
    n = _norm(y)
    return artanh(n) * y / n


def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den.clamp_min(_MIN)


def poincare_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * artanh(mobius_add(-x, y).norm(dim=-1))


def lambda_x(x: torch.Tensor) -> torch.Tensor:
    return 2.0 / (1.0 - (x * x).sum(-1, keepdim=True)).clamp_min(_MIN)


def mobius_scalar(r: float, x: torch.Tensor) -> torch.Tensor:
    n = _norm(x)
    return project(torch.tanh(r * artanh(n)) * x / n)


def gyromidpoint(x: torch.Tensor, node_idx: torch.Tensor, edge_idx: torch.Tensor, num_edges: int) -> torch.Tensor:
    """Paper eq. 13. x: [..., N, D] -> [..., E, D]."""
    lam = lambda_x(x)                                           # [..., N, 1]
    lead = x.shape[:-2]
    num = x.new_zeros(*lead, num_edges, x.shape[-1]).index_add(-2, edge_idx, (lam * x).index_select(-2, node_idx))
    den = x.new_zeros(*lead, num_edges, 1).index_add(-2, edge_idx, (lam - 1).index_select(-2, node_idx))
    return mobius_scalar(0.5, num / den.clamp_min(_MIN))
