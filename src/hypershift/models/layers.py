"""Poincare FC (HNN++), beta-concatenation, hyperbolic and Euclidean temporal convolutions (paper eq. 9-12)."""
from __future__ import annotations

import math

import torch
from torch import nn

from hypershift.geometry.poincare import expmap0, lambda_x, logmap0, project


def beta_fn(a: float, b: float) -> float:
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def beta_concat(xs: list[torch.Tensor]) -> torch.Tensor:
    n = sum(x.shape[-1] for x in xs)
    bn = beta_fn(n / 2, 0.5)
    parts = [logmap0(x) * (bn / beta_fn(x.shape[-1] / 2, 0.5)) for x in xs]
    return expmap0(torch.cat(parts, dim=-1))


class PoincareLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_gain: float = 1.0):
        super().__init__()
        self.z = nn.Parameter(torch.randn(in_dim, out_dim) * (2 * in_dim * out_dim) ** -0.5 * init_gain)
        self.r = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = lambda_x(x)                                           # [..., 1]
        z_norm = self.z.norm(dim=0).clamp_min(1e-15)                # [out]
        inner = (x @ self.z) / z_norm                               # <x, z_k/|z_k|>
        v = 2 * z_norm * torch.asinh(lam * inner * torch.cosh(2 * self.r) - (lam - 1) * torch.sinh(2 * self.r))
        w = torch.sinh(v.clamp(-15, 15))
        return project(w / (1 + torch.sqrt(1 + (w * w).sum(-1, keepdim=True))))


def _windows(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """[B,T,N,C] -> [B,T//K,N,K*C] (non-overlapping windows, time-major inside the window)."""
    b, t, n, c = x.shape
    if t % kernel:
        raise ValueError(f"sequence length {t} not divisible by kernel {kernel}")
    return x.reshape(b, t // kernel, kernel, n, c).permute(0, 1, 3, 2, 4).reshape(b, t // kernel, n, kernel * c)


class HypTemporalConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, init_gain: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.scale = beta_fn(kernel * in_dim / 2, 0.5) / beta_fn(in_dim / 2, 0.5)
        self.fc = PoincareLinear(kernel * in_dim, out_dim, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(expmap0(_windows(logmap0(x) * self.scale, self.kernel)))


class EucTemporalConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, activation: bool = True):
        super().__init__()
        self.kernel = kernel
        self.activation = activation
        self.lin = nn.Linear(kernel * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(_windows(x, self.kernel))
        return torch.relu(y) if self.activation else y
