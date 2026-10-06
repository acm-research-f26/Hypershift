"""Small dense hypergraphs and Poincare-ball operations, curvature -1.

This is an educational THINK-inspired implementation, not the THINK architecture.
All learned weights live in ordinary tangent coordinates, so PyTorch AdamW works.
"""
from itertools import combinations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def correlation_hyperedges(train_returns, neighbors=5):
    """One stock plus its strongest positively correlated neighbors; deduplicate.

    Caller must supply ONLY observations available in the training block.
    Unlike pairwise edges, each column of H represents a complete group.
    """
    n = train_returns.shape[1]
    if not 0 <= neighbors < n:
        raise ValueError('neighbors must be in [0, number of stocks).')
    z = train_returns - train_returns.mean(0)
    z = z / np.maximum(np.linalg.norm(z, axis=0), 1e-12)
    corr = np.clip(z.T @ z, -1, 1)
    np.fill_diagonal(corr, -np.inf)
    groups = []
    for i in range(n):
        selected = [j for j in np.argsort(-corr[i], kind='stable')[:neighbors]
                    if corr[i, j] > 0]
        group = tuple(sorted([i] + selected))
        if group not in groups:
            groups.append(group)
    h = np.zeros((n, len(groups)), dtype=np.float32)
    for j, group in enumerate(groups):
        h[list(group), j] = 1
    return h


def four_point_delta(distances):
    """Exact vertex-metric delta: half the largest gap of the top two sums.

    O(n^4): appropriate for this 12-stock example, not an exchange-sized graph.
    Distinct quadruples suffice; repeated vertices cannot increase the maximum.
    """
    d = np.asarray(distances, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError('Disconnected distances: compute each component separately.')
    delta = 0.0
    for a, b, c, e in combinations(range(len(d)), 4):
        sums = sorted([d[a, b] + d[c, e], d[a, c] + d[b, e], d[a, e] + d[b, c]])
        delta = max(delta, (sums[2] - sums[1]) / 2)
    diameter = float(d.max())
    return {'delta': float(delta), 'diameter': diameter,
            'relative_delta': 2 * delta / diameter if diameter else None}


def hypergraph_geometry(h, s=1):
    """Paper's node s-walk: adjacent nodes share at least s hyperedges."""
    if s < 1:
        raise ValueError('s must be positive.')
    n = len(h)
    distance = np.where(h @ h.T >= s, 1.0, np.inf)
    np.fill_diagonal(distance, 0)
    for k in range(n):  # Floyd-Warshall shortest paths.
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    remaining, components = set(range(n)), []
    while remaining:
        nodes = np.flatnonzero(np.isfinite(distance[min(remaining)])).tolist()
        remaining.difference_update(nodes)
        components.append({'nodes': nodes, **four_point_delta(distance[np.ix_(nodes, nodes)])})
    return {'s': s, 'connected': len(components) == 1,
            'global_delta': components[0]['delta'] if len(components) == 1 else None,
            'components': components}


def project(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x * ((1 - 1e-5) / norm).clamp_max(1)


def exp0(v):
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    return project(torch.tanh(norm) * v / norm)


def log0(x):
    x = project(x)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.atanh(norm) * x / norm


def poincare_distance(x, y):
    """2 atanh(||(-x) Mobius-plus y||), stable including identical points."""
    x, y = project(x), project(y)
    xx, yy = (x * x).sum(-1, keepdim=True), (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = -(1 - 2 * xy + yy) * x + (1 - xx) * y
    denominator = (1 - 2 * xy + xx * yy).clamp_min(1e-8)
    norm = torch.linalg.vector_norm(numerator / denominator, dim=-1).clamp_max(1 - 1e-5)
    return 2 * torch.atanh(norm)


def einstein_midpoint(points, weights):
    """[B,N,D] points and [N,E] or [B,E,N] weights -> [B,E,D].

    Map Poincare -> Klein; take a Lorentz-weighted mean; map back.
    This is not an ordinary arithmetic mean of Poincare coordinates.
    """
    points = project(points)
    klein = 2 * points / (1 + points.square().sum(-1, keepdim=True))
    gamma = torch.rsqrt((1 - klein.square().sum(-1)).clamp_min(1e-7))
    if weights.ndim == 2:
        weights = weights.T.unsqueeze(0)
    effective = weights * gamma[:, None, :]
    mean = (effective @ klein) / effective.sum(-1, keepdim=True).clamp_min(1e-8)
    return project(mean / (1 + torch.sqrt((1 - mean.square().sum(-1, keepdim=True)).clamp_min(1e-7))))


class DirectionNet(nn.Module):
    """Shared lag encoder + optional relationship layer + up/down logit.

    The Euclidean and hyperbolic hypergraph variants have identical parameter
    counts and inputs. Their group averaging and distance geometry differ.
    'mlp' supplies its own encoding as context; 'gcn' uses neighbor averages.
    """
    def __init__(self, features, hidden, kind, incidence, adjacency):
        super().__init__()
        if kind not in ('logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic'):
            raise ValueError('Unknown model kind.')
        self.kind = kind
        self.register_buffer('incidence', torch.as_tensor(incidence, dtype=torch.float32))
        self.register_buffer('adjacency', torch.as_tensor(adjacency, dtype=torch.float32))
        if kind == 'logistic':
            self.linear = nn.Linear(features, 1)
        else:
            self.encoder = nn.Linear(features, hidden)
            self.mix = nn.Linear(2 * hidden, hidden)
            self.output = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(0.1)
            if kind in ('hypergraph', 'hyperbolic'):
                self.temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if self.kind == 'logistic':
            return self.linear(x).squeeze(-1)
        # Bounded vectors keep hyperbolic operations away from the ball boundary.
        z = torch.tanh(self.encoder(x)) / self.encoder.out_features ** 0.5
        h = self.incidence
        if self.kind == 'mlp':
            context = z
        elif self.kind == 'gcn':
            context = self.adjacency @ z
        elif self.kind == 'hypergraph':
            groups = h.T @ z / h.sum(0)[None, :, None]
            distance = 2 * torch.linalg.vector_norm(z[:, :, None] - groups[:, None], dim=-1)
            attention = (-F.softplus(self.temperature) * distance).masked_fill(h[None] == 0, -torch.inf)
            context = attention.softmax(-1) @ groups
        else:
            points = exp0(z)
            groups = einstein_midpoint(points, h)
            distance = poincare_distance(points[:, :, None], groups[:, None])
            attention = (-F.softplus(self.temperature) * distance).masked_fill(h[None] == 0, -torch.inf)
            context = log0(einstein_midpoint(groups, attention.softmax(-1)))
        combined = torch.cat([z, context], dim=-1)
        return self.output(self.dropout(torch.relu(self.mix(combined)))).squeeze(-1)
