"""Small adaptations of research mechanisms around repository THINK layers.

These are mechanism tests, not reproductions of the named papers.
"""
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hypershift.data.hypergraph import Hypergraph
from hypershift.models.layers import EucTemporalConv, HypTemporalConv
from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention
from hypershift.geometry.poincare import expmap0, logmap0


class ScreenModel(nn.Module):
    def __init__(self, kind, nodes, hidden=16, groups=None):
        super().__init__()
        self.kind = kind
        self.hyp = kind == 'hyp'
        self.groups = np.asarray(groups) if groups is not None else np.arange(nodes) % 4
        if kind == 'random_graph':
            self.groups = np.random.default_rng(7).permutation(self.groups)
        seq = 64 if kind == 'long' else 16
        self.seq = seq
        temporal = HypTemporalConv if self.hyp else EucTemporalConv
        self.t1 = temporal(5, hidden, 4)
        self.t2 = temporal(hidden, 1, seq//4) if self.hyp else temporal(hidden, 1, seq//4, activation=False)
        spatial = HypHypergraphAttention if self.hyp else EucHypergraphAttention
        self.spatial = spatial(hidden)
        if kind in ('two', 'two_residual'):
            self.second = spatial(hidden)
        if kind in ('soft', 'residual_soft'):
            self.prototypes = nn.Parameter(torch.randn(16, hidden)*.1)
            self.message = nn.Linear(hidden, hidden)
        if kind in ('mix', 'gated'):
            self.mixer = nn.Sequential(nn.Linear(nodes, 16), nn.GELU(), nn.Linear(16, nodes))
            if kind == 'gated':
                self.gate = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        if kind == 'lowrank':
            self.query = nn.Linear(hidden, 8)
            self.value = nn.Linear(hidden, hidden)
        if kind in ('multiscale', 'summary'):
            self.extra = nn.Linear(15, hidden)
        if kind == 'master':
            self.market_gate = nn.Sequential(nn.Linear(5, 5), nn.Sigmoid())
            self.attention = nn.MultiheadAttention(hidden, 2, batch_first=True, dropout=0.)
        if kind == 'mlp':
            self.mlp = nn.Sequential(nn.Flatten(-2), nn.Linear(80, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if kind == 'pretrain':
            self.decoder = nn.Linear(hidden, 20)
        if kind == 'star':
            # Explicit star expansion: one hub per cluster with its own node->hub and hub->node maps.
            self.to_hub = nn.Linear(hidden, hidden)
            self.from_hub = nn.Linear(hidden, hidden)
            self.self_fc = nn.Linear(hidden, hidden)
        self._graphs = {}

    def graph(self, eligible, history=None):
        ids = np.flatnonzero(eligible.detach().cpu().numpy())
        if self.kind == 'clique':
            # Clique (pairwise) expansion: every within-cluster pair is a 2-node hyperedge. Cached per
            # eligible set because the pair list is large and rebuilt on every forward otherwise.
            key = ids.tobytes()
            if key not in self._graphs:
                groups = [tuple(i for i in ids if self.groups[i] == g) for g in np.unique(self.groups)]
                pairs = tuple(p for g in groups for p in combinations(g, 2))
                self._graphs[key] = Hypergraph(len(eligible), pairs + tuple((int(i),) for i in ids)).to_torch(eligible.device)
            return self._graphs[key]
        if history is None:
            groups = [tuple(i for i in ids if self.groups[i] == g) for g in np.unique(self.groups)]
        else:
            # Causal rolling history; assign each stock to its most-correlated
            # deterministic anchor. No future labels and no dense N*N graph.
            r = history[ids, 1:, 4] - history[ids, :-1, 4]
            r = r-r.mean(-1, keepdim=True)
            r = F.normalize(r, dim=-1)
            anchor = torch.linspace(0, len(ids)-1, min(16, len(ids)), device=r.device).long()
            assignment = (r @ r[anchor].T).argmax(-1).detach().cpu().numpy()
            groups = [tuple(ids[assignment == g]) for g in range(len(anchor))]
        # A partition alone gives incidence degree1 and erases within-group
        # identity under segment softmax. Common self hyperedges preserve each
        # stock's own history in ALL graph arms, including rolling/random.
        edges = tuple(g for g in groups if len(g) >= 2) + tuple((int(i),) for i in ids)
        return Hypergraph(len(eligible), edges).to_torch(eligible.device)

    def encode(self, x, eligible):
        x = torch.where(eligible[:, :, None, None], x, 0.)
        if self.kind == 'adaptive':
            # Signal-local normalization: no learned future statistics.
            x = (x-x.mean(2, keepdim=True))/x.std(2, keepdim=True).clamp_min(.25)
        if self.kind == 'master':
            market = x.sum(1, keepdim=True)/eligible.sum(1)[:, None, None, None].clamp_min(1)
            x = x * (2*self.market_gate(market))
        u = x[:, :, -self.seq:].permute(0, 2, 1, 3)
        h = self.t1(expmap0(u) if self.hyp else u)
        if self.kind in ('multiscale', 'summary'):
            lengths = (4, 8, 16) if self.kind == 'multiscale' else (16, 32, 64)
            extra = torch.cat([x[:, :, -s:].mean(2) for s in lengths], -1)
            h = h + self.extra(extra)[:, None]
        return h, x

    def reconstruct(self, x, eligible):
        h, _ = self.encode(x, eligible)
        return self.decoder(h).reshape(h.shape[0], 4, h.shape[2], 4, 5).permute(0, 2, 1, 3, 4).reshape(x.shape[0], x.shape[1], 16, 5)

    def forward(self, x, eligible):
        x = torch.where(eligible[:, :, None, None], x, 0.)
        if self.kind == 'mlp':
            return self.mlp(x[:, :, -16:]).squeeze(-1)
        h, x = self.encode(x, eligible)
        initial = h
        if self.kind != 'none':
            if self.kind == 'soft':
                pass
            elif self.kind == 'star':
                member = torch.as_tensor(np.eye(self.groups.max()+1)[self.groups], dtype=h.dtype, device=h.device)
                member = member[None] * eligible[:, :, None]                                  # B x N x G
                count = member.sum(1)
                hub = torch.einsum('bng,btnc->btgc', member, h) / count.clamp_min(1)[:, None, :, None]
                hub = F.relu(self.to_hub(hub))
                back = torch.einsum('bng,btgc->btnc', member, hub)
                active = (member * (count[:, None, :] >= 2)).sum(-1) > 0
                h = torch.where(active[:, None, :, None], F.relu(self.self_fc(h) + self.from_hub(back)), h)
            else:
                h = torch.cat([self.spatial(h[b:b+1], self.graph(eligible[b], x[b] if self.kind == 'rolling' else None))
                               for b in range(len(x))], 0)
            if self.kind in ('two', 'two_residual'):
                if self.kind == 'two_residual':
                    h = .5*(h+initial)
                h = torch.cat([self.second(h[b:b+1], self.graph(eligible[b])) for b in range(len(x))], 0)
                if self.kind == 'two_residual':
                    h = .5*(h+initial)
            if self.kind in ('soft', 'residual_soft'):
                assignment = (initial @ self.prototypes.T / initial.shape[-1]**.5).softmax(-1)
                assignment = assignment * eligible[:, None, :, None]
                factors = torch.einsum('btnm,btnh->btmh', assignment, initial)
                factors = factors / assignment.sum(-2).clamp_min(1e-6)[..., None]
                message = assignment @ self.message(factors)
                h = initial+message if self.kind == 'soft' else h+message
                self.assignment_entropy = float((-(assignment.clamp_min(1e-9).log()*assignment).sum(-1)).detach().mean())
            if self.kind == 'lowrank':
                a = self.query(initial).softmax(-1) * eligible[:, None, :, None]
                factors = torch.einsum('btnr,btnh->btrh', a, self.value(initial))
                factors /= a.sum(-2).clamp_min(1e-6)[..., None]
                h = h+a@factors
            if self.kind == 'master':
                b, t, n, c = h.shape
                flat = h.reshape(b*t, n, c)
                pad = (~eligible)[:, None].expand(-1, t, -1).reshape(b*t, n)
                a, _ = self.attention(flat, flat, flat, key_padding_mask=pad, need_weights=False)
                h = h+a.reshape(b, t, n, c)
        self.embedding_variance = float(h.detach().var(-2).mean())
        self.representation_cosine = float(F.cosine_similarity(h.detach(), h.detach().mean(-2, keepdim=True), dim=-1).mean())
        score = self.t2(h)
        if self.hyp:
            score = logmap0(score)
        score = score[:, 0, :, 0]
        if self.kind in ('mix', 'gated'):
            correction = self.mixer(score*eligible)
            if self.kind == 'gated':
                market = x[:, :, -1].sum(1)/eligible.sum(1)[:, None].clamp_min(1)
                correction = correction*self.gate(market)
            score = score+correction
        return score


def objective(pred, target, mask, kind='mse'):
    if not mask.any():
        raise ValueError('no mature labels in batch')
    mse = ((pred-target)[mask]**2).mean()
    if kind == 'mse':
        return mse
    if kind == 'pairwise':
        # Deterministic cyclic pairs, same across methods/seeds.
        dp = pred-pred.roll(1, -1); dy = target-target.roll(1, -1)
        good = mask & mask.roll(1, -1)
        return mse + .1*F.softplus(-dp[good]*dy[good].sign()).mean()
    rows = mask.any(-1)
    pred, target, mask = pred[rows], target[rows], mask[rows]
    q = target.masked_fill(~mask, -torch.inf).softmax(-1)
    logp = pred.masked_fill(~mask, -1e9).log_softmax(-1)
    loss = -(q*logp).sum(-1)
    loss = loss[mask.any(-1)].mean()
    return loss if kind == 'listwise' else mse+.1*loss
