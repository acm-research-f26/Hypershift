"""Predeclared loss and mixer ablations; local adaptations, not paper replication."""
import torch
from torch import nn
from torch.nn import functional as F
from research_models import RankModel, VARIANTS, objective

EXPERIMENTS = {kind: dict(base=kind, loss='mse') for kind in VARIANTS}
EXPERIMENTS.update({
    'think_huber': dict(base='think', loss='huber'),
    'think_mix_huber': dict(base='think_mix', loss='huber'),
    'stockmixer_huber': dict(base='stockmixer', loss='huber'),
    'think_mix_shrink': dict(base='think_mix', loss='mse', shrink=.1),
    'think_mix_shrink_huber': dict(base='think_mix', loss='huber', shrink=.1),
    'multiscale_mse': dict(base='multiscale', loss='mse'),
    'multiscale_huber': dict(base='multiscale', loss='huber'),
})


class CausalMix(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.a = nn.Linear(length, length)
        self.b = nn.Linear(length, length)
        self.register_buffer('causal', torch.ones(length, length).tril())

    def forward(self, x):
        # Inputs contain only signal-time history. Also prevent later positions
        # within that history from entering earlier temporal representations.
        h = F.hardswish(F.linear(x, self.a.weight*self.causal, self.a.bias))
        return x + F.linear(h, self.b.weight*self.causal, self.b.bias)


class MultiscaleMixer(nn.Module):
    def __init__(self, nodes, features=5, lookback=16):
        super().__init__()
        self.scales = [1, 2, 4]
        self.indicators = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(features), nn.Linear(features, 16),
                          nn.Hardswish(), nn.Linear(16, features)) for _ in self.scales])
        self.temporal = nn.ModuleList([CausalMix(lookback//s) for s in self.scales])
        self.compress = nn.Linear(features*sum(lookback//s for s in self.scales), 16)
        self.market = nn.Sequential(nn.Linear(nodes, 8), nn.Hardswish(), nn.Linear(8, nodes))
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(32, 1)

    def forward(self, x, eligible):
        b, t, n, c = x.shape
        source = (x*eligible[:, None, :, None]).permute(0, 2, 3, 1)
        pieces = []
        for s, indicator, time in zip(self.scales, self.indicators, self.temporal):
            patch = F.avg_pool1d(source.reshape(b*n, c, t), s, s).reshape(b, n, c, t//s)
            patch = patch + indicator(patch.transpose(-1, -2)).transpose(-1, -2)
            pieces.append(time(patch).flatten(-2))
        local = self.compress(torch.cat(pieces, -1))*eligible[:, :, None]
        market = self.market((self.norm(local)*eligible[:, :, None]).transpose(1, 2)).transpose(1, 2)
        return self.head(torch.cat([local, local+market], -1)).squeeze(-1), None


def make_model(kind, nodes, groups):
    spec = EXPERIMENTS[kind]
    if spec['base'] == 'multiscale':
        return MultiscaleMixer(nodes)
    model = RankModel(spec['base'], nodes, groups)
    if 'shrink' in spec:
        # Same initialization/parameters as the unconstrained mixer; only its
        # residual contribution changes. Saved in the experiment specification.
        model.market = nn.Sequential(model.market, FixedScale(spec['shrink']))
    return model


class FixedScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale*x


def robust_objective(score, y, mask, kind, auxiliary=None, risk=None):
    spec = EXPERIMENTS[kind]
    base = spec['base']
    if spec['loss'] == 'mse' and base != 'multiscale':
        return objective(score, y, mask, base, auxiliary, risk)
    # Targets/scores are percentage points. 2*Huber(delta=1pp) matches
    # MSE curvature near zero while capping tail gradients at magnitude 2.
    point = (2*F.huber_loss(score, y, reduction='none', delta=1.)
             if spec['loss'] == 'huber' else (score-y).square())
    loss = (point*mask).sum()/mask.sum().clamp_min(1)
    if base in ['stockmixer', 'multiscale']:
        dp = score[:, :, None]-score[:, None, :]
        dy = y[:, :, None]-y[:, None, :]
        pair = mask[:, :, None]*mask[:, None, :]
        loss = loss + .1*(F.relu(-dp*dy)*pair).sum()/pair.sum().clamp_min(1)
    return loss
