"""Controlled architecture comparisons, not certified paper reproductions."""
import torch
from torch import nn
from think_model import ThinkReconstruction

VARIANTS = ['think', 'think_rank', 'think_gate', 'think_mix', 'think_risk',
            'mlp', 'lstm', 'stockmixer', 'market_transformer']


class RankModel(nn.Module):
    def __init__(self, kind, nodes, groups, features=5, lookback=16, hidden=16):
        super().__init__()
        self.kind = kind
        self.gate = nn.Sequential(nn.Linear(features, features), nn.Sigmoid())
        if kind.startswith('think'):
            self.base = ThinkReconstruction(features, nodes, groups, hidden,
                                             kernels=(4, 4), variant='think')
            if kind == 'think_mix':
                self.market = nn.Sequential(nn.Linear(nodes, 16), nn.GELU(), nn.Linear(16, nodes))
            if kind == 'think_risk':
                self.risk = nn.Linear(hidden, 1)
        elif kind == 'mlp':
            self.base = nn.Sequential(nn.Flatten(-2), nn.Linear(lookback*features, hidden),
                                      nn.GELU(), nn.Linear(hidden, 1))
        elif kind == 'lstm':
            self.base = nn.LSTM(features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        elif kind == 'stockmixer':
            self.temporal = nn.Sequential(nn.Linear(lookback, lookback), nn.GELU(),
                                          nn.Linear(lookback, lookback))
            self.channel = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, features))
            self.market = nn.Sequential(nn.Linear(nodes, 16), nn.GELU(), nn.Linear(16, nodes))
            self.head = nn.Linear(lookback*features, 1)
        elif kind == 'market_transformer':
            self.embed = nn.Linear(features, hidden)
            self.pos = nn.Parameter(torch.randn(1, lookback, hidden)*.01)
            self.temporal = nn.TransformerEncoderLayer(hidden, 2, hidden*2, dropout=0., batch_first=True)
            self.cross = nn.MultiheadAttention(hidden, 2, dropout=0., batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError(kind)

    def forward(self, x, eligible):
        # All market information is contemporaneous and masked by signal-time eligibility.
        mask = eligible[:, None, :, None]
        x = x * mask
        market = x.sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)
        if self.kind == 'think_gate' or self.kind == 'market_transformer':
            x = x * (2*self.gate(market))
        auxiliary = None
        if self.kind.startswith('think'):
            score = self.base(x)
            if self.kind == 'think_mix':
                score = score + self.market(score*eligible)
            if self.kind == 'think_risk':
                # Shared first hyperbolic temporal block; volatility head in tangent space.
                from think_model import exp0, log0
                h = log0(self.base.first(exp0(x))).mean(1)
                auxiliary = self.risk(h).squeeze(-1)
        elif self.kind == 'mlp':
            score = self.base(x.permute(0, 2, 1, 3)).squeeze(-1)
        elif self.kind == 'lstm':
            b, t, n, c = x.shape
            h, _ = self.base(x.permute(0, 2, 1, 3).reshape(b*n, t, c))
            score = self.head(h[:, -1]).reshape(b, n)
        elif self.kind == 'stockmixer':
            h = x.permute(0, 2, 3, 1)
            h = h + self.temporal(h)
            h = h.permute(0, 1, 3, 2)
            h = h + self.channel(h)
            h = h + self.market(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            score = self.head(h.flatten(-2)).squeeze(-1)
        else:
            b, t, n, c = x.shape
            h = self.embed(x).permute(0, 2, 1, 3).reshape(b*n, t, -1) + self.pos
            h = self.temporal(h)[:, -1].reshape(b, n, -1)
            # At least two eligible names are enforced upstream.
            cross, _ = self.cross(h, h, h, key_padding_mask=~eligible.bool(), need_weights=False)
            score = self.head(self.norm(h+cross)).squeeze(-1)
        return score, auxiliary


def objective(score, y, mask, kind, auxiliary=None, risk=None):
    error = ((score-y).square()*mask).sum()/mask.sum().clamp_min(1)
    if kind in ('think_rank', 'stockmixer', 'market_transformer'):
        dp = score[:, :, None]-score[:, None, :]
        dy = y[:, :, None]-y[:, None, :]
        pairs = mask[:, :, None]*mask[:, None, :]
        error = error + .1*(torch.relu(-dp*dy)*pairs).sum()/pairs.sum().clamp_min(1)
    if kind == 'think_risk':
        error = error + .1*((auxiliary-risk).square()*mask).sum()/mask.sum().clamp_min(1)
    return error
