"""Masked ranking consistency and parameter EMA for a controlled continuation.

R-Drop-inspired adaptation: cross-sectional stock probabilities, input dropout
for both families, and the inherited supervised ranking objective.
"""
import torch
from torch.nn import functional as F


def symmetric_rank_kl(first, second, eligible, temperature=.5):
    if first.ndim == 1:
        first, second, eligible = first[None], second[None], eligible[None]
    mask = eligible.bool()
    if first.shape != second.shape or first.shape != mask.shape or (mask.sum(1) < 2).any():
        raise ValueError('Invalid cross-sectional distributions')
    a = (first/temperature).masked_fill(~mask, -1e4)
    b = (second/temperature).masked_fill(~mask, -1e4)
    la, lb = F.log_softmax(a, dim=-1), F.log_softmax(b, dim=-1)
    return (.5*((la.exp()-lb.exp())*(la-lb)).sum(-1)).mean()


def stochastic_forward(model, prepared, ids, family, probability=.1):
    eligible = prepared['eligible'][ids]
    if family == 'think':
        x = F.dropout(prepared['x'][ids], p=probability, training=model.training)
        extra = F.dropout(prepared['extra'][ids], p=probability, training=model.training)
        return model(x, extra, eligible.float())[0]
    if family != 'dynamic':
        raise ValueError(family)
    features = F.dropout(prepared['features'][ids], p=probability, training=model.training)
    return model(features, None, eligible)[0]


class ParameterEMA:
    """Average floating parameters after each optimizer update; copy buffers."""
    def __init__(self, model, decay=.99):
        if not 0 <= decay < 1:
            raise ValueError('Invalid EMA decay')
        self.decay = decay
        self.updates = 0
        self.shadow = {name: value.detach().clone() for name, value in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1-self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates,
                    shadow={name: value.detach().cpu().clone() for name, value in self.shadow.items()})

    def load_state_dict(self, saved, device):
        if saved['decay'] != self.decay or saved['shadow'].keys() != self.shadow.keys():
            raise ValueError('EMA state mismatch')
        self.updates = saved['updates']
        self.shadow = {name: value.to(device).clone() for name, value in saved['shadow'].items()}

    @torch.no_grad()
    def averaged_state(self, model):
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for name, value in self.shadow.items():
            state[name] = value.detach().cpu().clone()
        return state
