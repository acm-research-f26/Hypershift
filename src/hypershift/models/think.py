"""THINK (paper eq. 17): log0(TConv2(DHHAN(TConv1(exp0(X)), G))) with ablation switches."""
from __future__ import annotations

import torch
from torch import nn

from hypershift.data.hypergraph import TorchHypergraph
from hypershift.geometry.poincare import expmap0, logmap0
from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention
from hypershift.models.layers import EucTemporalConv, HypTemporalConv


class THINK(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 32, seq: int = 16, kernel: int = 4,
                 temporal: str = "hyp", spatial: str = "hyp", structure: str = "hyper",
                 attn_score: str = "mobius", attn_dist: str = "mult", out_dim: int = 1):
        super().__init__()
        if seq % kernel:
            raise ValueError("seq must be a multiple of kernel")
        k2 = seq // kernel
        self.temporal_hyp = temporal == "hyp"
        self.spatial_hyp = spatial == "hyp"
        self.use_spatial = structure != "none"
        if self.temporal_hyp:
            self.tconv1 = HypTemporalConv(in_dim, hidden, kernel)
            self.tconv2 = HypTemporalConv(hidden, out_dim, k2)
        else:
            self.tconv1 = EucTemporalConv(in_dim, hidden, kernel, activation=True)
            self.tconv2 = EucTemporalConv(hidden, out_dim, k2, activation=False)
        if self.use_spatial:
            cls = HypHypergraphAttention if self.spatial_hyp else EucHypergraphAttention
            self.spatial = cls(hidden, score=attn_score, dist=attn_dist)

    def forward(self, x: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        h = x.permute(0, 2, 1, 3)                      # [B,T,N,C]
        if self.temporal_hyp:
            h = expmap0(h)
        h = self.tconv1(h)                             # [B,T/K,N,H]
        if self.use_spatial and hg.num_edges > 0:
            if self.spatial_hyp and not self.temporal_hyp:
                h = self.spatial(expmap0(h), hg)
                h = logmap0(h)
            elif self.temporal_hyp and not self.spatial_hyp:
                h = expmap0(self.spatial(logmap0(h), hg))
            else:
                h = self.spatial(h, hg)
        h = self.tconv2(h)                             # [B,1,N,out]
        if self.temporal_hyp:
            h = logmap0(h)
        h = h[:, 0]
        return h.squeeze(-1) if h.shape[-1] == 1 else h
