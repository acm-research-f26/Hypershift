"""Paper-named stock baselines reimplemented in PyTorch: RSR-I (Feng 2019) and STHGCN (Sawhney 2020/21).

Both take x [B, N, T, C] and return [B, N] (same contract as THINK). See docs/phase1/R8_baselines.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hypershift.data.hypergraph import TorchHypergraph
from hypershift.data.rsr import read_ticker_file


@functools.lru_cache(maxsize=4)
def _raw_relation_entries(root: str, market: str):
    """Multi-hot relation entries of RSR industry + wiki tensors: (i, j, channel) arrays in full-universe indices.
    Channels: industry (108 NYSE) then wiki (33 NYSE); each includes its own trailing self-relation channel."""
    ii, jj, kk, off = [], [], [], 0
    for sub, name in (("sector_industry", "industry"), ("wikidata", "wiki")):
        rel = np.load(Path(root) / "relation" / sub / f"{market}_{name}_relation.npy", mmap_mode="r")
        for k in range(rel.shape[2]):
            i, j = np.nonzero(np.asarray(rel[:, :, k]))
            ii.append(i)
            jj.append(j)
            kk.append(np.full(len(i), off + k))
        off += rel.shape[2]
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(kk), off


def relation_entries(root: str, market: str, tickers: list[str]):
    """Restrict to the current universe (by ticker) and remap.
    Returns (pair_i, pair_j, entry_pair, entry_channel, K): unique related pairs and the multi-hot entries."""
    full = read_ticker_file(Path(root) / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    pos = {t: n for n, t in enumerate(full)}
    remap = np.full(len(full), -1, dtype=np.int64)
    for new, t in enumerate(tickers):
        remap[pos[t]] = new
    i, j, k, K = _raw_relation_entries(str(root), market)
    ni, nj = remap[i], remap[j]
    keep = (ni >= 0) & (nj >= 0)
    ni, nj, k = ni[keep], nj[keep], k[keep]
    n = len(tickers)
    pid, inv = np.unique(ni * n + nj, return_inverse=True)
    return pid // n, pid % n, inv, k, K


class RSRI(nn.Module):
    """RSR-I: LSTM -> inner-product relational propagation -> concat -> leaky-relu dense (Feng et al. 2019).

    Follows relation_rank_lstm.py: rel_weight = leaky_relu(dense(multi-hot relation)); weight = <f_i,f_j> * rel_weight;
    softmax(mask + weight, dim=0) (sic, over axis 0 of the [N,N] matrix); prop = W @ f; pred = leaky_relu(dense([f, prop])).
    """

    def __init__(self, in_dim, hidden, pair_i, pair_j, entry_pair, entry_channel, num_channels, softmax_axis: int = 0):
        super().__init__()
        self.softmax_axis = softmax_axis
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.rel = nn.Linear(num_channels, 1)
        self.fc = nn.Linear(2 * hidden, 1)
        for name, t in (("pair_i", pair_i), ("pair_j", pair_j), ("entry_pair", entry_pair),
                        ("entry_channel", entry_channel)):
            self.register_buffer(name, torch.as_tensor(np.asarray(t), dtype=torch.long), persistent=False)
        self.num_pairs = len(pair_i)

    def rel_weight(self) -> torch.Tensor:
        s = torch.zeros(self.num_pairs, device=self.rel.weight.device)
        s = s.index_add(0, self.entry_pair, self.rel.weight[0][self.entry_channel]) + self.rel.bias
        return F.leaky_relu(s, 0.2)

    def forward(self, x: torch.Tensor, hg: TorchHypergraph | None = None) -> torch.Tensor:
        B, N, T, C = x.shape
        _, (h, _) = self.lstm(x.reshape(B * N, T, C))
        f = h[-1].reshape(B, N, -1)
        rw = torch.zeros(N, N, device=x.device).index_put((self.pair_i, self.pair_j), self.rel_weight())
        mask = torch.full((N, N), -1e9, device=x.device).index_put(
            (self.pair_i, self.pair_j), torch.zeros(self.num_pairs, device=x.device))
        w = torch.softmax(torch.matmul(f, f.transpose(1, 2)) * rw + mask, dim=1 + self.softmax_axis)
        prop = torch.matmul(w, f)
        return F.leaky_relu(self.fc(torch.cat([f, prop], dim=-1)), 0.2).squeeze(-1)


class TimeBlock(nn.Module):
    """Gated temporal conv of STHGCN/STGCN: relu(conv1(x) + sigmoid(conv2(x)) + conv3(x)); valid conv over time."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.c1, self.c2, self.c3 = (nn.Conv2d(cin, cout, (1, k)) for _ in range(3))

    def forward(self, x):                                        # [B, N, T, C]
        x = x.permute(0, 3, 1, 2)
        y = F.relu(self.c1(x) + torch.sigmoid(self.c2(x)) + self.c3(x))
        return y.permute(0, 2, 3, 1)


def hgnn_operator(hg: TorchHypergraph, device) -> torch.Tensor:
    """Dense G = Dv^-1/2 H De^-1 H^T Dv^-1/2 (unit hyperedge weights); isolated nodes get G_ii = 1."""
    N = hg.num_nodes
    if hg.num_edges == 0:
        return torch.eye(N, device=device)
    H = torch.zeros(N, hg.num_edges, device=device)
    H[hg.node_idx, hg.edge_idx] = 1.0
    dv, de = H.sum(1), H.sum(0)
    dvi = torch.where(dv > 0, dv.clamp(min=1).pow(-0.5), torch.zeros_like(dv))
    G = (dvi[:, None] * H) @ ((1.0 / de.clamp(min=1))[:, None] * (H.t() * dvi[None, :]))
    return G + torch.diag((dv == 0).float())


class STHGCN(nn.Module):
    """[TimeBlock -> hypergraph conv -> TimeBlock -> BN] x2 -> TimeBlock -> flatten -> linear (STGCN layout)."""

    def __init__(self, in_dim: int, seq: int, num_nodes: int, out_ch: int = 64, spatial_ch: int = 16, k: int = 3):
        super().__init__()
        if seq - 5 * (k - 1) < 1:
            raise ValueError("seq too short for STHGCN")
        self.t1a, self.t1b = TimeBlock(in_dim, out_ch, k), TimeBlock(spatial_ch, out_ch, k)
        self.t2a, self.t2b = TimeBlock(out_ch, out_ch, k), TimeBlock(spatial_ch, out_ch, k)
        self.th1 = nn.Parameter(torch.empty(out_ch, spatial_ch))
        self.th2 = nn.Parameter(torch.empty(out_ch, spatial_ch))
        for p in (self.th1, self.th2):
            nn.init.uniform_(p, -1 / spatial_ch ** 0.5, 1 / spatial_ch ** 0.5)
        self.bn1, self.bn2 = nn.BatchNorm2d(num_nodes), nn.BatchNorm2d(num_nodes)
        self.last = TimeBlock(out_ch, out_ch, k)
        self.fc = nn.Linear((seq - 5 * (k - 1)) * out_ch, 1)
        self._G, self._key, self._dev = None, None, None

    def _op(self, hg, device):
        if self._key is not hg or self._dev != str(device):   # hold the object itself (no id reuse)
            self._G, self._key, self._dev = hgnn_operator(hg, device), hg, str(device)
        return self._G

    @staticmethod
    def _spatial(G, x, theta):                                   # relu(G X Theta) at every time step
        return F.relu(torch.einsum("ij,bjtc->bitc", G, x @ theta))

    def forward(self, x: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        G = self._op(hg, x.device)
        h = self.bn1(self.t1b(self._spatial(G, self.t1a(x), self.th1)))
        h = self.bn2(self.t2b(self._spatial(G, self.t2a(h), self.th2)))
        h = self.last(h)
        return self.fc(h.reshape(h.shape[0], h.shape[1], -1)).squeeze(-1)
