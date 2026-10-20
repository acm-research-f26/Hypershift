"""Causal local-panel preparation and five-session mature-label splits."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from sklearn.cluster import KMeans


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def forward_labels(price, mask, horizon=5):
    y = np.zeros_like(price, dtype=np.float32)
    ok = np.zeros_like(mask, dtype=bool)
    y[:, :-horizon] = price[:, horizon:] / np.maximum(price[:, :-horizon], 1e-12) - 1
    valid = np.ones_like(mask[:, :-horizon], dtype=bool)
    for j in range(horizon + 1):
        valid &= mask[:, j:mask.shape[1] - horizon + j]
    ok[:, :-horizon] = valid
    y[~ok] = 0
    return y, ok


def split_signals(dates, year, horizon=5, history=64, train_stride=5):
    t = np.arange(history - 1, len(dates) - horizon)
    mature = dates[t + horizon]
    vi = np.datetime64(f'{year-1}-01-01')
    ti = np.datetime64(f'{year}-01-01')
    end = np.datetime64(f'{year+1}-01-01')
    return (t[mature < vi][::train_stride], t[(dates[t] >= vi) & (mature < ti)],
            t[(dates[t] >= ti) & (mature < end)])


def usable_signals(signals, usable, minimum=40):
    return signals[usable[:, signals].sum(0) >= minimum]


@dataclass
class Panel:
    features: np.ndarray
    price: np.ndarray
    observed: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray
    y: np.ndarray
    label_ok: np.ndarray
    eligible: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    evaluate: np.ndarray
    groups: np.ndarray
    scale: np.ndarray
    yscale: float
    beta: np.ndarray
    digest: str
    sectors: np.ndarray = None

    def batch(self, signals):
        indices = np.asarray(signals)[:, None] - np.arange(63, -1, -1)
        f = self.features[:, indices, :].transpose(1, 0, 2, 3).copy()
        base = self.price[:, signals].T[:, :, None, None]
        f = (f / np.maximum(base, 1e-12) - 1) / self.scale
        e = self.eligible[:, signals].T
        f = np.where(e[:, :, None, None], np.clip(f, -10, 10), 0)
        return f.astype(np.float32), e

    def target(self, signals, kind='raw'):
        y = self.y[:, signals].T.copy()
        mask = (self.label_ok[:, signals] & self.eligible[:, signals]).T
        if kind != 'raw':
            market = (y * mask).sum(1) / np.maximum(mask.sum(1), 1)
            y -= self.beta[:, signals].T * market[:, None]
            if kind in ('cluster', 'sector'):
                labels = self.groups if kind == 'cluster' else self.sectors
                for g in np.unique(labels):
                    ix = labels == g
                    residual = (y[:, ix] * mask[:, ix]).sum(1) / np.maximum(mask[:, ix].sum(1), 1)
                    y[:, ix] -= residual[:, None]
        return y.astype(np.float32) / self.yscale, mask


def load_panel(path, year, train_stride=5):
    z = np.load(Path(path), allow_pickle=False)
    ident = z['identity_ok'].astype(bool)
    # Truncate before constructing features, labels, statistics or relationships.
    dates = z['dates'].astype('datetime64[D]')
    keep = dates < np.datetime64(f'{year+1}-01-01')
    dates = dates[keep]
    f = z['features'][ident][:, keep].astype(np.float32)
    observed = z['mask'][ident][:, keep].astype(bool)
    price = f[:, :, -1]
    y, ok = forward_labels(price, observed)
    eligible = np.zeros_like(observed)
    # Same 64-session eligibility for all methods, including 16-session models.
    cs = np.pad(np.cumsum(~observed, axis=1), ((0, 0), (1, 0)))
    eligible[:, 63:] = cs[:, 64:] - cs[:, :-64] == 0
    train, valid, evaluation = split_signals(dates, year, train_stride=train_stride)
    train, valid, evaluation = [usable_signals(ts, eligible & ok) for ts in (train, valid, evaluation)]
    if not len(train) or not len(valid) or not len(evaluation):
        raise ValueError('panel does not cover requested fold')
    boundary = int(np.searchsorted(dates, np.datetime64(f'{year-1}-01-01')))
    daily = np.zeros_like(price)
    daily[:, 1:] = np.where(observed[:, 1:] & observed[:, :-1],
                            price[:, 1:] / np.maximum(price[:, :-1], 1e-12) - 1, 0)
    train_returns = daily[:, :boundary]
    # Training-only clusters, NOT historical sectors. Never filter by future survival.
    standardized = train_returns / np.maximum(train_returns.std(1, keepdims=True), 1e-4)
    groups = KMeans(n_clusters=16, random_state=7, n_init=5).fit_predict(standardized)
    p = Panel(f, price, observed, dates, z['tickers'][ident], y, ok, eligible,
              train, valid, evaluation, groups, np.ones(5), 1., np.ones_like(price), sha256(path),
              z['sectors'][ident].astype(int) if 'sectors' in z.files else None)
    sample = train[::max(1, len(train)//32)]
    xs, es = p.batch(sample)
    p.scale = np.maximum(xs[es].std(axis=(0, 1)), .01).astype(np.float32)
    yy, mm = p.target(train)
    p.yscale = max(float(yy[mm].std()), .001)
    # Rolling beta uses only trailing observed daily returns, including signal day.
    market = (daily * observed).sum(0) / np.maximum(observed.sum(0), 1)
    for t in range(63, len(dates)):
        start = max(1, t - 125)
        r, q = daily[:, start:t+1], market[start:t+1]
        valid_r = observed[:, start:t+1] & observed[:, start-1:t]
        count = np.maximum(valid_r.sum(1), 1)
        mr = (r * valid_r).sum(1) / count
        mq = (q * valid_r).sum(1) / count
        cov = ((r-mr[:, None]) * (q-mq[:, None]) * valid_r).sum(1) / count
        var = (((q-mq[:, None])**2) * valid_r).sum(1) / count
        p.beta[:, t] = np.clip(cov / np.maximum(var, 1e-8), -3, 3)
    return p
