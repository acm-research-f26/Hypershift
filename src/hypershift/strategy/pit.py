"""PIT re-test of the style-only book (docs/strategy_pit_20261023/PREREG.md).

load(): the Alpaca PIT panel as engine data + an FB-like object (style features with raw dollar volume, 5-day
labels, PIT eligibility = member & 64 observed sessions, beta vs the EW of observed members).
annual_scores(): weekly signals from the first trading day of 2018; one model per year, trained at that year's first
signal on every signal whose label matured by then.
"""
from types import SimpleNamespace

import numpy as np

from hypershift.featbench.data import FEATURES, compute_features
from hypershift.featbench.prospective import fit_fixed
from hypershift.fbaudit.leakfree import corrected_columns
from hypershift.fbaudit.matched import nested_inputs
from hypershift.fbaudit.refit import target_values
from hypershift.screening.data import forward_labels

NPZ = 'data/pit/pit_sp500_2016_2023.npz'
START = np.datetime64('2018-01-01')
HISTORY = 64


def _beta(price, observed, member, window=126):
    T = price.shape[1]
    daily = np.zeros_like(price)
    daily[:, 1:] = np.where(observed[:, 1:] & observed[:, :-1], price[:, 1:] / np.maximum(price[:, :-1], 1e-12) - 1, 0)
    w = observed & member
    market = (daily * w).sum(0) / np.maximum(w.sum(0), 1)
    beta = np.ones_like(price)
    for t in range(63, T):
        s = max(1, t - window + 1)
        r, q = daily[:, s:t + 1], market[s:t + 1]
        v = observed[:, s:t + 1] & observed[:, s - 1:t]
        n = np.maximum(v.sum(1), 1)
        mr, mq = (r * v).sum(1) / n, (q * v).sum(1) / n
        cov = ((r - mr[:, None]) * (q - mq[:, None]) * v).sum(1) / n
        var = (((q - mq[:, None]) ** 2) * v).sum(1) / n
        beta[:, t] = np.clip(cov / np.maximum(var, 1e-8), -3, 3)
    return beta


def load(path=NPZ):
    z = np.load(path)
    dates = z['dates'].astype('datetime64[D]')
    if dates.max() > np.datetime64('2023-12-31'):
        raise ValueError('PIT panel must end by 2023-12-31 (2024+ protected)')
    observed, member = z['observed'], z['member']
    close, open_ = np.nan_to_num(z['close'], nan=0.), np.nan_to_num(z['open'], nan=0.)
    cs = np.pad(np.cumsum(~observed, 1), ((0, 0), (1, 0)))
    hist = np.zeros_like(observed)
    hist[:, HISTORY - 1:] = cs[:, HISTORY:] - cs[:, :-HISTORY] == 0
    eligible = hist & member
    beta = _beta(close, observed, member)
    sectors = z['sectors'].astype(int)
    ohlcv = np.stack([np.nan_to_num(z[k], nan=0.) for k in ('open', 'high', 'low', 'close', 'volume')], -1)
    X = compute_features(ohlcv, observed, beta, sectors, eligible)
    ohlcv_raw = ohlcv.copy()
    ohlcv_raw[..., 4] = np.nan_to_num(z['raw_volume'], nan=0.)
    ldv, ami = corrected_columns(ohlcv_raw, observed, eligible, z['raw_close'])     # raw close x raw volume
    X[..., FEATURES.index('log_dvol_20')] = ldv
    X[..., FEATURES.index('amihud_20')] = ami
    y, ok = forward_labels(close, observed, 5)
    fb = SimpleNamespace(X=X, y=y, y_ok=ok, eligible=eligible, beta=beta, sectors=sectors, dates=dates, horizon=5)
    d = SimpleNamespace(close=np.where(observed, close, np.nan), open=np.where(observed, open_, np.nan),
                        price=close, observed=observed, eligible=eligible, beta=beta, sectors=sectors, dates=dates,
                        labels=z['labels'], survivor=z['survivor_index'] >= 0, failure=z['failure'], last=z['last'],
                        member=member)
    return d, fb


def restrict(d, fb, keep):
    """Copies of (d, fb) whose eligibility is limited to names in `keep` [N] (arm S)."""
    d2, fb2 = SimpleNamespace(**vars(d)), SimpleNamespace(**vars(fb))
    d2.eligible = d.eligible & keep[:, None]
    fb2.eligible = fb.eligible & keep[:, None]
    return d2, fb2


def signal_days(dates, start=START, end_year=2023):
    s0 = int(np.searchsorted(dates, start))
    s = np.arange(s0, len(dates) - 1, 5)
    return s[dates[s] <= np.datetime64(f'{end_year}-12-31')]


def train_signals(fb, asof, minimum=40):
    t = np.arange(HISTORY - 1, asof - fb.horizon + 1)
    t = t[(fb.y_ok[:, t] & fb.eligible[:, t]).sum(0) >= minimum]
    if len(t) and t.max() + fb.horizon > asof:
        raise ValueError('training labels must mature on or before the as-of date')
    return t


def annual_scores(fb, model, years=range(2018, 2024), seeds=(7, 19, 42), rounds=50, alpha=1.0, threads=8):
    """{execution day: score [N]} (NaN where ineligible at the signal close); rank-average over seeds."""
    sig = signal_days(fb.dates, end_year=max(years))
    out = {}
    for yr in years:
        sy = sig[fb.dates[sig].astype('datetime64[Y]') == np.datetime64(str(yr), 'Y')]
        if not len(sy):
            continue
        tr = train_signals(fb, sy[0])
        Xtr = nested_inputs(fb, tr, 'style')
        ytr, mtr = target_values(fb, tr, 'residual')
        mtr = mtr & fb.eligible[:, tr].T
        Xev = nested_inputs(fb, sy, 'style')
        e = fb.eligible[:, sy].T
        runs = [fit_fixed('lgbm', Xtr, ytr, mtr, Xev, {'n_estimators': rounds}, seed=s, threads=threads) for s in seeds] \
            if model == 'lgbm' else [fit_fixed('ridge', Xtr, ytr, mtr, Xev, {'alpha': alpha})]
        ranks = []
        for p in runs:
            r = np.full(p.shape, np.nan)
            for k in range(len(sy)):
                m = e[k]
                r[k, m] = np.argsort(np.argsort(p[k, m])) / max(m.sum() - 1, 1)
            ranks.append(r)
        avg = np.mean(ranks, 0)
        for k, s in enumerate(sy):
            out[int(s) + 1] = avg[k]
    return out
