"""Causal OHLCV features, horizon-h labels, targets and per-date transforms for Study FB."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from scipy.stats import rankdata

from hypershift.screening.data import forward_labels, load_panel, split_signals, usable_signals
from hypershift.screening.metrics import corr

PANEL = 'data/fresh/sp500_daily/screen_sp500_2015_2026.npz'
OHLCV = 'data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz'
CACHE = Path('results/FEATBENCH_SP500_20261022/_cache')
LAST_DEV_YEAR = 2023  # 2024+ was spent for Study J arms only

FEATURES = ['ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_60', 'mom_20_60', 'vol_5', 'vol_20', 'vol_60',
            'downvol_20', 'mdd_60', 'beta', 'log_dvol_20', 'relvol_5_60', 'volchg_5_20', 'amihud_20',
            'hl_range_20', 'sec_ret_5', 'sec_ret_20']
RETURN_FEATURES = ['ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_60', 'mom_20_60', 'sec_ret_5', 'sec_ret_20']


@dataclass
class FB:
    X: np.ndarray          # [N, T, F] causal features at each close
    y: np.ndarray          # [N, T] h-day forward return from close t
    y_ok: np.ndarray       # [N, T] label observed on every day of the horizon
    eligible: np.ndarray   # [N, T] 64-session eligibility (screening Panel)
    beta: np.ndarray       # [N, T] trailing-126 beta (screening Panel)
    sectors: np.ndarray    # [N] GICS sector ids (current, not point-in-time)
    dates: np.ndarray
    price: np.ndarray
    observed: np.ndarray
    neutral: np.ndarray    # [N, T] y minus beta x market, then sector-demeaned (common primary target)
    train: np.ndarray
    valid: np.ndarray
    evaluate: np.ndarray
    horizon: int
    year: int
    mu: np.ndarray         # train-period feature mean (for the 'raw' transform)
    sd: np.ndarray
    panel: object = None


def _lag(a, k):
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def _roll_mean(a, k):
    c = np.cumsum(np.nan_to_num(a, nan=0.), 1, dtype=np.float64)
    c = np.pad(c, ((0, 0), (1, 0)))
    out = np.full(a.shape, np.nan)
    out[:, k-1:] = (c[:, k:] - c[:, :-k]) / k
    return out


def _sector_mean(v, sectors, eligible):
    out = np.zeros_like(v)
    for g in np.unique(sectors):
        ix = sectors == g
        w = eligible[ix]
        m = np.where(w, np.nan_to_num(v[ix]), 0).sum(0) / np.maximum(w.sum(0), 1)
        out[ix] = m[None]
    return out


def compute_features(ohlcv, observed, beta, sectors, eligible):
    """[N, T, 20] features; value at t uses only data at or before close t."""
    op, hi, lo, cl, vo = (ohlcv[..., k].astype(np.float64) for k in range(5))
    both = observed.copy(); both[:, 0] = False
    both[:, 1:] &= observed[:, :-1]
    r = np.zeros_like(cl)
    r[:, 1:] = np.where(both[:, 1:], cl[:, 1:] / np.maximum(cl[:, :-1], 1e-12) - 1, 0)
    ret = {k: cl / np.maximum(_lag(cl, k), 1e-12) - 1 for k in (1, 3, 5, 10, 20, 60)}
    f = {f'ret_{k}': v for k, v in ret.items()}
    f['mom_20_60'] = _lag(cl, 20) / np.maximum(_lag(cl, 60), 1e-12) - 1
    for k in (5, 20, 60):
        m1, m2 = _roll_mean(r, k), _roll_mean(r**2, k)
        f[f'vol_{k}'] = np.sqrt(np.maximum(m2 - m1**2, 0))
    f['downvol_20'] = np.sqrt(_roll_mean(np.minimum(r, 0)**2, 20))
    peak = cl.copy()
    for k in range(1, 60):
        peak = np.fmax(peak, _lag(cl, k))
    f['mdd_60'] = cl / np.maximum(peak, 1e-12) - 1
    f['beta'] = beta.astype(np.float64)
    dvol = cl * vo
    f['log_dvol_20'] = np.log1p(_roll_mean(dvol, 20))
    f['relvol_5_60'] = np.log((_roll_mean(vo, 5) + 1) / (_roll_mean(vo, 60) + 1))
    f['volchg_5_20'] = np.log((_roll_mean(vo, 5) + 1) / (_roll_mean(vo, 20) + 1))
    f['amihud_20'] = _roll_mean(np.abs(r) / (dvol + 1), 20) * 1e9
    f['hl_range_20'] = _roll_mean((hi - lo) / np.maximum(cl, 1e-12), 20)
    for k in (5, 20):
        f[f'sec_ret_{k}'] = ret[k] - _sector_mean(ret[k], sectors, eligible)
    x = np.stack([f[name] for name in FEATURES], -1)
    x = np.where(np.isfinite(x), x, 0.)
    return np.where(eligible[..., None], x, 0.).astype(np.float32)


def _neutralise(y, mask, beta, sectors):
    """y [S, N] -> minus beta x masked-mean market, then masked sector demeaning (Panel.target('sector'))."""
    y = y.copy()
    market = (y * mask).sum(1) / np.maximum(mask.sum(1), 1)
    y -= beta * market[:, None]
    for g in np.unique(sectors):
        ix = sectors == g
        res = (y[:, ix] * mask[:, ix]).sum(1) / np.maximum(mask[:, ix].sum(1), 1)
        y[:, ix] -= res[:, None]
    return y


def _source_hash():
    h = hashlib.sha256(Path(__file__).read_bytes())
    for p in (PANEL, OHLCV):
        h.update(str(Path(p).stat().st_size).encode())
    return h.hexdigest()[:12]


def load_fb(year, horizon=5, cache=True):
    if year > LAST_DEV_YEAR:
        raise ValueError(f'Study FB never scores {year}: 2024+ is spent for Study J arms only')
    panel = load_panel(PANEL, year, train_stride=1)
    o = np.load(OHLCV)
    z = np.load(PANEL)
    ident = z['identity_ok'].astype(bool)
    keep = z['dates'].astype('datetime64[D]') < np.datetime64(f'{year+1}-01-01')
    assert (o['tickers'][ident] == panel.tickers).all()
    path = CACHE / f'X_{year}_{_source_hash()}.npy'
    if cache and path.exists():
        X = np.load(path)
    else:
        X = compute_features(o['ohlcv'][ident][:, keep], panel.observed, panel.beta, panel.sectors, panel.eligible)
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True); np.save(path, X)
    y, ok = forward_labels(panel.price, panel.observed, horizon)
    tr, va, ev = split_signals(panel.dates, year, horizon=horizon, train_stride=1)
    usable = panel.eligible & ok
    tr, va, ev = [usable_signals(s, usable) for s in (tr, va, ev)]
    mask = (ok & panel.eligible).T
    neutral = _neutralise(y.T, mask, panel.beta.T, panel.sectors).T
    xs = X[:, tr][panel.eligible[:, tr]]
    return FB(X=X, y=y, y_ok=ok, eligible=panel.eligible, beta=panel.beta, sectors=panel.sectors,
              dates=panel.dates, price=panel.price, observed=panel.observed, neutral=neutral.astype(np.float32),
              train=tr, valid=va, evaluate=ev, horizon=horizon, year=year,
              mu=xs.mean(0), sd=np.maximum(xs.std(0), 1e-8), panel=panel)


def target(fb, signals, kind):
    y = fb.y[:, signals].T.astype(np.float64)
    mask = (fb.y_ok[:, signals] & fb.eligible[:, signals]).T
    if kind == 'raw':
        return np.where(mask, y, 0.), mask
    if kind == 'sector':
        out = y.copy()
        for g in np.unique(fb.sectors):
            ix = fb.sectors == g
            out[:, ix] -= ((y[:, ix] * mask[:, ix]).sum(1) / np.maximum(mask[:, ix].sum(1), 1))[:, None]
        return np.where(mask, out, 0.), mask
    if kind == 'residual':
        out = np.zeros_like(y)
        dummies = (fb.sectors[:, None] == np.unique(fb.sectors)[None]).astype(np.float64)
        j = FEATURES.index
        for d, s in enumerate(signals):
            m = mask[d]
            if m.sum() < 30:
                continue
            style = [fb.beta[:, s], fb.X[:, s, j('log_dvol_20')], fb.X[:, s, j('ret_60')], fb.X[:, s, j('vol_60')]]
            style = [(v[m] - v[m].mean()) / max(v[m].std(), 1e-12) for v in style]
            A = np.column_stack([dummies[m]] + style)
            coef, *_ = np.linalg.lstsq(A, y[d, m], rcond=None)
            out[d, m] = y[d, m] - A @ coef
        return out, mask
    raise ValueError(kind)


def _cs_z(x, m):
    w = m[..., None]
    n = np.maximum(m.sum(1), 1)[:, None]
    mu = (x * w).sum(1) / n
    sd = np.sqrt((((x - mu[:, None]) * w)**2).sum(1) / n)
    z = (x - mu[:, None]) / np.maximum(sd, 1e-12)[:, None]
    return np.where(w, np.clip(z, -5, 5), 0.)


def transform(xs, mask, kind, fb):
    """xs [S, N, F] -> transformed [S, N, F]; ineligible cells are 0."""
    xs = xs.astype(np.float64)
    if kind == 'raw':
        z = np.clip((xs - fb.mu) / fb.sd, -10, 10)
        return np.where(mask[..., None], z, 0.).astype(np.float32)
    if kind == 'cs_z':
        return _cs_z(xs, mask).astype(np.float32)
    if kind == 'cs_rank':
        out = np.zeros_like(xs)
        for d in range(len(xs)):
            m = mask[d]
            if m.sum() > 1:
                out[d, m] = (rankdata(xs[d, m], axis=0) - 1) / (m.sum() - 1) - .5
        return out.astype(np.float32)
    if kind == 'sector_z':
        y = xs.copy()
        for g in np.unique(fb.sectors):
            ix = fb.sectors == g
            w = mask[:, ix][..., None]
            y[:, ix] -= ((xs[:, ix] * w).sum(1) / np.maximum(w.sum(1), 1))[:, None]
        return _cs_z(y, mask).astype(np.float32)
    if kind == 'volscaled_z':
        y = xs.copy()
        v = np.maximum(xs[..., FEATURES.index('vol_60')], 1e-4)
        for name in RETURN_FEATURES:
            y[..., FEATURES.index(name)] /= v
        return _cs_z(y, mask).astype(np.float32)
    raise ValueError(kind)


def neutral_rankic(pred, neutral, mask):
    out = np.zeros(len(pred))
    for d, (p, y, m) in enumerate(zip(pred, neutral, mask)):
        if m.sum() >= 20:
            out[d] = corr(rankdata(p[m]), rankdata(y[m]))
    return out
