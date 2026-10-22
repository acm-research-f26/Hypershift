"""Ex-ante exposures, prediction neutralisation and shifted-execution labels for the FB audit."""
import numpy as np

from hypershift.featbench.data import FEATURES


def _z(v, m):
    out = np.zeros_like(v, dtype=np.float64)
    if m.sum() > 1:
        out[m] = (v[m] - v[m].mean()) / max(v[m].std(), 1e-12)
    return out


def exposures(fb, signals, kind='full'):
    """[S, N, K] exposures known at the signal close: sector dummies + z-scored style columns over eligible stocks."""
    sec = np.unique(fb.sectors)
    dummies = (fb.sectors[:, None] == sec[None]).astype(np.float64)
    cols = ['beta'] if kind == 'sb' else ['beta', 'log_dvol_20', 'ret_60', 'vol_60']
    out = np.zeros((len(signals), fb.X.shape[0], len(sec) + len(cols)))
    for d, s in enumerate(signals):
        m = fb.eligible[:, s]
        style = [_z(fb.beta[:, s].astype(np.float64) if c == 'beta' else fb.X[:, s, FEATURES.index(c)].astype(np.float64), m)
                 for c in cols]
        out[d] = np.column_stack([dummies] + style)
    return out


def neutralise_pred(pred, expo, mask):
    """Per-date OLS residual of pred on [1, expo] over masked stocks; thin dates returned unchanged."""
    out = np.array(pred, dtype=np.float64, copy=True)
    K = expo.shape[-1]
    for d in range(len(pred)):
        m = mask[d]
        if m.sum() <= K + 1:
            continue
        A = np.column_stack([np.ones(m.sum()), expo[d, m]])
        coef, *_ = np.linalg.lstsq(A, out[d, m], rcond=None)
        out[d, m] = out[d, m] - A @ coef
    return out


def shifted_neutral(fb, signals, mode, ohlcv=None):
    """Labels with delayed execution, neutralised like featbench (beta at t, masked market, sector demeaning).

    close1: close t+1 -> close t+1+h.  open1: open t+1 -> close t+h (needs ohlcv [N, T, 5], any per-stock scale).
    Signals whose label needs an index beyond FB's touched range (min(T-1, max(evaluate)+h)) get keep=False.
    Returns (neutral [S, N], raw [S, N], ok [S, N], keep [S]).
    """
    from hypershift.featbench.data import _neutralise
    h = fb.horizon
    T = fb.price.shape[1]
    limit = min(T - 1, int(np.max(fb.evaluate)) + h)
    need = h + 1 if mode == 'close1' else h
    signals = np.asarray(signals)
    keep = signals + need <= limit
    S, N = len(signals), fb.price.shape[0]
    raw = np.zeros((S, N))
    ok = np.zeros((S, N), bool)
    for d, t in enumerate(signals):
        if not keep[d]:
            continue
        seen = fb.observed[:, t:t + need + 1].all(1) & fb.eligible[:, t]
        if mode == 'close1':
            r = fb.price[:, t + 1 + h] / np.maximum(fb.price[:, t + 1], 1e-12) - 1
        elif mode == 'open1':
            r = ohlcv[:, t + h, 3] / np.maximum(ohlcv[:, t + 1, 0], 1e-12) - 1
        else:
            raise ValueError(mode)
        raw[d] = np.where(seen, r, 0.)
        ok[d] = seen
    neu = _neutralise(raw, ok, fb.beta[:, signals].T.astype(np.float64), fb.sectors)
    return np.where(ok, neu, 0.), raw, ok, keep


def style_r2(pred, expo, mask):
    """Per-date R^2 of prediction ranks on [1, exposures] over masked stocks."""
    from scipy.stats import rankdata
    out = np.full(len(pred), np.nan)
    for d in range(len(pred)):
        m = mask[d]
        if m.sum() <= expo.shape[-1] + 1:
            continue
        y = rankdata(pred[d, m])
        A = np.column_stack([np.ones(m.sum()), expo[d, m]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ coef
        out[d] = 1 - (e @ e) / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return out
