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
