"""BC review diagnostics: residual reversal (predeclared D1) and the nonlinear style-residual metric (D2)."""
import numpy as np
from scipy.stats import rankdata

from hypershift.featbench.data import FEATURES, _neutralise
from hypershift.fbaudit.diag import neutralise_pred

STYLES = ('beta', 'log_dvol_20', 'ret_60', 'vol_60')


def residual_reversal(fb, signals, k=21):
    """RR = -(past k-day return minus beta x EW market, sector-demeaned over eligible) / vol_60, at each signal."""
    signals = np.asarray(signals)
    r = (fb.price[:, signals] / np.maximum(fb.price[:, signals - k], 1e-12) - 1).T
    m = (fb.eligible[:, signals] & fb.observed[:, signals] & fb.observed[:, signals - k]).T
    n = _neutralise(np.where(m, r, 0.), m, fb.beta[:, signals].T.astype(np.float64), fb.sectors)
    vol = np.maximum(fb.X[:, signals, FEATURES.index('vol_60')].T.astype(np.float64), 1e-4)
    return np.where(m, -n / vol, 0.)


def _style(fb, s, name):
    return fb.beta[:, s].astype(np.float64) if name == 'beta' else fb.X[:, s, FEATURES.index(name)].astype(np.float64)


def quintile_dummies(fb, signals):
    """[S, N, n_sectors + 16]: sector dummies + quintile 2..5 dummies of each style over eligible stocks."""
    sec = np.unique(fb.sectors)
    dummies = (fb.sectors[:, None] == sec[None]).astype(np.float64)
    out = np.zeros((len(signals), fb.X.shape[0], len(sec) + 4 * len(STYLES)))
    out[:, :, :len(sec)] = dummies[None]
    for d, s in enumerate(signals):
        e = fb.eligible[:, s]
        for j, name in enumerate(STYLES):
            v = _style(fb, s, name)[e]
            if len(np.unique(v)) < 2:
                continue
            q = np.minimum((rankdata(v) - 1) / len(v) * 5, 4.999).astype(int)
            col = len(sec) + 4 * j
            for b in range(1, 5):
                out[d, e, col + b - 1] = (q == b)
    return out


def nl_resid(y, fb, signals, mask):
    """Per-date OLS residual of y on [1, sector dummies, style quintile dummies]."""
    return neutralise_pred(y, quintile_dummies(fb, signals), mask)
