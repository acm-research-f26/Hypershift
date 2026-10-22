"""Leak-free dollar-volume features: unadjusted Close x Volume instead of dividend back-adjusted Close x Volume."""
import numpy as np

from hypershift.featbench.data import _roll_mean


def corrected_columns(ohlcv, observed, eligible, raw_close):
    """Return corrected (log_dvol_20, amihud_20) [N, T]; same formulas as featbench compute_features.

    Returns stay total-return (adjusted closes); only the dollar-volume level uses the unadjusted close."""
    cl, vo = ohlcv[..., 3].astype(np.float64), ohlcv[..., 4].astype(np.float64)
    both = observed.copy(); both[:, 0] = False
    both[:, 1:] &= observed[:, :-1]
    r = np.zeros_like(cl)
    r[:, 1:] = np.where(both[:, 1:], cl[:, 1:] / np.maximum(cl[:, :-1], 1e-12) - 1, 0)
    rc = np.where(np.isfinite(raw_close), raw_close, cl)          # fall back to adjusted where raw is missing
    dvol = rc * vo
    ldv = np.log1p(_roll_mean(dvol, 20))
    ami = _roll_mean(np.abs(r) / (dvol + 1), 20) * 1e9
    clean = lambda x: np.where(eligible & np.isfinite(x), x, 0.).astype(np.float32)
    return clean(ldv), clean(ami)
