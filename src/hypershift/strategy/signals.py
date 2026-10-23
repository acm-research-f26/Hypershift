"""Predetermined price-based signals at a signal close t (returns only; no adjusted-price levels)."""
import numpy as np


def momentum(price, observed, t):
    """12-1 momentum: P(t-21) / P(t-252) - 1 (NaN if either price unobserved)."""
    ok = observed[:, t - 21] & observed[:, t - 252]
    return np.where(ok, price[:, t - 21] / np.maximum(price[:, t - 252], 1e-12) - 1, np.nan)


def _daily(price, observed, a, b):
    """Daily returns for days a..b (inclusive), NaN where either day is unobserved: [N, b-a+1]."""
    r = price[:, a:b + 1] / np.maximum(price[:, a - 1:b], 1e-12) - 1
    return np.where(observed[:, a:b + 1] & observed[:, a - 1:b], r, np.nan)


def lowvol(price, observed, t, window=252, min_obs=200):
    """Minus the std (ddof=1) of daily returns over the last `window` days."""
    r = _daily(price, observed, t - window + 1, t)
    n = np.isfinite(r).sum(1)
    sd = np.nanstd(r, axis=1, ddof=1)
    return np.where(n >= min_obs, -sd, np.nan)


def rr21(price, observed, beta, sectors, eligible, t, k=21):
    """Residual reversal: -(k-day return - beta x EW market, sector-demeaned) / 60-day vol."""
    r = price[:, t] / np.maximum(price[:, t - k], 1e-12) - 1
    m = eligible[:, t] & observed[:, t] & observed[:, t - k]
    out = np.full(len(r), np.nan)
    if m.sum() < 5:
        return out
    res = r - beta[:, t] * r[m].mean()
    for g in np.unique(sectors[m]):
        ix = m & (sectors == g)
        res[ix] -= res[ix].mean()
    vol = np.nanstd(_daily(price, observed, t - 59, t), axis=1, ddof=1)
    out[m] = -res[m] / np.maximum(vol[m], 1e-4)
    return out


def composite(sigs, eligible, clip=3.0, min_avail=2):
    """Mean of per-signal cross-sectional z-scores (over eligible, finite), clipped at +-clip; NaN if < min_avail."""
    zs = []
    for s in sigs:
        z = np.full(len(s), np.nan)
        m = eligible & np.isfinite(s)
        if m.sum() > 1 and s[m].std() > 0:
            z[m] = np.clip((s[m] - s[m].mean()) / s[m].std(), -clip, clip)
        zs.append(z)
    zs = np.array(zs)
    n = np.isfinite(zs).sum(0)
    c = np.nansum(zs, 0) / np.maximum(n, 1)
    return np.where(n >= min_avail, c, np.nan)
