"""Point-in-time S&P 500 panel from fja05680 membership + Alpaca SIP daily bars (docs/strategy_pit_20261023).

Each membership interval (ticker, start, end) is fetched with Alpaca `asof` = its last member day, which maps the
ticker to the company holding it then. Intervals whose bar series have identical daily returns are one entity.
"""
import csv

import numpy as np
import pandas as pd

COMPONENTS = 'data/audit/membership/components.csv'


def load_snapshots(path=COMPONENTS):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    snaps = sorted((np.datetime64(r['date']), frozenset(t.strip() for t in r['tickers'].split(','))) for r in rows)
    return np.array([s[0] for s in snaps]), [s[1] for s in snaps]


def intervals(snap_dates, snaps, lo, hi):
    """Contiguous membership runs [(ticker, start, end_exclusive)] that overlap [lo, hi]; an open run ends at hi+1."""
    k0 = max(int(np.searchsorted(snap_dates, lo, side='right')) - 1, 0)
    stop = hi + np.timedelta64(1, 'D')
    open_, out = {}, []
    for k in range(k0, len(snaps)):
        d = max(snap_dates[k], snap_dates[k0])
        if d > hi:
            break
        for t in list(open_):
            if t not in snaps[k]:
                out.append((t, open_.pop(t), d))
        for t in snaps[k]:
            open_.setdefault(t, d)
    out += [(t, s, stop) for t, s in open_.items()]
    return out


def member_from_intervals(entity_intervals, dates):
    m = np.zeros((len(entity_intervals), len(dates)), bool)
    for i, ivs in enumerate(entity_intervals):
        for s, e in ivs:
            m[i] |= (dates >= s) & (dates < e)
    return m


def group_entities(series, tol=1e-5, min_overlap=20):
    """Group keys whose close series have identical daily returns on their overlap (scale-free)."""
    keys = list(series)
    R = pd.concat({k: series[k].pct_change(fill_method=None) for k in keys}, axis=1).to_numpy().T   # [K, T]
    parent = {k: k for k in keys}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for i, a in enumerate(keys):
        D = np.abs(R[i + 1:] - R[i])
        ok = np.isfinite(D)
        n = ok.sum(1)
        bad = (np.where(ok, D, 0) >= tol).sum(1)          # days that differ
        for j in np.flatnonzero((n >= min_overlap) & (bad <= .01 * n)):
            b = keys[i + 1 + j]
            if root(a) != root(b):
                parent[root(b)] = root(a)
    groups = {}
    for k in keys:
        groups.setdefault(root(k), []).append(k)
    return list(groups.values())


def sector_proxy(ret, observed, ref, window=126):
    """Sector id per name: argmax correlation of its first `window` observed returns with each reference sector's
    equal-weight return (reference = names with ref >= 0, the name itself left out). Same rule for every name."""
    secs = np.unique(ref[ref >= 0])
    r = np.where(observed, ret, 0.)
    out = np.full(len(ret), -1)
    tot = {g: r[ref == g].sum(0) for g in secs}
    cnt = {g: observed[ref == g].sum(0) for g in secs}
    for i in range(len(ret)):
        idx = np.flatnonzero(observed[i])[:window]
        if len(idx) < 20:
            continue
        best = (-2., -1)
        for g in secs:
            own = ref[i] == g
            s = (tot[g][idx] - (r[i, idx] if own else 0)) / np.maximum(cnt[g][idx] - (1 if own else 0), 1)
            c = np.corrcoef(r[i, idx], s)[0, 1]
            if np.isfinite(c) and c > best[0]:
                best = (c, g)
        out[i] = best[1]
    return out


def failure_flags(close, observed, last, lookback=60, ratio=.5):
    """Bars ending after a collapse: last close < ratio x the close `lookback` observed sessions earlier."""
    out = np.zeros(len(close), bool)
    for i, L in enumerate(last):
        idx = np.flatnonzero(observed[i, :L + 1])
        if len(idx) > lookback:
            out[i] = close[i, L] < ratio * close[i, idx[-1 - lookback]]
    return out


def bars_frame(bars):
    """Alpaca v2 bar dicts -> DataFrame o,h,l,c,v indexed by session date."""
    idx = pd.DatetimeIndex([pd.Timestamp(b['t']).tz_convert('UTC').tz_localize(None).normalize() for b in bars])
    return pd.DataFrame({k: [float(b[k]) for b in bars] for k in 'ohlcv'}, index=idx).sort_index()
