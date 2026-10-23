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


FRACTIONS = (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2., 3., 4., 5., 10.)   # 2/3 dropped: SIVB/SEDG crashes sat there


def clean_returns(raw_c, raw_o, all_c, observed, max_adj=.15, gap_min=.3, tol=.01):
    """Daily total returns from RAW bars, using Alpaca's adjustment only where it is plausible.

    Alpaca records some spin-offs as reverse splits (TGNA 2017, WRK 2016, UTX 2020, DLPH 2017), so its adjusted
    series can show -80% days. Rules per stock-day (PREREG amendment A1):
    - no adjustment change: raw close-to-close return ('raw');
    - adjustment change whose adjusted return is within +-max_adj: Alpaca's adjusted return ('adjusted';
      dividends, consistent splits);
    - adjustment change of more than 5% whose adjusted return is beyond +-max_adj or larger in size than the raw
      return (an adjustment without a matching price event, e.g. IR 2020-03-02), or no adjustment change but a raw
      overnight gap beyond gap_min that equals a simple share ratio within tol (unrecorded split or distribution):
      zero return ('neutralised': the event is treated as value-neutral; the day's own bars can be spliced,
      e.g. ARNC 2020-04-01, so its open-to-close return is not trusted either).
    Returns r [N, T] (0 where not observed on both days) and kind [N, T]."""
    N, T = raw_c.shape
    r = np.zeros((N, T))
    kind = np.full((N, T), 'missing', dtype='<U11')
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = raw_c[:, 1:] / raw_c[:, :-1]
        F = all_c / raw_c
        dF = F[:, 1:] / F[:, :-1]
        ar = rr * dF - 1
        intra = raw_c[:, 1:] / raw_o[:, 1:] - 1
        gap = raw_o[:, 1:] / raw_c[:, :-1]
    flagged = np.abs(dF - 1) > 1e-6
    out = np.where(flagged, ar, rr - 1)
    k = np.where(flagged, 'adjusted', 'raw')
    bad = (np.abs(dF - 1) > .05) & ((np.abs(ar) > max_adj) | (np.abs(ar) > np.abs(rr - 1)))
    near = np.zeros_like(bad)
    for f in FRACTIONS:
        near |= np.abs(gap / f - 1) < tol
    unflagged = ~flagged & near & (np.abs(gap - 1) > gap_min)
    neut = bad | unflagged
    out = np.where(neut, 0., out)
    k = np.where(neut, 'neutralised', k)
    ok = observed[:, 1:] & observed[:, :-1] & np.isfinite(out)
    r[:, 1:] = np.where(ok, out, 0.)
    kind[:, 1:] = np.where(ok, k, 'missing')
    return r, kind
