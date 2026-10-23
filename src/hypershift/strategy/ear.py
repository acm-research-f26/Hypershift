"""Earnings-announcement-return (EAR) overlay (docs/strategy_ear_20261023/PREREG.md).

Events: original 8-K filings with Item 2.02, dated by SEC's official filing date (the clock part of the submissions-
JSON acceptance time is unreliable; TIMING_AUDIT.md). Signal: market-adjusted return over trading days d0-1..d0+1
around the filing day d0, usable only after the close of d0+1. Matched control: the same statistic over the three
sessions before the window (d0-4..d0-2), with identical availability, expiry and portfolio rules.
"""
import numpy as np

from hypershift.data import edgar as ED


def event_day(filing_date, cal):
    """Trading-day index of the official filing date (the next trading day if the market was closed)."""
    return int(np.searchsorted(cal, np.datetime64(filing_date, 'D'), side='left'))


def events(block, cal, collapse=5):
    """Original 8-K filings with Item 2.02 from a submissions-JSON block; filings within `collapse` trading days of a
    kept event are the same announcement (first one kept)."""
    out = []
    n = len(block['form'])
    for k in range(n):
        if block['form'][k] != '8-K':
            continue
        items = {x.strip() for x in str(block.get('items', [''] * n)[k] or '').split(',')}
        if '2.02' not in items:
            continue
        d0 = event_day(block['filingDate'][k], cal)
        out.append({'accession': block['accessionNumber'][k], 'filingDate': block['filingDate'][k],
                    'acceptance_et': ED.acceptance_et(block['acceptanceDateTime'][k]), 'd0': d0})
    out.sort(key=lambda e: (e['d0'], e['acceptance_et']))
    kept = []
    for e in out:
        if kept and e['d0'] - kept[-1]['d0'] < collapse:
            continue
        kept.append(e)
    return kept


def market_returns(close, observed, member):
    """Equal-weight daily return of members observed on both days."""
    both = observed[:, 1:] & observed[:, :-1]
    r = np.where(both, close[:, 1:] / np.where(both, close[:, :-1], 1.) - 1, 0.)
    w = both & member[:, 1:]
    m = np.zeros(close.shape[1])
    m[1:] = (r * w).sum(0) / np.maximum(w.sum(0), 1)
    return m


def window_ar(close, observed, member, i, d0, win, m=None):
    """Sum of (r_i - r_market) over trading days d0+win[0] .. d0+win[1]; NaN unless every bar is observed."""
    lo, hi = d0 + win[0], d0 + win[1]
    if lo < 1 or hi >= close.shape[1] or not observed[i, lo - 1:hi + 1].all():
        return np.nan
    if m is None:
        m = market_returns(close, observed, member)
    r = close[i, lo:hi + 1] / close[i, lo - 1:hi] - 1
    return float((r - m[lo:hi + 1]).sum())


def rank_bins(sig, eligible, frac=.2):
    """(bottom, top): the round(frac x n) lowest and highest signals among eligible names with a signal."""
    idx = np.flatnonzero(eligible & np.isfinite(sig))
    k = int(round(frac * len(idx)))
    bot, top = np.zeros(len(sig), bool), np.zeros(len(sig), bool)
    if k:
        bot[idx[np.argsort(sig[idx], kind='stable')[:k]]] = True
        top[idx[np.argsort(-sig[idx], kind='stable')[:k]]] = True
    return bot, top


def event_bins(rows, lookback=252, min_n=100, lo=.2, hi=.8):
    """Replace each row's value by its bin at availability: -1 (<= the lo quantile), +1 (>= the hi quantile) or 0,
    with breakpoints from all finite values that became available in the `lookback` trading days before it (causal;
    NaN if fewer than `min_n`). rows: (entity, available_index, value, tag)."""
    a = np.array([r[1] for r in rows])
    v = np.array([r[2] for r in rows], float)
    order = np.argsort(a, kind='stable')
    fa, fv = a[order], v[order]
    ok = np.isfinite(fv)
    out = []
    for e, ai, vi, tag in rows:
        if not np.isfinite(vi):
            out.append((e, ai, np.nan, tag))
            continue
        k0, k1 = np.searchsorted(fa, ai - lookback, 'left'), np.searchsorted(fa, ai, 'left')
        prior = fv[k0:k1][ok[k0:k1]]
        if len(prior) < min_n:
            out.append((e, ai, np.nan, tag))
            continue
        ql, qh = np.quantile(prior, [lo, hi])
        out.append((e, ai, -1. if vi <= ql else 1. if vi >= qh else 0., tag))
    return out


def tilt_weights(eligible, bottom, top, industry, cap=None):
    """(B, A): B equal weight. In each industry the bottom names' weight moves pro rata to its top names, then to its
    other non-bottom names (if no top name, or once top names reach `cap` x their B weight); whatever cannot move
    under the cap stays on the bottom names pro rata. Industry weights equal B's."""
    b = eligible / max(eligible.sum(), 1)
    a = b.copy()
    for g in np.unique(industry[eligible]):
        ix = eligible & (industry == g)
        bw = b[ix & bottom].sum()
        if bw <= 0:
            continue
        left = bw
        for rec in (ix & top, ix & ~bottom & ~top):     # top names first, then the other non-bottom names
            for _ in range(int(rec.sum()) + 1):          # pro rata, re-spread whatever the cap stops
                room = np.full(len(b), np.inf) if cap is None else (cap - 1) * b - (a - b)
                r = rec & (room > 1e-15)
                if not r.any() or left <= 1e-15:
                    break
                give = np.minimum(left * b[r] / b[r].sum(), room[r])
                a[r] += give
                left -= give.sum()
        left = left if left > 1e-12 else 0.
        a[ix & bottom] = b[ix & bottom] * left / bw     # what cannot move under the cap stays on the bottom names
    return b, a
