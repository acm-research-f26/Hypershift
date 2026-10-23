"""Filing-text changer avoidance (docs/strategy_lazyprices_20261023/PREREG.md): signal selection at monthly
formation dates, within-form bottom-quintile changers, and FF12-industry-neutral avoidance weights."""
import numpy as np


def formation_days(cal, start, end):
    """Indices of the first trading day of each month with start <= day <= end."""
    m = cal.astype('datetime64[M]')
    first = np.flatnonzero(np.r_[True, m[1:] != m[:-1]])
    return first[(cal[first] >= start) & (cal[first] <= end)]


def signals_at(rows, n, F, cal, months=3):
    """Latest signal per entity at formation index F. rows: (entity, available_index, similarity, form).
    Usable if available on or before F-1 and available no earlier than the first day of the month `months` before F."""
    floor = (cal[F].astype('datetime64[M]') - months).astype('datetime64[D]')
    sim = np.full(n, np.nan)
    form = np.full(n, '', dtype='<U4')
    best = np.full(n, -1)
    for e, a, s, f in rows:
        if a <= F - 1 and cal[a] >= floor and a > best[e] and np.isfinite(s):
            best[e], sim[e], form[e] = a, s, f
    return sim, form


def changers(sim, form, eligible, frac=.2):
    """Bottom `frac` (round(frac x n) lowest similarities) among eligible names with a signal, within each form."""
    out = np.zeros(len(sim), bool)
    for f in ('10-K', '10-Q'):
        idx = np.flatnonzero(eligible & (form == f) & np.isfinite(sim))
        k = int(round(frac * len(idx)))
        if k:
            out[idx[np.argsort(sim[idx], kind='stable')[:k]]] = True
    return out


def avoid_weights(eligible, changer, industry):
    """(benchmark B, strategy A): B = equal weight of eligible names; A moves each changer's weight pro rata to the
    non-changers of its industry (an industry with no non-changer keeps its changers)."""
    b = eligible / max(eligible.sum(), 1)
    a = b.copy()
    for g in np.unique(industry[eligible]):
        ix = eligible & (industry == g)
        keep = ix & ~changer
        if keep.any():
            a[ix & changer] = 0.
            a[keep] = b[keep] * b[ix].sum() / b[keep].sum()
    return b, a
