"""Shared construction for the EAR pre-checks and the single evaluation (docs/strategy_ear_20261023/PREREG.md)."""
import json
from pathlib import Path

import numpy as np

from hypershift.data import edgar as ED
from hypershift.strategy import ear as EA, lazyprices as LP

START, END = np.datetime64('2016-07-01'), np.datetime64('2023-12-31')
SPLICE = {'IR': np.datetime64('2020-03-03')}     # entity priced as Gardner Denver before this date (DATA_CHECKS.md)


def eligibility(d):
    e = d.eligible.copy()
    labels = list(d.labels)
    for lab, until in SPLICE.items():
        if lab in labels:
            e[labels.index(lab), d.dates < until] = False
    return e


def signal_rows(d, window):
    """(entity, available_index, market-adjusted window return, 'EAR') per Item 2.02 event; available after d0+1."""
    ev = json.loads(Path('data/edgar/ear_events.json').read_text())
    labels = list(d.labels)
    close = np.nan_to_num(d.close, nan=1.)
    m = EA.market_returns(close, d.observed, d.member)
    rows = []
    for lab, es in ev.items():
        if lab not in labels:
            continue
        i = labels.index(lab)
        for e in es:
            d0 = e['d0']
            if d0 + 1 >= len(d.dates):
                continue
            v = EA.window_ar(close, d.observed, d.member, i, d0, window, m)
            rows.append((i, d0 + 1, v, 'EAR'))
    return rows


def industries(d, F):
    """FF12 from the SIC in the EDGAR header of the entity's CIK, known from that header's filing date on; else 12."""
    cmap = json.loads(Path('data/edgar/cik_map.json').read_text())
    sic = json.loads(Path('data/edgar/sic_hist.json').read_text())
    day = d.dates[F - 1]
    out = np.full(len(d.labels), 12)
    for i, lab in enumerate(d.labels):
        for s, e, c in cmap.get(str(lab), {}).get('segments', []):
            h = sic.get(str(c))
            if h and h[0] and np.datetime64(h[1]) <= day and np.datetime64(s) - np.timedelta64(120, 'D') <= day:
                out[i] = ED.ff12(h[0])
    return out


def targets(d, rows, months=4, cap=2.):
    """{F: (B, A, bottom, top, coverage)}: each event's bin is fixed at its availability against trailing breakpoints
    (`ear.event_bins`), the latest valid event per name counts, FF12-neutral tilt capped at `cap` x B weight."""
    elig = eligibility(d)
    binned = EA.event_bins(rows)
    out = {}
    for F in LP.formation_days(d.dates, START, END):
        e = elig[:, F - 1]
        sig, _ = LP.signals_at(binned, len(d.labels), F, d.dates, months=months)
        bot, top = e & (sig == -1), e & (sig == 1)
        b, a = EA.tilt_weights(e, bot, top, industries(d, F), cap=cap)
        out[int(F)] = (b, a, bot, top, float(np.isfinite(sig[e]).mean()))
    return out
