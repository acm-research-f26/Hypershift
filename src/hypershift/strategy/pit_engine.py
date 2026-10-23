"""Point-in-time variant of `strategy.engine.run` (engine.py is frozen by the paper-trading manifest).

Differences, both needed once eligibility follows index membership (docs/strategy_pit_20261023/PREREG.md):
- entry needs eligibility at t-1; exit needs only an observed bar and a finite price at t, so a held name that left
  the index is sold at the next rebalance instead of being kept as an unhedged stub;
- a held name whose bars end for good (delisting) is liquidated at its last close plus `terminal[i]` (0 by default)
  and the proceeds go to cash. `delisted_weight` reports the |weight| liquidated each day.
With every name always eligible and observed this reproduces engine.run exactly.
"""
import numpy as np

from hypershift.strategy.engine import _drift, _lo_targets, _ret, ls_weights


def last_bars(observed):
    """Index of each name's last observed bar (-1 if never observed)."""
    T = observed.shape[1]
    rev = observed[:, ::-1].argmax(1)
    return np.where(observed.any(1), T - 1 - rev, -1)


def run_pit(scores, data, mode='LO', top=.2, band=.4, execution='open', cost_bp=10., borrow_bp=50., end=None,
            equal=False, trace=False, terminal=None):
    T = data.close.shape[1]
    N = data.close.shape[0]
    days = sorted(scores)
    t0, t1 = days[0], (T - 1 if end is None else end)
    last = last_bars(data.observed)
    term = np.zeros(N) if terminal is None else np.asarray(terminal, float)
    w = np.zeros(N)
    held_l, held_s = np.zeros(N, bool), np.zeros(N, bool)
    out = {k: [] for k in ('gross', 'traded', 'borrow', 'net', 'delisted_weight')}
    dates, ws, rs = [], [], []
    for t in range(t0, t1 + 1):
        ok = data.observed[:, t] & data.observed[:, t - 1]
        gone = (last == t - 1) & (w != 0)
        if trace:
            ws.append(w.copy())
            rs.append(_ret(data.close[:, t - 1], data.close[:, t], ok))
        g_term = float(w[gone] @ term[gone])
        out['delisted_weight'].append(float(np.abs(w[gone]).sum()))
        if gone.any():
            w = w.copy()
            w[gone] = 0.
            w = w / (1 + g_term) if 1 + g_term > 1e-9 else w * 0   # NAV moved by the terminal return
        target = None
        if t in scores:
            px = data.open[:, t] if execution == 'open' else data.close[:, t]
            can = data.observed[:, t] & np.isfinite(px)
            tradable = data.eligible[:, t - 1] & can
            target = w.copy()
            free = 1.0 - float(np.abs(w[~can]).sum()) if mode == 'LO' else 1.0
            target[can] = 0.
            if equal:
                idx = np.flatnonzero(tradable)
                target[idx] = free / max(len(idx), 1)
            elif mode == 'LO':
                pick = _lo_targets(scores[t], tradable, held_l, top, band)
                target[pick] = free / max(len(pick), 1)
                held_l[:] = False
                held_l[pick] = True
            else:
                wl, lo, sh = ls_weights(scores[t], tradable, data.sectors, data.beta[:, t - 1], held_l, held_s, top, band)
                target[tradable] = wl[tradable]
                held_l[:] = False
                held_l[lo] = True
                held_s[:] = False
                held_s[sh] = True
        traded = 0.
        if execution == 'open' and target is not None:
            w, p1 = _drift(w, _ret(data.close[:, t - 1], data.open[:, t], ok))
            traded = float(np.abs(target - w).sum())
            w = target
            w, p2 = _drift(w, _ret(data.open[:, t], data.close[:, t], ok))
            g = (1 + p1) * (1 + p2) - 1
        else:
            w, g = _drift(w, _ret(data.close[:, t - 1], data.close[:, t], ok))
            if target is not None:
                traded = float(np.abs(target - w).sum())
                w = target
        g = (1 + g_term) * (1 + g) - 1 if g_term else g
        borrow = borrow_bp * 1e-4 / 252 * float(-w[w < 0].sum())
        out['gross'].append(g)
        out['traded'].append(traded)
        out['borrow'].append(borrow)
        out['net'].append(g - cost_bp * 1e-4 * traded - borrow)
        dates.append(data.dates[t])
    res = {k: np.array(v) for k, v in out.items()}
    res['dates'] = np.array(dates)
    if trace:
        res['w_start'], res['r'] = np.array(ws), np.array(rs)
    return res


def benchmark_ew(scores, data, execution='open', cost_bp=10., end=None, terminal=None):
    return run_pit(scores, data, mode='LO', execution=execution, cost_bp=cost_bp, borrow_bp=0., end=end, equal=True,
                   terminal=terminal)


def run_targets(targets, data, execution='open', cost_bp=10., borrow_bp=0., end=None, terminal=None):
    """Like run_pit, but each rebalance sets given long-only target weights {t: w [N], sum 1}. Names without a bar or
    price at t keep their current weight; the targets of tradable names are rescaled to the remaining NAV."""
    T, N = data.close.shape[1], data.close.shape[0]
    days = sorted(targets)
    t0, t1 = days[0], (T - 1 if end is None else end)
    last = last_bars(data.observed)
    term = np.zeros(N) if terminal is None else np.asarray(terminal, float)
    w = np.zeros(N)
    out = {k: [] for k in ('gross', 'traded', 'borrow', 'net', 'delisted_weight')}
    dates = []
    for t in range(t0, t1 + 1):
        ok = data.observed[:, t] & data.observed[:, t - 1]
        gone = (last == t - 1) & (w != 0)
        g_term = float(w[gone] @ term[gone])
        out['delisted_weight'].append(float(np.abs(w[gone]).sum()))
        if gone.any():
            w = w.copy()
            w[gone] = 0.
            w = w / (1 + g_term) if 1 + g_term > 1e-9 else w * 0
        target = None
        if t in targets:
            px = data.open[:, t] if execution == 'open' else data.close[:, t]
            can = data.observed[:, t] & np.isfinite(px)
            tw = np.where(can, np.asarray(targets[t], float), 0.)
            target = w.copy()
            target[can] = 0.
            free = 1.0 - float(np.abs(w[~can]).sum())
            if tw.sum() > 0:
                target[can] = tw[can] * free / tw.sum()
        traded = 0.
        if execution == 'open' and target is not None:
            w, p1 = _drift(w, _ret(data.close[:, t - 1], data.open[:, t], ok))
            traded = float(np.abs(target - w).sum())
            w = target
            w, p2 = _drift(w, _ret(data.open[:, t], data.close[:, t], ok))
            g = (1 + p1) * (1 + p2) - 1
        else:
            w, g = _drift(w, _ret(data.close[:, t - 1], data.close[:, t], ok))
            if target is not None:
                traded = float(np.abs(target - w).sum())
                w = target
        g = (1 + g_term) * (1 + g) - 1 if g_term else g
        borrow = borrow_bp * 1e-4 / 252 * float(-w[w < 0].sum())
        out['gross'].append(g)
        out['traded'].append(traded)
        out['borrow'].append(borrow)
        out['net'].append(g - cost_bp * 1e-4 * traded - borrow)
        dates.append(data.dates[t])
    res = {k: np.array(v) for k, v in out.items()}
    res['dates'] = np.array(dates)
    return res
