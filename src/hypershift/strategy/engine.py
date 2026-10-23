"""Portfolio engine: long-only (LO) or sector-neutral beta-hedged long-short (LS), buffered, with drift,
next-open or next-close execution, linear costs per unit traded, and short borrow accrual.

data: close/open [N, T] (same per-stock scale), observed, eligible, beta [N, T], sectors [N], dates [T].
scores: {t: [N]} where t is the trading day of execution; scores use information up to the close of t-1.
Weights are fractions of NAV; cash earns 0. Net return = gross - cost_bp*1e-4*traded - borrow.
"""
import numpy as np


def month_starts(dates):
    m = dates.astype('datetime64[M]')
    return np.flatnonzero(np.r_[False, m[1:] != m[:-1]])


def _ret(a, b, ok):
    good = ok & np.isfinite(a) & np.isfinite(b) & (np.abs(np.nan_to_num(a)) > 0)
    return np.where(good, np.nan_to_num(b) / np.where(good, a, 1.) - 1, 0.)


def _drift(w, r):
    pr = float(w @ r)
    return w * (1 + r) / (1 + pr), pr


def _lo_targets(score, tradable, held, top, band):
    idx = np.flatnonzero(tradable & np.isfinite(score))
    order = idx[np.argsort(-score[idx], kind='stable')]
    n = max(1, int(round(top * len(idx))))
    bn = max(n, int(round(band * len(idx))))
    keep = [i for i in order[:bn] if held[i]][:n]
    kept = set(keep)
    pick = keep + [i for i in order if i not in kept][:n - len(keep)]
    return np.array(pick, int)


def _ls_targets(score, tradable, sectors, held_l, held_s, top, band):
    longs, shorts = [], []
    for g in np.unique(sectors):
        idx = np.flatnonzero(tradable & np.isfinite(score) & (sectors == g))
        if len(idx) < 10:
            continue
        order = idx[np.argsort(-score[idx], kind='stable')]
        n = max(1, int(round(top * len(idx))))
        bn = max(n, int(round(band * len(idx))))
        kl = [i for i in order[:bn] if held_l[i]][:n]
        ks = [i for i in order[::-1][:bn] if held_s[i]][:n]
        taken = set(kl) | set(ks)
        lo = kl + [i for i in order if i not in taken][:n - len(kl)]
        taken |= set(lo)
        sh = ks + [i for i in order[::-1] if i not in taken][:n - len(ks)]
        longs += lo
        shorts += sh
    return np.array(longs, int), np.array(shorts, int)


def run(scores, data, mode='LO', top=.2, band=.4, execution='open', cost_bp=10., borrow_bp=50., end=None, equal=False):
    T = data.close.shape[1]
    days = sorted(scores)
    t0, t1 = days[0], (T - 1 if end is None else end)
    N = data.close.shape[0]
    w = np.zeros(N)
    held_l, held_s = np.zeros(N, bool), np.zeros(N, bool)
    out = {k: [] for k in ('gross', 'traded', 'borrow', 'net')}
    dates = []
    for t in range(t0, t1 + 1):
        ok = data.observed[:, t] & data.observed[:, t - 1]
        target = None
        if t in scores:
            px = data.open[:, t] if execution == 'open' else data.close[:, t]
            tradable = data.eligible[:, t - 1] & data.observed[:, t] & np.isfinite(px)
            target = w.copy()
            free = 1.0 - float(np.abs(w[~tradable]).sum()) if mode == 'LO' else 1.0
            target[tradable] = 0.
            if equal:
                idx = np.flatnonzero(tradable)
                target[idx] = free / max(len(idx), 1)
            elif mode == 'LO':
                pick = _lo_targets(scores[t], tradable, held_l, top, band)
                target[pick] = free / max(len(pick), 1)
                held_l[:] = False
                held_l[pick] = True
            else:
                lo, sh = _ls_targets(scores[t], tradable, data.sectors, held_l, held_s, top, band)
                if len(lo) and len(sh):
                    target[lo] = .5 / len(lo)
                    target[sh] = -.5 / len(sh)
                    e = np.flatnonzero(tradable)
                    b = np.nan_to_num(data.beta[:, t - 1])
                    bm = b[e].mean()
                    if abs(bm) > 1e-12:
                        target[e] -= (target @ b) / (len(e) * bm)
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
        borrow = borrow_bp * 1e-4 / 252 * float(-w[w < 0].sum())
        out['gross'].append(g)
        out['traded'].append(traded)
        out['borrow'].append(borrow)
        out['net'].append(g - cost_bp * 1e-4 * traded - borrow)
        dates.append(data.dates[t])
    res = {k: np.array(v) for k, v in out.items()}
    res['dates'] = np.array(dates)
    return res


def benchmark_ew(scores, data, execution='open', cost_bp=10., end=None):
    """Equal weight of all tradable eligible stocks, rebalanced on the same days as `scores`."""
    return run(scores, data, mode='LO', execution=execution, cost_bp=cost_bp, borrow_bp=0., end=end, equal=True)
