"""Cost-aware market-neutral book for the BC review (predeclared D4): drift, next-close/next-open execution, hedge.

Book: long top 10% / short bottom 10% of each sector (sectors with < 10 eligible stocks skipped), legs 0.5 gross
each, plus an equal-weight basket of eligible stocks sized to zero ex-ante beta (hedge trades are costed).
Weights are fractions of NAV and drift with prices between trades. Net return = gross - c * traded.
"""
import numpy as np

TOP, EXIT = .10, .30


def _legs(score, idx, frac):
    order = idx[np.argsort(-score[idx], kind='stable')]
    n = max(1, int(round(frac * len(idx))))
    return order[:n], order[-n:], order


def book_targets(score, eligible, sectors, beta, held_long=None, held_short=None):
    """Return (pre-hedge weights, hedged weights). Buffer when held_* given: keep held names inside the exit band."""
    N = len(score)
    longs, shorts = [], []
    for g in np.unique(sectors):
        idx = np.flatnonzero((sectors == g) & eligible)
        if len(idx) < 10:
            continue
        lo, sh, order = _legs(score, idx, TOP)
        n = len(lo)
        if held_long is not None:
            band = max(n, int(round(EXIT * len(idx))))
            keep_l = [i for i in order[:band] if held_long[i]][:n]
            keep_s = [i for i in order[-band:] if held_short[i]][:n]
            lo = keep_l + [i for i in order if i not in keep_l and i not in keep_s][:n - len(keep_l)]
            rest = [i for i in order[::-1] if i not in lo and i not in keep_s]
            sh = keep_s + rest[:n - len(keep_s)]
        longs += list(lo); shorts += list(sh)
    pre = np.zeros(N)
    if longs:
        pre[longs] = .5 / len(longs)
        pre[shorts] = -.5 / len(shorts)
    e = np.flatnonzero(eligible)
    hedged = pre.copy()
    bmean = beta[e].mean()
    if len(e) and abs(bmean) > 1e-12:
        hedged[e] -= (pre @ beta) / (len(e) * bmean)
    return pre, hedged


def _drift(w, r):
    pr = float(w @ r)
    return w * (1 + r) / (1 + pr), pr


def simulate_book(pred, fb, signals, schedule, execution, ohlcv=None):
    """pred [S, N] scores at signal closes. Returns gross [S], traded [S], beta_ex_ante [S], weights [S, N]."""
    S, N = pred.shape
    T = fb.price.shape[1]
    w = np.zeros(N)
    held_l, held_s = np.zeros(N, bool), np.zeros(N, bool)
    gross, traded, bex, W = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros((S, N))
    beta_now = np.zeros(N)
    for d, t in enumerate(signals):
        if t + 1 >= T:
            gross, traded, bex, W = gross[:d], traded[:d], bex[:d], W[:d]
            break
        seen = fb.observed[:, t] & fb.observed[:, t + 1]
        rebalance = schedule == 'daily' or d % 5 == 0
        target = None
        if rebalance:
            e = fb.eligible[:, t] & seen
            score = np.where(e, pred[d], -np.inf)
            beta = np.nan_to_num(fb.beta[:, t].astype(np.float64))
            buf = schedule == 'weekly_buffer' and (held_l.any() or held_s.any())
            pre, target = book_targets(np.nan_to_num(score, neginf=-1e18), e, fb.sectors, beta,
                                       held_l if buf else None, held_s if buf else None)
            target = np.where(seen, target, w)        # untradable names keep their current weight
            held_l, held_s, beta_now = pre > 0, pre < 0, beta
        if execution == 'close1':
            r = np.where(seen, fb.price[:, t + 1] / np.maximum(fb.price[:, t], 1e-12) - 1, 0.)
            w, pr = _drift(w, np.nan_to_num(r))
            if target is not None:
                traded[d] = np.abs(target - w).sum(); w = target.copy()
            gross[d] = pr
        else:
            op, cl = ohlcv[:, t + 1, 0], ohlcv[:, t + 1, 3]
            rg = np.where(seen, op / np.maximum(ohlcv[:, t, 3], 1e-12) - 1, 0.)
            ri = np.where(seen, cl / np.maximum(op, 1e-12) - 1, 0.)
            w, p1 = _drift(w, np.nan_to_num(rg))
            if target is not None:
                traded[d] = np.abs(target - w).sum(); w = target.copy()
            w, p2 = _drift(w, np.nan_to_num(ri))
            gross[d] = (1 + p1) * (1 + p2) - 1
        bex[d] = w @ beta_now
        W[d] = w
    return {'gross': gross, 'traded': traded, 'beta_ex_ante': bex, 'weights': W}
