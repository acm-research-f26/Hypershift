"""Study FB portfolio rules on frozen predictions and subgroup RankIC (descriptive, no retraining)."""
import numpy as np
from scipy.stats import rankdata

from hypershift.featbench.data import FEATURES
from hypershift.screening.metrics import corr

RULES = ('A_top10', 'B_sector_ls', 'C_volscaled', 'D_beta_ls', 'E_buffer')


def _top(score, k):
    return np.argsort(-score, kind='stable')[:k]


def weights_rule(pred, fb, signals, rule, k=10):
    """Per-signal target weights [S, N]; long leg sums to 1, short leg (if any) is negative."""
    S, N = pred.shape
    w = np.zeros((S, N))
    held = np.zeros(0, int)
    for d, t in enumerate(signals):
        e = fb.eligible[:, t]
        score = np.where(e, pred[d], -np.inf)
        idx = np.flatnonzero(e)
        if rule == 'A_top10':
            w[d, _top(score, k)] = 1 / k
        elif rule == 'C_volscaled':
            top = _top(score, k)
            inv = 1 / np.maximum(fb.X[top, t, FEATURES.index('vol_60')], 1e-4)
            w[d, top] = inv / inv.sum()
        elif rule == 'B_sector_ls':
            longs, shorts = [], []
            for g in np.unique(fb.sectors[idx]):
                members = idx[fb.sectors[idx] == g]
                n = max(1, int(round(.2 * len(members))))
                order = members[np.argsort(-pred[d, members], kind='stable')]
                if len(members) >= 2 * n:
                    longs += list(order[:n]); shorts += list(order[-n:])
            w[d, longs] = 1 / len(longs); w[d, shorts] = -1 / len(shorts)
        elif rule == 'D_beta_ls':
            n = max(1, len(idx) // 10)
            order = idx[np.argsort(-pred[d, idx], kind='stable')]
            lo, sh = order[:n], order[-n:]
            b = fb.beta[:, t].astype(np.float64)
            w[d, lo] = 1 / n
            bl, bs = b[lo].mean(), b[sh].mean()
            w[d, sh] = -(bl / bs) / n if abs(bs) > 1e-6 else -1 / n
        elif rule == 'E_buffer':
            rank = np.empty(N); rank[np.argsort(-score, kind='stable')] = np.arange(N)
            keep = [i for i in held if e[i] and rank[i] < 2 * k]
            fill = [i for i in np.argsort(-score, kind='stable') if i not in keep][:k - len(keep)]
            held = np.array(sorted(keep + fill), int)
            w[d, held] = 1 / k
        else:
            raise ValueError(rule)
    return w


def simulate(weights, fb, signals, hold=5, cost=.001):
    """Daily book = sum of the last `hold` sleeve weights / hold (no drift); return t->t+1; cost x turnover."""
    price, seen = fb.price, fb.observed
    book = np.zeros(weights.shape[1])
    gross, net, turns, mkt = [], [], [], []
    for d, t in enumerate(signals):
        new = weights[max(0, d - hold + 1):d + 1].sum(0) / hold
        turn = float(np.abs(new - book).sum())
        book = new
        if t + 1 >= price.shape[1]:
            break
        good = seen[:, t + 1] & seen[:, t]
        r = np.where(good, price[:, t + 1] / np.maximum(price[:, t], 1e-12) - 1, 0)
        g = float(book @ r)
        gross.append(g); net.append(g - cost * turn); turns.append(turn)
        e = fb.eligible[:, t] & good
        mkt.append(float(r[e].mean()) if e.any() else 0.)
    gross, net, mkt = map(np.array, (gross, net, mkt))
    sh = lambda x: float(x.mean() / max(x.std(ddof=1), 1e-12) * np.sqrt(252))
    eq = np.cumprod(1 + net)
    return {'gross_Sharpe': sh(gross), 'net_Sharpe': sh(net), 'ann_return': float(net.mean() * 252),
            'max_drawdown': float((eq / np.maximum.accumulate(eq) - 1).min()), 'turnover': float(np.mean(turns)),
            'beta_to_market': float(np.cov(net, mkt)[0, 1] / max(mkt.var(ddof=1), 1e-12)), 'net': net}


def subgroup_rankic(pred, fb, signals, char, minimum=10):
    """Per-date terciles of a feature known at t; neutral RankIC within each tercile."""
    j = FEATURES.index(char)
    out = {'low': [], 'mid': [], 'high': []}
    for d, t in enumerate(signals):
        m = fb.eligible[:, t] & fb.y_ok[:, t]
        c = fb.X[:, t, j]
        q = rankdata(c[m]) / m.sum()
        ix = np.flatnonzero(m)
        for name, lo, hi in (('low', 0, 1 / 3), ('mid', 1 / 3, 2 / 3), ('high', 2 / 3, 1.01)):
            sel = ix[(q > lo) & (q <= hi)]
            out[name].append(corr(rankdata(pred[d, sel]), rankdata(fb.neutral[sel, t])) if len(sel) >= minimum else 0.)
    return {k: np.array(v) for k, v in out.items()}
