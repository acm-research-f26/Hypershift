"""Date-level ranking and staggered-sleeve exploratory trading metrics."""
import numpy as np
from scipy.stats import rankdata


def corr(a, b):
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-15 else 0.


def rank_metrics(pred, target, mask):
    fields = ['IC', 'RankIC', 'MSE', 'spread10', 'prediction_std', 'unique_fraction']
    fields += [f'{prefix}NDCG{k}' for k in (5, 10, 20) for prefix in ('', 'random_')]
    daily = {k: [] for k in fields}
    for p, y, m in zip(pred, target, mask):
        p, y = p[m], y[m]
        if len(y) < 20 or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite predictions or fewer than 20 eligible stocks')
        # Stable fixed tie rule; explicitly retain tie diagnostics.
        order = np.argsort(-p, kind='stable')
        relevance = (rankdata(y)-1) / (len(y)-1)
        daily['IC'].append(corr(p, y))
        daily['RankIC'].append(corr(rankdata(p), rankdata(y)))
        daily['MSE'].append(float(np.mean((p-y)**2)))
        daily['spread10'].append(float(y[order[:10]].mean()-y[order[-10:]].mean()))
        daily['prediction_std'].append(float(p.std()))
        daily['unique_fraction'].append(float(len(np.unique(p))/len(p)))
        for k in (5, 10, 20):
            discount = 1 / np.log2(np.arange(k)+2)
            ideal = np.sort(relevance)[-k:][::-1] @ discount
            daily[f'NDCG{k}'].append(float(relevance[order[:k]] @ discount / ideal))
            daily[f'random_NDCG{k}'].append(float(relevance.mean()*discount.sum()/ideal))
    daily = {k: np.array(v) for k, v in daily.items()}
    return {k: float(v.mean()) for k, v in daily.items()}, daily


def trading(pred, signals, panel, k=10, cost=.001):
    # Five staggered sleeves; enter at signal close, hold five closes.
    # Eligibility is known at entry; future missing bars never used to choose assets.
    # Missing held-stock returns are zero-marked and counted, NOT treated as delisting payouts.
    start, end = int(signals[0]), int(signals[-1])+5
    prices, seen = panel.price, panel.observed
    holdings = np.zeros((5, len(prices)))  # Dollar values, NOT normalized weights.
    cash = np.full(5, .2)
    previous_nav = 1.
    gross, net, turns, market = [], [], [], []
    missing_marks = 0
    sigmap = {int(t): i for i, t in enumerate(signals)}
    for t in range(start, end+1):
        if t > start:
            good = seen[:, t] & seen[:, t-1]
            r = np.where(good, prices[:, t]/np.maximum(prices[:, t-1], 1e-12)-1, 0)
            w = holdings.sum(0)
            missing_marks += int(((w>0) & ~good).sum())
            # Each sleeve retains its own wealth; no implicit cross-sleeve transfer.
            holdings *= 1+r
            market.append(float(r[good].mean()) if good.any() else 0.)
        else:
            market.append(0.)
        before_trade = float(holdings.sum()+cash.sum())
        turnover = 0.
        if t in sigmap:
            i = sigmap[t]
            eligible = panel.eligible[:, t]
            score = np.where(eligible, pred[i], -np.inf)
            chosen = np.argsort(-score, kind='stable')[:k]
            slot = t % 5
            budget = float(holdings[slot].sum()+cash[slot])
            weights = np.zeros(len(prices)); weights[chosen] = 1/k
            amount = budget
            # Solve net investment plus both-side turnover costs = sleeve NAV.
            for _ in range(20):
                turnover = float(np.abs(amount*weights-holdings[slot]).sum())
                amount = budget-cost*turnover
            holdings[slot] = amount*weights
            cash[slot] = 0.
        elif t > int(signals[-1]):
            # Mature each last signal's five-day position and liquidate at close.
            slot = t % 5
            turnover = float(holdings[slot].sum())
            cash[slot] += turnover*(1-cost)
            holdings[slot] = 0
        after_trade = float(holdings.sum()+cash.sum())
        gross.append(before_trade/previous_nav-1)
        net.append(after_trade/previous_nav-1)
        turns.append(turnover/previous_nav)
        previous_nav = after_trade
    a, b = np.array(gross), np.array(net)
    equity = np.cumprod(1+b)
    sharpe = lambda x: float(x.mean()/max(x.std(ddof=1), 1e-12)*np.sqrt(252))
    return {'gross_Sharpe': sharpe(a), 'net_Sharpe': sharpe(b),
            'CAGR': float(equity[-1]**(252/len(b))-1),
            'max_drawdown': float((equity/np.maximum.accumulate(np.r_[1., equity])[1:]-1).min()),
            'turnover': float(np.mean(turns)), 'missing_held_marks': missing_marks,
            'market_correlation': corr(b, np.array(market))}, b
