"""Quants.md-inspired causal adaptations to the existing NDX portfolio.

These are adaptations, not reproductions of the site's multi-asset examples.
All market features stop at the known signal close; execution is next close.
"""
import numpy as np
import pandas as pd

from ndx_drawdown_engine import ISSUERS, capped_weights, issuer_weights
from ndx_retention import hedged_targets


RECIPES = [
    dict(name='baseline'),
    dict(name='inverse_vol63', sizing='invvol'),
    dict(name='capped_erc63', sizing='erc'),
    dict(name='momentum126_25', signal='momentum126'),
    dict(name='momentum252_25', signal='momentum252'),
    dict(name='reversal21_25', signal='reversal21'),
    dict(name='trend200_half', off=.5),
    dict(name='trend200_cash', off=0.),
    dict(name='drift_band02', band=.02),
    dict(name='drift_band05', band=.05),
    dict(name='invvol_band02', sizing='invvol', band=.02),
    dict(name='erc_band02', sizing='erc', band=.02),
    dict(name='momentum126_invvol', signal='momentum126', sizing='invvol'),
    dict(name='momentum252_erc', signal='momentum252', sizing='erc'),
]


def equal_risk_weights(covariance):
    """Positive ERC solution by convex risk-budget coordinate descent."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError('Invalid covariance')
    if np.min(np.linalg.eigvalsh(c)) <= 0:
        raise ValueError('Covariance must be positive definite')
    c = c / np.mean(np.diag(c))
    x = 1 / np.sqrt(np.diag(c))
    b = 1 / len(x)
    for _ in range(1000):
        old = x.copy()
        for j in range(len(x)):
            cross = c[j] @ x - c[j, j] * x[j]
            x[j] = 2*b / (cross + np.sqrt(cross*cross + 4*c[j, j]*b))
        if np.max(np.abs(x-old)) < 1e-12:
            break
    w = x / x.sum()
    contribution = w * (c @ w)
    assert np.max(np.abs(contribution / contribution.sum() - b)) < 1e-8
    return w


def market_ranks(data, ids, lookback, reverse=False):
    """Unsupported endpoint histories receive a neutral percentile of 0.5."""
    result = np.full((len(ids), len(data['panel'].tickers)), .5)
    for row, i in enumerate(ids):
        day = int(data['origin'][i])
        if day < lookback:
            continue
        before, now = data['marks'][day-lookback], data['marks'][day]
        valid = (data['mask'][i] & data['tradable'][day-lookback]
                 & data['tradable'][day] & np.isfinite(before) & (before > 0)
                 & np.isfinite(now))
        values = now[valid]/before[valid]-1
        if reverse:
            values = -values
        result[row, valid] = pd.Series(values).rank(method='average').to_numpy()/max(1, valid.sum())
    return result


def size_issuers(data, ids, stock, method):
    """63 known closes, 50% diagonal covariance shrinkage for capped ERC.

    The 30% cap applies to issuers. Tied share classes split their issuer budget.
    Missing risk observations are zero as in the inherited basket risk engine.
    """
    output = np.zeros_like(stock)
    for row, i in enumerate(ids):
        day = int(data['origin'][i])
        groups = {}
        for j in np.flatnonzero(stock[row]):
            ticker = data['panel'].tickers[j]
            groups.setdefault(ISSUERS.get(ticker, ticker), []).append(j)
        members = list(groups.values())
        history = np.nan_to_num(data['returns'][max(0, day-62):day+1], nan=0.)
        returns = np.column_stack([history[:, js] @ (stock[row, js]/stock[row, js].sum()) for js in members])
        covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))*252
        diagonal = np.maximum(np.diag(covariance), .10**2)
        if method == 'invvol':
            budgets = 1/np.sqrt(diagonal)
        elif method == 'erc':
            covariance = .5*covariance + .5*np.diag(diagonal)
            budgets = equal_risk_weights(covariance)
        else:
            raise ValueError(method)
        budgets = capped_weights(budgets, .30)
        for js, budget in zip(members, budgets):
            output[row, js] = budget*stock[row, js]/stock[row, js].sum()
    return output


def portfolio_targets(data, ids, score, market, recipe):
    stock = np.array([issuer_weights(s, data['mask'][i], data['panel'].tickers, 5)
                      for i, s in zip(ids, score)])
    if recipe.get('sizing'):
        stock = size_issuers(data, ids, stock, recipe['sizing'])
    # Same 20-session basket estimator as the inherited risk engine, evaluated
    # only at decisions instead of constructing unused daily allocations.
    volatility = []
    for row, i in enumerate(ids):
        day = int(data['origin'][i]); active = stock[row] > 0
        r = np.nan_to_num(data['returns'][day-19:day+1, active], nan=0.) @ stock[row, active]
        volatility.append(max(float(r.std(ddof=1)*np.sqrt(252)), .05))
    origin = data['origin'][ids]
    long = np.minimum(1., .20/np.asarray(volatility))
    result = hedged_targets(stock, long, market['qqqvol20'][origin], .75)
    if 'off' in recipe:
        weak = market['qqq'][origin] < market['ma200'][origin]
        # Reduce the whole signed sleeve; moving to cash does not add a short.
        result[weak] *= recipe['off']
    assert np.max(np.abs(result).sum(1)) <= 1+1e-12
    assert np.max(np.abs(result[:, :-1][~data['mask'][ids]]), initial=0.) == 0
    return result
