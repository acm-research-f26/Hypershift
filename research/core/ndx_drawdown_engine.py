"""Causal long-only allocation and exposure controls for saved NDX forecasts.

New research engine; frozen studies are never modified. A decision row uses the
preceding close's marks/NAV/risk inputs and fills at the following session close.
Weekly AH+PM rankings become available at 09:20 on that fill date, as in parent.
"""
import numpy as np
import pandas as pd
from research_engine import target_weights, performance
from pit_research_engine import cost_rate

ISSUERS = {'GOOG':'ALPHABET', 'GOOGL':'ALPHABET', 'FOX':'FOX', 'FOXA':'FOX',
           'LBTYA':'LIBERTY_GLOBAL', 'LBTYK':'LIBERTY_GLOBAL'}


def issuer_weights(scores, eligible, tickers, k=5):
    """Rank issuers by their best class; share ties equally at both boundaries."""
    members = {}
    for j in np.flatnonzero(eligible):
        members.setdefault(ISSUERS.get(tickers[j], tickers[j]), []).append(j)
    names = sorted(members)
    best = np.array([np.max(scores[members[name]]) for name in names])
    issuer = target_weights(best, np.ones(len(names), dtype=bool), k, 'equal')
    result = np.zeros(len(scores))
    for name, budget, value in zip(names, issuer, best):
        ids = [j for j in members[name] if scores[j] == value]
        result[ids] = budget/len(ids)
    return result


def capped_weights(raw, cap=.30):
    """Normalize positive budgets with a feasible upper limit."""
    raw = np.maximum(np.asarray(raw, dtype=float), 0.)
    active = raw > 0
    if not active.any():
        return raw
    cap = max(cap, 1/int(active.sum()))
    result = np.zeros_like(raw)
    remaining = 1.
    while active.any():
        proposed = raw[active]/raw[active].sum()*remaining
        excessive = proposed > cap+1e-14
        ids = np.flatnonzero(active)
        if not excessive.any():
            result[ids] = proposed
            break
        fixed = ids[excessive]
        result[fixed] = cap
        remaining -= cap*len(fixed)
        active[fixed] = False
    return result


def allocations(data, ids, scores, kind):
    returns = pd.DataFrame(data['returns'])
    vol = returns.rolling(20, min_periods=15).std().to_numpy()*np.sqrt(252)
    weights = np.zeros_like(scores, dtype=float)
    for row, i in enumerate(ids):
        if kind in ['equal5', 'equal10']:
            weights[row] = target_weights(scores[row], data['mask'][i], 5 if kind == 'equal5' else 10, 'equal')
        else:
            k = 8 if kind == 'issuer8_soft' else 5
            weights[row] = issuer_weights(scores[row], data['mask'][i], data['panel'].tickers, k)
            chosen = weights[row] > 0
            if kind == 'issuer_invvol5':
                risk = np.maximum(np.nan_to_num(vol[data['origin'][i]], nan=.30), .10)
                weights[row] = capped_weights(weights[row]/risk, .30)
            elif kind == 'issuer8_soft':
                raw = weights[row]*np.exp(4*(scores[row]-np.max(scores[row, chosen])))
                weights[row] = capped_weights(raw, .25)
        if not np.isfinite(weights[row]).all() or np.any(weights[row] < 0):
            raise ValueError('Invalid allocation')
        np.testing.assert_allclose(weights[row].sum(), 1., atol=1e-12)
    return weights


def market_inputs(data):
    root = data['manifest']['root']
    q = pd.read_csv(root+'/etf_adj_close.csv', index_col=0, parse_dates=True)['QQQ']
    r = q.pct_change(fill_method=None)
    result = {'qqq':q.to_numpy()}
    for length in [50, 100, 150, 200, 250]:
        result['ma'+str(length)] = q.rolling(length, min_periods=length).mean().to_numpy()
    for length in [20, 60]:
        result['qqqvol'+str(length)] = (r.rolling(length, min_periods=length).std()*np.sqrt(252)).to_numpy()
    result['qqqdd'] = (q/q.rolling(252, min_periods=200).max()-1).to_numpy()
    return result


def daily_allocation(data, ids, weekly):
    n = len(data['panel'].dates)
    result = np.zeros((n, weekly.shape[1]))
    lookup = {int(data['origin'][i]):row for row,i in enumerate(ids)}
    current = np.zeros(weekly.shape[1])
    for day in range(n):
        if day in lookup:
            current = weekly[lookup[day]]
        result[day] = current
    return result


def risk_inputs(data, daily_weights):
    result = {}
    r = np.asarray(data['returns'])
    for length in [20, 60]:
        values = np.full(len(r), .30)
        for day in range(length-1, len(r)):
            weights = daily_weights[day]
            if not weights.any():
                continue
            active = weights > 0
            history = np.nan_to_num(r[day-length+1:day+1, active], nan=0.)@weights[active]
            values[day] = max(float(history.std(ddof=1)*np.sqrt(252)), .05)
        result['basketvol'+str(length)] = values
    return result


def exposure_series(spec, market, risk):
    n = len(market['qqq'])
    exposure = np.ones(n)*spec.get('constant', 1.)
    if 'ma' in spec:
        weak = market['qqq'] < market['ma'+str(spec['ma'])]
        exposure = np.minimum(exposure, np.where(weak, spec.get('off', .5), 1.))
    if spec.get('multitrend'):
        healthy = np.mean([market['qqq'] >= market['ma'+str(k)] for k in [50,100,200]], axis=0)
        exposure = np.minimum(exposure, .25+.75*healthy)
    if 'vol' in spec:
        observed = (risk if spec['vol'] == 'basket' else market)[spec['vol']+'vol'+str(spec.get('window',20))]
        exposure = np.minimum(exposure, spec['target']/np.maximum(observed, .05))
    if 'market_dd' in spec:
        exposure = np.minimum(exposure, np.where(market['qqqdd'] < -spec['market_dd'], spec['off'], 1.))
    return np.clip(np.nan_to_num(exposure, nan=.5), 0., 1.)


def simulate(data, ids, weekly, exposure, cadence='weekly', multiplier=1., band=.10,
             detailed=False, aum=1_000_000.):
    """Close execution with frozen quantities, parent costs, caps and cash rules."""
    dates = data['panel'].dates
    marks = data['marks']
    n = len(data['panel'].tickers)
    units = np.zeros(n)
    cash = nav = aum
    lookup = {int(data['origin'][i]):row for row,i in enumerate(ids)}
    events = {dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}
    first, last = int(data['entry'][ids[0]]), len(dates)-1
    base = np.zeros(n)
    previous_exposure = None
    last_weights = np.zeros(n)
    total_cost = turnover = 0.
    paths, records = [], []

    def order(day):
        nonlocal base, previous_exposure, last_weights
        weekly_update = day in lookup
        if weekly_update:
            base = weekly[lookup[day]].copy()
        changed = previous_exposure is None or abs(float(exposure[day])-previous_exposure) >= band-1e-12
        if not weekly_update and not (cadence == 'daily' and changed):
            return None
        previous_exposure = float(exposure[day])
        last_weights = base*previous_exposure
        desired = np.divide(last_weights*nav, marks[day], out=np.zeros(n), where=np.isfinite(marks[day])&(marks[day]>0))
        cap = np.divide(.01*np.nan_to_num(data['adv'][day]), marks[day], out=np.zeros(n), where=np.isfinite(marks[day])&(marks[day]>0))
        return np.clip(desired-units, -cap, cap), data['adv'][day].copy()

    pending = order(first-1)
    for day in range(first, last+1):
        old = nav
        fees = traded = 0.
        if day in events:
            event = events[day]
            node = data['panel'].tickers.index(event['security_id'])
            value = units[node]*marks[day,node]
            if value:
                fee = float(value*event['stock_fraction_at_completion']*cost_rate(value*event['stock_fraction_at_completion'], data['adv'][day-1,node], multiplier))
                cash += value-fee
                fees += fee
                units[node] = 0.
        if day == last:
            pending = None
        if pending is not None:
            delta, adv = pending
            delta = delta.copy()
            delta[~data['tradable'][day]] = 0.
            delta = np.maximum(delta, -units)
            dollars = np.nan_to_num(delta*marks[day])
            sell, buy = np.minimum(dollars,0.), np.maximum(dollars,0.)
            rates = cost_rate(dollars,adv,multiplier)
            available = cash-sell.sum()-float(np.abs(sell)@rates)
            need = buy.sum()+float(buy@rates)
            if need > available and need > 0:
                delta[dollars>0] *= max(0., available/need)
                dollars = np.nan_to_num(delta*marks[day])
                rates = cost_rate(dollars,adv,multiplier)
            fee = float(np.abs(dollars)@rates)
            cash -= float(dollars.sum())+fee
            units += delta
            fees += fee
            traded += float(np.abs(dollars).sum())
            pending = None
        if day == last:
            if np.any((units>1e-12)&~data['tradable'][day]):
                raise ValueError('Unsettled terminal position')
            dollars = np.nan_to_num(units*marks[day])
            fee = float(dollars@cost_rate(dollars,data['adv'][day-1],multiplier))
            cash += float(dollars.sum())-fee
            fees += fee
            traded += float(dollars.sum())
            units[:] = 0.
        held = np.where(units!=0,np.nan_to_num(units*marks[day]),0.)
        nav = cash+float(held.sum())
        if cash < -1e-5 or not np.isfinite(nav) or nav <= 0:
            raise ValueError('Invalid account')
        if np.any((units>1e-12)&~data['tradable'][day]):
            raise ValueError('Unpriced held security')
        paths.append(nav/old-1)
        if detailed:
            issuer_values = {}
            for j in np.flatnonzero(held):
                name = ISSUERS.get(data['panel'].tickers[j],data['panel'].tickers[j])
                issuer_values[name] = issuer_values.get(name,0.)+held[j]
            records.append(dict(date=str(dates[day].date()), nav=nav, cash=cash, fee=fees,
                                traded=traded, equity_exposure=float(held.sum()/nav),
                                target_exposure=previous_exposure,
                                largest_issuer_weight=max(issuer_values.values(),default=0.)/nav))
        total_cost += fees
        turnover += traded/old
        if day < last:
            pending = order(day)
    result = performance(paths)
    result.update(cost_dollars=total_cost,turnover_annual=turnover*252/len(paths),terminal_nav=nav)
    return result, np.asarray(paths), pd.DataFrame(records)
