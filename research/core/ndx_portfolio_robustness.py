"""Fixed target-averaging and causal actual-holdings no-trade-band experiments."""
import numpy as np


def band_target(target, current, eligible, band):
    """Keep small changes in current NAV weights; eligibility and gross win.

    Both inputs use preceding-close economic marks/NAV, never fill prices.
    Exiting an ineligible security is mandatory, including residual positions.
    If keeping weights breaches the target gross budget, scale the full proposal
    to one. Subsequent fills retain the inherited ADV/funding constraints.
    """
    target, current = np.asarray(target, float), np.asarray(current, float)
    eligible = np.asarray(eligible, bool)
    if (target.shape != current.shape or eligible.shape != target.shape
            or target.ndim != 1 or not np.isfinite(target).all()
            or not np.isfinite(current).all() or not np.isfinite(band) or band < 0
            or np.abs(target).sum() > 1+1e-12 or np.any(target[~eligible] != 0)):
        raise ValueError('Invalid band inputs')
    keep = (np.abs(target-current) <= band) & eligible & (target*current >= 0)
    proposed = np.where(keep, current, target)
    proposed[~eligible] = 0
    return proposed/max(1., np.abs(proposed).sum())


def average_targets(targets):
    targets = np.asarray(targets, float)
    if (targets.ndim != 3 or not np.isfinite(targets).all()
            or targets.shape[0] < 1 or np.max(np.abs(targets).sum(-1)) > 1+1e-12):
        raise ValueError('Invalid constituent targets')
    result = targets.mean(0)
    assert np.max(np.abs(result).sum(-1)) <= 1+1e-12
    return result


def simulate_banded(data, ids, weights, eligible, band=0., borrow=.03, cost=1.):
    """Replay live-position bands; zero-band paths must match the old engine."""
    return _simulate(data, ids, weights, eligible, band, borrow, cost)


def _simulate(data, ids, weights, eligible, band, borrow_rate, multiplier):
    # Deliberately separate from the completed experiment's frozen engine.
    import pandas as pd
    from pit_research_engine import cost_rate
    from research_engine import performance
    dates, marks = data['panel'].dates, data['marks']
    weights = np.asarray(weights, float); eligible = np.asarray(eligible, bool)
    if weights.shape != eligible.shape or len(weights) != len(ids):
        raise ValueError('Target/eligibility schedule mismatch')
    units = np.zeros(marks.shape[1]); cash = nav = 1e6
    first, last = int(data['entry'][ids[0]]), len(dates)-1
    schedule = {int(data['origin'][i]): row for row, i in enumerate(ids)}
    events = {dates.get_loc(pd.Timestamp(e['date'])): e for e in data['events']}
    paths, ledger, decisions = [], [], []
    effective = np.zeros_like(weights)
    fees_total = borrow_total = turnover = max_error = 0.
    gross_sum = max_gross = short_sum = 0.; min_equity = 1.; min_cash = 1e6

    def order(day):
        row = schedule[day]
        current = np.nan_to_num(units*marks[day])/nav
        w = band_target(weights[row], current, eligible[row], band)
        effective[row] = w
        decisions.append(dict(row=row, day=day, nav=nav,
                              current=current.copy(), target=w.copy()))
        valid = np.isfinite(marks[day]) & (marks[day] > 0)
        desired = np.divide(w*nav, marks[day], out=np.zeros_like(units), where=valid)
        caps = np.divide(.01*np.nan_to_num(data['adv'][day]), marks[day],
                         out=np.zeros_like(units), where=valid)
        return np.clip(desired-units, -caps, caps), data['adv'][day].copy()

    pending = order(first-1)
    for day in range(first, last+1):
        old = nav
        before = np.nan_to_num(units*marks[day-1])
        held_mark = np.nan_to_num(units*marks[day])
        pnl = float((held_mark-before).sum())
        borrow = float(np.maximum(-before, 0).sum()*borrow_rate*(dates[day]-dates[day-1]).days/365)
        cash -= borrow
        fees = traded = 0.
        event = events.get(day)
        if event:
            j = data['panel'].tickers.index(event['security_id'])
            value = float(held_mark[j])
            notional = abs(value)*event['stock_fraction_at_completion']
            fee = float(notional*cost_rate(notional, data['adv'][day-1, j], multiplier))
            cash += value-fee; fees += fee; units[j] = 0.
        if day == last: pending = None
        if pending is not None:
            delta, adv = pending
            delta = delta.copy(); delta[~data['tradable'][day]] = 0.
            dollars = np.nan_to_num(delta*marks[day]); rates = cost_rate(dollars, adv, multiplier)
            sells, buys = np.minimum(dollars, 0), np.maximum(dollars, 0)
            available = cash-sells.sum()-float(np.abs(sells)@rates)
            short_before_buys = float(np.maximum(-np.nan_to_num((units+np.minimum(delta,0))*marks[day]),0).sum())
            available -= short_before_buys*borrow_rate*28/365
            need = buys.sum()+float(buys@rates)
            if need > available and need > 0:
                delta[dollars > 0] *= max(0., available/need)
                dollars = np.nan_to_num(delta*marks[day]); rates = cost_rate(dollars, adv, multiplier)
            fee = float(np.abs(dollars)@rates)
            cash -= float(dollars.sum())+fee; units += delta
            fees += fee; traded += float(np.abs(dollars).sum()); pending = None
        if np.any((np.abs(units) > 1e-12) & ~data['tradable'][day]):
            raise ValueError('Unpriced signed holding')
        if day == last:
            dollars = np.nan_to_num(units*marks[day])
            fee = float(np.abs(dollars)@cost_rate(dollars, data['adv'][day-1], multiplier))
            cash += float(dollars.sum())-fee
            fees += fee; traded += float(np.abs(dollars).sum()); units[:] = 0.
        held = np.nan_to_num(units*marks[day]); nav = cash+float(held.sum())
        error = abs((nav-old)-(pnl-fees-borrow)); max_error = max(max_error, error)
        if error > 1e-6 or cash < -1e-5 or not np.isfinite(nav) or nav <= 0:
            raise ValueError('Invalid self-financing account')
        gross = float(np.abs(held).sum())/nav; short = float(np.maximum(-held, 0).sum())/nav
        gross_sum += gross; short_sum += short; max_gross = max(max_gross, gross)
        min_equity = min(min_equity, 1/max(gross, 1e-12)); min_cash = min(min_cash, cash)
        paths.append(nav/old-1)
        ledger.append(dict(date=str(dates[day].date()), nav=nav, cash=cash,
                           long=float(np.maximum(held, 0).sum()), short=short*nav,
                           fee=fees, borrow=borrow, traded=traded, pnl=pnl))
        fees_total += fees; borrow_total += borrow; turnover += traded/old
        if day < last and day in schedule: pending = order(day)
    metrics = performance(paths)
    metrics.update(terminal_nav=nav, cost_dollars=fees_total, borrow_dollars=borrow_total,
                   turnover_annual=turnover*252/len(paths), mean_gross=gross_sum/len(paths),
                   mean_short=short_sum/len(paths), max_gross=max_gross,
                   minimum_equity_to_gross=min_equity, minimum_cash=min_cash,
                   max_accounting_error=max_error)
    assert len(decisions) == len(weights)
    return metrics, np.asarray(paths), pd.DataFrame(ledger), effective, decisions
