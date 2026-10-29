"""Weekly causal market-short targets and a signed, self-financing NAV account."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pit_research_engine import cost_rate
from research_engine import performance


def weak_signals(price):
    p = pd.Series(np.asarray(price, dtype=float))
    return {'ma200': (p < p.rolling(200).mean()).to_numpy(),
            'ma100': (p < p.rolling(100).mean()).to_numpy(),
            'ma200_mom63': ((p < p.rolling(200).mean()) & (p < p.shift(63))).to_numpy()}


def targets(weekly, exposure, origins, weak, short_vol, mode):
    """Cash/short switch replaces longs; hedge uses only unallocated budget."""
    result = np.zeros((len(weekly), weekly.shape[1]+1))
    for row, day in enumerate(origins):
        long = float(exposure[day])
        short = 0.
        if weak[day] and mode != 'baseline':
            if mode == 'cash':
                long = 0.
            elif mode.startswith('switch'):
                long = 0.
                short = min(int(mode[6:])/100, .20/max(short_vol[day], .05))
            elif mode.startswith('hedge'):
                short = min(int(mode[5:])/100, 1-long, .20/max(short_vol[day], .05))
            else:
                raise ValueError('Unknown short mode')
        result[row, :-1] = weekly[row]*long
        result[row, -1] = -short
    if not np.isfinite(result).all() or np.max(np.abs(result).sum(1)) > 1+1e-12:
        raise ValueError('Invalid target gross budget')
    return result


def append_qqq(data, etf):
    result = dict(data)
    result['panel'] = SimpleNamespace(dates=data['panel'].dates,
                                     tickers=data['panel'].tickers+['ETF:QQQ'])
    for name in ['marks', 'tradable', 'adv']:
        result[name] = np.concatenate([data[name], etf[name]], axis=1)
    return result


def simulate_signed(data, ids, weights, borrow_rate=.03, multiplier=1., aum=1e6,
                    detailed=False):
    """Prior-close sized orders, next-close fills, calendar-day short fees.

    Adjusted economic units embed dividend obligations on the short leg. Short
    proceeds earn no rebate and never increase the gross target budget. Realized
    gross exposure can drift above one. No broker margin/recall simulation.
    """
    dates, marks = data['panel'].dates, data['marks']
    if borrow_rate < 0 or multiplier < 0 or not np.isfinite(weights).all():
        raise ValueError('Invalid costs/weights')
    if np.any(np.abs(weights).sum(1) > 1+1e-12):
        raise ValueError('Levered target')
    units = np.zeros(marks.shape[1]); cash = nav = aum
    first, last = int(data['entry'][ids[0]]), len(dates)-1
    schedule = {int(data['origin'][i]): weights[row] for row, i in enumerate(ids)}
    events = {dates.get_loc(pd.Timestamp(e['date'])): e for e in data['events']}
    paths, ledger = [], []
    fees_total = borrow_total = turnover = max_error = 0.
    gross_sum = max_gross = 0.; min_equity = 1.; min_cash = aum
    short_sum = 0.

    def order(day):
        w = schedule[day]
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
        if day == last:
            pending = None
        if pending is not None:
            delta, adv = pending
            delta = delta.copy(); delta[~data['tradable'][day]] = 0.
            dollars = np.nan_to_num(delta*marks[day])
            rates = cost_rate(dollars, adv, multiplier)
            sells, buys = np.minimum(dollars, 0), np.maximum(dollars, 0)
            available = cash-sells.sum()-float(np.abs(sells)@rates)
            # Keep funding for accrued borrow between rebalances. In particular,
            # cash-scaled covering can leave a small short even at a zero target.
            # Budget 2x current short value for fourteen calendar days, before
            # positive trades (covers included); no future prices enter reserve.
            short_before_buys = float(np.maximum(-np.nan_to_num((units+np.minimum(delta,0))*marks[day]),0).sum())
            reserve = short_before_buys*borrow_rate*28/365
            available -= reserve
            need = buys.sum()+float(buys@rates)
            if need > available and need > 0:
                delta[dollars > 0] *= max(0., available/need)
                dollars = np.nan_to_num(delta*marks[day])
                rates = cost_rate(dollars, adv, multiplier)
            fee = float(np.abs(dollars)@rates)
            cash -= float(dollars.sum())+fee
            units += delta; fees += fee; traded += float(np.abs(dollars).sum())
            pending = None
        if np.any((np.abs(units) > 1e-12) & ~data['tradable'][day]):
            raise ValueError('Unpriced signed holding')
        if day == last:
            dollars = np.nan_to_num(units*marks[day])
            fee = float(np.abs(dollars)@cost_rate(dollars, data['adv'][day-1], multiplier))
            cash += float(dollars.sum())-fee
            fees += fee; traded += float(np.abs(dollars).sum()); units[:] = 0.
        held = np.nan_to_num(units*marks[day])
        nav = cash+float(held.sum())
        error = abs((nav-old)-(pnl-fees-borrow))
        max_error = max(max_error, error)
        if error > 1e-6 or cash < -1e-5 or not np.isfinite(nav) or nav <= 0:
            raise ValueError('Invalid self-financing account')
        gross = float(np.abs(held).sum())/nav
        short = float(np.maximum(-held, 0).sum())/nav
        gross_sum += gross; short_sum += short; max_gross = max(max_gross, gross)
        min_equity = min(min_equity, 1/max(gross, 1e-12)); min_cash = min(min_cash, cash)
        paths.append(nav/old-1)
        if detailed:
            ledger.append(dict(date=str(dates[day].date()), nav=nav, cash=cash,
                               long=float(np.maximum(held, 0).sum()), short=short*nav,
                               fee=fees, borrow=borrow, traded=traded, pnl=pnl))
        fees_total += fees; borrow_total += borrow; turnover += traded/old
        if day < last and day in schedule:
            pending = order(day)
    metrics = performance(paths)
    metrics.update(terminal_nav=nav, cost_dollars=fees_total, borrow_dollars=borrow_total,
                   turnover_annual=turnover*252/len(paths), mean_gross=gross_sum/len(paths),
                   mean_short=short_sum/len(paths), max_gross=max_gross,
                   minimum_equity_to_gross=min_equity, minimum_cash=min_cash,
                   max_accounting_error=max_error)
    return metrics, np.asarray(paths), pd.DataFrame(ledger)
