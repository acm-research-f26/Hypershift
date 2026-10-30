"""Independent dollar-position replay with exact additive asset attribution."""
import numpy as np
import pandas as pd

from pit_research_engine import cost_rate


def attribute_account(data, ids, weights, borrow=.03, cost=1., aum=1e6):
    dates, marks = data['panel'].dates, data['marks']
    first, last = int(data['entry'][ids[0]]), len(marks)-1
    schedule = {int(data['origin'][i]):weights[row] for row, i in enumerate(ids)}
    events = {dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}
    value = np.zeros(marks.shape[1]); cash = nav = aum
    contributions = []; returns = []; beginning_weights = []; annual = {}

    def order(day):
        valid = np.isfinite(marks[day]) & (marks[day] > 0)
        desired = np.divide(nav*schedule[day]-value, marks[day], out=np.zeros_like(value), where=valid)
        cap = np.divide(.01*np.nan_to_num(data['adv'][day]), marks[day], out=np.zeros_like(value), where=valid)
        return np.clip(desired, -cap, cap), data['adv'][day].copy()

    pending = order(first-1)
    for day in range(first, last+1):
        old = nav; before = value.copy(); beginning_weights.append(before/old)
        charge = np.maximum(-value, 0)*borrow*(dates[day]-dates[day-1]).days/365
        cash -= charge.sum()
        ratio = np.divide(marks[day], marks[day-1], out=np.ones_like(value),
                          where=np.isfinite(marks[day]) & np.isfinite(marks[day-1]))
        value *= ratio; profit = value-before; fees = np.zeros_like(value)
        if day in events:
            event = events[day]; node = data['panel'].tickers.index(event['security_id'])
            stock = abs(value[node])*event['stock_fraction_at_completion']
            fees[node] += stock*cost_rate(stock, data['adv'][day-1, node], cost)
            cash += value[node]-fees[node]; value[node] = 0.
        if day == last: pending = None
        if pending is not None:
            shares, adv = pending
            trade = np.nan_to_num(shares*marks[day]); trade[~data['tradable'][day]] = 0.
            sells, buys = np.minimum(trade, 0), np.maximum(trade, 0)
            reserve = np.maximum(-(value+sells), 0).sum()*borrow*28/365
            available = cash-sells.sum()-(np.abs(sells)*cost_rate(sells, adv, cost)).sum()-reserve
            required = (buys*(1+cost_rate(buys, adv, cost))).sum()
            scale = min(1., max(0., available/required)) if required > 0 else 1.
            actual = sells+buys*scale; trade_fees = np.abs(actual)*cost_rate(actual, adv, cost)
            cash -= actual.sum()+trade_fees.sum(); value += actual; fees += trade_fees
        pending = None
        if day == last:
            terminal = np.abs(value)*cost_rate(value, data['adv'][day-1], cost)
            cash += value.sum()-terminal.sum(); fees += terminal; value[:] = 0.
        nav = cash+value.sum(); daily = nav/old-1
        contribution = (profit-fees-charge)/old
        assert abs(contribution.sum()-daily) < 1e-12
        contributions.append(contribution); returns.append(daily)
        year = int(dates[day].year)
        if year not in annual: annual[year] = np.zeros((5, len(value)))
        annual[year] += np.stack([profit, fees, charge, contribution, before/old])
        if day < last and day in schedule: pending = order(day)
    rows = []
    for year, totals in annual.items():
        days = int(np.count_nonzero(dates[first:last+1].year == year))
        for j, ticker in enumerate(data['panel'].tickers):
            rows.append(dict(year=year, ticker=ticker, pnl_dollars=float(totals[0, j]),
                fees=float(totals[1, j]), borrow=float(totals[2, j]),
                sum_daily_return_contribution=float(totals[3, j]),
                annualized_arithmetic_contribution=float(totals[3, j]*252/days),
                mean_beginning_weight=float(totals[4, j]/days)))
    return np.asarray(returns), pd.DataFrame(rows), np.asarray(contributions), np.asarray(beginning_weights)
