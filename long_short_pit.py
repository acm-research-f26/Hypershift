"""Signed-position extension; leave the previously frozen long-only engine intact."""
import numpy as np
import pandas as pd
from pit_research_engine import cost_rate
from research_engine import target_weights, performance


def signed_weights(score, mask, k, long_budget, short_budget):
    if k < 1 or min(long_budget, short_budget) < 0 or long_budget+short_budget > 1+1e-12:
        raise ValueError('Invalid allocation')
    if not np.any(mask) or (short_budget and np.sum(mask) < 2*k):
        return np.zeros(len(score))
    long = target_weights(score, mask, k, 'equal')*long_budget
    short = target_weights(-np.asarray(score), mask, k, 'equal')*short_budget
    return long-short


def signed_backtest(data, ids, scores, k=10, long_budget=.5, short_budget=.5,
                    borrow_rate=.03, cost_multiplier=1., aum=1_000_000.):
    """Frozen orders, signed economic-price units, calendar-day borrow accrual.

    Opening shorts increases cash; covering shorts reduces cash. Short proceeds
    do not expand target budgets. All absolute traded dollars incur positive fees.
    Synthetic adjusted units embed total-return effects, including dividends.
    """
    if borrow_rate < 0 or cost_multiplier < 0:
        raise ValueError('Negative fee')
    dates=data['panel'].dates; ids=np.asarray(ids)
    if len(scores)!=len(ids) or np.any(np.diff(ids)!=1):
        raise ValueError('Forecast alignment')
    n=len(data['panel'].tickers); units=np.zeros(n); cash=aum; nav=aum
    marks=data['marks']; rows=[]; ledger=[]; turn=fees_total=borrow_total=0.
    first,last=int(data['entry'][ids[0]]),len(dates)-1
    schedule={int(data['origin'][i]):(i,scores[j]) for j,i in enumerate(ids)}
    events={dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}

    def order(day, i, score):
        weight=signed_weights(score,data['mask'][i],k,long_budget,short_budget)
        valid=np.isfinite(marks[day])&(marks[day]>0)
        desired=np.divide(weight*nav,marks[day],out=np.zeros(n),where=valid)
        cap=np.divide(.01*np.nan_to_num(data['adv'][day]),marks[day],out=np.zeros(n),where=valid)
        return dict(delta=np.clip(desired-units,-cap,cap),adv=data['adv'][day].copy())

    i,score=schedule[first-1]; pending=order(first-1,i,score)
    for day in range(first,last+1):
        old=nav
        before=np.where(units!=0,np.nan_to_num(units*marks[day-1]),0.)
        held_before=np.where(units!=0,np.nan_to_num(units*marks[day]),0.)
        profit=float((held_before-before).sum())
        days=(dates[day]-dates[day-1]).days
        borrow=float(np.maximum(-before,0).sum()*borrow_rate*days/365)
        cash-=borrow; fees=traded=settled=0.
        event=events.get(day)
        if event:
            node=data['panel'].tickers.index(event['security_id'])
            value=float(units[node]*marks[day,node])
            if value:
                stock_dollars=abs(value)*event['stock_fraction_at_completion']
                fee=float(stock_dollars*cost_rate(stock_dollars,data['adv'][day-1,node],cost_multiplier))
                cash+=value-fee; fees+=fee; settled+=value; units[node]=0.
        if day==last:
            pending=None
        if pending:
            delta=pending['delta'].copy(); delta[~data['tradable'][day]]=0.
            dollars=np.nan_to_num(delta*marks[day])
            sell=np.minimum(dollars,0.); buy=np.maximum(dollars,0.)
            rates=cost_rate(dollars,pending['adv'],cost_multiplier)
            available=cash-sell.sum()-float(np.abs(sell)@rates)
            need=buy.sum()+float(buy@rates)
            if need>available and need>0:
                delta[dollars>0]*=max(0.,available/need)
                dollars=np.nan_to_num(delta*marks[day]); rates=cost_rate(dollars,pending['adv'],cost_multiplier)
            fee=float(np.abs(dollars)@rates)
            cash-=float(dollars.sum())+fee; units+=delta; fees+=fee; traded+=float(np.abs(dollars).sum())
            pending=None
        if np.any((np.abs(units)>1e-12)&~data['tradable'][day]):
            raise ValueError('Missing held price without settlement')
        if day==last:
            dollars=np.nan_to_num(units*marks[day])
            fee=float(np.abs(dollars)@cost_rate(dollars,data['adv'][day-1],cost_multiplier))
            cash+=float(dollars.sum())-fee; units[:]=0.; fees+=fee; traded+=float(np.abs(dollars).sum())
        held=np.where(units!=0,np.nan_to_num(units*marks[day]),0.)
        long=float(np.maximum(held,0).sum()); short=float(np.maximum(-held,0).sum())
        nav=cash+long-short
        if cash < -1e-5 or not np.isfinite(nav) or nav<=0:
            raise ValueError('Invalid cash/NAV')
        # Self-financing identity must hold even through signed merger settlements.
        error=(nav-old)-(profit-fees-borrow)
        if abs(error)>1e-7+1e-9*abs(profit-fees-borrow):
            raise ValueError('Signed account is not self-financing')
        rows.append(dict(date=str(dates[day].date()),net=nav/old-1,gross=profit/old))
        ledger.append(dict(date=str(dates[day].date()),nav=nav,cash=cash,long_dollars=long,
            short_dollars=short,gross_exposure=(long+short)/nav,net_exposure=(long-short)/nav,
            fee=fees,borrow_fee=borrow,traded_dollars=traded,settled_dollars=settled))
        fees_total+=fees; borrow_total+=borrow; turn+=traded/old
        if day in schedule:
            i,score=schedule[day]; pending=order(day,i,score)
    daily=pd.DataFrame(rows); accounts=pd.DataFrame(ledger)
    np.testing.assert_allclose(aum*np.cumprod(1+daily.net),accounts.nav,rtol=1e-10)
    metrics=performance(daily.net)
    metrics.update(cost_dollars=fees_total,borrow_dollars=borrow_total,turnover_annual=turn*252/len(daily),
        terminal_nav=nav,mean_gross_exposure=accounts.gross_exposure.mean(),
        max_gross_exposure=accounts.gross_exposure.max(),mean_net_exposure=accounts.net_exposure.mean(),
        max_abs_net_exposure=accounts.net_exposure.abs().max(),minimum_cash=accounts.cash.min())
    return metrics,daily,accounts
