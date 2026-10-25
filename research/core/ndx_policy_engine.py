"""Versioned risk-exposure adaptation of independently audited account engines.

The original engines stay unchanged. The only policy intervention multiplies
signal-close target weights by an ex-ante exposure fraction before orders freeze.
"""
import numpy as np
import pandas as pd
from research_engine import target_weights, performance
from pit_research_engine import cost_rate

def policy_backtest(data,ids,scores,k=10,cost_multiplier=1.,basket=False,aum=1_000_000.,exposure=None):
    """Frozen signal-close share orders; next-close fills; one cash account.

    Acquisition marks include actual cash/share consideration. Stock consideration
    is liquidated at that day's close and charged trading costs; cash is credited
    at completion close as an explicitly disclosed settlement approximation.
    Missing fills cancel, never rerank. Cash shortages scale buys pro rata.
    """
    dates=data['panel'].dates;ids=np.asarray(ids)
    if len(scores)!=len(ids) or np.any(np.diff(ids)!=1):raise ValueError('Forecast alignment')
    n=len(data['panel'].tickers);units=np.zeros(n);cash=aum;nav=aum
    marks=data['marks'];rows=[];ledger=[];pending=None;turn=0.;total_cost=0.
    first,last=int(data['entry'][ids[0]]),len(dates)-1
    schedule={int(data['origin'][i]):(i,scores[j]) for j,i in enumerate(ids)}
    events={dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}
    last_basket_month=None

    def order(day,i,score):
        nonlocal last_basket_month
        if basket:
            month=dates[data['entry'][i]].to_period('M')
            if month==last_basket_month:return None
            last_basket_month=month
            mask=data['membership'][data['entry'][i]] & data['tradable'][day]
            w=mask.astype(float)/max(mask.sum(),1)
        else:
            mask=data['mask'][i]
            if mask.sum()==0:return None
            w=target_weights(score,mask,k,'equal')
        if exposure is not None:
            w=w*float(exposure[i])
        desired=np.divide(w*nav,marks[day],out=np.zeros(n),where=np.isfinite(marks[day])&(marks[day]>0))
        cap=np.divide(.01*np.nan_to_num(data['adv'][day]),marks[day],out=np.zeros(n),where=np.isfinite(marks[day])&(marks[day]>0))
        return dict(delta=np.clip(desired-units,-cap,cap),adv=data['adv'][day].copy(),signal=day)

    i,score=schedule[first-1];pending=order(first-1,i,score)
    for day in range(first,last+1):
        old=nav
        ratio=np.divide(marks[day],marks[day-1],out=np.ones(n),where=np.isfinite(marks[day])&np.isfinite(marks[day-1]))
        profit=float(np.nan_to_num(units*marks[day-1]*(ratio-1)).sum()) if day>first else 0.
        fees=0.;traded=0.;settled=0.
        event=events.get(day)
        if event:
            node=data['panel'].tickers.index(event['security_id'])
            value=units[node]*marks[day,node]
            if value:
                stock_fraction=event['stock_fraction_at_completion']
                fee=float(value*stock_fraction*cost_rate(value*stock_fraction,data['adv'][day-1,node],cost_multiplier))
                cash+=value-fee;fees+=fee;settled+=value;units[node]=0.
        if day==last:pending=None  # Predetermined terminal liquidation, no new terminal-close purchase.
        if pending:
            delta=pending['delta'].copy()
            delta[~data['tradable'][day]]=0.
            delta=np.maximum(delta,-units)
            dollars=np.nan_to_num(delta*marks[day])
            sell=np.minimum(dollars,0.);buy=np.maximum(dollars,0.)
            rates=cost_rate(dollars,pending['adv'],cost_multiplier)
            available=cash-sell.sum()-float(np.abs(sell)@rates)
            need=buy.sum()+float(buy@rates)
            if need>available and need>0:
                factor=max(0.,available/need);delta[dollars>0]*=factor
                dollars=np.nan_to_num(delta*marks[day]);rates=cost_rate(dollars,pending['adv'],cost_multiplier)
            fee=float(np.abs(dollars)@rates)
            cash-=float(dollars.sum())+fee;units+=delta;fees+=fee;traded+=float(np.abs(dollars).sum())
            pending=None
        if day==last:
            if np.any((units>1e-12)&~data['tradable'][day]):
                raise ValueError('Unsettled non-tradable terminal position')
            dollars=np.nan_to_num(units*marks[day]);fee=float(dollars@cost_rate(dollars,data['adv'][day-1],cost_multiplier))
            cash+=float(dollars.sum())-fee;units[:]=0.;fees+=fee;traded+=float(dollars.sum())
        held=np.where(units!=0,np.nan_to_num(units*marks[day]),0.)
        nav=cash+float(held.sum())
        if cash < -1e-5 or not np.isfinite(nav) or nav<=0:raise ValueError('Invalid cash/NAV')
        if np.any((units>1e-12)&~data['tradable'][day]):raise ValueError('Missing held price without settlement')
        rows.append(dict(date=str(dates[day].date()),net=nav/old-1,gross=profit/old))
        ledger.append(dict(date=str(dates[day].date()),nav=nav,cash=cash,holdings=float(held.sum()),
                           fee=fees,traded_dollars=traded,settled_dollars=settled))
        total_cost+=fees;turn+=traded/old
        if day in schedule:
            i,score=schedule[day];pending=order(day,i,score)
    daily=pd.DataFrame(rows);accounts=pd.DataFrame(ledger)
    np.testing.assert_allclose(aum*np.cumprod(1+daily.net),accounts.nav,rtol=1e-10)
    metrics=performance(daily.net)
    metrics.update(cost_dollars=total_cost,turnover_annual=turn*252/len(daily),terminal_nav=nav)
    return metrics,daily,accounts


def policy_replay(data,ids,scores,k,mult,basket=False,exposure=None):
    """Independent daily unit ledger: no production allocation/backtest calls."""
    names=data['panel'].tickers;dates=data['panel'].dates;n=len(names)
    shares=np.zeros(n);bank=1e6;previous=1e6;values=[];new_order=None
    quotes=data['marks'];first=int(data['entry'][ids[0]]);last=len(dates)-1
    forecasts={int(data['origin'][i]):(i,np.asarray(scores[j])) for j,i in enumerate(ids)}
    actions={str(e['date']):e for e in data['events']};last_month=None
    def fee(trades,adv):
        rate=.0007+.0003*np.sqrt(np.abs(trades)/np.maximum(np.nan_to_num(adv,nan=1.),1.)/.01)
        return float((np.abs(trades)*rate).sum()*mult)
    def freeze(day,i,s):
        nonlocal last_month
        weights=np.zeros(n)
        if basket:
            month=str(dates[data['entry'][i]].to_period('M'))
            if month==last_month:return None
            last_month=month
            selected=data['membership'][data['entry'][i]] & data['tradable'][day]
            weights[selected]=1/max(int(selected.sum()),1)
        else:
            selected=data['mask'][i];available=np.flatnonzero(selected)
            if not len(available):return None
            cutoff=np.sort(s[available])[max(0,len(available)-k)]
            strict=selected&(s>cutoff);tie=selected&(s==cutoff)
            weights[strict]=1.;weights[tie]=(min(k,len(available))-strict.sum())/max(tie.sum(),1)
            weights/=weights.sum()
        if exposure is not None:
            weights*=float(exposure[i])
        target=np.zeros(n);cap=np.zeros(n)
        for j in range(n):
            if np.isfinite(quotes[day,j]) and quotes[day,j]>0:
                target[j]=weights[j]*previous/quotes[day,j]
                cap[j]=.01*np.nan_to_num(data['adv'][day,j])/quotes[day,j]
        return np.maximum(np.minimum(target-shares,cap),-cap),data['adv'][day].copy()
    i,s=forecasts[first-1];new_order=freeze(first-1,i,s)
    for day in range(first,last+1):
        action=actions.get(str(dates[day].date()))
        if action:
            node=names.index(action['security_id']);value=shares[node]*quotes[day,node]
            stock=value*action['stock_fraction_at_completion']
            bank+=value-fee(np.array([stock]),np.array([data['adv'][day-1,node]]));shares[node]=0.
        if day==last:new_order=None
        if new_order is not None:
            quantities,adv=new_order
            quantities=np.where(data['tradable'][day],quantities,0.)
            quantities=np.maximum(quantities,-shares)
            trades=np.nan_to_num(quantities*quotes[day])
            sells=np.minimum(trades,0.);buys=np.maximum(trades,0.)
            sellfee=fee(sells,adv);buyfee=fee(buys,adv)
            available=bank-sells.sum()-sellfee
            if buys.sum()+buyfee>available and buys.sum()+buyfee>0:
                quantities[trades>0]*=max(0.,available/(buys.sum()+buyfee))
                trades=np.nan_to_num(quantities*quotes[day])
            bank-=trades.sum()+fee(trades,adv);shares+=quantities;new_order=None
        if day==last:
            trades=np.nan_to_num(shares*quotes[day]);bank+=trades.sum()-fee(trades,data['adv'][day-1]);shares[:]=0.
        worth=bank+float(np.nan_to_num(shares*quotes[day]).sum())
        values.append(worth);previous=worth
        if day in forecasts:
            i,s=forecasts[day];new_order=freeze(day,i,s)
    return np.asarray(values)
