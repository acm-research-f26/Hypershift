"""Historical monthly membership, causal graph construction and continuous NAV.

This engine is separate from the archived annual-reset portfolio experiments.
No result is valid until the input review manifest is explicitly complete.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import Panel, prepare_panel, target_weights, performance

ROOT = Path('data/nasdaq_monthly_pit/assembled')


def load_data(root=ROOT):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'reviewed_research_inputs':
        raise ValueError('Historical input review is not complete; training is blocked.')
    read = lambda field: pd.read_csv(root/(field+'.csv'),index_col=0,parse_dates=True)
    close, dv, tradable = read('economic_close'), read('dollar_volume'), read('tradable').astype(bool)
    members=read('membership').astype(bool)
    panel=Panel('NASDAQ100_historical_monthly',close.index,list(close.columns),close.to_numpy(),
                dv.to_numpy(),read('etf_adj_close')['SPY'].to_numpy(),manifest['limitations'])
    data=prepare_panel(panel,horizon=5)
    # Shift the five-session calendar so the first test fill is Jan 3, 2022.
    first_fill=panel.dates.get_loc(pd.Timestamp('2022-01-03'))
    residue=(first_fill-1)%5
    origins=np.arange(260+(residue-260)%5,len(close)-1,5)
    # Reuse historical feature calculations; build all requested forecast windows,
    # including a final partial holding period ending at the 2025 terminal close.
    marks=close.ffill()
    r=marks.pct_change(fill_method=None)
    adv=dv.rolling(20,min_periods=20).mean()
    features=np.stack([r,marks/marks.rolling(5).mean()-1,marks/marks.rolling(20).mean()-1,
                       r.rolling(20).std(),np.log((dv/adv).clip(lower=.01,upper=100))],axis=-1)
    x=np.stack([features[t-15:t+1] for t in origins])
    entry=origins+1
    exit_=np.minimum(entry+5,len(close)-1)
    # Membership can change at a month boundary only when the new snapshot's
    # formation close has already occurred. The reviewed file records that rule.
    mask=members.to_numpy()[entry] & tradable.to_numpy()[origins]
    mask &= (tradable.rolling(40,min_periods=40).sum().to_numpy()[origins]==40)
    mask &= np.isfinite(x).all((1,3)) & (adv.to_numpy()[origins]>=1_000_000)
    if np.any(mask[panel.dates[origins]>=pd.Timestamp('2019-07-01')].sum(1)<2):
        raise ValueError('Too few eligible historical members')
    y=close.to_numpy()[exit_]/close.to_numpy()[entry]-1
    data.update(x=np.nan_to_num(x).astype('float32'),y=np.nan_to_num(y).astype('float32'),
                mask=mask,labelmask=mask & np.isfinite(y) & ((exit_-entry)==5)[:,None],risk=np.zeros_like(y,dtype='float32'),
                origin=origins,entry=entry,exit=exit_,adv=adv.to_numpy(),
                returns=r.to_numpy(),marks=marks.to_numpy(),stale=~tradable.to_numpy(),
                membership=members.to_numpy(),tradable=tradable.to_numpy(),
                momentum=(marks/marks.shift(126)-1).to_numpy()[origins],
                reversal=-(marks/marks.shift(5)-1).to_numpy()[origins],
                events=manifest['test_settlements'],manifest=manifest)
    return data


def fold(data,year):
    dates=data['panel'].dates
    signal=dates[data['origin']];outcome=dates[data['exit']];fill=dates[data['entry']]
    test=np.flatnonzero(fill.year==year)
    cutoff=signal[test[0]]
    vstart=pd.Timestamp(year-1,1,1)
    lower=max(pd.Timestamp('2019-07-01'),pd.Timestamp(year-5,1,1))
    train=np.flatnonzero((signal>=lower)&(outcome<vstart))
    val=np.flatnonzero((signal>=vstart)&(outcome<cutoff))
    assert outcome[train].max()<signal[val].min()
    assert outcome[val].max()<signal[test].min()
    return train,val,test


def historical_groups(data,train):
    stop=data['origin'][train[-1]]
    start=data['origin'][train[0]]
    ret=pd.DataFrame(data['returns'][start:stop+1])
    known=data['membership'][start:stop+1] & data['tradable'][start:stop+1]
    ret=ret.where(known)
    correlation=ret.corr(min_periods=30).to_numpy()
    groups=[]
    for node in range(len(data['panel'].tickers)):
        values=correlation[node].copy();values[node]=np.nan
        available=np.flatnonzero(np.isfinite(values)&(values>0))
        near=available[np.argsort(-values[available],kind='stable')[:4]]
        groups.append(set([node,*near]))
    return groups


def cost_rate(dollars,adv,multiplier):
    participation=np.abs(dollars)/np.maximum(np.nan_to_num(adv,nan=1.),1.)
    return multiplier*(.0007+.0003*np.sqrt(participation/.01))


def continuous_backtest(data,ids,scores,k=10,cost_multiplier=1.,basket=False,aum=1_000_000.):
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


def etf_backtest(data,ticker,cost_multiplier=1.,aum=1_000_000.):
    root=Path(data['manifest']['root'])
    prices=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)[ticker]
    dv=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0,parse_dates=True)[ticker]
    dates=data['panel'].dates;first=dates.get_loc(pd.Timestamp('2022-01-03'));last=len(dates)-1
    units=0.;cash=aum;nav=aum;rows=[]
    # Buy once at the same first fill close; reserve actual execution fees.
    rate=float(cost_rate(aum,dv.iloc[:first].tail(20).mean(),cost_multiplier))
    for day in range(first,last+1):
        old=nav
        if day==first:
            dollars=aum/(1+rate);units=dollars/prices.iloc[day];cash=aum-dollars*(1+rate)
        if day==last:
            dollars=units*prices.iloc[day]
            cash+=dollars*(1-float(cost_rate(dollars,dv.iloc[:day].tail(20).mean(),cost_multiplier)))
            units=0.
        nav=cash+units*prices.iloc[day]
        rows.append(dict(date=str(dates[day].date()),net=nav/old-1))
    frame=pd.DataFrame(rows)
    return performance(frame.net),frame
