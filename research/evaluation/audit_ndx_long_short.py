"""Independent signed-cash replay; no calls to the portfolio engine's helpers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def replay(data, ids, scores, k, long_budget, short_budget, borrow_rate, multiplier, aum=1e6):
    dates=data['panel'].dates; prices=data['marks']; count=prices.shape[1]
    position=np.zeros(count); money=float(aum); equity=float(aum); pending=None; nav=[]
    fees=[]; borrows=[]; cash=[]
    signals={int(data['origin'][i]):(i,scores[j]) for j,i in enumerate(ids)}
    events={dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}
    first=int(data['entry'][ids[0]]); last=len(dates)-1

    def charges(value, adv):
        return multiplier*(.0007+.0003*np.sqrt(np.abs(value)/np.maximum(np.nan_to_num(adv,nan=1.),1.)/.01))

    def instruction(day):
        i,score=signals[day]; allowed=np.flatnonzero(data['mask'][i]); weights=np.zeros(count)
        if len(allowed) and (not short_budget or len(allowed)>=2*k):
            for direction,budget in [(1,long_budget),(-1,short_budget)]:
                values=direction*np.asarray(score)
                boundary=sorted(values[allowed],reverse=True)[min(k,len(allowed))-1]
                better=allowed[values[allowed]>boundary]; ties=allowed[values[allowed]==boundary]
                raw=np.zeros(count); raw[better]=1.; raw[ties]=(min(k,len(allowed))-len(better))/len(ties)
                weights+=direction*budget*raw/raw.sum()
        target=np.zeros(count); caps=np.zeros(count)
        valid=np.isfinite(prices[day])&(prices[day]>0)
        target[valid]=weights[valid]*equity/prices[day,valid]
        caps[valid]=np.nan_to_num(data['adv'][day,valid])*.01/prices[day,valid]
        return np.minimum(np.maximum(target-position,-caps),caps),data['adv'][day].copy()

    pending=instruction(first-1)
    for day in range(first,last+1):
        borrow=float(np.nansum(np.maximum(-position*prices[day-1],0))*borrow_rate*(dates[day]-dates[day-1]).days/365)
        money-=borrow; fee=0.
        if day in events:
            event=events[day]; node=data['panel'].tickers.index(event['security_id'])
            amount=float(position[node]*prices[day,node])
            stock=abs(amount)*event['stock_fraction_at_completion']
            charge=float(stock*charges(stock,data['adv'][day-1,node]))
            money+=amount-charge; fee+=charge; position[node]=0.
        if pending is not None and day!=last:
            trade,liquidity=pending; trade=trade.copy(); trade[~data['tradable'][day]]=0.
            value=np.nan_to_num(trade*prices[day]); sell=value<0; buy=value>0
            quote=charges(value,liquidity)
            room=money-float(value[sell].sum())-float((np.abs(value[sell])*quote[sell]).sum())
            required=float((value[buy]*(1+quote[buy])).sum())
            if required>room and required>0:
                trade[buy]*=max(0.,room/required); value=np.nan_to_num(trade*prices[day]); quote=charges(value,liquidity)
            charge=float((np.abs(value)*quote).sum())
            money-=float(value.sum())+charge; fee+=charge; position+=trade
        pending=None
        if day==last:
            values=np.nan_to_num(position*prices[day]); charge=float((np.abs(values)*charges(values,data['adv'][day-1])).sum())
            money+=float(values.sum())-charge; fee+=charge; position[:]=0.
        equity=money+float(np.nansum(position*prices[day]))
        nav.append(equity); fees.append(fee); borrows.append(borrow); cash.append(money)
        if day in signals:
            pending=instruction(day)
    return np.column_stack([nav,cash,fees,borrows])


def main():
    from run_ndx_long_short import RUN, verify_lock, prepared_forecasts
    verify_lock(); data,ids,predictions=prepared_forecasts()
    rows=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}); rows=rows[~rows.parent_reused]
    saved=np.load(RUN/'ledgers.npz'); largest=0.
    for number,row in enumerate(rows.itertuples(),1):
        score=predictions[row.model,row.seed]
        d=data if row.model not in ['momentum126','reversal5'] else dict(data,
            mask=data['mask']&np.isfinite(data['momentum' if row.model=='momentum126' else 'reversal']))
        expected=replay(d,ids,score,int(row.k),row.long_budget,row.short_budget,row.borrow_rate,row.cost_multiplier)
        actual=saved[str(row.path_id)][:,[0,1,6,7]]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-11)
        largest=max(largest,float(np.max(np.abs(actual-expected))))
        assert np.isclose(actual[-1,0],row.terminal_nav,atol=1e-6)
        if number%60==0:
            print(f'Independently replayed {number}/{len(rows)} signed accounts',flush=True)
    # Perturb only future prices/liquidity/scores: earlier fills and wealth cannot move.
    key=('think_mix_pseudohuber','ensemble'); score=predictions[key]
    cut=int(data['entry'][ids[len(ids)//2]])
    changed=dict(data,marks=data['marks'].copy(),adv=data['adv'].copy()); changed_score=score.copy()
    changed['marks'][cut:]*=1.1; changed['adv'][cut:]*=.7
    changed_score[data['origin'][ids]>=cut]*=-1
    reference=replay(data,ids,score,10,.5,.5,.03,1)
    perturbed=replay(changed,ids,changed_score,10,.5,.5,.03,1)
    first=int(data['entry'][ids[0]])
    np.testing.assert_array_equal(reference[:cut-first],perturbed[:cut-first])
    # Check every unchanged long-only parent curve against the same signed engine.
    from long_short_pit import signed_backtest
    from run_robust_research import model_data
    all_rows=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str})
    parent_curves=pd.read_csv('runs/nasdaq_topk_pseudohuber_20261011/curves.csv',dtype={'seed':str})
    for row in all_rows[all_rows.parent_reused].itertuples():
        _,daily,_=signed_backtest(model_data(data,row.model),ids,predictions[row.model,'ensemble'],int(row.k),1,0,0,1)
        original=parent_curves[(parent_curves.model==row.model)&(parent_curves.seed=='ensemble')&
            (parent_curves.k==row.k)&(parent_curves.cost_multiplier==1)]
        np.testing.assert_allclose(daily.net,original.net,rtol=1e-10,atol=1e-12)
    # Return-derived diagnostics and yearly compounding independently recomputed.
    from research_engine import performance
    curves=pd.read_csv(RUN/'curves.csv')
    by_id=all_rows.set_index('path_id')
    for path,sub in curves.groupby('path_id',sort=False):
        result=performance(sub.net)
        for field in ['annualized_return','maximum_drawdown','sharpe','cumulative_return']:
            np.testing.assert_allclose(result[field],by_id.loc[path,field],atol=1e-10)
        assert len(sub)==1003 and sub.date.is_unique
    annual=pd.read_csv(RUN/'annual.csv')
    for path,sub in annual.groupby('path_id',sort=False):
        np.testing.assert_allclose((1+sub.cumulative_return).prod(),1+by_id.loc[path,'cumulative_return'],atol=1e-10)
    result=dict(status='passed',independent_signed_cash_paths=len(rows),parent_long_only_paths_unchanged=30,
        largest_ledger_difference_dollars=largest,future_perturbation_unchanged_daily_marks=cut-first,
        new_model_fits=0,annual_resets=0,price_type='adjusted economic-price proxy',
        historical_short_availability_verified=False)
    (RUN/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
