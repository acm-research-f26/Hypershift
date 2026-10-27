"""Explain top-five/top-ten differences using frozen forecasts and reconciled P&L."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from research_engine import target_weights, performance
from run_robust_research import ensemble_predictions, bootstrap_difference
from run_ndx_topk_pseudohuber import load_forecasts, RUN as SOURCE
from audit_historical_ndx import replay
from pit_research_engine import continuous_backtest

RUN=Path('runs/nasdaq_rank_depth_20261011')
MODELS=['stockmixer','think_mix','think_mix_pseudohuber']
LABELS={'stockmixer':'StockMixer','think_mix':'THINK + mixer',
        'think_mix_pseudohuber':'THINK + mixer pseudo-Huber'}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace(data, ids, weights, multiplier=1., aum=1e6):
    """Independent long-only ledger with daily stock-level P&L and fees.

    Summed stock P&L less stock fees equals change in account equity. Aggregate
    dollar contributions divided by initial capital reconcile cumulative return;
    they must not be presented as additive annualized CAGR contributions.
    """
    marks=data['marks']; dates=data['panel'].dates; n=marks.shape[1]
    units=np.zeros(n); cash=nav=float(aum); rows=[]; profit=[]; fees=[]; exposures=[]
    first=int(data['entry'][ids[0]]); last=len(dates)-1
    schedule={int(data['origin'][i]):weights[j] for j,i in enumerate(ids)}
    events={dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}

    def rates(values,adv):
        return multiplier*(.0007+.0003*np.sqrt(np.abs(values)/np.maximum(np.nan_to_num(adv,nan=1.),1.)/.01))

    def order(day):
        valid=np.isfinite(marks[day])&(marks[day]>0)
        desired=np.divide(schedule[day]*nav,marks[day],out=np.zeros(n),where=valid)
        cap=np.divide(.01*np.nan_to_num(data['adv'][day]),marks[day],out=np.zeros(n),where=valid)
        return np.clip(desired-units,-cap,cap),data['adv'][day].copy()

    pending=order(first-1)
    for day in range(first,last+1):
        old=nav
        pnl=np.nan_to_num(units*(marks[day]-marks[day-1])); fee=np.zeros(n)
        if day in events:
            event=events[day]; node=data['panel'].tickers.index(event['security_id'])
            value=float(units[node]*marks[day,node]); stock=value*event['stock_fraction_at_completion']
            charge=float(stock*rates(stock,data['adv'][day-1,node]))
            cash+=value-charge; fee[node]+=charge; units[node]=0.
        if day==last: pending=None
        if pending is not None:
            delta,adv=pending; delta=delta.copy(); delta[~data['tradable'][day]]=0.
            delta=np.maximum(delta,-units); dollars=np.nan_to_num(delta*marks[day])
            sells=np.minimum(dollars,0.); buys=np.maximum(dollars,0.)
            rate=rates(dollars,adv); available=cash-sells.sum()-float(np.abs(sells)@rate)
            need=buys.sum()+float(buys@rate)
            if need>available and need>0:
                delta[dollars>0]*=max(0.,available/need)
                dollars=np.nan_to_num(delta*marks[day]); rate=rates(dollars,adv)
            charge=np.abs(dollars)*rate
            cash-=float(dollars.sum()+charge.sum()); units+=delta; fee+=charge; pending=None
        if np.any((units>1e-12)&~data['tradable'][day]): raise ValueError('Missing held price')
        if day==last:
            value=np.nan_to_num(units*marks[day]); charge=value*rates(value,data['adv'][day-1])
            cash+=float(value.sum()-charge.sum()); fee+=charge; units[:]=0.
        held=np.nan_to_num(units*marks[day]); nav=cash+held.sum()
        if cash < -1e-5 or nav<=0: raise ValueError('Invalid account')
        np.testing.assert_allclose(nav-old,pnl.sum()-fee.sum(),atol=1e-7,rtol=1e-9)
        rows.append(dict(date=str(dates[day].date()),net=nav/old-1,nav=nav,cash=cash))
        profit.append(pnl); fees.append(fee); exposures.append(held/nav)
        if day in schedule: pending=order(day)
    return pd.DataFrame(rows),np.asarray(profit),np.asarray(fees),np.asarray(exposures)


def main():
    data,ids,forecasts=load_forecasts()
    RUN.mkdir(exist_ok=True)
    lock=json.loads((SOURCE/'INPUT_CODE_LOCK.json').read_text())
    lock.update(json.loads((SOURCE/'SOURCE_ARTIFACT_LOCK.json').read_text()))
    for name,expected in json.loads((SOURCE/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():
        lock[str(SOURCE/name)]=expected
    for name in ['metrics.csv','curves.csv','annual.csv','ranking.csv']:
        lock[str(SOURCE/name)]=digest(SOURCE/name)
    lock[Path(__file__).name]=digest(__file__)
    for name,expected in lock.items(): assert digest(name)==expected,name
    if (RUN/'INPUT_CODE_LOCK.json').exists():
        assert json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())==lock
    else: (RUN/'INPUT_CODE_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    source_metrics=pd.read_csv(SOURCE/'metrics.csv',dtype={'seed':str})
    source_curves=pd.read_csv(SOURCE/'curves.csv',dtype={'seed':str})
    dates=data['panel'].dates; metric_rows=[]; attribution=[]; yearly_attr=[]; curve_rows=[]
    buckets=[]; paired=[]; seed_pairs=[]; concentration=[]; checks=[]; saved={}
    for model in MODELS:
        available=forecasts[model]; score=ensemble_predictions(available,data['mask'][ids])
        w5=np.stack([target_weights(s,m,5,'equal') for s,m in zip(score,data['mask'][ids])])
        w10=np.stack([target_weights(s,m,10,'equal') for s,m in zip(score,data['mask'][ids])])
        band=2*w10-w5
        assert band.min()>-1e-12
        np.testing.assert_allclose(band.sum(1),1.,atol=1e-12)
        np.testing.assert_allclose(.5*w5+.5*band,w10,atol=1e-12)
        seed_values=source_metrics[(source_metrics.model==model)&(source_metrics.seed!='ensemble')&
            (source_metrics.cost_multiplier==1)&source_metrics.k.isin([5,10])].pivot(index='seed',columns='k',values='annualized_return')
        for seed,row in seed_values.iterrows():
            seed_pairs.append(dict(model=model,seed=seed,top5_CAGR=row[5],top10_CAGR=row[10],gap=row[5]-row[10]))
        # Forecast-level diagnostic excludes unobserved labels, without reranking.
        for j,i in enumerate(ids):
            observed=data['labelmask'][i]; row=dict(model=model,entry_date=str(dates[data['entry'][i]].date()),
                year=int(dates[data['entry'][i]].year))
            for name,weight in [('top5',w5[j]),('ranks6_10',band[j]),('top10',w10[j])]:
                mass=float(weight[observed].sum())
                row[name+'_observed_weight']=mass
                row[name+'_observed_return']=float(weight[observed]@data['y'][i,observed]/mass) if mass else np.nan
            buckets.append(row)
        for policy,weights,k in [('top5',w5,5),('top10',w10,10),('ranks6_10',band,None)]:
            for mult in [1.,0.]:
                daily,pnl,fee,exposure=trace(data,ids,weights,mult)
                # Supply only the fixed allocation plan to the frozen production
                # engine; its price, order, cost and settlement accounting stays intact.
                with patch('pit_research_engine.target_weights',side_effect=iter(weights)):
                    _,production,_=continuous_backtest(data,ids,score,k=5,cost_multiplier=mult)
                np.testing.assert_allclose(daily.net,production.net,atol=1e-12,rtol=1e-10)
                values=performance(daily.net); tag=dict(model=model,policy=policy,cost_multiplier=mult)
                metric_rows.append(dict(**tag,**values,cost_dollars=float(fee.sum())))
                curve_rows.append(daily.assign(**tag))
                if k is not None:
                    independent=replay(data,ids,score,k,mult)
                    np.testing.assert_allclose(daily.nav,independent,atol=1e-6,rtol=1e-11)
                    checks.append(dict(**tag,independent_nav_max_difference=float(np.abs(daily.nav-independent).max())))
                if k is not None and mult==1:
                    previous=source_curves[(source_curves.model==model)&(source_curves.seed=='ensemble')&
                        (source_curves.k==k)&(source_curves.cost_multiplier==1)]
                    np.testing.assert_allclose(daily.net,previous.net,atol=1e-12,rtol=1e-10)
                net=pnl-fee
                np.testing.assert_allclose(net.sum()/1e6,values['cumulative_return'],atol=1e-10)
                if mult==1:
                    saved[model,policy]=daily.net.to_numpy()
                    for node,ticker in enumerate(data['panel'].tickers):
                        attribution.append(dict(**tag,ticker=ticker,gross_PnL_dollars=float(pnl[:,node].sum()),
                            fees_dollars=float(fee[:,node].sum()),net_PnL_dollars=float(net[:,node].sum()),
                            cumulative_return_contribution=float(net[:,node].sum()/1e6),
                            held_days=int((exposure[:,node]>1e-12).sum()),mean_weight=float(exposure[:,node].mean())))
                    years=pd.to_datetime(daily.date).dt.year.to_numpy()
                    for year in [2022,2023,2024,2025]:
                        mask=years==year; first=np.flatnonzero(mask)[0]
                        starting=1e6 if first==0 else daily.nav.iloc[first-1]
                        np.testing.assert_allclose(net[mask].sum()/starting,
                            performance(daily.net.to_numpy()[mask])['cumulative_return'],atol=1e-10)
                        for node,ticker in enumerate(data['panel'].tickers):
                            yearly_attr.append(dict(**tag,year=year,ticker=ticker,
                                net_PnL_dollars=float(net[mask,node].sum()),
                                year_return_contribution=float(net[mask,node].sum()/starting)))
                        for count in [1,3,5]:
                            contribution=net[mask].sum(0)/starting
                            idx=np.argsort(-contribution)[:count]
                            concentration.append(dict(model=model,policy=policy,year=year,top_contributors=count,
                                contributor_return=float(contribution[idx].sum()),
                                annual_return=float(contribution.sum()),tickers=','.join(np.asarray(data['panel'].tickers)[idx])))
        paired.append(dict(model=model,comparison='top5 minus top10 Sharpe',
            **bootstrap_difference(saved[model,'top5'],saved[model,'top10'])))
        print(f'Reconciled top5, top10 and ranks6–10 attribution: {model}',flush=True)
    tables={'metrics':metric_rows,'curves':pd.concat(curve_rows,ignore_index=True),
        'stock_attribution':attribution,'year_stock_attribution':yearly_attr,'forecast_buckets':buckets,
        'paired_bootstrap':paired,'seed_pairs':seed_pairs,'year_concentration':concentration}
    for name,values in tables.items(): pd.DataFrame(values).to_csv(RUN/(name+'.csv'),index=False)
    audit=dict(status='passed',new_model_fits=0,diagnostic_paths=18,
        original_paths_unchanged=6,independent_top5_top10_paths=12,
        diagnostic_paths_matched_frozen_production_engine=18,
        signal_weight_identity='top10 = 0.5 * top5 + 0.5 * ranks6_10, exact including fractional boundary ties',
        stock_dollar_contributions_reconcile=True,year_stock_contributions_reconcile=True,
        checks=checks,limitations=['All analysis is post hoc on already inspected 2022–2025',
            'Ranks6–10 is a diagnostic allocation, not a new recommendation',
            'CAGRs do not add; dollar P&L contributions divided by initial NAV reconcile cumulative return',
            'Different K changes weights, compounding, turnover, cash scaling and liquidity limits',
            'Observed forecast returns exclude missing outcomes without reranking; account attribution uses verified settlements'])
    (RUN/'AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({key:audit[key] for key in ['status','new_model_fits','diagnostic_paths','original_paths_unchanged']}))


if __name__=='__main__': main()
