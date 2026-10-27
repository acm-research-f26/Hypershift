"""Independent value-position replay of fixed short accounts after completion."""
import numpy as np
import pandas as pd
import run_ndx_weak_market_short as study
from ndx_drawdown_engine import allocations, daily_allocation, risk_inputs, exposure_series, market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from pit_research_engine import cost_rate


def value_replay(data,ids,w,borrow,cost):
    """Carry dollar positions through returns; never call the signed engine."""
    p,dates = data['marks'],data['panel'].dates
    first,last = int(data['entry'][ids[0]]),len(p)-1
    schedule = {int(data['origin'][i]):w[k] for k,i in enumerate(ids)}
    actions = {dates.get_loc(pd.Timestamp(e['date'])):e for e in data['events']}
    value = np.zeros(p.shape[1]); cash = nav = 1e6
    output=[]
    def submit(day):
        price = p[day]; valid = np.isfinite(price)&(price>0)
        desired_dollars = nav*schedule[day]-value
        shares = np.divide(desired_dollars,price,out=np.zeros_like(value),where=valid)
        cap = np.divide(.01*np.nan_to_num(data['adv'][day]),price,out=np.zeros_like(value),where=valid)
        return np.clip(shares,-cap,cap),data['adv'][day].copy()
    pending=submit(first-1)
    for day in range(first,last+1):
        old=nav
        cash -= np.maximum(-value,0).sum()*borrow*(dates[day]-dates[day-1]).days/365
        ratio=np.divide(p[day],p[day-1],out=np.ones_like(value),where=np.isfinite(p[day])&np.isfinite(p[day-1]))
        value *= ratio
        if day in actions:
            e=actions[day]; node=data['panel'].tickers.index(e['security_id'])
            principal=value[node]; stock=abs(principal)*e['stock_fraction_at_completion']
            cash += principal-stock*cost_rate(stock,data['adv'][day-1,node],cost)
            value[node]=0.
        if day==last:pending=None
        if pending is not None:
            shares,adv=pending
            trade=np.nan_to_num(shares*p[day]);trade[~data['tradable'][day]]=0.
            sold=np.minimum(trade,0.); bought=np.maximum(trade,0.)
            reserve=np.maximum(-(value+sold),0).sum()*borrow*28/365
            cash_after_sales=cash-sold.sum()-(np.abs(sold)*cost_rate(sold,adv,cost)).sum()
            required=(bought*(1+cost_rate(bought,adv,cost))).sum()
            scale=min(1.,max(0.,(cash_after_sales-reserve)/required)) if required>0 else 1.
            actual=sold+bought*scale
            cash -= actual.sum()+(np.abs(actual)*cost_rate(actual,adv,cost)).sum()
            value += actual
        pending=None
        if day==last:
            cash += value.sum()-(np.abs(value)*cost_rate(value,data['adv'][day-1],cost)).sum()
            value[:]=0.
        nav=cash+value.sum();output.append(nav/old-1)
        if day<last and day in schedule:pending=submit(day)
    return np.asarray(output)


def audit():
    study.verify()
    complete=__import__('json').loads((study.RUN/'COMPLETE.json').read_text())
    for name,expected in complete['output_sha256'].items():
        assert study.parent.sha(study.RUN/name)==expected,name
    data=study.parent.load_data()
    ids=np.concatenate([study.parent.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    scores=study.parent.saved_scores(data,ids);market=market_inputs(data)
    q=pd.Series(market['qqq'])
    flags=dict(ma200=(q<q.rolling(200).mean()).to_numpy(),ma100=(q<q.rolling(100).mean()).to_numpy(),
               ma200_mom63=((q<q.rolling(200).mean())&(q/q.shift(63)-1<0)).to_numpy())
    combined=append_qqq(data,etf_data(data,'QQQ'))
    prepared={}
    for group in study.parent.GROUPS:
        base_w=allocations(data,ids,scores[group],'issuer5')
        risk=risk_inputs(data,daily_allocation(data,ids,base_w))
        for s in study.recipes():
            e=exposure_series(study.RULES[s['base']],market,risk)[data['origin'][ids]].copy()
            short=np.zeros(len(e));on=flags.get(s['gate'],np.zeros(len(q),bool))[data['origin'][ids]]
            cap=.20/np.maximum(market['qqqvol20'][data['origin'][ids]],.05)
            if s['mode']=='cash':e[on]=0.
            elif s['mode'].startswith('switch'):
                short[on]=np.minimum(int(s['mode'][6:])/100,cap[on]);e[on]=0.
            elif s['mode'].startswith('hedge'):
                short[on]=np.minimum(np.minimum(int(s['mode'][5:])/100,1-e[on]),cap[on])
            prepared[s['id'],group]=np.column_stack([base_w*e[:,None],-short])
    targets=pd.read_csv(study.RUN/'targets.csv')
    for (recipe,group),block in targets.groupby(['recipe','group'],sort=False):
        w=prepared[recipe,group]
        np.testing.assert_allclose(block.long_target,w[:,:-1].sum(1),atol=1e-14)
        np.testing.assert_allclose(block.short_target,-w[:,-1],atol=1e-14)
    metrics=pd.read_csv(study.RUN/'metrics.csv')
    fixed=['full/baseline','vol20/baseline','vol20/ma200/cash','vol20/ma200/switch50',
           'vol20/ma200/switch100','vol20/ma200/hedge25','vol25trend/ma100/switch50']
    sample=metrics[(metrics.recipe.isin(fixed))&(metrics.cost_multiplier==1)&(metrics.borrow_rate==.03)]
    sample=pd.concat([sample,metrics[(metrics.recipe.isin(['vol20/ma200/switch100','vol20/ma200/hedge50']))&
                                   ((metrics.cost_multiplier==4)|(metrics.borrow_rate==.10))]])
    errors=[]
    with np.load(study.RUN/'paths.npz') as saved:
        paths=dict(zip(saved['ids'].astype(str),saved['returns'].T))
        for row in sample.itertuples():
            actual=value_replay(combined,ids,prepared[row.recipe,row.group],row.borrow_rate,row.cost_multiplier)
            nav_error=float(np.max(np.abs(1e6*np.cumprod(1+actual)-1e6*np.cumprod(1+paths[row.id]))))
            assert nav_error<1e-5,(row.id,nav_error)
            errors.append(dict(id=row.id,max_nav_error=nav_error))
    result=dict(status='passed',all_117_target_schedules_independently_reconstructed=True,
                replayed_accounts=len(errors),max_nav_error=max(r['max_nav_error'] for r in errors),
                accounts=errors,source_sha256=study.parent.sha(__file__),
                limitations='Independent dollar-position accounting replay; same reviewed prices, parent forecasts/allocations and cost formula. No independent vendor or broker validation.')
    study.parent.write(study.RUN/'INDEPENDENT_AUDIT.json',result)
    study.parent.write(study.DOC/'INDEPENDENT_AUDIT.json',result)
    print('Independent target checks and',len(errors),'value-position replays passed; max NAV error',result['max_nav_error'],flush=True)


if __name__=='__main__':audit()
