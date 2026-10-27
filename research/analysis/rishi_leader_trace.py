"""Read-only tracing of the twelve published leader accounts; no policy edits."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
import pit_research_engine as pit


def _anchors(function,fragments):
    lines,start=inspect.getsourcelines(function);result={}
    for offset,line in enumerate(lines):
        for key,text in fragments.items():
            if line.strip().startswith(text):result[start+offset]=key
    if len(result)!=len(fragments):raise ValueError('Frozen trace anchors changed')
    return result


def _reconcile(data,snapshots,fills,submitted,path,source_fees,aum):
    n=len(data['panel'].tickers);count=len(snapshots)
    units=np.stack([s['units'] for s in snapshots]);fees=np.zeros((count,n));deltas=np.zeros_like(fees);flow=np.zeros(count)
    day_to_row={s['day']:i for i,s in enumerate(snapshots)}
    for item in fills:
        row=day_to_row[item['day']];node=data['panel'].tickers.index(item['security_id'])
        fees[row,node]+=item['fee'];deltas[row,node]+=item['units'];flow[row]-=item['notional']+item['fee']
    cash=aum+np.cumsum(flow);recorded_cash=np.array([s['cash'] for s in snapshots])
    cash_error=float(np.max(np.abs(cash-recorded_cash)))
    share_error=float(np.max(np.abs(np.cumsum(deltas,axis=0)-units)))
    marks=data['marks'][[s['day'] for s in snapshots]]
    nav=cash+np.where(units!=0,np.nan_to_num(units*marks),0).sum(1)
    recorded_nav=np.array([s['nav'] for s in snapshots]);nav_error=float(np.max(np.abs(nav-recorded_nav))/aum)
    old=np.array([s['old'] for s in snapshots]);gross=np.zeros_like(units)
    gross[1:]=np.nan_to_num(units[:-1]*(marks[1:]-marks[:-1]))
    contribution=(gross-fees)/old[:,None]
    np.testing.assert_allclose(contribution.sum(1),path,rtol=0,atol=1e-10)
    np.testing.assert_allclose(nav/np.r_[aum,nav[:-1]]-1,path,rtol=0,atol=1e-10)
    if source_fees is not None:np.testing.assert_allclose(fees.sum(1),source_fees,rtol=1e-10,atol=1e-6)
    if cash_error>1e-6 or share_error>1e-8 or nav_error>1e-8:raise AssertionError('Recorded operations do not reconstruct account')
    # Source cancels any pending purchase on its predetermined terminal close.
    completed={(item.get('signal_day'),item['security_id']) for item in fills if item['kind']=='fill'}
    for item in submitted:
        if (item['signal_day'],item['security_id']) not in completed:
            status='terminal_cancel' if item['fill_day']==snapshots[-1]['day'] else 'cap_zero'
            if item['frozen_delta']!=0 and status!='terminal_cancel':raise AssertionError('Unexplained pending order')
            fills.append(dict(day=item['fill_day'],signal_day=item['signal_day'],security_id=item['security_id'],
                kind='cancellation',status=status,units=0.,notional=0.,fee=0.))
    return dict(units=units,marks=marks,cash=cash,nav=nav,fees=fees,gross_dollars=gross,net_contribution=contribution,
        submitted=submitted,fills=fills,dates=[str(data['panel'].dates[s['day']].date()) for s in snapshots],
        security_ids=list(data['panel'].tickers),cash_error=cash_error,share_error=share_error,relative_nav_error=nav_error)


def trace_risk(data,ids,weekly,exposure,aum=1_000_000.):
    function=engine.simulate
    outer=_anchors(function,dict(settlement='cash += value-fee',fill='cash -= float(dollars.sum())+fee',
        terminal='cash += float(dollars.sum())-fee',snapshot='paths.append(nav/old-1)'))
    order_code=next(c for c in function.__code__.co_consts if isinstance(c,types.CodeType) and c.co_name=='order')
    lines,start=inspect.getsourcelines(function)
    order_line=next(start+i for i,line in enumerate(lines) if line.strip().startswith('return np.clip(desired-units,'))
    submitted=[];fills=[];snapshots=[];order_days=set();n=len(data['panel'].tickers)
    def record(day,kind,delta,dollars,charge,frozen=None):
        for j in np.flatnonzero((delta!=0)|(charge!=0)|((frozen!=0) if frozen is not None else False)):
            status='filled'
            if frozen is not None:
                if not data['tradable'][day,j]:status='missing_fill'
                elif abs(delta[j]-frozen[j])>1e-10:status='cash_scaled' if frozen[j]>0 else 'position_limited'
            fills.append(dict(day=day,signal_day=day-1 if kind=='fill' else None,security_id=data['panel'].tickers[j],
                kind=kind,status=status,units=float(delta[j]),notional=float(dollars[j]),fee=float(charge[j])))
    def observe(frame,event,arg):
        if frame.f_code not in (function.__code__,order_code):return None
        if event!='line':return observe
        v=frame.f_locals
        if frame.f_code is order_code:
            if frame.f_lineno==order_line and v['day'] not in order_days:
                day=v['day'];order_days.add(day);raw=v['desired']-v['units'];frozen=np.clip(raw,-v['cap'],v['cap'])
                for j in np.flatnonzero((raw!=0)|(frozen!=0)):
                    submitted.append(dict(signal_day=day,fill_day=day+1,security_id=data['panel'].tickers[j],
                        desired_delta=float(raw[j]),frozen_delta=float(frozen[j]),cap=float(v['cap'][j]),
                        cap_binding=bool(abs(raw[j])>v['cap'][j]+1e-10),target_weight=float(v['last_weights'][j])))
            return observe
        if frame.f_lineno not in outer:return observe
        kind=outer[frame.f_lineno];day=v['day']
        if kind=='snapshot':
            snapshots.append(dict(day=day,old=v['old'],nav=v['nav'],cash=v['cash'],units=v['units'].copy()))
        elif kind=='fill':record(day,'fill',v['delta'].copy(),v['dollars'].copy(),np.abs(v['dollars'])*v['rates'],v['pending'][0])
        elif kind=='terminal':
            dollars=v['dollars'].copy();record(day,'terminal',-v['units'].copy(),-dollars,
                dollars*engine.cost_rate(dollars,data['adv'][day-1],1.))
        else:
            delta=np.zeros(n);delta[v['node']]=-v['units'][v['node']]
            dollars=np.zeros(n);dollars[v['node']]=-v['value'];fee=np.zeros(n);fee[v['node']]=v['fee']
            record(day,'settlement',delta,dollars,fee)
        return observe
    previous=sys.gettrace()
    try:
        sys.settrace(observe);metrics,path,ledger=function(data,ids,weekly,exposure,'weekly',detailed=True,aum=aum)
    finally:sys.settrace(previous)
    return metrics,path,ledger,_reconcile(data,snapshots,fills,submitted,path,ledger.fee.to_numpy(),aum)


def trace_etf(data,ticker,aum=1_000_000.):
    function=pit.etf_backtest
    anchors=_anchors(function,dict(after_buy='if day==last:',terminal='cash+=dollars*(1-float(cost_rate(',snapshot='rows.append(dict('))
    snapshots=[];fills=[];prices=None
    def observe(frame,event,arg):
        nonlocal prices
        if frame.f_code is not function.__code__:return None
        if event!='line' or frame.f_lineno not in anchors:return observe
        v=frame.f_locals;day=v['day'];kind=anchors[frame.f_lineno]
        if kind=='after_buy' and day==v['first']:
            fills.append(dict(day=day,signal_day=None,security_id=ticker,kind='fill',status='dollar_budget_fill',
                units=float(v['units']),notional=float(v['dollars']),fee=float(v['dollars']*v['rate'])))
        elif kind=='terminal':
            rate=float(pit.cost_rate(v['dollars'],v['dv'].iloc[:day].tail(20).mean(),1.))
            fills.append(dict(day=day,signal_day=None,security_id=ticker,kind='terminal',status='filled',
                units=-float(v['units']),notional=-float(v['dollars']),fee=float(v['dollars']*rate)))
        elif kind=='snapshot':
            snapshots.append(dict(day=day,old=v['old'],nav=v['nav'],cash=v['cash'],units=np.array([v['units']])))
            prices=v['prices'].to_numpy()[:,None]
        return observe
    previous=sys.gettrace()
    try:
        sys.settrace(observe);metrics,curve=function(data,ticker,aum=aum)
    finally:sys.settrace(previous)
    observed=dict(panel=types.SimpleNamespace(dates=data['panel'].dates,tickers=[ticker]),marks=prices)
    trace=_reconcile(observed,snapshots,fills,[],curve.net.to_numpy(),None,aum)
    ledger=pd.DataFrame(dict(date=trace['dates'],nav=trace['nav'],cash=trace['cash'],fee=trace['fees'].sum(1)))
    return metrics,curve,ledger,trace


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch9_leader_trace'
    large=Path('results/RISHI_MECHANISM_LEADER_TRACE_20261027')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=pit.load_data();market=engine.market_inputs(data)
    workbook=pd.read_csv(doc/'execution_batch1/WORKBOOK_PATHS.csv')
    reference=pd.read_csv(doc/'execution_batch1/WORKBOOK_METRICS.csv')
    protocol=json.loads(Path('docs/ndx_drawdown_20261026/PROTOCOL.json').read_text())
    recipes={r['id']:r for r in protocol['recipes']};timing=pd.read_csv('runs/ndx_extended_hours_20261024/TIMING.csv')
    receipts=[];stock=[]
    for key in reference.id:
        if key in ['SPY','QQQ']:
            metrics,curve,ledger,trace=trace_etf(data,key);path=curve.net.to_numpy();contract='first_close_dollar_budget'
        else:
            recipe,group=key.rsplit('/',1);spec=recipes[recipe]
            with np.load(f'results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_E_{group}.npz') as saved:
                ids=saved['ids'];scores=saved['scores']
            weights=engine.allocations(data,ids,scores,spec['allocation'])
            risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,weights))
            exposure=engine.exposure_series(spec,market,risk)
            metrics,path,ledger,trace=trace_risk(data,ids,weights,exposure);contract='frozen_shares_next_close'
            lookup={int(data['origin'][i]):timing.iloc[i] for i in ids}
            for order in trace['submitted']:
                t=lookup[order['signal_day']]
                order.update(decision_utc=t.decision_utc,feature_cutoff_utc=t.feature_cutoff_utc,fill_utc=t.fill_utc)
        expected=workbook[workbook.id==key]
        if trace['dates']!=expected.date.tolist():raise AssertionError('Workbook dates')
        error=float(np.max(np.abs(path-expected.net.to_numpy())))
        if error>1e-10:raise AssertionError('Workbook return mismatch')
        row=reference[reference.id==key].iloc[0]
        for field in ['sharpe','annualized_return','annualized_volatility','maximum_drawdown']:
            np.testing.assert_allclose(metrics[field],float(row[field]),rtol=0,atol=1e-10)
        name=key.replace('/','__')
        np.savez_compressed(large/(name+'.npz'),dates=np.asarray(trace['dates'],dtype='U10'),
            security_ids=np.asarray(trace['security_ids'],dtype=str),**{k:trace[k] for k in ['units','marks','cash','nav','fees','gross_dollars','net_contribution']})
        pd.DataFrame(trace['submitted']).to_csv(large/(name+'_orders.csv'),index=False)
        pd.DataFrame(trace['fills']).to_csv(large/(name+'_fills.csv'),index=False)
        for j,ticker in enumerate(trace['security_ids']):
            stock.append(dict(id=key,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),
                annual_arithmetic_contribution=float(trace['net_contribution'][:,j].mean()*252),
                gross_dollars=float(trace['gross_dollars'][:,j].sum()),fees=float(trace['fees'][:,j].sum()),
                holding_days=int((trace['units'][:,j]>1e-12).sum())))
        receipts.append(dict(id=key,order_contract=contract,max_daily_error=error,cash_error=trace['cash_error'],
            share_error=trace['share_error'],relative_nav_error=trace['relative_nav_error'],
            submitted=len(trace['submitted']),fills=len(trace['fills']),binding_caps=sum(o['cap_binding'] for o in trace['submitted']),
            missing_fills=sum(o['status']=='missing_fill' for o in trace['fills']),
            cash_scaled=sum(o['status']=='cash_scaled' for o in trace['fills']),
            terminal_canceled=sum(o['status']=='terminal_cancel' for o in trace['fills'])))
        print(f'Reconciled leader {key}',flush=True)
    sf=pd.DataFrame(stock);sf.to_csv(out/'STOCK_ATTRIBUTION.csv',index=False)
    sf.groupby(['id','issuer'],as_index=False)[['annual_arithmetic_contribution','gross_dollars','fees']].sum().to_csv(out/'ISSUER_ATTRIBUTION.csv',index=False)
    pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),doc/'LEADER_TRACE_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py')]}
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(receipts),stock_rows=len(sf),source_hashes=hashes,
        scope='Canonical-model orders/holdings and accounting identities; not broker records or independent execution policy'),indent=2)+'\n')


if __name__=='__main__':main()
