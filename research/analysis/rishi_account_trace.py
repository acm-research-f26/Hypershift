"""Observe frozen canonical accounting without modifying its execution policy."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pit_research_engine import continuous_backtest, cost_rate, load_data
from ndx_drawdown_engine import ISSUERS


def write_axes(path, dates, tickers):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(dict(dates=list(dates),security_ids=list(tickers),
                       axes=['date','security_id']),stream,indent=2)
        stream.write('\n')


def traced_backtest(data, ids, scores, k, aum=1_000_000.):
    source,start=inspect.getsourcelines(continuous_backtest)
    anchors={}
    for offset,line in enumerate(source):
        stripped=line.strip()
        for key,fragment in [('settlement','cash+=value-fee;'),('fill','cash-=float(dollars.sum())+fee;'),
                             ('terminal','cash+=float(dollars.sum())-fee;units[:]'),
                             ('snapshot','ledger.append(dict(')]:
            if stripped.startswith(fragment):anchors[start+offset]=key
    if len(anchors)!=4:raise ValueError('Canonical trace anchors changed')
    n=len(data['panel'].tickers);snapshots=[];orders=[];flows={};fees={};changes={}
    def observe(frame,event,arg):
        if frame.f_code is not continuous_backtest.__code__:return None
        if event!='line' or frame.f_lineno not in anchors:return observe
        v=frame.f_locals;day=v['day'];kind=anchors[frame.f_lineno]
        if kind=='snapshot':
            if not snapshots or snapshots[-1][0]!=day:
                snapshots.append((day,v['old'],v['cash'],v['units'].copy()))
            return observe
        if kind=='fill':
            delta=v['delta'].copy();dollars=v['dollars'].copy()
            charge=np.abs(dollars)*v['rates'];signal=v['pending']['signal']
        elif kind=='terminal':
            delta=-v['units'].copy();dollars=-v['dollars'].copy()
            charge=-dollars*cost_rate(-dollars,data['adv'][day-1],1.);signal=None
        else:
            delta=np.zeros(n);delta[v['node']]=-v['units'][v['node']]
            dollars=np.zeros(n);dollars[v['node']]=-v['value']
            charge=np.zeros(n);charge[v['node']]=v['fee'];signal=None
        fees.setdefault(day,np.zeros(n))[:]+=charge
        changes.setdefault(day,np.zeros(n))[:]+=delta
        flows[day]=flows.get(day,0.)-float(dollars.sum())-float(charge.sum())
        for node in np.flatnonzero((delta!=0)|(charge!=0)):
            orders.append(dict(day=day,date=str(data['panel'].dates[day].date()),
                security_id=data['panel'].tickers[node],kind=kind,signal_day=signal,
                units=float(delta[node]),notional=float(dollars[node]),fee=float(charge[node])))
        return observe
    previous=sys.gettrace()
    try:
        sys.settrace(observe)
        metrics,curve,ledger=continuous_backtest(data,ids,scores,k,aum=aum)
    finally:sys.settrace(previous)
    units=np.stack([s[3] for s in snapshots]);charge=np.stack([fees.get(s[0],np.zeros(n)) for s in snapshots])
    gross=np.zeros_like(units);running_cash=aum;running_units=np.zeros(n);cash_error=share_error=0.
    for row,(day,old,cash,position) in enumerate(snapshots):
        if row:gross[row]=np.nan_to_num(units[row-1]*(data['marks'][day]-data['marks'][day-1]))
        running_cash+=flows.get(day,0.);running_units+=changes.get(day,np.zeros(n))
        cash_error=max(cash_error,abs(running_cash-cash))
        share_error=max(share_error,float(np.max(np.abs(running_units-position))))
    contribution=(gross-charge)/np.array([s[1] for s in snapshots])[:,None]
    np.testing.assert_allclose(contribution.sum(1),curve.net,rtol=0,atol=1e-10)
    np.testing.assert_allclose(charge.sum(1),ledger.fee,rtol=1e-10,atol=1e-6)
    if cash_error>1e-6 or share_error>1e-8:raise AssertionError('Cash/share trace mismatch')
    return metrics,curve,ledger,dict(units=units,net_contribution=contribution,gross_dollars=gross,
        fees=charge,orders=orders,cash_reconciliation_error=cash_error,share_reconciliation_error=share_error)


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch3_trace')
    large=Path('results/RISHI_MECHANISM_TRACE_20261027')
    for path in (out,large):
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True)
    data=load_data();source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    expected=pd.read_csv(source/'ACCOUNT_CURVES.csv');rows=[];receipts=[]
    for candidate in ['B','D','E','Hlabel','E_Hlabel']:
        score=np.load(source/f'scores_{candidate}_ten_seed.npz')
        for k in (5,10):
            metrics,curve,ledger,trace=traced_backtest(data,score['ids'],score['scores'],k)
            key=f'{candidate}_ten_seed_k{k}'
            prior=expected[expected.id==f'{candidate}/ten_seed/{k}']
            if curve.date.tolist()!=prior.date.tolist():raise AssertionError('Prior date mismatch')
            error=float(np.max(np.abs(curve.net.to_numpy()-prior.net.to_numpy())))
            if error>1e-10:raise AssertionError('Prior account mismatch')
            if not (large/'AXES.json').exists():write_axes(large/'AXES.json',curve.date,data['panel'].tickers)
            np.savez_compressed(large/f'{key}.npz',**{x:trace[x] for x in ['units','net_contribution','gross_dollars','fees']})
            pd.DataFrame(trace['orders']).to_csv(large/f'{key}_fills.csv',index=False)
            for j,ticker in enumerate(data['panel'].tickers):
                rows.append(dict(candidate=candidate,k=k,security_id=ticker,
                    annual_arithmetic_contribution=float(trace['net_contribution'][:,j].mean()*252),
                    gross_dollars=float(trace['gross_dollars'][:,j].sum()),fees=float(trace['fees'][:,j].sum()),
                    holding_days=int((trace['units'][:,j]>1e-12).sum())))
            receipts.append(dict(account=key,daily_error=error,cash_error=trace['cash_reconciliation_error'],
                share_error=trace['share_reconciliation_error'],fills=len(trace['orders'])))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'STOCK_CONTRIBUTIONS.csv',index=False)
    frame['issuer']=frame.security_id.map(lambda x:ISSUERS.get(x,x))
    frame.groupby(['candidate','k','issuer'],as_index=False).agg(
        annual_arithmetic_contribution=('annual_arithmetic_contribution','sum'),
        gross_dollars=('gross_dollars','sum'),fees=('fees','sum')).to_csv(out/'ISSUER_CONTRIBUTIONS.csv',index=False)
    (out/'RECONCILIATION.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts,indent=2))


if __name__=='__main__':main()
