"""Retrospective nontradable interest sidecar on saved account cash."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_mechanism_batch1 import account_metrics

DOC=Path('docs/rishi_mechanism_20261027')
ROOT=Path('results/RISHI_MECHANISM_RISK_20261027')
FACTOR=Path('data/factors/F-F_Research_Data_5_Factors_2x3_daily.csv')
BOOKS=[('E','ten_seed','equal5'),('E','ten_seed','issuer5'),('B','ten_seed','equal5'),
       ('D','ten_seed','equal5'),('REV5','ten_seed','equal5'),('Hlabel','ten_seed','equal5'),
       ('EW','benchmark','eligible_weekly'),('QQQ','benchmark','one_etf')]


def sidecar(nav,cash,rf):
    nav,cash,rf=[np.asarray(a,dtype=float) for a in (nav,cash,rf)]
    if not (len(nav)==len(cash)==len(rf)) or len(nav)<2 or not all(np.isfinite(a).all() for a in (nav,cash,rf)):
        raise ValueError('Invalid sidecar inputs')
    if (nav<=0).any() or (cash < -1e-5).any() or (rf<=-1).any():raise ValueError('Invalid economic values')
    interest=np.zeros(len(nav));balance=np.zeros(len(nav))
    for i in range(1,len(nav)):
        interest[i]=rf[i]*(cash[i-1]+balance[i-1])
        balance[i]=balance[i-1]+interest[i]
    wealth=nav+balance
    if (wealth<=0).any():raise ValueError('Invalid augmented wealth')
    returns=wealth/np.r_[1e6,wealth[:-1]]-1
    return interest,balance,wealth,returns


def rf_series():
    rows=[]
    with FACTOR.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.reader(stream):
            if row and len(row[0].strip())==8 and row[0].strip().isdigit():
                if len(row)!=7:raise ValueError('Malformed factor row')
                rows.append((pd.to_datetime(row[0].strip(),format='%Y%m%d'),float(row[6])/100))
    s=pd.Series(dict(rows),dtype=float)
    if len(s)!=len(rows):raise ValueError('Duplicate factor date')
    return s


def main():
    out=DOC/'execution_batch29_cash_yield'
    out.mkdir(parents=True,exist_ok=False)
    rates=rf_series();inputs=[FACTOR,Path(__file__),Path('research/analysis/rishi_mechanism_batch1.py'),DOC/'CASH_YIELD_PROTOCOL.md']
    rows=[];ledgers=[];dates=None;max_zero_error=0.
    for book in BOOKS:
        for policy in ('full','vol20','fixed'):
            key='/'.join((*book,policy,'modeled1'));path=ROOT/(key.replace('/','__')+'.csv');inputs.append(path)
            df=pd.read_csv(path,parse_dates=['date'])
            index=pd.DatetimeIndex(df.date)
            if len(index)!=1003 or index.has_duplicates or not index.is_monotonic_increasing:raise ValueError('Invalid calendar')
            if dates is None:dates=index
            elif not index.equals(dates):raise ValueError('Calendar mismatch')
            rf=rates.reindex(index).to_numpy()
            nav=df.nav.to_numpy();cash=df.cash.to_numpy();net=df.net.to_numpy()
            if not np.isfinite(net).all() or not np.allclose(nav/np.r_[1e6,nav[:-1]]-1,net,rtol=0,atol=1e-12):raise ValueError('Source NAV/return mismatch')
            interest,balance,wealth,ret=sidecar(nav,cash,rf)
            _,zero,zero_nav,zero_ret=sidecar(nav,cash,np.zeros(len(nav)))
            err=float(np.max(np.abs(zero_ret-net)));max_zero_error=max(max_zero_error,err)
            if not np.array_equal(zero_nav,nav) or not np.all(zero==0) or err>1e-12:raise ValueError('Zero-yield identity fails')
            source=account_metrics(net);aug=account_metrics(ret)
            rows.append(dict(id=key,source_sharpe=source['sharpe'],sidecar_sharpe=aug['sharpe'],
                source_cagr=source['annualized_return'],sidecar_cagr=aug['annualized_return'],
                source_mdd=source['maximum_drawdown'],sidecar_mdd=aug['maximum_drawdown'],
                source_terminal=nav[-1],sidecar_terminal=wealth[-1],interest_total=interest.sum(),
                annual_arithmetic_return_difference=float(252*np.mean(ret-net))))
            ledgers.append(pd.DataFrame(dict(id=key,date=index.strftime('%Y-%m-%d'),source_nav=nav,source_cash=cash,
                rf=rf,interest=interest,sidecar=balance,augmented_wealth=wealth,source_return=net,augmented_return=ret)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'SUMMARY.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(out/'DAILY_LEDGER.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(rows),days=len(dates),max_zero_return_error=max_zero_error,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2)+'\n')
    print(frame[['id','source_sharpe','sidecar_sharpe','source_cagr','sidecar_cagr','interest_total']].to_string(index=False))


if __name__=='__main__':main()
