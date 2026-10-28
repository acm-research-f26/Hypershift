"""Trace saved training-selected issuer screens and target replacements."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_leader_trace import trace_risk

DOC=Path('docs/rishi_mechanism_20261027');SOURCE=Path('results/RISHI_TRAINING_EXCLUSION_20261028')


def main():
    out=DOC/'execution_batch40_exclusion_trace';large=Path('results/RISHI_TRAINING_EXCLUSION_TRACE_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();tickers=np.asarray(data['panel'].tickers);issuer=np.array([engine.ISSUERS.get(t,t) for t in tickers])
    saved=np.load(SOURCE/'PATHS.npz',allow_pickle=False);dates=saved['dates'];year=pd.DatetimeIndex(dates).year.to_numpy()
    inputs=[SOURCE/'PATHS.npz',Path(__file__),DOC/'TRAINING_EXCLUSION_TRACE_PROTOCOL.md',Path('research/analysis/rishi_leader_trace.py'),
        Path('research/core/ndx_drawdown_engine.py'),DOC/'execution_batch39_training_exclusion/CONTRASTS.csv']
    records=[];receipts=[];traces={};targets=[];stockrows=[];annual=[]
    for model in ('E','E_Hlabel'):
        base_weights=None
        for n in (0,1,3):
            key=f'{model}_k10_exclude{n}';weight_path=SOURCE/f'{key}_weights.npz';ledger_path=SOURCE/f'{key}_ledger.csv';inputs.extend([weight_path,ledger_path])
            with np.load(weight_path,allow_pickle=False) as z:
                ids=z['ids'].copy();weights=z['weights'].copy();np.testing.assert_array_equal(z['security_ids'],tickers)
            if n==0:base_weights=weights.copy()
            else:
                for i,origin in enumerate(ids):
                    for j in np.flatnonzero(weights[i]!=base_weights[i]):
                        targets.append(dict(model=model,exclude=n,origin=int(origin),security_id=str(tickers[j]),issuer=str(issuer[j]),
                            original_weight=float(base_weights[i,j]),screen_weight=float(weights[i,j]),change=float(weights[i,j]-base_weights[i,j])))
            metric,net,ledger,trace=trace_risk(data,ids,weights,np.ones(len(data['panel'].dates)))
            old=pd.read_csv(ledger_path)
            if trace['dates']!=dates.tolist() or old.date.tolist()!=dates.tolist():raise ValueError('Dates mismatch')
            daily=float(np.max(abs(net-saved[key])));nav=float(np.max(abs(trace['nav']-old.nav.to_numpy()))/1e6)
            if daily>1e-10 or nav>1e-8 or np.max(abs(trace['units'][-1]))>1e-8:raise ValueError('Trace mismatch')
            prev=np.r_[1e6,trace['nav'][:-1]];gross=trace['gross_dollars']/prev[:,None];fees=trace['fees']/prev[:,None];netc=trace['net_contribution']
            exposure=np.where(trace['units']!=0,np.nan_to_num(trace['units']*trace['marks']),0.)/trace['nav'][:,None]
            np.testing.assert_allclose(gross-fees,netc,atol=1e-12,rtol=0)
            np.testing.assert_allclose(netc.sum(1),net,atol=1e-10,rtol=0)
            np.savez_compressed(large/f'{key}.npz',net=netc,gross=gross,fees=fees,exposure=exposure,dates=dates,security_ids=tickers.astype('U'))
            traces[model,n]=netc
            receipts.append(dict(model=model,exclude=n,daily_error=daily,nav_error=nav,cash_error=trace['cash_error'],share_error=trace['share_error']))
            records.append(dict(model=model,exclude=n,annual_net=float(net.mean()*252),annual_gross=float(gross.sum(1).mean()*252),
                annual_fee=float(fees.sum(1).mean()*252),mean_exposure=float(exposure.sum(1).mean())))
            for j,ticker in enumerate(tickers):stockrows.append(dict(model=model,exclude=n,security_id=ticker,issuer=issuer[j],
                net=float(netc[:,j].mean()*252),gross=float(gross[:,j].mean()*252),fee=float(fees[:,j].mean()*252),exposure=float(exposure[:,j].mean())))
            print(key,flush=True)
        for n in (1,3):
            diff=traces[model,n]-traces[model,0]
            np.testing.assert_allclose(diff.sum(1),saved[f'{model}_k10_exclude{n}']-saved[f'{model}_k10_exclude0'],atol=1e-10,rtol=0)
            for y in (2022,2023,2024,2025):
                for j,ticker in enumerate(tickers):annual.append(dict(model=model,exclude=n,year=y,security_id=ticker,issuer=issuer[j],difference=float(diff[year==y,j].mean()*252)))
    stocks=pd.DataFrame(stockrows);changes=[]
    for model in ('E','E_Hlabel'):
        base=stocks[(stocks.model==model)&(stocks['exclude']==0)].set_index('security_id')
        for n in (1,3):
            b=stocks[(stocks.model==model)&(stocks['exclude']==n)].set_index('security_id')
            for ticker in tickers:changes.append(dict(model=model,exclude=n,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),
                net_difference=float(b.loc[ticker,'net']-base.loc[ticker,'net']),exposure_difference=float(b.loc[ticker,'exposure']-base.loc[ticker,'exposure'])))
    stocks.to_csv(out/'STOCKS.csv',index=False);pd.DataFrame(records).to_csv(out/'ACCOUNTS.csv',index=False)
    change=pd.DataFrame(changes);change.to_csv(out/'PAIRED_STOCKS.csv',index=False)
    change.groupby(['model','exclude','issuer'],as_index=False)[['net_difference','exposure_difference']].sum().to_csv(out/'PAIRED_ISSUERS.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'PAIRED_CALENDAR_STOCKS.csv',index=False);pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    pd.DataFrame(targets).to_csv(out/'TARGET_CHANGES.csv',index=False)
    input_paths=inputs+[Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    base=Path('data/nasdaq_monthly_pit/assembled');input_paths.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(records),target_change_rows=len(targets),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        limits='Selected-history post-score accounting and target differences, not model re-inference'),indent=2)+'\n')
    print(change.groupby(['model','exclude']).net_difference.sum().to_string())


if __name__=='__main__':main()
