"""Trace the fixed-validation and model risk-managed stock accounts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_risk import POLICIES
from rishi_leader_trace import trace_risk


def main():
    source=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch52_validation_trace');large=Path('results/RISHI_VALIDATION_FIXED_TRACE_20261029')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);paths=np.load(source/'PATHS.npz',allow_pickle=False);targets=np.load(source/'TARGETS.npz',allow_pickle=False)
    ids=targets['ids'];dates=paths['dates'];years=pd.DatetimeIndex(dates).year;names=np.asarray(data['panel'].tickers);issuers=np.asarray([engine.ISSUERS.get(x,x) for x in names]);np.testing.assert_array_equal(names,targets['security_ids'])
    stocks=[];receipts=[];accounts=[];annual=[];changed=[];hashes={};traces={}
    for group in ('original_five','new_five','ten_seed'):
        for arm in ('original','fixed'):
            key=f'{group}__{arm}__vol20';w=targets[f'{group}__{arm}'];risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,w));exp=engine.exposure_series(POLICIES['vol20'],market,risk)
            metric,net,ledger,trace=trace_risk(data,ids,w,exp);path=source/f'{key}_ledger.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();old=pd.read_csv(path)
            assert trace['dates']==dates.tolist()==old.date.tolist()
            error=float(np.max(abs(net-paths[key])));nav_error=float(np.max(abs(trace['nav']-old.nav))/1e6)
            assert error<=1e-10 and nav_error<=1e-8 and np.max(abs(trace['units'][-1]))<=1e-8
            prev=np.r_[1e6,trace['nav'][:-1]];gross=trace['gross_dollars']/prev[:,None];fees=trace['fees']/prev[:,None];netc=trace['net_contribution']
            held=np.where(trace['units']!=0,np.nan_to_num(trace['units']*trace['marks']),0.)/trace['nav'][:,None]
            np.testing.assert_allclose(gross-fees,netc,rtol=0,atol=1e-12);np.testing.assert_allclose(netc.sum(1),net,rtol=0,atol=1e-10)
            np.savez_compressed(large/f'{key}.npz',net=netc,gross=gross,fees=fees,exposure=held,dates=dates,security_ids=names)
            traces[group,arm]=netc;receipts.append(dict(group=group,arm=arm,daily_error=error,nav_error=nav_error,cash_error=trace['cash_error'],share_error=trace['share_error']))
            accounts.append(dict(group=group,arm=arm,net=252*net.mean(),gross=252*gross.sum(1).mean(),fee=252*fees.sum(1).mean(),exposure=held.sum(1).mean(),turnover=metric['turnover_annual']))
            for j,name in enumerate(names):stocks.append(dict(group=group,arm=arm,security_id=name,issuer=issuers[j],net=252*netc[:,j].mean(),gross=252*gross[:,j].mean(),fee=252*fees[:,j].mean(),exposure=held[:,j].mean()))
            print(key,flush=True)
        difference=traces[group,'original']-traces[group,'fixed'];np.testing.assert_allclose(difference.sum(1),paths[f'{group}__original__vol20']-paths[f'{group}__fixed__vol20'],atol=1e-10,rtol=0)
        for j,name in enumerate(names):
            for year in (2022,2023,2024,2025):annual.append(dict(group=group,year=year,security_id=name,issuer=issuers[j],difference=252*difference[years==year,j].mean()))
        a=targets[f'{group}__original'];b=targets[f'{group}__fixed']
        for j,i in enumerate(ids):
            for n in np.flatnonzero(a[j]!=b[j]):changed.append(dict(group=group,origin=int(i),security_id=names[n],difference=float(a[j,n]-b[j,n])))
    frame=pd.DataFrame(stocks);frame.to_csv(out/'STOCKS.csv',index=False);paired=[]
    for group in ('original_five','new_five','ten_seed'):
        a=frame[(frame.group==group)&(frame.arm=='original')].set_index('security_id');b=frame[(frame.group==group)&(frame.arm=='fixed')].set_index('security_id')
        for name in names:paired.append(dict(group=group,security_id=name,issuer=engine.ISSUERS.get(name,name),difference=float(a.loc[name,'net']-b.loc[name,'net']),exposure_difference=float(a.loc[name,'exposure']-b.loc[name,'exposure'])))
    p=pd.DataFrame(paired);p.to_csv(out/'PAIRED_STOCKS.csv',index=False);p.groupby(['group','issuer'],as_index=False)[['difference','exposure_difference']].sum().to_csv(out/'PAIRED_ISSUERS.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'CALENDAR_STOCKS.csv',index=False);pd.DataFrame(changed).to_csv(out/'TARGET_CHANGES.csv',index=False);pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False);pd.DataFrame(accounts).to_csv(out/'ACCOUNTS.csv',index=False)
    for path in (source/'PATHS.npz',source/'TARGETS.npz',Path('docs/rishi_mechanism_20261027/VALIDATION_FIXED_TRACE_PROTOCOL_20261029.md'),Path(__file__)):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');print('target changes',len(changed))


if __name__=='__main__':main()
