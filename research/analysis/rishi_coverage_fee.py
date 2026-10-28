"""Exact fee/gross accounting identity for already-saved coverage controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_mechanism_batch1 import stationary_indices

DOC=Path('docs/rishi_mechanism_20261027')
ROOT=Path('results/RISHI_INDUSTRY_COVERAGE_20261028')


def components(nav,net,fee,traded):
    nav,net,fee,traded=[np.asarray(v,float) for v in (nav,net,fee,traded)]
    if any(v.ndim!=1 or len(v)!=len(nav) or not np.isfinite(v).all() for v in (nav,net,fee,traded)):
        raise ValueError('Invalid ledger arrays')
    prev=np.r_[1e6,nav[:-1]]
    if (nav<=0).any() or (fee<0).any() or (traded<0).any():raise ValueError('Invalid ledger economics')
    if not np.allclose(nav/prev-1,net,rtol=0,atol=1e-10):raise ValueError('NAV and returns differ')
    drag=fee/prev
    return dict(net=net,gross=net+drag,fee=drag,turnover=traded/prev)


def main():
    out=DOC/'execution_batch35_coverage_fee';out.mkdir(parents=True,exist_ok=False)
    paths=np.load(ROOT/'PATHS.npz',allow_pickle=False);dates=paths['dates'];frame=[];summary=[];calendar=[];intervals=[]
    prior=pd.read_csv(DOC/'execution_batch34_industry_coverage/PAIRED_CONTRASTS.csv');inputs=[ROOT/'PATHS.npz',DOC/'execution_batch34_industry_coverage/PAIRED_CONTRASTS.csv',Path(__file__),DOC/'COVERAGE_FEE_ATTRIBUTION_PROTOCOL.md',Path('research/analysis/rishi_mechanism_batch1.py')]
    for model in ('E','E_Hlabel'):
        accounts={}
        for arm in ('original','coverage6'):
            ledger_path=ROOT/f'{model}_{arm}_ledger.csv';inputs.append(ledger_path)
            ledger=pd.read_csv(ledger_path);net=paths[f'{model}_{arm}']
            if ledger.date.tolist()!=dates.tolist():raise ValueError('Date mismatch')
            a=components(ledger.nav,net,ledger.fee,ledger.traded);accounts[arm]=a
            if not np.allclose(ledger.nav,1e6*np.cumprod(1+net),rtol=1e-8,atol=.01):raise ValueError('Cumulative NAV mismatch')
            summary.append(dict(model=model,arm=arm,**{k+'_annual':float(v.mean()*252) for k,v in a.items()}))
            for year in (2022,2023,2024,2025):
                mask=pd.DatetimeIndex(dates).year==year
                calendar.append(dict(model=model,arm=arm,year=year,**{k+'_annual':float(v[mask].mean()*252) for k,v in a.items()}))
        a,b=accounts['original'],accounts['coverage6']
        differences={k:a[k]-b[k] for k in ('net','gross','fee')}
        if not np.allclose(differences['net'],differences['gross']-differences['fee'],rtol=0,atol=1e-14):raise ValueError('Identity failure')
        frame.append(pd.DataFrame(dict(model=model,date=dates,**{f'{arm}_{k}':v for arm,account in accounts.items() for k,v in account.items()})))
        for block in (5,21,63):
            draws=stationary_indices(len(dates),10000,block,27102028)
            row=dict(model=model,block=block)
            for k,v in differences.items():
                samples=v[draws].mean(1)*252;row.update({k+'_difference':float(v.mean()*252),k+'_low':float(np.quantile(samples,.025)),k+'_high':float(np.quantile(samples,.975))})
            old=prior[(prior.model==model)&(prior.block==block)].iloc[0]
            np.testing.assert_allclose([row['net_difference'],row['net_low'],row['net_high']],[old.mean_difference,old.mean_low,old.mean_high],rtol=0,atol=1e-12)
            intervals.append(row)
    pd.concat(frame,ignore_index=True).to_csv(out/'DAILY.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'ACCOUNTS.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'CONTRASTS.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=4,pairs=2,days=len(dates),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Fee accounting on fee-affected positions, not a no-fee replay'),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(intervals).query('block==21').to_string(index=False))


if __name__=='__main__':main()
