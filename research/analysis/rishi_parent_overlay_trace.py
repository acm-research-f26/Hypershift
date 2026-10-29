"""Saved issuer5/vol20 parent-view subgroup accounting trace."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_risk import POLICIES
from rishi_leader_trace import trace_risk

DOC=Path('docs/rishi_mechanism_20261027');SOURCE=Path('results/RISHI_PARENT_OVERLAY_20261028')
GROUPS=('original_five','new_five','ten_seed');VIEWS=('daily','after_hours_premarket')


def main():
    out=DOC/'execution_batch45_overlay_trace';large=Path('results/RISHI_PARENT_OVERLAY_TRACE_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);tickers=np.asarray(data['panel'].tickers);issuer=np.asarray([engine.ISSUERS.get(t,t) for t in tickers])
    paths=np.load(SOURCE/'PATHS.npz',allow_pickle=False);targets=np.load(SOURCE/'TARGETS.npz',allow_pickle=False);ids=targets['ids'];dates=paths['dates'];years=pd.DatetimeIndex(dates).year
    np.testing.assert_array_equal(targets['security_ids'],tickers)
    prior=pd.read_csv(DOC/'execution_batch44_parent_overlay/CONTRASTS.csv');receipts=[];accounts=[];stocks=[];daily=[];targets_changed=[];annual=[];traces={}
    inputs=[SOURCE/'PATHS.npz',SOURCE/'TARGETS.npz',DOC/'execution_batch44_parent_overlay/CONTRASTS.csv',Path(__file__),DOC/'PARENT_OVERLAY_TRACE_PROTOCOL.md',
        Path('research/analysis/rishi_leader_trace.py'),Path('research/analysis/rishi_mechanism_risk.py'),
        Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    base=Path('data/nasdaq_monthly_pit/assembled');inputs.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    for group in GROUPS:
        exposures={}
        for view in VIEWS:
            key=f'{view}__{group}__issuer5__vol20';weights=targets[f'{view}__{group}__issuer5'];ledger_path=SOURCE/f'{key}_ledger.csv';inputs.append(ledger_path)
            risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,weights));exposure=engine.exposure_series(POLICIES['vol20'],market,risk)
            metrics,net,ledger,trace=trace_risk(data,ids,weights,exposure);old=pd.read_csv(ledger_path)
            if trace['dates']!=dates.tolist() or old.date.tolist()!=dates.tolist():raise ValueError('Date mismatch')
            err=float(np.max(abs(net-paths[key])));nav=float(np.max(abs(trace['nav']-old.nav.to_numpy()))/1e6)
            if err>1e-10 or nav>1e-8 or np.max(abs(trace['units'][-1]))>1e-8:raise ValueError('Trace mismatch')
            prev=np.r_[1e6,trace['nav'][:-1]];gross=trace['gross_dollars']/prev[:,None];fee=trace['fees']/prev[:,None];netc=trace['net_contribution']
            held=np.where(trace['units']!=0,np.nan_to_num(trace['units']*trace['marks']),0.)/trace['nav'][:,None]
            np.testing.assert_allclose(gross-fee,netc,rtol=0,atol=1e-12);np.testing.assert_allclose(netc.sum(1),net,rtol=0,atol=1e-10)
            np.savez_compressed(large/f'{key}.npz',net=netc,gross=gross,fees=fee,exposure=held,dates=dates,security_ids=tickers.astype('U'))
            traces[group,view]=netc;exposures[view]=held.sum(1)
            accounts.append(dict(group=group,view=view,net_annual=float(net.mean()*252),gross_annual=float(gross.sum(1).mean()*252),
                fee_annual=float(fee.sum(1).mean()*252),mean_exposure=float(held.sum(1).mean()),turnover_annual=metrics['turnover_annual']))
            receipts.append(dict(group=group,view=view,daily_error=err,nav_error=nav,cash_error=trace['cash_error'],share_error=trace['share_error']))
            for j,ticker in enumerate(tickers):stocks.append(dict(group=group,view=view,security_id=ticker,issuer=issuer[j],
                annual_net=float(netc[:,j].mean()*252),annual_gross=float(gross[:,j].mean()*252),annual_fee=float(fee[:,j].mean()*252),mean_exposure=float(held[:,j].mean())))
            print(key,flush=True)
        diff=traces[group,'after_hours_premarket']-traces[group,'daily'];expected=paths[f'after_hours_premarket__{group}__issuer5__vol20']-paths[f'daily__{group}__issuer5__vol20']
        np.testing.assert_allclose(diff.sum(1),expected,rtol=0,atol=1e-10)
        ref=prior[(prior['group']==group)&(prior.allocation=='issuer5')&(prior.policy=='vol20')&(prior.comparator=='daily')&(prior.block==21)].iloc[0]
        np.testing.assert_allclose(diff.sum(1).mean()*252,ref.mean_difference,rtol=0,atol=1e-12)
        for j,ticker in enumerate(tickers):
            for y in (2022,2023,2024,2025):annual.append(dict(group=group,year=y,security_id=ticker,issuer=issuer[j],annual_difference=float(diff[years==y,j].mean()*252)))
        daily.append(pd.DataFrame(dict(group=group,date=dates,net_difference=expected,ahpm_exposure=exposures['after_hours_premarket'],daily_exposure=exposures['daily'])))
        a=targets[f'after_hours_premarket__{group}__issuer5'];b=targets[f'daily__{group}__issuer5']
        for i,origin in enumerate(ids):
            for j in np.flatnonzero(a[i]!=b[i]):targets_changed.append(dict(group=group,origin=int(origin),security_id=tickers[j],issuer=issuer[j],ahpm_weight=float(a[i,j]),daily_weight=float(b[i,j]),difference=float(a[i,j]-b[i,j])))
    stock=pd.DataFrame(stocks);stock.to_csv(out/'STOCKS.csv',index=False);changes=[]
    for group in GROUPS:
        a=stock[(stock['group']==group)&(stock.view=='after_hours_premarket')].set_index('security_id');b=stock[(stock['group']==group)&(stock.view=='daily')].set_index('security_id')
        for ticker in tickers:changes.append(dict(group=group,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),
            annual_difference=float(a.loc[ticker,'annual_net']-b.loc[ticker,'annual_net']),exposure_difference=float(a.loc[ticker,'mean_exposure']-b.loc[ticker,'mean_exposure'])))
    changes=pd.DataFrame(changes);changes.to_csv(out/'PAIRED_STOCKS.csv',index=False)
    changes.groupby(['group','issuer'],as_index=False)[['annual_difference','exposure_difference']].sum().to_csv(out/'PAIRED_ISSUERS.csv',index=False)
    pd.DataFrame(accounts).to_csv(out/'ACCOUNTS.csv',index=False);pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'PAIRED_CALENDAR_STOCKS.csv',index=False);pd.concat(daily).to_csv(out/'DAILY_EXPOSURES.csv',index=False)
    pd.DataFrame(targets_changed).to_csv(out/'TARGET_CHANGES.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(accounts),target_changes=len(targets_changed),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Post-hoc subgroup accounting under basket-dependent risk overlays; not causal'),indent=2)+'\n')
    print(changes.groupby('group').annual_difference.sum().to_string())


if __name__=='__main__':main()
