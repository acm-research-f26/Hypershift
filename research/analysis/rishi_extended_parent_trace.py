"""Trace saved parent-view/dynamic blend portfolios by stock and target."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_leader_trace import trace_risk

DOC=Path('docs/rishi_mechanism_20261027');SOURCE=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028')
VIEWS=('daily','after_hours','after_hours_premarket')


def main():
    out=DOC/'execution_batch43_parent_trace';large=Path('results/RISHI_EXTENDED_PARENT_TRACE_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();tickers=np.asarray(data['panel'].tickers);issuer=np.asarray([engine.ISSUERS.get(t,t) for t in tickers])
    saved=np.load(SOURCE/'PATHS.npz',allow_pickle=False);targets=np.load(SOURCE/'TARGETS.npz',allow_pickle=False)
    ids=targets['ids'];dates=saved['dates'];years=pd.DatetimeIndex(dates).year.to_numpy();np.testing.assert_array_equal(targets['security_ids'],tickers)
    prior=pd.read_csv(DOC/'execution_batch42_extended_parent/CONTRASTS.csv')
    inputs=[SOURCE/'PATHS.npz',SOURCE/'TARGETS.npz',DOC/'execution_batch42_extended_parent/CONTRASTS.csv',
        Path(__file__),DOC/'EXTENDED_PARENT_TRACE_PROTOCOL.md',Path('research/analysis/rishi_leader_trace.py'),
        Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    base=Path('data/nasdaq_monthly_pit/assembled');inputs.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    receipts=[];stock=[];account=[];traces={};changes=[];annual=[];target_changes=[]
    for k in (5,10):
        for view in VIEWS:
            key=f'{view}__ten_seed__k{k}';weights=targets[key];ledger_path=SOURCE/f'{key}_ledger.csv';inputs.append(ledger_path)
            metrics,net,ledger,trace=trace_risk(data,ids,weights,np.ones(len(data['panel'].dates)))
            old=pd.read_csv(ledger_path)
            if trace['dates']!=dates.tolist() or old.date.tolist()!=dates.tolist():raise ValueError('Date mismatch')
            error=float(np.max(abs(net-saved[key])));nav=float(np.max(abs(trace['nav']-old.nav.to_numpy()))/1e6)
            if error>1e-10 or nav>1e-8 or np.max(abs(trace['units'][-1]))>1e-8:raise ValueError('Trace mismatch')
            prev=np.r_[1e6,trace['nav'][:-1]];gross=trace['gross_dollars']/prev[:,None];fees=trace['fees']/prev[:,None];netc=trace['net_contribution']
            exposure=np.where(trace['units']!=0,np.nan_to_num(trace['units']*trace['marks']),0.)/trace['nav'][:,None]
            np.testing.assert_allclose(gross-fees,netc,atol=1e-12,rtol=0);np.testing.assert_allclose(netc.sum(1),net,atol=1e-10,rtol=0)
            np.savez_compressed(large/f'{key}.npz',net=netc,gross=gross,fees=fees,exposure=exposure,dates=dates,security_ids=tickers.astype('U'))
            traces[view,k]=netc
            receipts.append(dict(view=view,k=k,daily_error=error,nav_error=nav,cash_error=trace['cash_error'],share_error=trace['share_error']))
            account.append(dict(view=view,k=k,net_annual=float(net.mean()*252),gross_annual=float(gross.sum(1).mean()*252),
                fee_annual=float(fees.sum(1).mean()*252),mean_exposure=float(exposure.sum(1).mean()),turnover_annual=metrics['turnover_annual']))
            for j,ticker in enumerate(tickers):stock.append(dict(view=view,k=k,security_id=ticker,issuer=issuer[j],
                net=float(netc[:,j].mean()*252),gross=float(gross[:,j].mean()*252),fee=float(fees[:,j].mean()*252),exposure=float(exposure[:,j].mean())))
            print(key,flush=True)
        left=targets[f'after_hours_premarket__ten_seed__k{k}']
        for view in ('daily','after_hours'):
            right=targets[f'{view}__ten_seed__k{k}'];difference=traces['after_hours_premarket',k]-traces[view,k]
            expected=saved[f'after_hours_premarket__ten_seed__k{k}']-saved[f'{view}__ten_seed__k{k}']
            np.testing.assert_allclose(difference.sum(1),expected,atol=1e-10,rtol=0)
            ref=prior[(prior['group']=='ten_seed')&(prior.k==k)&(prior.comparator==view)&(prior.block==21)].iloc[0]
            np.testing.assert_allclose(difference.sum(1).mean()*252,ref.mean_difference,atol=1e-12,rtol=0)
            for j,ticker in enumerate(tickers):changes.append(dict(k=k,comparator=view,security_id=ticker,issuer=issuer[j],annual_difference=float(difference[:,j].mean()*252)))
            for y in (2022,2023,2024,2025):
                for j,ticker in enumerate(tickers):annual.append(dict(k=k,comparator=view,year=y,security_id=ticker,issuer=issuer[j],annual_difference=float(difference[years==y,j].mean()*252)))
            for i,origin in enumerate(ids):
                for j in np.flatnonzero(left[i]!=right[i]):target_changes.append(dict(k=k,comparator=view,origin=int(origin),security_id=tickers[j],issuer=issuer[j],
                    ahpm_weight=float(left[i,j]),comparator_weight=float(right[i,j]),difference=float(left[i,j]-right[i,j])))
    stock=pd.DataFrame(stock);stock.to_csv(out/'STOCKS.csv',index=False);pd.DataFrame(account).to_csv(out/'ACCOUNTS.csv',index=False)
    change=pd.DataFrame(changes);change.to_csv(out/'PAIRED_STOCKS.csv',index=False)
    change.groupby(['k','comparator','issuer'],as_index=False).annual_difference.sum().to_csv(out/'PAIRED_ISSUERS.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'PAIRED_CALENDAR_STOCKS.csv',index=False);pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    pd.DataFrame(target_changes).to_csv(out/'TARGET_CHANGES.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(account),target_change_rows=len(target_changes),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Post-hoc saved-account attribution, not a causal AH/PM feature intervention'),indent=2)+'\n')
    print(change.groupby(['k','comparator']).annual_difference.sum().to_string())


if __name__=='__main__':main()
