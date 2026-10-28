"""Trace existing original and industry-matched coverage accounts by security."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_leader_trace import trace_risk


def paired_contributions(left,right):
    left=np.asarray(left,float);right=np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=2 or not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('Aligned finite contributions required')
    return (left-right).mean(axis=0)*252


def aggregate_issuers(tickers,values,mapping):
    result={}
    for ticker,value in zip(tickers,values):
        issuer=mapping.get(ticker,ticker);result[issuer]=result.get(issuer,0.)+float(value)
    return result


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch37_coverage_trace';large=Path('results/RISHI_INDUSTRY_COVERAGE_TRACE_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');simple=Path('results/RISHI_INDUSTRY_COVERAGE_20261028')
    saved=np.load(simple/'PATHS.npz',allow_pickle=False);account_summary=doc/'execution_batch35_coverage_fee/ACCOUNTS.csv';inputs=[simple/'PATHS.npz',account_summary,Path(__file__),doc/'COVERAGE_STOCK_TRACE_PROTOCOL.md',Path('research/analysis/rishi_leader_trace.py'),Path('research/core/ndx_drawdown_engine.py')]
    traces={};receipts=[];stocks=[];cal=[];years=pd.DatetimeIndex(saved['dates']).year.to_numpy()
    for model in ('E','E_Hlabel'):
        score_path=source/f'scores_{model}_ten_seed.npz';weight_path=simple/f'{model}_coverage6_weights.npz';inputs.extend([score_path,weight_path])
        with np.load(score_path,allow_pickle=False) as z:ids=z['ids'].copy();weekly=engine.allocations(data,ids,z['scores'],'equal10')
        with np.load(weight_path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['security_ids'],data['panel'].tickers);matched=z['weights'].copy()
        for arm,weights in [('original',weekly),('coverage6',matched)]:
            metrics,path,ledger,trace=trace_risk(data,ids,weights,np.ones(len(data['panel'].dates)))
            ref=saved[f'{model}_original' if arm=='original' else f'{model}_coverage6'];expected_ledger=pd.read_csv(simple/f'{model}_original_ledger.csv' if arm=='original' else simple/f'{model}_coverage6_ledger.csv')
            inputs.append(simple/f'{model}_{arm}_ledger.csv')
            if trace['dates']!=saved['dates'].tolist() or trace['dates']!=expected_ledger.date.tolist():raise ValueError('Trace calendar mismatch')
            error=float(np.max(abs(path-ref)));nav_error=float(np.max(abs(trace['nav']-expected_ledger.nav.to_numpy()))/1_000_000.)
            if error>1e-10 or nav_error>1e-8 or np.any(abs(trace['units'][-1])>1e-8):
                (out/'TRACE_FAILED.json').write_text(json.dumps(dict(model=model,arm=arm,error=error,nav_error=nav_error),indent=2)+'\n');raise ValueError('Trace replay failed')
            old=np.r_[1_000_000.,trace['nav'][:-1]];gross=trace['gross_dollars']/old[:,None];fees=trace['fees']/old[:,None]
            exposure=np.where(trace['units']!=0,np.nan_to_num(trace['units']*trace['marks']),0.)/trace['nav'][:,None]
            np.testing.assert_allclose(gross-fees,trace['net_contribution'],atol=1e-12,rtol=0)
            traces[model,arm]=trace
            np.savez_compressed(large/f'{model}_{arm}.npz',net=trace['net_contribution'],gross=gross,fees=fees,exposure=exposure,
                dates=np.asarray(trace['dates'],dtype='U'),security_ids=np.asarray(trace['security_ids'],dtype='U'))
            for j,ticker in enumerate(trace['security_ids']):
                stocks.append(dict(model=model,arm=arm,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),
                    annual_net=float(trace['net_contribution'][:,j].mean()*252),annual_gross=float(gross[:,j].mean()*252),
                    annual_fee=float(fees[:,j].mean()*252),mean_exposure=float(exposure[:,j].mean()),holding_days=int((trace['units'][:,j]>1e-12).sum())))
            receipts.append(dict(model=model,arm=arm,daily_error=error,relative_nav_error=nav_error,cash_error=trace['cash_error'],share_error=trace['share_error']))
            print(model,arm,'trace complete',flush=True)
        a=traces[model,'original']['net_contribution'];b=traces[model,'coverage6']['net_contribution'];paired=paired_contributions(a,b)
        for year in (2022,2023,2024,2025):
            delta=(a[years==year]-b[years==year]).mean(0)*252
            for j,ticker in enumerate(data['panel'].tickers):cal.append(dict(model=model,year=year,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),annual_difference=float(delta[j])))
        expected_difference=float((saved[f'{model}_original']-saved[f'{model}_coverage6']).mean()*252)
        np.testing.assert_allclose(paired.sum(),expected_difference,atol=1e-10,rtol=0)
    stock=pd.DataFrame(stocks);stock.to_csv(out/'STOCKS.csv',index=False)
    expected=pd.read_csv(account_summary)
    for (model,arm),group in stock.groupby(['model','arm']):
        row=expected[(expected.model==model)&(expected.arm==arm)].iloc[0]
        np.testing.assert_allclose(group[['annual_net','annual_gross','annual_fee']].sum().to_numpy(),
            [row.net_annual,row.gross_annual,row.fee_annual],rtol=0,atol=1e-10)
    stock.groupby(['model','arm','issuer'],as_index=False)[['annual_net','annual_gross','annual_fee','mean_exposure']].sum().to_csv(out/'ISSUERS.csv',index=False)
    changes=[]
    for model in ('E','E_Hlabel'):
        a=stock[(stock.model==model)&(stock.arm=='original')].set_index('security_id');b=stock[(stock.model==model)&(stock.arm=='coverage6')].set_index('security_id')
        for ticker in a.index:changes.append(dict(model=model,security_id=ticker,issuer=engine.ISSUERS.get(ticker,ticker),annual_difference=float(a.loc[ticker,'annual_net']-b.loc[ticker,'annual_net']),
            exposure_difference=float(a.loc[ticker,'mean_exposure']-b.loc[ticker,'mean_exposure']),fee_difference=float(a.loc[ticker,'annual_fee']-b.loc[ticker,'annual_fee'])))
    changes=pd.DataFrame(changes);changes.to_csv(out/'PAIRED_STOCKS.csv',index=False)
    changes.groupby(['model','issuer'],as_index=False)[['annual_difference','exposure_difference','fee_difference']].sum().to_csv(out/'PAIRED_ISSUERS.csv',index=False)
    pd.DataFrame(cal).to_csv(out/'PAIRED_CALENDAR_STOCKS.csv',index=False);pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=4,days=len(years),inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Reconstructed accounting attribution of selected historical accounts, not a security-removal counterfactual'),indent=2)+'\n')
    print(changes.groupby('model').annual_difference.sum().to_string());print('Reconciled four paths and all security contributions')


if __name__=='__main__':main()
