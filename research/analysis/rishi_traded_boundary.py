"""Exact additive decision-category accounting on existing reconciled E traces."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import target_weights
from ndx_drawdown_engine import ISSUERS
from rishi_rank_behavior import selection_categories

CATEGORIES=['both','parent_only','dynamic_only','neither','exit_cost','residual']


def partition_account(days,decisions,units,marks,old,nav,gross,fees,operations,security_ids):
    count,n=units.shape;before=np.empty((count,n),int);after=np.empty_like(before);current=np.full(n,5)
    for row,day in enumerate(days):
        before[row]=current
        if int(day) in decisions:current=decisions[int(day)].copy()
        after[row]=current
    cost=np.zeros((count,6));fee_assets=np.zeros((6,n));observed=np.zeros_like(fees)
    lookup={int(day):row for row,day in enumerate(days)};nodes={name:j for j,name in enumerate(security_ids)}
    for op in operations:
        row=lookup[int(op['day'])];j=nodes[op['security_id']]
        if op['kind']=='fill':
            if int(op['day']) not in decisions:raise ValueError('Ordinary fill lacks frozen decision')
            category=after[row,j] if after[row,j]!=5 else 4
        elif op['kind'] in ['terminal','settlement']:category=before[row,j]
        else:raise ValueError('Unknown operation type')
        charge=float(op['fee']);observed[row,j]+=charge;cost[row,category]+=charge/old[row];fee_assets[category,j]+=charge/old[row]
    np.testing.assert_allclose(observed,fees,rtol=1e-10,atol=1e-6)
    holding=np.where(units!=0,np.nan_to_num(units*marks),0)/nav[:,None]
    gross_return=gross/old[:,None];g=np.zeros((count,6));exposure=np.zeros_like(g)
    gross_assets=np.zeros((6,n));exposure_assets=np.zeros((6,n))
    for category in range(6):
        value=np.where(before==category,gross_return,0);g[:,category]=value.sum(1);gross_assets[category]=value.sum(0)
        value=np.where(after==category,holding,0);exposure[:,category]=value.sum(1);exposure_assets[category]=value.mean(0)
    np.testing.assert_allclose(g.sum(1)-cost.sum(1),(gross-fees).sum(1)/old,atol=1e-10)
    np.testing.assert_allclose(exposure.sum(1),holding.sum(1),atol=1e-12)
    return dict(gross=g,cost=cost,net=g-cost,exposure=exposure,gross_by_asset=gross_assets,
                cost_by_asset=fee_assets,mean_exposure_by_asset=exposure_assets)


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch12_boundary'
    large=Path('results/RISHI_MECHANISM_BOUNDARY_20261027')
    for path in (out,large):
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True)
    data=load_data();source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');scores={};inputs=[]
    for name in ['B','D','E']:
        file=source/f'scores_{name}_ten_seed.npz';inputs.append(file)
        with np.load(file) as saved:
            if scores:np.testing.assert_array_equal(ids,saved['ids'])
            ids=saved['ids'].copy();scores[name]=saved['scores'].copy()
    curves=pd.read_csv(source/'ACCOUNT_CURVES.csv');inputs.append(source/'ACCOUNT_CURVES.csv')
    axes_path=Path('results/RISHI_MECHANISM_TRACE_20261027/AXES.json');axes=json.loads(axes_path.read_text());inputs.append(axes_path)
    dates=pd.DatetimeIndex(axes['dates']);days=data['panel'].dates.get_indexer(dates);assert (days>=0).all()
    assert axes['security_ids']==data['panel'].tickers
    summary=[];calendar=[];issuers=[];receipts=[]
    for k in (5,10):
        trace_path=Path(f'results/RISHI_MECHANISM_TRACE_20261027/E_ten_seed_k{k}.npz')
        fills_path=trace_path.with_name(trace_path.stem+'_fills.csv');inputs.extend([trace_path,fills_path])
        with np.load(trace_path,allow_pickle=False) as saved:trace={name:saved[name].copy() for name in ['units','gross_dollars','fees','net_contribution']}
        decisions={}
        for row,i in enumerate(ids):
            weights={name:target_weights(value[row],data['mask'][i],k,'equal') for name,value in scores.items()}
            labels=selection_categories(weights['B'],weights['D']);labels[weights['E']==0]=5
            decisions[int(data['entry'][i])]=labels
        actual=curves[curves.id==f'E/ten_seed/{k}'];assert actual.date.tolist()==axes['dates']
        nav=1e6*np.cumprod(1+actual.net.to_numpy());old=np.r_[1e6,nav[:-1]]
        result=partition_account(days,decisions,trace['units'],data['marks'][days],old,nav,trace['gross_dollars'],trace['fees'],
            pd.read_csv(fills_path).to_dict('records'),axes['security_ids'])
        error=float(np.max(np.abs(result['net'].sum(1)-actual.net.to_numpy())))
        if error>1e-10:raise AssertionError('Category daily return mismatch')
        np.savez_compressed(large/f'top{k}.npz',dates=np.asarray(axes['dates'],dtype='U10'),categories=np.asarray(CATEGORIES,dtype=str),
            gross=result['gross'],cost=result['cost'],net=result['net'],exposure=result['exposure'])
        for c,category in enumerate(CATEGORIES):
            summary.append(dict(k=k,category=category,annual_gross=float(result['gross'][:,c].mean()*252),
                annual_cost=float(result['cost'][:,c].mean()*252),annual_net=float(result['net'][:,c].mean()*252),
                mean_equity_exposure=float(result['exposure'][:,c].mean())))
            for year in [2022,2023,2024,2025]:
                use=dates.year==year
                calendar.append(dict(k=k,category=category,year=year,annual_arithmetic_contribution=float(result['net'][use,c].mean()*252),
                    gross_dollars=float((result['gross'][use,c]*old[use]).sum()),fee_dollars=float((result['cost'][use,c]*old[use]).sum())))
            for j,name in enumerate(data['panel'].tickers):
                issuers.append(dict(k=k,category=category,issuer=ISSUERS.get(name,name),
                    annual_gross=float(result['gross_by_asset'][c,j]/len(dates)*252),
                    annual_cost=float(result['cost_by_asset'][c,j]/len(dates)*252),mean_equity_exposure=float(result['mean_exposure_by_asset'][c,j])))
        receipts.append(dict(k=k,max_daily_error=error,original_annual_mean=float(actual.net.mean()*252),
            classified_annual_mean=float(result['net'].sum(1).mean()*252)))
    pd.DataFrame(summary).to_csv(out/'CATEGORY_ATTRIBUTION.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR_ATTRIBUTION.csv',index=False)
    issuer=pd.DataFrame(issuers).groupby(['k','category','issuer'],as_index=False).sum()
    issuer['annual_net']=issuer.annual_gross-issuer.annual_cost;issuer.to_csv(out/'CATEGORY_ISSUERS.csv',index=False)
    pd.DataFrame(receipts).to_csv(out/'RECONCILIATION.csv',index=False)
    inputs.extend([Path(__file__),doc/'TRADED_BOUNDARY_PROTOCOL.md'])
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=2,categories=6,calendar_rows=48,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        scope='Exact additive convention on existing operations; no new strategy or causal category contribution'),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
