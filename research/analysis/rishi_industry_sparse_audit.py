"""Rebuild sparse-control constraints and outcome summaries from saved artifacts."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch25_industry_sparse';large=Path('results/RISHI_INDUSTRY_SPARSE_20261028')
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');data=load_data();tickers=data['panel'].tickers
    c=pd.read_csv(doc/'execution_batch21_saved_industry/DETAIL.csv',float_precision='round_trip');c=c[c.model=='E']
    cut=pd.read_csv(doc/'execution_batch21_saved_industry/SUMMARY.csv');cut=cut[cut.model=='E'].sort_values('year');cutoffs=np.array(cut.signal_date,dtype='datetime64[D]')
    metrics=pd.read_csv(out/'CONTROL_METRICS.csv');paired=pd.read_csv(out/'PAIRED_AVERAGES.csv');receipts=pd.read_csv(out/'GROUP_CONSTRAINTS.csv')
    paths=np.load(large/'PATHS.npz',allow_pickle=False);checks=0;max_diff=0.;weight_hashes={}
    for model in ('E','E_Hlabel'):
        score=np.load(source/f'scores_{model}_ten_seed.npz',allow_pickle=False);ids=score['ids'];base=engine.allocations(data,ids,score['scores'],'equal10')
        labels=[];years=[]
        for origin in ids:
            day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(cutoffs,day,side='right')-1);year=int(cut.iloc[idx].year)
            v=c[c.year==year].set_index('security_id').industry.reindex(tickers).to_numpy(float);labels.append(np.where(np.isfinite(v),v,-1));years.append(year)
        labels=np.stack(labels);controls=[]
        for j in range(20):
            wp=large/f'{model}_{j:02d}_weights.npz';weight_hashes[str(wp)]=hashlib.sha256(wp.read_bytes()).hexdigest()
            z=np.load(wp,allow_pickle=False);np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['security_ids'],tickers);w=z['weights'];
            for i,origin in enumerate(ids):
                elig=data['mask'][origin];priority=np.random.default_rng(np.random.SeedSequence([27102028,years[i],j])).random(len(tickers));expected=np.zeros(len(tickers))
                if np.any(w[i,~elig]!=0) or np.any(w[i]<0):raise ValueError('Invalid holdings')
                for group in np.unique(labels[i,elig]):
                    members=np.flatnonzero(elig&(labels[i]==group));orig=base[i,members];actual=w[i,members]
                    vals=np.sort(orig[orig>0])[::-1];selected=members[np.argsort(priority[members],kind='stable')[:len(vals)]];expected[selected]=vals
                    max_diff=max(max_diff,abs(float(actual.sum()-orig.sum())))
                    if not np.array_equal(np.sort(actual[actual>0]),np.sort(vals)):raise ValueError('Weight multiset differs')
                    checks+=1
                if not np.array_equal(expected,w[i]):raise ValueError('Priority allocation differs')
            arr=paths[f'{model}_{j:02d}'];controls.append(arr);row=metrics[(metrics.model==model)&(metrics.control==j)]
            if len(row)!=1:raise ValueError('Metric row mismatch')
            row=row.iloc[0];m=performance(arr)
            for field in ('sharpe','annualized_return','maximum_drawdown'):np.testing.assert_allclose(m[field],row[field],atol=1e-12)
            l=pd.read_csv(large/f'{model}_{j:02d}_ledger.csv');np.testing.assert_array_equal(l.date.to_numpy(),paths['dates'])
            np.testing.assert_allclose(l.nav.to_numpy(),1_000_000*np.cumprod(1+arr),rtol=1e-11,atol=1e-6)
            prev=np.r_[1_000_000.,l.nav.to_numpy()[:-1]]
            np.testing.assert_allclose([l.fee.sum(),(l.traded.to_numpy()/prev).sum()*252/len(l),l.equity_exposure.mean(),l.cash.min()],
                [row.cost_dollars,row.turnover_annual,row.mean_exposure,row.min_cash],rtol=1e-11,atol=1e-7)
            if l.cash.min()<-1e-5 or l.equity_exposure.iloc[-1]!=0:raise ValueError('Infeasible account')
        average=np.mean(np.stack(controls),axis=0);np.testing.assert_array_equal(average,paths[f'{model}_diagnostic_average'])
        diff=paths[f'{model}_original']-average
        for block in (5,21,63):
            draw=stationary_indices(len(diff),10000,block,27102028);lo,hi=np.quantile(diff[draw].mean(1)*252,[.025,.975])
            row=paired[(paired.model==model)&(paired.block==block)].iloc[0]
            np.testing.assert_allclose([diff.mean()*252,lo,hi],[row.mean_difference,row.mean_low,row.mean_high],atol=1e-12)
    if checks!=len(receipts) or max_diff>1e-12:raise ValueError('Group constraint count mismatch')
    result=dict(group_constraints=checks,max_group_error=max_diff,controls=40,bootstrap_contrasts=6,weights_sha256=weight_hashes,
        paths_sha256=hashlib.sha256((large/'PATHS.npz').read_bytes()).hexdigest())
    (out/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print('Validated',checks,'group constraints, 40 accounts and six paired intervals')


if __name__=='__main__':main()
