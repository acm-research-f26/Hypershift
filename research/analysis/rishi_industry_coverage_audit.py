"""Independently reconstruct saved simple-signal target choices and metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch34_industry_coverage';large=Path('results/RISHI_INDUSTRY_COVERAGE_20261028');root=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    data=load_data();tickers=data['panel'].tickers;c=pd.read_csv(doc/'execution_batch21_saved_industry/DETAIL.csv',float_precision='round_trip');c=c[c.model=='E']
    cut=pd.read_csv(doc/'execution_batch21_saved_industry/SUMMARY.csv');cut=cut[cut.model=='E'].sort_values('year');cutoffs=np.array(cut.signal_date,dtype='datetime64[D]')
    results=pd.read_csv(out/'CONTROL_METRICS.csv');contrasts=pd.read_csv(out/'PAIRED_CONTRASTS.csv');overlap=pd.read_csv(out/'OVERLAP.csv');cons=pd.read_csv(out/'GROUP_CONSTRAINTS.csv')
    paths=np.load(large/'PATHS.npz',allow_pickle=False);source=load_data();constraints=0;hashes={};maxerr=0.
    for model in ('E','E_Hlabel'):
        z=np.load(root/f'scores_{model}_ten_seed.npz',allow_pickle=False);ids=z['ids'];original=engine.allocations(source,ids,z['scores'],'equal10');class_rows=[];years=[]
        for origin in ids:
            day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(cutoffs,day,side='right')-1);year=int(cut.iloc[idx].year)
            labels=c[c.year==year].set_index('security_id').industry.reindex(tickers).to_numpy(float);class_rows.append(np.where(np.isfinite(labels),labels,-1));years.append(year)
        class_rows=np.stack(class_rows)
        for signal in ('coverage6',):
            s=np.load(root/f'scores_{signal}_ten_seed.npz',allow_pickle=False);np.testing.assert_array_equal(s['ids'],ids);score=s['scores']
            weight_path=large/f'{model}_{signal}_weights.npz';hashes[str(weight_path)]=hashlib.sha256(weight_path.read_bytes()).hexdigest()
            saved=np.load(weight_path,allow_pickle=False);np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['security_ids'],tickers);weights=saved['weights']
            for row,origin in enumerate(ids):
                eligible=data['mask'][origin];priority=np.random.default_rng(np.random.SeedSequence([27102028,years[row],0])).random(len(tickers));expected=np.zeros(len(tickers))
                for group in np.unique(class_rows[row,eligible]):
                    members=np.flatnonzero(eligible&(class_rows[row]==group));values=np.sort(original[row,members][original[row,members]>0])[::-1]
                    selected=sorted(members,key=lambda j:(-score[row,j],priority[j],j))[:len(values)];expected[selected]=values
                    maxerr=max(maxerr,abs(float(expected[members].sum()-original[row,members].sum())));constraints+=1
                if not np.array_equal(expected,weights[row]):raise ValueError('Saved signal choice mismatch')
                rec=overlap[(overlap.model==model)&(overlap.signal==signal)&(overlap.origin==origin)]
                if len(rec)!=1 or int(rec.iloc[0].shared_names)!=int(((original[row]>0)&(weights[row]>0)).sum()):raise ValueError('Overlap mismatch')
                np.testing.assert_allclose(rec.iloc[0].target_weight_overlap,np.minimum(original[row],weights[row]).sum(),rtol=0,atol=1e-12)
            arr=paths[f'{model}_{signal}'];m=performance(arr);rec=results[(results.model==model)&(results.signal==signal)].iloc[0]
            for k in ('sharpe','annualized_return','maximum_drawdown'):np.testing.assert_allclose(m[k],rec[k],atol=1e-12)
            ledger=pd.read_csv(large/f'{model}_{signal}_ledger.csv');np.testing.assert_array_equal(ledger.date.to_numpy(),paths['dates'])
            np.testing.assert_allclose(ledger.nav,1_000_000*np.cumprod(1+arr),rtol=1e-11,atol=1e-6)
            previous=np.r_[1_000_000.,ledger.nav.to_numpy()[:-1]]
            np.testing.assert_allclose([ledger.fee.sum(),(ledger.traded.to_numpy()/previous).sum()*252/len(ledger),ledger.equity_exposure.mean(),ledger.cash.min()],
                [rec.cost_dollars,rec.turnover_annual,rec.mean_exposure,rec.min_cash],rtol=1e-11,atol=1e-7)
            delta=paths[f'{model}_original']-arr
            for block in (5,21,63):
                draw=stationary_indices(len(delta),10000,block,27102028);lo,hi=np.quantile(delta[draw].mean(1)*252,[.025,.975]);r=contrasts[(contrasts.model==model)&(contrasts.signal==signal)&(contrasts.block==block)].iloc[0]
                np.testing.assert_allclose([delta.mean()*252,lo,hi],[r.mean_difference,r.mean_low,r.mean_high],atol=1e-12)
    if constraints!=len(cons) or maxerr>1e-12 or not (cons.original_count==cons.alternative_count).all():raise ValueError('Constraint count or matching')
    (out/'INDEPENDENT_AUDIT.json').write_text(json.dumps({'group_constraints':constraints,'max_group_error':maxerr,'weights_sha256':hashes,
        'paths_sha256':hashlib.sha256((large/'PATHS.npz').read_bytes()).hexdigest(),'account_paths':2,'paired_mean_intervals':6},indent=2)+'\n')
    print('Verified',constraints,'group constraints, 2 accounts and 6 paired intervals')


if __name__=='__main__':main()
