"""Rebuild matched target constraints and paired statistics from saved results."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch24_industry_matched';large=Path('results/RISHI_INDUSTRY_MATCHED_20261028')
    data=load_data();detail=pd.read_csv(doc/'execution_batch21_saved_industry/DETAIL.csv',float_precision='round_trip');detail=detail[detail.model=='E']
    cut=pd.read_csv(doc/'execution_batch21_saved_industry/SUMMARY.csv');cut=cut[cut.model=='E'].sort_values('year');dates=np.array(cut.signal_date,dtype='datetime64[D]')
    path=np.load(large/'PATHS.npz',allow_pickle=False);metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'PAIRED_CONTRASTS.csv')
    receipt=pd.read_csv(out/'GROUP_TARGETS.csv');maximum=0.;uniform=0.;count=0;holdings=[]
    for name in ('E','E_Hlabel'):
        z=np.load(f'results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_{name}_ten_seed.npz',allow_pickle=False)
        ids=z['ids'];original=engine.allocations(data,ids,z['scores'],'equal10');saved=np.load(large/f'{name}_weights.npz',allow_pickle=False)
        np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['security_ids'],data['panel'].tickers)
        w=saved['weights'];
        for j,origin in enumerate(ids):
            day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(dates,day,side='right')-1)
            labels=detail[detail.year==int(cut.iloc[idx].year)].set_index('security_id').industry.reindex(data['panel'].tickers).to_numpy(float)
            groups=np.where(np.isfinite(labels),labels,-1);eligible=data['mask'][origin]
            if np.any(w[j,~eligible]!=0) or np.any(w[j]<0):raise ValueError('Ineligible or negative control weight')
            holdings.append(int((w[j]>0).sum()))
            for g in np.unique(groups[eligible]):
                use=eligible&(groups==g);actual=w[j,use];maximum=max(maximum,abs(float(actual.sum()-original[j,use].sum())))
                uniform=max(uniform,float(np.ptp(actual)));count+=1
                label='unclassified' if g==-1 else str(int(g));record=receipt[(receipt.model==name)&(receipt.origin==origin)&(receipt.industry==label)]
                if len(record)!=1 or abs(float(record.iloc[0].control_mass)-float(actual.sum()))>1e-12:raise ValueError('Target ledger mismatch')
        if maximum>1e-12 or uniform>1e-12:raise ValueError('Target matching failed')
        for arm in ('original','matched'):
            arr=path[f'{name}_{arm}'];m=performance(arr);row=metrics[(metrics.model==name)&(metrics.arm==arm)].iloc[0]
            for field in ('sharpe','annualized_return','maximum_drawdown'):np.testing.assert_allclose(m[field],row[field],atol=1e-12)
            l=pd.read_csv(large/f'{name}_{arm}_ledger.csv');np.testing.assert_array_equal(l.date.to_numpy(),path['dates']);
            np.testing.assert_allclose(l.nav.to_numpy(),1_000_000*np.cumprod(1+arr),rtol=1e-11,atol=1e-6)
            prior=np.r_[1_000_000.,l.nav.to_numpy()[:-1]]
            np.testing.assert_allclose([l.fee.sum(),(l.traded.to_numpy()/prior).sum()*252/len(l),l.equity_exposure.mean(),l.cash.min()],
                [row.cost_dollars,row.turnover_annual,row.mean_exposure,row.min_cash],rtol=1e-11,atol=1e-7)
            if l.cash.min() < -1e-5 or l.equity_exposure.iloc[-1]!=0:raise ValueError('Cash/terminal feasibility failed')
        left=path[f'{name}_original'];right=path[f'{name}_matched'];delta=left-right
        for block in (5,21,63):
            draws=stationary_indices(len(delta),10000,block,27102028);boot=delta[draws].mean(1)*252
            row=contrasts[(contrasts.model==name)&(contrasts.block==block)].iloc[0]
            np.testing.assert_allclose([delta.mean()*252,*np.quantile(boot,[.025,.975])],[row.mean_difference,row.mean_low,row.mean_high],atol=1e-12)
    (out/'INDEPENDENT_AUDIT_V2.json').write_text(json.dumps(dict(group_records=count,max_group_error=maximum,max_within_group_range=uniform,
        mean_selected_count=float(np.mean(holdings)),min_selected_count=int(np.min(holdings)),max_selected_count=int(np.max(holdings)),
        path_sha256=hashlib.sha256((large/'PATHS.npz').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Verified',count,'group constraints; selected count mean',float(np.mean(holdings)),'range',int(np.min(holdings)),int(np.max(holdings)))


if __name__=='__main__':main()
