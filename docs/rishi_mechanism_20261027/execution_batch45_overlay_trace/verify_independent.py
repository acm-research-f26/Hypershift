"""Independently rebuild six paths, group issuer differences and target changes."""
from pathlib import Path
import numpy as np
import pandas as pd

source=Path('results/RISHI_PARENT_OVERLAY_20261028');root=Path('results/RISHI_PARENT_OVERLAY_TRACE_20261028')
out=Path('docs/rishi_mechanism_20261027/execution_batch45_overlay_trace')
paths=np.load(source/'PATHS.npz',allow_pickle=False);targets=np.load(source/'TARGETS.npz',allow_pickle=False)
stocks=pd.read_csv(out/'STOCKS.csv');issuers=pd.read_csv(out/'PAIRED_ISSUERS.csv');changes=pd.read_csv(out/'TARGET_CHANGES.csv');daily=pd.read_csv(out/'DAILY_EXPOSURES.csv')
calendar=pd.read_csv(out/'PAIRED_CALENDAR_STOCKS.csv')
maximum=0.;target_rows=0
for group in ('original_five','new_five','ten_seed'):
    nets={};exposures={}
    for view in ('daily','after_hours_premarket'):
        key=f'{view}__{group}__issuer5__vol20';z=np.load(root/f'{key}.npz',allow_pickle=False);ledger=pd.read_csv(source/f'{key}_ledger.csv')
        assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
        net=z['net'];nets[view]=net;exposures[view]=z['exposure'].sum(1);prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]]
        for value in (np.max(abs(net.sum(1)-paths[key])),np.max(abs(z['gross']-z['fees']-net)),np.max(abs(z['fees'].sum(1)-ledger.fee.to_numpy()/prev))):maximum=max(maximum,float(value))
        rows=stocks[(stocks['group']==group)&(stocks.view==view)].set_index('security_id').loc[z['security_ids'].tolist()]
        for col,arr in [('annual_net',net),('annual_gross',z['gross']),('annual_fee',z['fees']),('mean_exposure',z['exposure'])]:
            maximum=max(maximum,float(np.max(abs(rows[col].to_numpy()-arr.mean(0)*(1 if col=='mean_exposure' else 252)))))
    delta=(nets['after_hours_premarket']-nets['daily']).mean(0)*252
    dates=pd.to_datetime(paths['dates'])
    for year in (2022,2023,2024,2025):
        subset=calendar[(calendar['group']==group)&(calendar['year']==year)].set_index('security_id').loc[z['security_ids'].tolist()]
        expected_calendar=(nets['after_hours_premarket'][dates.year==year]-nets['daily'][dates.year==year]).mean(0)*252
        maximum=max(maximum,float(np.max(abs(subset.annual_difference.to_numpy()-expected_calendar))))
    grouped=pd.DataFrame({'issuer':rows.issuer.to_numpy(),'v':delta}).groupby('issuer').v.sum()
    actual=issuers[issuers['group']==group].set_index('issuer').annual_difference
    maximum=max(maximum,float(np.max(abs(grouped.sort_index()-actual.sort_index()))))
    observed=daily[daily['group']==group]
    for key,view in [('ahpm_exposure','after_hours_premarket'),('daily_exposure','daily')]:
        maximum=max(maximum,float(np.max(abs(observed[key]-exposures[view]))))
    expected=targets[f'after_hours_premarket__{group}__issuer5']-targets[f'daily__{group}__issuer5']
    subset=changes[changes['group']==group];reconstructed=np.zeros_like(expected)
    origins={int(v):i for i,v in enumerate(targets['ids'])};names={v:i for i,v in enumerate(targets['security_ids'])}
    assert not subset.duplicated(['origin','security_id']).any()
    for row in subset.itertuples():reconstructed[origins[row.origin],names[row.security_id]]=row.difference
    maximum=max(maximum,float(np.max(abs(expected-reconstructed))));target_rows+=len(subset)
assert maximum<1e-10
print('Six paths, three issuer/exposure comparisons, 12 calendar partitions and',target_rows,'target changes verified; max error',maximum)
