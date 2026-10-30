"""Rebuild all daily traces, issuer contrasts and target changes."""
from pathlib import Path
import numpy as np
import pandas as pd

source=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028');root=Path('results/RISHI_EXTENDED_PARENT_TRACE_20261028')
out=Path('docs/rishi_mechanism_20261027/execution_batch43_parent_trace')
paths=np.load(source/'PATHS.npz',allow_pickle=False);targets=np.load(source/'TARGETS.npz',allow_pickle=False)
stocks=pd.read_csv(out/'STOCKS.csv');issuer=pd.read_csv(out/'PAIRED_ISSUERS.csv');changes=pd.read_csv(out/'TARGET_CHANGES.csv')
maximum=0.;target_rows=0
for k in (5,10):
    nets={}
    for view in ('daily','after_hours','after_hours_premarket'):
        key=f'{view}__ten_seed__k{k}';z=np.load(root/f'{key}.npz',allow_pickle=False);ledger=pd.read_csv(source/f'{key}_ledger.csv')
        assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
        net=z['net'];nets[view]=net;prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]]
        for v in (np.max(abs(net.sum(1)-paths[key])),np.max(abs(z['gross']-z['fees']-net)),np.max(abs(z['fees'].sum(1)-ledger.fee.to_numpy()/prev))):maximum=max(maximum,float(v))
        rows=stocks[(stocks.view==view)&(stocks.k==k)].set_index('security_id').loc[z['security_ids'].tolist()]
        for col,arr in [('net',net),('gross',z['gross']),('fee',z['fees']),('exposure',z['exposure'])]:
            maximum=max(maximum,float(np.max(abs(rows[col].to_numpy()-arr.mean(0)*(1 if col=='exposure' else 252)))))
    for view in ('daily','after_hours'):
        diff=(nets['after_hours_premarket']-nets[view]).mean(0)*252
        grouped=pd.DataFrame({'issuer':rows.issuer.to_numpy(),'net':diff}).groupby('issuer').net.sum()
        actual=issuer[(issuer.k==k)&(issuer.comparator==view)].set_index('issuer').annual_difference
        maximum=max(maximum,float(np.max(abs(grouped.sort_index()-actual.sort_index()))))
        expected=targets[f'after_hours_premarket__ten_seed__k{k}']-targets[f'{view}__ten_seed__k{k}']
        subset=changes[(changes.k==k)&(changes.comparator==view)]
        reconstructed=np.zeros_like(expected);origins={int(v):i for i,v in enumerate(targets['ids'])};names={v:i for i,v in enumerate(targets['security_ids'])}
        assert not subset.duplicated(['origin','security_id']).any()
        for row in subset.itertuples():reconstructed[origins[row.origin],names[row.security_id]]=row.difference
        maximum=max(maximum,float(np.max(abs(expected-reconstructed))));target_rows+=len(subset)
assert maximum<1e-10
print('Six daily traces, four issuer comparisons and',target_rows,'target changes verified; max error',maximum)
