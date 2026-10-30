"""Rebuild six daily traces, four issuer contrasts and target changes."""
from pathlib import Path
import numpy as np
import pandas as pd

source=Path('results/RISHI_TRAINING_EXCLUSION_20261028');root=Path('results/RISHI_TRAINING_EXCLUSION_TRACE_20261028')
out=Path('docs/rishi_mechanism_20261027/execution_batch40_exclusion_trace')
paths=np.load(source/'PATHS.npz',allow_pickle=False);stocks=pd.read_csv(out/'STOCKS.csv')
issuer=pd.read_csv(out/'PAIRED_ISSUERS.csv');changes=pd.read_csv(out/'TARGET_CHANGES.csv');maximum=0.;target_rows=0
for model in ('E','E_Hlabel'):
    net={};weights={}
    for n in (0,1,3):
        key=f'{model}_k10_exclude{n}';z=np.load(root/f'{key}.npz',allow_pickle=False);w=np.load(source/f'{key}_weights.npz',allow_pickle=False)
        ledger=pd.read_csv(source/f'{key}_ledger.csv');assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
        net[n]=z['net'];weights[n]=w['weights'];prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]]
        for value in (np.max(abs(net[n].sum(1)-paths[key])),np.max(abs(z['gross']-z['fees']-net[n])),np.max(abs(z['fees'].sum(1)-ledger.fee.to_numpy()/prev))):maximum=max(maximum,float(value))
        rows=stocks[(stocks.model==model)&(stocks['exclude']==n)].set_index('security_id').loc[z['security_ids'].tolist()]
        for column,arr in [('net',net[n]),('gross',z['gross']),('fee',z['fees']),('exposure',z['exposure'])]:
            expected=arr.mean(0)*(1 if column=='exposure' else 252)
            maximum=max(maximum,float(np.max(abs(rows[column].to_numpy()-expected))))
    for n in (1,3):
        diff=(net[n]-net[0]).mean(0)*252
        table=stocks[(stocks.model==model)&(stocks['exclude']==n)].set_index('security_id').loc[z['security_ids'].tolist()]
        grouped=pd.DataFrame({'issuer':table.issuer.to_numpy(),'v':diff}).groupby('issuer').v.sum()
        actual=issuer[(issuer.model==model)&(issuer['exclude']==n)].set_index('issuer').net_difference
        maximum=max(maximum,float(np.max(abs(grouped.sort_index()-actual.sort_index()))))
        w=np.load(source/f'{model}_k10_exclude{n}_weights.npz',allow_pickle=False);expected=weights[n]-weights[0]
        subset=changes[(changes.model==model)&(changes['exclude']==n)]
        reconstructed=np.zeros_like(expected);origin_index={int(v):i for i,v in enumerate(w['ids'])};stock_index={v:i for i,v in enumerate(w['security_ids'])}
        assert not subset.duplicated(['origin','security_id']).any()
        for row in subset.itertuples():reconstructed[origin_index[row.origin],stock_index[row.security_id]]=row.change
        maximum=max(maximum,float(np.max(abs(expected-reconstructed))));target_rows+=len(subset)
assert maximum<1e-10
print('Six daily traces, four issuer comparisons and',target_rows,'target changes verified; max error',maximum)
