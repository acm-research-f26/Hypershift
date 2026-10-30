"""Independently verify traced paths, issuer sums, years and target changes."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    source=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029');root=Path('results/RISHI_VALIDATION_FIXED_TRACE_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch52_validation_trace')
    paths=np.load(source/'PATHS.npz',allow_pickle=False);targets=np.load(source/'TARGETS.npz',allow_pickle=False)
    stock=pd.read_csv(out/'STOCKS.csv');issuer=pd.read_csv(out/'PAIRED_ISSUERS.csv');years_table=pd.read_csv(out/'CALENDAR_STOCKS.csv');changes=pd.read_csv(out/'TARGET_CHANGES.csv')
    years=pd.DatetimeIndex(paths['dates']).year;maximum=0.;count=0
    for group in ('original_five','new_five','ten_seed'):
        values={}
        for arm in ('original','fixed'):
            key=f'{group}__{arm}__vol20';z=np.load(root/f'{key}.npz',allow_pickle=False);ledger=pd.read_csv(source/f'{key}_ledger.csv');assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
            prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]];net=z['net'];values[arm]=net
            for e in (np.max(abs(net.sum(1)-paths[key])),np.max(abs(z['gross']-z['fees']-net)),np.max(abs(z['fees'].sum(1)-ledger.fee/prev))):maximum=max(maximum,float(e))
            s=stock[(stock.group==group)&(stock.arm==arm)].set_index('security_id').loc[z['security_ids'].tolist()]
            for name,array in [('net',net),('gross',z['gross']),('fee',z['fees']),('exposure',z['exposure'])]:
                maximum=max(maximum,float(np.max(abs(s[name].to_numpy()-array.mean(0)*(1 if name=='exposure' else 252)))))
        delta=values['original']-values['fixed'];agg=pd.DataFrame({'issuer':s.issuer.to_numpy(),'value':252*delta.mean(0)}).groupby('issuer').value.sum().sort_index()
        got=issuer[issuer.group==group].set_index('issuer').difference.sort_index();maximum=max(maximum,float(np.max(abs(agg-got))))
        for year in (2022,2023,2024,2025):
            got=years_table[(years_table.group==group)&(years_table.year==year)].set_index('security_id').loc[z['security_ids'].tolist()].difference.to_numpy()
            maximum=max(maximum,float(np.max(abs(got-252*delta[years==year].mean(0)))))
        expected=targets[f'{group}__original']-targets[f'{group}__fixed'];reconstructed=np.zeros_like(expected)
        origins={int(v):i for i,v in enumerate(targets['ids'])};names={v:i for i,v in enumerate(targets['security_ids'])};subset=changes[changes.group==group]
        assert not subset.duplicated(['origin','security_id']).any()
        for row in subset.itertuples():reconstructed[origins[row.origin],names[row.security_id]]=row.difference
        maximum=max(maximum,float(np.max(abs(expected-reconstructed))));count+=len(subset)
    assert maximum<1e-10 and count==len(changes)
    print(f'6 paths, 3 issuer tables, 12 calendar partitions, {count} target changes verified; maximum error {maximum:.3g}')


if __name__=='__main__':main()
