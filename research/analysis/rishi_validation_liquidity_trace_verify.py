"""Verify four liquidity comparison paths, issuer totals and target changes."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    source=Path('results/RISHI_VALIDATION_STYLE_ACCOUNTS_20261029');root=Path('results/RISHI_VALIDATION_LIQUIDITY_TRACE_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch62_liquidity_trace')
    paths=np.load(source/'PATHS.npz',allow_pickle=False);targets=np.load(source/'TARGETS.npz',allow_pickle=False);stock=pd.read_csv(out/'STOCKS.csv');issuers=pd.read_csv(out/'PAIRED_ISSUERS.csv');calendar=pd.read_csv(out/'CALENDAR_STOCKS.csv');changes=pd.read_csv(out/'TARGET_CHANGES.csv');years=pd.DatetimeIndex(paths['dates']).year;maximum=0.;count=0
    for kind in ('issuer5','equal10'):
        values={}
        for arm in ('fixed','liquidity'):
            key=f'ten_seed__{kind}__{arm}';z=np.load(root/f'{key}.npz',allow_pickle=False);ledger=pd.read_csv(source/f'{key}_ledger.csv');assert z['dates'].tolist()==ledger.date.tolist()==paths['dates'].tolist()
            prev=np.r_[1e6,ledger.nav.to_numpy()[:-1]];net=z['net'];values[arm]=net
            for e in (np.max(abs(net.sum(1)-paths[key])),np.max(abs(z['gross']-z['fees']-net)),np.max(abs(z['fees'].sum(1)-ledger.fee/prev))):maximum=max(maximum,float(e))
            s=stock[(stock.kind==kind)&(stock.arm==arm)].set_index('security_id').loc[z['security_ids'].tolist()]
            for name,array in [('net',net),('gross',z['gross']),('fee',z['fees']),('exposure',z['exposure'])]:maximum=max(maximum,float(np.max(abs(s[name].to_numpy()-array.mean(0)*(1 if name=='exposure' else 252)))))
        diff=values['fixed']-values['liquidity'];expected=pd.DataFrame({'issuer':s.issuer.to_numpy(),'value':252*diff.mean(0)}).groupby('issuer').value.sum().sort_index();actual=issuers[issuers.kind==kind].set_index('issuer').difference.sort_index();maximum=max(maximum,float(np.max(abs(expected-actual))))
        for year in (2022,2023,2024,2025):
            actual=calendar[(calendar.kind==kind)&(calendar.year==year)].set_index('security_id').loc[z['security_ids'].tolist()].difference.to_numpy();maximum=max(maximum,float(np.max(abs(actual-252*diff[years==year].mean(0)))))
        expected=targets[f'ten_seed__{kind}__fixed']-targets[f'ten_seed__{kind}__liquidity'];rebuilt=np.zeros_like(expected);ix={int(v):i for i,v in enumerate(targets['ids'])};names={v:i for i,v in enumerate(targets['security_ids'])};rows=changes[changes.kind==kind];assert not rows.duplicated(['origin','security_id']).any()
        for r in rows.itertuples():rebuilt[ix[r.origin],names[r.security_id]]=r.difference
        maximum=max(maximum,float(np.max(abs(expected-rebuilt))));count+=len(rows)
    assert maximum<1e-10 and count==len(changes)
    print(f'4 paths, 2 issuer tables, 8 calendar partitions and {count} target changes verified; max error {maximum:.3g}')


if __name__=='__main__':main()
