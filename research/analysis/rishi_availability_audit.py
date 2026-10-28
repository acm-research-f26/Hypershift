"""Audit common-screen accounts and additive cohort identities; no strategy rescore."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold
from rishi_availability import common_screen
from rishi_mechanism_selection import CANDIDATES


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch11_availability'
    large=Path('results/RISHI_MECHANISM_AVAILABILITY_20261027')
    if (out/'AUDIT.json').exists():raise FileExistsError('Preserve audit')
    data=load_data();source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    with np.load(source/'scores_E_ten_seed.npz') as saved:ids=saved['ids']
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy();counts=np.zeros((len(ids),len(data['panel'].tickers)),int)
    for year in [2022,2023,2024,2025]:
        train,_,_=fold(data,year);counts[years==year]=data['labelmask'][train].sum(0)
    screen=common_screen(data,ids,counts);diagnostics=pd.read_csv(out/'MASK_DIAGNOSTICS.csv')
    np.testing.assert_array_equal(screen.sum(1),diagnostics.common_eligible)
    with (large/'COMMON_MASK_AXES.npz').open('xb') as stream:
        np.savez_compressed(stream,origin_ids=ids,security_ids=np.asarray(data['panel'].tickers,dtype=str),
            signal_dates=np.asarray(data['panel'].dates[data['origin'][ids]].strftime('%Y-%m-%d'),dtype='U10'),
            entry_dates=np.asarray(data['panel'].dates[data['entry'][ids]].strftime('%Y-%m-%d'),dtype='U10'),
            original_eligible=data['mask'][ids],common_eligible=screen,training_label_counts=counts)
    metrics=pd.read_csv(out/'ACCOUNT_METRICS.csv');assert len(metrics)==19 and not metrics.duplicated(['candidate','k']).any()
    with np.load(large/'PATHS.npz',allow_pickle=False) as archive:
        assert archive['returns'].shape==(1003,19)
        for j,key in enumerate(archive['ids']):
            candidate,k=key.split('/');row=metrics[(metrics.candidate==candidate)&(metrics.k==int(k))].iloc[0]
            ending=float(1e6*np.prod(1+archive['returns'][:,j]))
            np.testing.assert_allclose(ending,row.terminal_nav,rtol=1e-10)
            np.testing.assert_allclose((ending/1e6)**(252/1003)-1,row.annualized_return,rtol=1e-10,atol=1e-12)
    cohorts=pd.read_csv(out/'MEMBERSHIP_COHORTS.csv');assert len(cohorts)==57 and cohorts.id.nunique()==19
    original=pd.read_csv(source/'ACCOUNT_CURVES.csv');workbook=pd.read_csv(doc/'execution_batch1/WORKBOOK_PATHS.csv')
    max_error=0.
    for key,rows in cohorts.groupby('id'):
        path=original[original.id==key.removesuffix('/full')] if key.endswith('/full') else workbook[workbook.id==key]
        assert len(path)==1003
        error=abs(float(rows.annual_arithmetic_contribution.sum()-path.net.mean()*252));max_error=max(max_error,error)
        assert error<1e-12
        np.testing.assert_allclose(rows.gross_dollars.sum()-rows.fees.sum(),1e6*(np.prod(1+path.net)-1),rtol=1e-10,atol=1e-6)
    inputs=[source/f'scores_{name}_ten_seed.npz' for name in CANDIDATES]
    inputs.extend([source/'ACCOUNT_CURVES.csv',doc/'execution_batch1/WORKBOOK_PATHS.csv',
                   doc/'execution_batch10_selection/INPUT_AND_OUTPUT_AUDIT.json',Path('results/RISHI_MECHANISM_TRACE_20261027/AXES.json')])
    inputs.extend(Path('results/RISHI_MECHANISM_TRACE_20261027').glob('*.npz'))
    inputs.extend(Path('results/RISHI_MECHANISM_LEADER_TRACE_20261027').glob('*.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    with (out/'AUDIT.json').open('x') as stream:json.dump(dict(common_accounts=19,cohort_accounts=19,cohort_rows=57,
        max_cohort_annual_mean_error=max_error,input_sha256=hashes,original_masks_and_accounts_unchanged=True,
        source_membership_not_as_published_certification=True),stream,indent=2);stream.write('\n')
    print('19common accounts,57cohort rows,annual means and dollar profit identities verified; input hashes/mask axes recorded')


if __name__=='__main__':main()
