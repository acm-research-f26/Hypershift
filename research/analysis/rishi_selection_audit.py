"""Verify consumed input provenance and fixed selection/leader output identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from rishi_artifact_verify import verify_contract
from rishi_mechanism_selection import CANDIDATES,asof_style,corr


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch10_selection'
    if (out/'INPUT_AND_OUTPUT_AUDIT.json').exists():raise FileExistsError('Preserve audit')
    manifest=pd.read_csv('docs/ndx_artifacts_20261027/manifest.csv')
    verification=verify_contract({'files':manifest[['path','sha256']].to_dict('records')},Path.cwd())
    assert verification['matched']==len(manifest)==1184
    assert all(Path(r.path).stat().st_size==r.bytes for r in manifest.itertuples())
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    inputs=[source/f'scores_{name}_ten_seed.npz' for name in CANDIDATES]
    inputs.extend([source/'ACCOUNT_CURVES.csv',Path('results/RISHI_MECHANISM_RISK_20261027/PATHS_PORTABLE.npz'),
        doc/'execution_batch1/WORKBOOK_METRICS.csv',doc/'execution_batch1/WORKBOOK_PATHS.csv',
        Path('docs/ndx_drawdown_20261026/PROTOCOL.json'),Path('docs/ndx_artifacts_20261027/manifest.csv')])
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    data=load_data();style=pd.read_csv(out/'STYLE_PER_ORIGIN.csv')
    with np.load(source/'scores_Hlabel_ten_seed.npz') as saved:ids=saved['ids'];preference=saved['scores']
    q=pd.read_csv(Path(data['manifest']['root'])/'etf_adj_close.csv',index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).to_numpy()
    max_difference=0.
    for row,i in enumerate(ids):
        values=asof_style(data,int(data['origin'][i]),q);eligible=data['mask'][i]
        for j,name in enumerate(['beta252','momentum126','volatility20','log10adv20']):
            good=eligible&np.isfinite(values[:,j]);actual=corr(pd.Series(preference[row,good]).rank(),pd.Series(values[good,j]).rank())
            previous=style[(style.origin_id==i)&(style.feature==name)].iloc[0]
            assert previous.pairs==good.sum()
            np.testing.assert_allclose(actual,float(previous.spearman),atol=1e-12,equal_nan=True)
            if np.isfinite(actual):max_difference=max(max_difference,abs(actual-float(previous.spearman)))
    per=pd.read_csv(out/'PER_ORIGIN.csv');assert len(per)==1809 and not per.duplicated(['candidate','origin_id']).any()
    assert len(pd.read_csv(out/'PAIRED_SELECTION.csv'))==60 and len(pd.read_csv(out/'ACTIVE_VS_EW.csv'))==54
    bundles=pd.read_csv(out/'SEED_BUNDLE_ACCOUNTS.csv');assert len(bundles)==120 and not bundles.duplicated(['bundle','candidate','k']).any()
    plan=json.loads((doc/'SEED_BUNDLES.json').read_text());assert set(bundles.bundle)=={b['id'] for b in plan['bundles']}
    for row in bundles.itertuples():
        path=pd.read_csv(f'results/RISHI_MECHANISM_SELECTION_20261027/{row.bundle}_{row.candidate}_{row.k}.csv')
        np.testing.assert_allclose(1e6*np.prod(1+path.net),row.terminal_nav,rtol=1e-10)
    covariance=pd.read_csv(out/'RAW_COMPLEMENTARITY.csv');assert len(covariance)==15
    np.testing.assert_allclose(covariance.error_covariance,covariance.forecast_covariance-covariance.b_target_covariance-covariance.d_target_covariance+covariance.target_variance,rtol=1e-10)
    trace=pd.read_csv(doc/'execution_batch9_leader_trace/RECONCILIATION.csv');assert len(trace)==12
    with (out/'INPUT_AND_OUTPUT_AUDIT.json').open('x') as stream:
        json.dump(dict(publication_matched=1184,publication_bytes=int(manifest.bytes.sum()),consumed_generated_input_sha256=hashes,
            fixed_seed_accounts=120,style_recomputed_rows=804,max_style_guard_difference=max_difference,
            leader_accounts=12,leader_max_daily_error=float(trace.max_daily_error.max()),
            scope='Post-outcome provenance/identity audit; finite-positive momentum guard changes no current result; no account rescoring'),stream,indent=2)
        stream.write('\n')
    print('1184 original inputs hash-match;120 seed accounts,804 styles,15 covariance identities and12 leader accounts checked')


if __name__=='__main__':main()
