"""Verify copied long/short outputs, formulas and chart coverage."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import performance
from run_ndx_long_short import RUN, verify_lock


def verify(text):
    verify_lock()
    out=Path('docs/nasdaq_monthly_pit/long_short'); verified=[]
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['independent_signed_cash_paths']==570
    assert audit['parent_long_only_paths_unchanged']==30 and audit['new_model_fits']==0
    complete=json.loads((RUN/'completion.json').read_text())
    assert complete['total_portfolio_paths']==600 and complete['primary_ensemble_paths']==150
    assert complete['short_availability_assumed'] and complete['annual_resets']==0
    for name in ['PROTOCOL.json','INPUT_CODE_LOCK.json','AUDIT.json','completion.json','metrics.csv',
        'primary_metrics.csv','primary_curves.csv','annual.csv','monthly.csv','seed_summary.csv','paired_bootstrap.csv']:
        assert (out/name).read_bytes()==(RUN/name).read_bytes(),name
        verified.append(out/name)
    rows=pd.read_csv(out/'metrics.csv',dtype={'seed':str}).set_index('path_id')
    primary=pd.read_csv(out/'primary_metrics.csv',dtype={'seed':str}).set_index('path_id')
    curves=pd.read_csv(out/'primary_curves.csv'); annual=pd.read_csv(out/'annual.csv')
    assert len(rows)==600 and len(primary)==150 and rows.index.is_unique
    assert rows.parent_reused.sum()==30
    assert not rows.reset_index().duplicated(['model','seed','k','policy','cost_multiplier','borrow_rate']).any()
    assert set(primary.policy)==set(complete['policies']) and set(primary.k)=={5,10,20}
    for path,sub in curves.groupby('path_id',sort=False):
        assert len(sub)==1003 and sub.date.is_unique and sub.date.is_monotonic_increasing
        assert sub.date.iloc[0]=='2022-01-03' and sub.date.iloc[-1]=='2025-12-31'
        result=performance(sub.net)
        for field in ['sharpe','annualized_return','maximum_drawdown','cumulative_return']:
            np.testing.assert_allclose(result[field],primary.loc[path,field],atol=1e-10)
    for path,sub in annual.groupby('path_id',sort=False):
        assert list(sub.year)==[2022,2023,2024,2025]
        np.testing.assert_allclose((1+sub.cumulative_return).prod(),1+rows.loc[path,'cumulative_return'],atol=1e-10)
    for k in [5,10,20]:
        for policy in ['long75_short25','long50_short50']:
            for prefix in ['wealth','years']:
                image=out/'figures'/f'{prefix}_{policy}_k{k}.png'
                assert image.exists() and f'long_short/figures/{image.name}' in text
                verified.append(image)
        image=out/'figures'/f'allocations_k{k}.png'
        assert image.exists() and f'long_short/figures/{image.name}' in text
        verified.append(image)
    for phrase in ['600 disclosed portfolio paths','570 new signed accounts','50% long / 50% short',
        'same-long-allocation plus cash','not a raw-share broker replay','Historical stock locates']:
        assert phrase in text,phrase
    print('Verified long/short publication: 600 paths, 150 primary ensembles and 15 charts')
    return verified
