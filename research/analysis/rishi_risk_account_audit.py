"""Audit existing risk accounts; preserve original archives and add portable axes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_mechanism_risk import write_paths


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch8_risk')
    large=Path('results/RISHI_MECHANISM_RISK_20261027')
    if (out/'ACCOUNT_AUDIT.json').exists():raise FileExistsError('Preserve audit receipt')
    metrics=pd.read_csv(out/'ACCOUNT_METRICS.csv')
    assert len(metrics)==525 and metrics.id.nunique()==525
    assert (metrics.groupby('cost').size()==75).all()
    with np.load(large/'PATHS.npz',allow_pickle=False) as archive:
        ids=archive['ids'];paths=archive['returns'] # Original dates used object dtype; do not unpickle them.
    assert paths.shape==(1003,525) and ids.tolist()==metrics.id.tolist()
    dates=pd.read_csv(large/(ids[0].replace('/','__')+'.csv')).date.tolist()
    dates=pd.DatetimeIndex(dates)
    root=Path('data/nasdaq_monthly_pit/assembled')
    q=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).reindex(dates).to_numpy()
    downside=[];max_nav_error=0.;minimum_cash=float('inf')
    for j,key in enumerate(ids):
        detail=pd.read_csv(large/(key.replace('/','__')+'.csv'))
        assert detail.date.tolist()==dates.strftime('%Y-%m-%d').tolist()
        np.testing.assert_allclose(detail.net,paths[:,j],rtol=0,atol=1e-14)
        error=float(np.max(np.abs(1e6*np.cumprod(1+paths[:,j])-detail.nav)/1e6))
        max_nav_error=max(max_nav_error,error);assert error<1e-8
        assert detail.equity_exposure.iloc[-1]==0
        minimum_cash=min(minimum_cash,float(detail.cash.min()))
        rc=paths[:,j]-paths[:,j].mean();qc=q-q.mean();negative=q<0
        downside.append(dict(id=key,qqq_negative_sessions=int(negative.sum()),
            conditional_crossproduct_annual=float(np.mean(rc[negative]*qc[negative])*252),
            unconditional_downside_crossproduct_annual=float(np.mean(np.where(negative,rc*qc,0))*252)))
    assert minimum_cash>=-1e-5
    annual=pd.read_csv(out/'CALENDAR_ATTRIBUTION.csv');assert len(annual)==2100
    for key,rows in annual.groupby('id'):
        i=np.flatnonzero(ids==key)[0]
        np.testing.assert_allclose(rows.log_growth.sum(),np.log1p(paths[:,i]).sum(),atol=1e-12)
        np.testing.assert_allclose(np.prod(1+rows.calendar_return),np.prod(1+paths[:,i]),rtol=1e-12)
    parity=pd.read_csv(out/'SOURCE_REPLAY.csv');assert len(parity)==30 and parity.daily_error.max()==0
    fixed=pd.read_csv(out/'FIXED_CALIBRATION.csv');timing=pd.read_csv('runs/ndx_extended_hours_20261024/TIMING.csv')
    rows=[]
    for row in fixed.to_dict('records'):
        t=timing.iloc[row['first_origin_id']]
        assert row['first_signal_date']==t.prior_regular_close
        rows.append({**row,'decision_utc':t.decision_utc,'feature_cutoff_utc':t.feature_cutoff_utc,'fill_utc':t.fill_utc})
    pd.DataFrame(rows).to_csv(out/'CALIBRATION_TIMING.csv',index=False)
    pd.DataFrame(downside).to_csv(out/'DOWNSIDE_ZERO_THRESHOLD.csv',index=False)
    portable=large/'PATHS_PORTABLE.npz'
    write_paths(portable,dict(zip(ids,paths.T)),dates)
    with np.load(portable,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['returns'],paths)
        assert saved['ids'].tolist()==ids.tolist() and saved['dates'].tolist()==dates.strftime('%Y-%m-%d').tolist()
    receipt=dict(accounts=525,observations=525*1003,calendar_rows=len(annual),source_accounts=30,
        max_normalized_nav_error=max_nav_error,minimum_cash=minimum_cash,all_terminal_equity_zero=True,
        portable_returns_identical=True,original_archive_preserved=True,
        additional_downside_view='Actual QQQ returns below zero; original metric used QQQ return below its sample mean. Both preserved; attribution-only definition clarification.',
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (out/'ACCOUNT_AUDIT.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
