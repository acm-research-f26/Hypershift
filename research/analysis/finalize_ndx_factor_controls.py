"""Repair summary aggregation while preserving all198already-scored accounts.

The frozen runner's DataFrame.style name collides with pandas' Styler property.
This finalizer uses explicit column indexing, recovers paths from saved ledgers,
reconstructs the exact declared targets and checks every account independently.
No strategy, seed, feature, cost or outcome is changed.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

import run_ndx_factor_controls as study
from run_ndx_retention_replication import ROOT,sha,write
from ndx_rank_retraining import percentile_predictions
from ndx_causal_factors import factor_scores,factor_blend
from ndx_retention import retention_targets,hedged_targets
from ndx_drawdown_engine import allocations,daily_allocation,risk_inputs,exposure_series,market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from audit_ndx_weak_market_short import value_replay
from research_engine import performance


def main():
    cfg=study.verify();run=study.RUN
    if (run/'COMPLETE.json').exists():raise FileExistsError('Completed factor study')
    metrics=pd.read_csv(run/'METRICS.csv');assert len(metrics)==198
    amendment=run/'AGGREGATION_AMENDMENT.json'
    raw=[run/'METRICS.csv']+sorted((run/'ledgers').glob('*.csv'));assert len(raw)==199
    if not amendment.exists():
        snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'research/analysis/run_ndx_factor_controls.py',snapshot/'run_ndx_factor_controls.py')
        write(amendment,dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            reason='DataFrame.style is pandas Styler, not the style column; original accounts completed before summary IndexError',
            scope='Explicit column indexing and recovery of saved outputs only; no change to financial recipes or results',
            repair_after_account_outcomes=True,finalizer_sha256=sha(ROOT/'research/analysis/finalize_ndx_factor_controls.py'),
            protocol_sha256=sha(run/'PROTOCOL.json'),
            preserved_raw_sha256={p.relative_to(run).as_posix():sha(p) for p in raw}))
    audit_amendment=json.loads(amendment.read_text())
    assert audit_amendment['finalizer_sha256']==sha(ROOT/'research/analysis/finalize_ndx_factor_controls.py')
    for name,expected in audit_amendment['preserved_raw_sha256'].items():assert sha(run/name)==expected
    plan=study.check_fits(study.SOURCE);data=study.parent.load_data()
    ids=np.concatenate([study.parent.fold(data,y)[2] for y in plan['years']]);mask=data['mask'][ids]
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    factors=factor_scores(data,ids);families={};seeds=plan['groups']['fifteen_seed']
    for family in plan['families']:
        rows=[]
        for seed in seeds:
            parts=[]
            for year in plan['years']:
                with np.load(study.SOURCE/'fits'/family/str(year)/str(seed)/'rank_predictions.npz') as p:parts.append(p['test'])
            rows.append(percentile_predictions(np.concatenate(parts),mask))
        families[family]=np.asarray(rows)
    neural={}
    for group,members in plan['groups'].items():
        index=[seeds.index(seed) for seed in members]
        neural[group]=.5*families['think'][index].mean(0)+.5*families['dynamic'][index].mean(0)
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'))
    paths={};target_archive={};checks=[];books={}
    for row in metrics.to_dict('records'):
        key=(row['style'],row['neural_weight'],row['group'],row['allocation'])
        if key not in books:
            style,weight,group,allocation=key
            score=factors[style] if weight==0 else factor_blend(factors[style],neural[group],weight)
            w=retention_targets(score,mask,5) if allocation=='retain5' else allocations(data,ids,score,'issuer5')
            risk=risk_inputs(data,daily_allocation(data,ids,w))
            long=exposure_series(dict(vol='basket',window=20,target=.20),market,risk)[data['origin'][ids]]
            books[key]=hedged_targets(w,long,market['qqqvol20'][data['origin'][ids]],.75)
        desired=books[key];ledger=pd.read_csv(run/'ledgers'/(row['id']+'.csv'))
        np.testing.assert_array_equal(ledger['date'].to_numpy(),dates)
        nav=ledger['nav'].to_numpy();path=nav/np.r_[1e6,nav[:-1]]-1
        independent=value_replay(combined,ids,desired,row['borrow'],row['cost'])
        error=float(np.max(abs(path-independent)));assert error<1e-12
        for field,value in performance(path).items():np.testing.assert_allclose(value,row[field],rtol=1e-11,atol=1e-12)
        paths[row['id']]=path;target_archive[row['id']]=desired
        checks.append(dict(id=row['id'],max_daily_error=error))
    summaries=[]
    for (style,weight,allocation,cost,borrow),block in metrics[metrics['neural_weight']>0].groupby(
            ['style','neural_weight','allocation','cost','borrow']):
        pure=metrics[(metrics['style']==style)&(metrics['neural_weight']==0)&
            (metrics['allocation']==allocation)&(metrics['cost']==cost)&(metrics['borrow']==borrow)].iloc[0]
        row=dict(style=style,neural_weight=weight,allocation=allocation,cost=cost,borrow=borrow);values=[]
        for group in plan['groups']:
            value=float(block.loc[block['group']==group,'sharpe'].iloc[0]);values.append(value);row[group+'_sharpe']=value
        row.update(pure_factor_sharpe=float(pure['sharpe']),minimum_group_sharpe=min(values),
            minimum_increment_over_pure=min(values)-float(pure['sharpe']),goal_passed=min(values)>1.5)
        summaries.append(row)
    summary=pd.DataFrame(summaries).sort_values('minimum_group_sharpe',ascending=False)
    summary.to_csv(run/'CONSISTENCY.csv',index=False);pd.DataFrame(checks).to_csv(run/'ACCOUNT_CHECKS.csv',index=False)
    np.savez_compressed(run/'PATHS.npz',dates=dates,**paths)
    np.savez_compressed(run/'TARGETS.npz',ids=ids,security_ids=np.asarray(combined['panel'].tickers,dtype=str),**target_archive)
    base=summary[(summary['cost']==1)&(summary['borrow']==.03)]
    write(run/'AUDIT.json',dict(status='passed',accounts=198,max_independent_daily_error=max(c['max_daily_error'] for c in checks),
        passing_hybrid_recipes=int(base['goal_passed'].sum()),source_fit_lock_sha256=sha(study.SOURCE/'FITS_COMPLETE.json'),
        aggregation_repair='Explicit style-column access; all198metrics/ledgers byte-preserved; paths recovered from ledger NAV and checked against independent target/account replay',
        warning='Shared deterministic factors are not independent seed evidence; compare pure-factor controls; spent historical data'))
    for name,expected in audit_amendment['preserved_raw_sha256'].items():assert sha(run/name)==expected
    study.verify()
    write(run/'COMPLETE.json',dict(output_sha256={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}))
    print('Finalized198unchangedaccounts;max independent return error',max(c['max_daily_error'] for c in checks),flush=True)
    print(base.to_string(index=False),flush=True)


if __name__=='__main__':main()
