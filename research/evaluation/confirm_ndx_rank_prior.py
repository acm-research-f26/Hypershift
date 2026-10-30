"""Freeze a secondary all-group improvement candidate before reserve outcomes.

The original strict source gate failed. This secondary development selection is
explicitly disclosed and cannot retroactively turn that failure into a pass.
"""
import argparse
from datetime import datetime,timezone
import json

import numpy as np
import pandas as pd

import run_ndx_prior_robustness as prior
import confirm_ndx_portfolio_robustness as reserve
from ndx_portfolio_robustness import average_targets,simulate_banded
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from ndx_drawdown_engine import market_inputs
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha,write

ROOT=prior.ROOT
RUN=prior.RUN/'confirmation'
CODE=ROOT/'research/evaluation/confirm_ndx_rank_prior.py'


def freeze():
    plan=prior.verify();assert not (reserve.CONF/'METRICS.csv').exists()
    summary=pd.read_csv(prior.RUN/'CONSISTENCY.csv')
    base=summary[(summary.cost==1)&(summary.borrow==.03)]
    # Secondary criterion adopted from development outcomes, then frozen before
    # the unused cohort. Do not label it the original stricter primary gate.
    eligible=base[(base.minimum_issuer_delta>=.01)&(base.min_all_sharpe>=1.15)&(base.disjoint_gap<=.25)]
    if eligible.empty:raise ValueError('No secondary all-group improvement candidate')
    choice=eligible.sort_values(['min_all_sharpe','mean_all_sharpe','arm','allocation','aggregation'],
                               ascending=[False,False,True,True,True]).iloc[0].to_dict()
    half=plan['half_lives'][plan['architectures'].index(choice['arm'])]
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),candidate=choice,half_life=half,
        source_selection='Secondary/post-hoc selection after source primary gate failed: every group improves issuer baseline by>=.01, minSR>=1.15, source disjoint gap<=.25; maximize min then meanSR. No reserve account outcomes observed.',
        source_primary_passed=False,source_summary_sha256=sha(prior.RUN/'CONSISTENCY.csv'),
        source_protocol_sha256=sha(prior.RUN/'PROTOCOL.json'),reserve_protocol_sha256=sha(reserve.CONF/'PROTOCOL.json'),
        code_sha256=sha(CODE),new_fits=0,accounts=6,
        goal='Unused-five and expanded20candidateSR exceed matched original issuer baseline by>=.01 at base fees. Neither is worse under2xfees/6%borrow. Four disjoint five-seed cohortSR>=1.15, spread<=.30. All outcomes reported, no reselection.',
        limits='Unused seed cohort only; market history is spent, secondary selection does not establish market generalization or reliable futureSR1.5.'))
    write(RUN/'FREEZE.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),reserve_accounts_absent=True))
    print('Frozen secondary candidate before unused outcomes:',choice['arm'],choice['allocation'],choice['aggregation'],flush=True)


def verify():
    prior.verify();p=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    assert sha(prior.RUN/'CONSISTENCY.csv')==p['source_summary_sha256']
    assert sha(prior.RUN/'PROTOCOL.json')==p['source_protocol_sha256']
    assert sha(reserve.CONF/'PROTOCOL.json')==p['reserve_protocol_sha256']
    assert sha(CODE)==p['code_sha256']
    return p


def evaluate():
    p=verify();plan=reserve.verify();lock=json.loads((reserve.CONF/'FITS_COMPLETE.json').read_text())
    for name,expected in lock['hashes'].items():assert sha(reserve.CONF/name)==expected
    data=prior.parent.parent.load_data();ids=np.concatenate([prior.parent.parent.fold(data,y)[2] for y in plan['years']])
    neural=reserve.scores(plan,data,ids,'original');market=market_inputs(data)
    _,rank,_=prior.priors(dict(years=plan['years']),data,ids,p['half_life'])
    combined=append_qqq(data,etf_data(data,'QQQ'));eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    def pool(index):return .5*rank+.25*neural[0,index].mean(0)+.25*neural[1,index].mean(0)
    dest=RUN/'ledgers';dest.mkdir(exist_ok=False);rows=[];paths={};targets={}
    for group,index in [('unused_five',list(range(15,20))),('expanded_twenty',list(range(20)))]:
        candidate=p['candidate'];allocation=candidate['allocation']
        w=(prior.parent.signed_targets(data,ids,pool(index),allocation,market) if candidate['aggregation']=='rank_pool' else
           average_targets([prior.parent.signed_targets(data,ids,pool(index[k:k+5]),allocation,market) for k in range(0,len(index),5)]))
        for setting in plan['cost_regimes']:
            cost,borrow=setting['cost'],setting['borrow'];key=f'{group}__prior_candidate__c{cost}__b{int(borrow*100):02d}'
            measured,path,ledger,effective,_=simulate_banded(combined,ids,w,eligible,0.,borrow,cost)
            error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))));assert error<1e-12
            ledger.to_csv(dest/(key+'.csv'),index=False);paths[key]=path;targets[key]=effective
            rows.append(dict(id=key,group=group,cost=cost,borrow=borrow,max_independent_daily_error=error,**measured))
    frame=pd.DataFrame(rows);frame.to_csv(RUN/'METRICS.csv',index=False)
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    np.savez_compressed(RUN/'PATHS.npz',dates=dates,**paths);np.savez_compressed(RUN/'TARGETS.npz',ids=ids,**targets)
    baseline=pd.read_csv(reserve.CONF/'METRICS.csv')
    deltas={};stress={};srs={}
    for group in ('unused_five','expanded_twenty'):
        for cost,borrow,container in ((1,.03,deltas),(2,.06,stress)):
            a=frame[(frame.group==group)&(frame.cost==cost)&(frame.borrow==borrow)].iloc[0]
            b=baseline[(baseline.group==group)&(baseline.strategy=='baseline_issuer')&(baseline.cost==cost)&(baseline.borrow==borrow)].iloc[0]
            container[group]=float(a.sharpe-b.sharpe)
            if cost==1:srs[group]=float(a.sharpe)
    disjoint=[float(p['candidate'][g+'_sharpe']) for g in ('original_five','new_five','fresh_five')]+[srs['unused_five']]
    passed=min(deltas.values())>=.01 and min(stress.values())>=0 and min(disjoint)>=1.15 and max(disjoint)-min(disjoint)<=.30
    result=dict(passed=passed,secondary_confirmation=True,original_source_primary_passed=False,
        base_issuer_deltas=deltas,stress_issuer_deltas=stress,four_disjoint_sharpes=disjoint,
        disjoint_gap=max(disjoint)-min(disjoint),candidate_sharpes=srs,
        all_above_1_5=min(disjoint+[srs['expanded_twenty']])>1.5,
        limits='Seed-only confirmation of a secondary development-selected recipe; no untouched-market evidence')
    write(RUN/'RESULT.json',result);verify()
    write(RUN/'COMPLETE.json',dict(accounts=6,output_sha256={q.relative_to(RUN).as_posix():sha(q) for q in sorted(RUN.rglob('*')) if q.is_file()}))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','evaluate'],required=True)
    args=parser.parse_args();freeze() if args.stage=='freeze' else evaluate()
