"""Freeze and confirm one selected recipe on five unused initialization seeds.

This is initialization confirmation on spent market history, not a market holdout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_ndx_portfolio_robustness_v2 as study
import run_ndx_rank_retraining as trainer
from ndx_shared_think import SharedThink, initialize_shared
from ndx_portfolio_robustness import average_targets, simulate_banded
from ndx_rank_retraining import percentile_predictions
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from ndx_drawdown_engine import market_inputs
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha, write
from training_device import configure_device

ROOT = study.ROOT
CONF = study.RUN/'confirmation'
ORIGINAL_MODEL = trainer.ExtendedModel
ORIGINAL_INITIALIZE = trainer.neutralize_stock_initialization


def retry_atomic(path,value):
    temporary=path.with_suffix('.tmp');write(temporary,value)
    for attempt in range(50):
        try:
            temporary.replace(path);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


def freeze(family=False):
    plan=study.verify()
    selection_path=(ROOT/'runs/ndx_family_robustness_20261030/SELECTION.json') if family else study.RUN/'SELECTION.json'
    selection=json.loads(selection_path.read_text())
    candidate=selection['selected']
    if candidate is None: raise ValueError('No qualifying recipe; cannot claim successful confirmation')
    expansion=json.loads((ROOT/'config/ndx_seed_expansion_20261020.json').read_text())
    seeds=expansion['new_seeds'][10:15]
    assert len(seeds)==5 and not set(seeds)&set(plan['seed_groups']['fifteen_seed'])
    CONF.mkdir(exist_ok=False)
    config=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),candidate=candidate,seeds=seeds,
        expanded_seeds=plan['seed_groups']['fifteen_seed']+seeds,years=plan['years'],epochs=12,
        fits=60 if candidate['arm']=='shared' else 40,
        strategies=['baseline_issuer','baseline_matched','candidate'],
        groups=['unused_five','expanded_twenty'],cost_regimes=plan['cost_regimes'],
        selection_relative=selection_path.relative_to(ROOT).as_posix(),
        selection_sha256=sha(selection_path),
        script_sha256=sha(ROOT/'research/training/confirm_ndx_portfolio_robustness.py'),
        study_protocol_sha256=sha(study.RUN/'PROTOCOL.json'),
        goal='Candidate exceeds baseline issuer SR by>=0.01 in unused_five and expanded_twenty at base costs; stressSR not worse in either group; four disjoint five-seed cohort spread<=0.20. All cases reported, no reselection.',
        limits='Seed-only confirmation. Same market history; no unseen-market generalization or reliable futureSR claim.')
    write(CONF/'PROTOCOL.json',config)
    write(CONF/'FREEZE.json',dict(protocol_sha256=sha(CONF/'PROTOCOL.json'),fits_absent=True,accounts_absent=True))
    print('Frozen confirmation seeds',seeds,'fits',config['fits'],flush=True)


def verify():
    study.verify(); plan=json.loads((CONF/'PROTOCOL.json').read_text())
    assert sha(CONF/'PROTOCOL.json')==json.loads((CONF/'FREEZE.json').read_text())['protocol_sha256']
    assert sha(ROOT/plan['selection_relative'])==plan['selection_sha256']
    assert sha(ROOT/'research/training/confirm_ndx_portfolio_robustness.py')==plan['script_sha256']
    assert sha(study.RUN/'PROTOCOL.json')==plan['study_protocol_sha256']
    return plan


def year_job(year):
    plan=verify(); device=configure_device('cuda'); prepared=trainer.prepare(year,device)
    progress=CONF/'worker_progress';progress.mkdir(exist_ok=True)
    def log(path,value):
        if path.name=='PROGRESS.json':path=progress/f'{path.parent.name}_{year}.json'
        retry_atomic(path,value)
    trainer.atomic_json=log
    trainer.ExtendedModel=ORIGINAL_MODEL
    trainer.neutralize_stock_initialization=ORIGINAL_INITIALIZE
    for family in ('think','dynamic'):
        for seed in plan['seeds']:
            trainer.fit(CONF/'base',prepared,family,year,seed,plan,device)
    if plan['candidate']['arm']=='shared':
        trainer.ExtendedModel=SharedThink;trainer.neutralize_stock_initialization=initialize_shared
        for seed in plan['seeds']:
            trainer.fit(CONF/'shared',prepared,'think',year,seed,plan,device)
    return year


def train():
    plan=verify()
    if (CONF/'FITS_COMPLETE.json').exists(): raise FileExistsError('Confirmation already fitted')
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(year_job,year) for year in reversed(plan['years'])]
        for job in as_completed(jobs):print('Confirmation year complete',job.result(),flush=True)
    records=list(CONF.glob('*/fits/*/*/*/RECORD.json'));assert len(records)==plan['fits']
    hashes={}
    for record in records:
        for name,expected in json.loads(record.read_text())['hashes'].items():
            path=record.parent/name;assert sha(path)==expected;hashes[path.relative_to(CONF).as_posix()]=expected
        hashes[record.relative_to(CONF).as_posix()]=sha(record)
    verify();write(CONF/'FITS_COMPLETE.json',dict(fits=len(records),hashes=hashes,accounts_absent=True))


def scores(plan,data,ids,arm):
    result=[]
    for family in ('think','dynamic'):
        values=[]
        for seed in plan['expanded_seeds']:
            fresh=seed in plan['seeds']
            base=(CONF/'shared' if fresh else study.RUN) if arm=='shared' and family=='think' else (
                CONF/'base' if fresh else study.SOURCE)
            annual=[]
            for year in plan['years']:
                with np.load(base/'fits'/family/str(year)/str(seed)/'loss_predictions.npz',allow_pickle=False) as p:
                    annual.append(p['test'].copy())
            values.append(percentile_predictions(np.concatenate(annual),data['mask'][ids]))
        result.append(np.asarray(values))
    return np.asarray(result)


def evaluate():
    plan=verify();lock=json.loads((CONF/'FITS_COMPLETE.json').read_text())
    for name,expected in lock['hashes'].items():assert sha(CONF/name)==expected
    dest=CONF/'accounts';dest.mkdir(exist_ok=False)
    data=study.parent.load_data();ids=np.concatenate([study.parent.fold(data,y)[2] for y in plan['years']])
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'))
    eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    baseline=scores(plan,data,ids,'original')
    candidate=baseline if plan['candidate']['arm']=='original' else scores(plan,data,ids,'shared')
    years=np.concatenate([np.full(len(study.parent.fold(data,y)[2]),y) for y in plan['years']])
    family_weights=plan['candidate'].get('family_weights_by_year',{str(y):.5 for y in plan['years']})
    candidate_weight=np.array([family_weights[str(y)] for y in years])[:,None]
    def pool(values,index,weight):return weight*values[0,index].mean(0)+(1-weight)*values[1,index].mean(0)
    def target(values,index,allocation,weight=.5):return study.signed_targets(data,ids,pool(values,index,weight),allocation,market)
    targets={}
    for group,index in [('unused_five',list(range(15,20))),('expanded_twenty',list(range(20)))]:
        allocation=plan['candidate']['allocation'];aggregation=plan['candidate']['aggregation']
        targets[group,'baseline_issuer']=target(baseline,index,'issuer5')
        targets[group,'baseline_matched']=target(baseline,index,allocation)
        if aggregation=='rank_pool':desired=target(candidate,index,allocation,candidate_weight)
        elif aggregation=='seed_targets':desired=average_targets([target(candidate,[j],allocation,candidate_weight) for j in index])
        else:desired=average_targets([target(candidate,index[k:k+5],allocation,candidate_weight) for k in range(0,len(index),5)])
        targets[group,'candidate']=desired
    rows=[];paths={};effective_targets={};checks=[]
    for (group,strategy),desired in targets.items():
        band=float(plan['candidate']['band']) if strategy=='candidate' else 0.
        for setting in plan['cost_regimes']:
            cost,borrow=setting['cost'],setting['borrow'];key=f'{group}__{strategy}__c{cost}__b{int(borrow*100):02d}'
            measured,path,ledger,effective,_=simulate_banded(combined,ids,desired,eligible,band,borrow,cost)
            error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))))
            assert error<1e-12
            ledger.to_csv(dest/(key+'.csv'),index=False);paths[key]=path;effective_targets[key]=effective
            checks.append(dict(id=key,max_independent_daily_error=error))
            rows.append(dict(id=key,group=group,strategy=strategy,band=band,cost=cost,borrow=borrow,**measured))
    frame=pd.DataFrame(rows);frame.to_csv(CONF/'METRICS.csv',index=False)
    pd.DataFrame(checks).to_csv(CONF/'ACCOUNT_CHECKS.csv',index=False)
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    np.savez_compressed(CONF/'PATHS.npz',dates=dates,**paths)
    np.savez_compressed(CONF/'TARGETS.npz',ids=ids,**effective_targets)
    def sr(group,strategy,cost=1,borrow=.03):
        return float(frame[(frame.group==group)&(frame.strategy==strategy)&(frame.cost==cost)&(frame.borrow==borrow)].iloc[0].sharpe)
    candidate_selection=plan['candidate']
    disjoint=[float(candidate_selection[g+'_sharpe']) for g in ('original_five','new_five','fresh_five')]+[sr('unused_five','candidate')]
    improvements={group:sr(group,'candidate')-sr(group,'baseline_issuer') for group in ('unused_five','expanded_twenty')}
    stress={group:sr(group,'candidate',2,.06)-sr(group,'baseline_issuer',2,.06) for group in improvements}
    passed=all(value>=.01 for value in improvements.values()) and min(stress.values())>=0 and max(disjoint)-min(disjoint)<=.20
    write(CONF/'RESULT.json',dict(passed=passed,base_issuer_deltas=improvements,stress_issuer_deltas=stress,
        four_disjoint_sharpes=disjoint,disjoint_gap=max(disjoint)-min(disjoint),
        limits='Confirmation of initialization robustness on repeatedly inspected history only; not market generalization'))
    verify();write(CONF/'COMPLETE.json',dict(accounts=len(rows),output_sha256={
        p.relative_to(CONF).as_posix():sha(p) for p in sorted(CONF.rglob('*')) if p.is_file()}))
    print(frame[['group','strategy','cost','borrow','sharpe','annualized_return','maximum_drawdown']].to_string(index=False),flush=True)
    print('Confirmation passed:',passed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','train','evaluate'],required=True)
    p.add_argument('--family',action='store_true',help='Freeze a qualifying candidate from the conditional family follow-up')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args.family)
    elif args.stage=='train':train()
    else:evaluate()


if __name__=='__main__':main()
