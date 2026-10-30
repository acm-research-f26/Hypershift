"""Fixed follow-up: isolate model families and choose blends on prior validation.

Only run if the first portfolio/shared study does not meet its improvement goal.
No new training; all loss checkpoints were fixed before this follow-up.
"""
import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

import run_ndx_portfolio_robustness_v2 as study
from ndx_rank_retraining import percentile_predictions,rank_statistics,split_validation
from ndx_portfolio_robustness import average_targets,simulate_banded
from ndx_drawdown_engine import market_inputs
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha,write

ROOT=study.ROOT
NAME='ndx_family_robustness_20261030'
RUN=ROOT/'runs'/NAME
CASES=['original_think','dynamic','shared_think','validation_original','validation_shared']


def freeze():
    parent_plan=study.verify()
    selection=json.loads((study.RUN/'SELECTION.json').read_text())
    if selection['primary_passes']:
        raise ValueError('The predefined first-stage goal passed; confirm it before expanding research')
    protocol=json.loads((study.RUN/'PROTOCOL.json').read_text())
    code=dict(protocol['source_sha256']);inputs=dict(protocol['input_sha256'])
    code['research/analysis/run_ndx_family_robustness.py']=sha(ROOT/'research/analysis/run_ndx_family_robustness.py')
    for path in [study.RUN/'SELECTION.json',study.RUN/'FITS_COMPLETE.json',study.RUN/'CONSISTENCY.csv']:
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in (study.RUN/'fits').rglob('loss_predictions.npz'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    plan=dict(parent_plan)
    plan.update(study='Family isolation after first-stage goal failure',architectures=CASES,
        allocations=['retain5','issuer5'],aggregations=['cohort_targets'],bands=[0.,.01],
        accounts_per_arm=60,accounts=300,new_fits=0,
        validation_grid=[0.,.25,.5,.75,1.],
        validation_rule='Per-year THINK weight maximizes first-half prior-year meanRankIC-0.5quarterSD using all15pooledseedranks. Exact ties prefer0.5thenlowerweight. One weight peryear shared by every seed group. Second half diagnostics only. No test returns or seed filtering choose weights.',
        cases='Standalone originalTHINK,dynamic,sharedTHINK plus validation-only original/dynamic and shared/dynamic blends. Every case retained; all use cohort-target averaging. The independent dynamic case is not duplicated under two architecture labels.',
        confirmation='If a recipe qualifies, fix its source architecture, per-year family weights, allocation, aggregation and band before the same5unusedseeds. Do not re-estimate family weights from new-seed outcomes.',
        trigger='First540account fixed study failed strict improvement goal; family separation tests whether one predictor dilutes the other. This expansion and all earlier searches are disclosed.')
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',dict(config=plan,source_sha256=code,input_sha256=inputs))
    write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'),financial_accounts_absent=True))
    print('Frozen300family/blend accounts; validation-only mixture selection',flush=True)


def verify():
    p=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256','input_sha256'):
        for name,expected in p[section].items():assert sha(ROOT/name)==expected,name
    return p['config']


def validation_weights(plan,data,arm):
    weights={};rows=[];seeds=plan['seed_groups']['fifteen_seed']
    for year in plan['years']:
        _,validation,_=study.parent.fold(data,year);selection,audit=split_validation(data,validation)
        si=np.searchsorted(validation,selection);ai=np.searchsorted(validation,audit)
        families=[]
        for family in ('think','dynamic'):
            base=study.RUN if arm=='shared' and family=='think' else study.SOURCE
            values=[]
            for seed in seeds:
                with np.load(base/'fits'/family/str(year)/str(seed)/'loss_predictions.npz',allow_pickle=False) as p:
                    values.append(percentile_predictions(p['validation'],data['mask'][validation]))
            families.append(np.asarray(values).mean(0))
        candidates=[]
        for w in plan['validation_grid']:
            prediction=w*families[0]+(1-w)*families[1]
            selected=rank_statistics(prediction[si],data['y'][selection]*100,data['labelmask'][selection])
            diagnostic=rank_statistics(prediction[ai],data['y'][audit]*100,data['labelmask'][audit])
            row=dict(arm=arm,year=year,think_weight=w,selection_criterion=selected['criterion'],
                     selection_rankic=selected['rankic'],audit_rankic=diagnostic['rankic'])
            candidates.append(row);rows.append(row)
        best=min(candidates,key=lambda r:(-r['selection_criterion'],abs(r['think_weight']-.5),r['think_weight']))
        weights[year]=best['think_weight']
    return weights,rows


def evaluate():
    plan=verify();data=study.parent.load_data()
    ids=np.concatenate([study.parent.fold(data,y)[2] for y in plan['years']])
    old=study.load_scores(plan,data,ids,'original');shared=study.load_scores(plan,data,ids,'shared')
    years=np.concatenate([np.full(len(study.parent.fold(data,y)[2]),y) for y in plan['years']])
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'))
    eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    vo,ro=validation_weights(plan,data,'original');vs,rs=validation_weights(plan,data,'shared')
    pd.DataFrame(ro+rs).to_csv(RUN/'VALIDATION.csv',index=False)
    weights={
        'original_think':{y:1. for y in plan['years']},'dynamic':{y:0. for y in plan['years']},
        'shared_think':{y:1. for y in plan['years']},'validation_original':vo,'validation_shared':vs}
    write(RUN/'FAMILY_WEIGHTS.json',weights)
    for case in CASES:
        dest=RUN/case;dest.mkdir(exist_ok=False);(dest/'ledgers').mkdir()
        scores=shared if case in ('shared_think','validation_shared') else old
        weight=np.array([weights[case][int(y)] for y in years])[:,None]
        def pool(index):return weight*scores[0,index].mean(0)+(1-weight)*scores[1,index].mean(0)
        rows=[];paths={};targets={};checks=[]
        for allocation in plan['allocations']:
            for group,members in plan['seed_groups'].items():
                subset=[plan['seed_groups']['fifteen_seed'].index(s) for s in members]
                w=average_targets([study.signed_targets(data,ids,pool(subset[k:k+5]),allocation,market)
                                   for k in range(0,len(subset),5)])
                for band in plan['bands']:
                    for setting in plan['cost_regimes']:
                        cost,borrow=setting['cost'],setting['borrow']
                        key=study.key_for(case,allocation,'cohort_targets',group,band,cost,borrow)
                        measured,path,ledger,effective,_=simulate_banded(combined,ids,w,eligible,band,borrow,cost)
                        error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))))
                        assert error<1e-12 and np.max(np.abs(effective).sum(1))<=1+1e-12
                        ledger.to_csv(dest/'ledgers'/(key+'.csv'),index=False)
                        rows.append(dict(id=key,arm=case,allocation=allocation,aggregation='cohort_targets',
                                         group=group,band=band,cost=cost,borrow=borrow,**measured))
                        checks.append(dict(id=key,max_independent_daily_error=error));paths[key]=path;targets[key]=effective
                print(f'Family {case}/{allocation}/{group}: {len(rows)}/60',flush=True)
        assert len(rows)==60
        pd.DataFrame(rows).to_csv(dest/'METRICS.csv',index=False)
        pd.DataFrame(checks).to_csv(dest/'ACCOUNT_CHECKS.csv',index=False)
        np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths)
        np.savez_compressed(dest/'TARGETS.npz',ids=ids,**targets)
        write(dest/'COMPLETE.json',dict(accounts=60,output_sha256={p.relative_to(dest).as_posix():sha(p)
            for p in sorted(dest.rglob('*')) if p.is_file()}))
    verify()


def summarize():
    verify();previous=study.RUN
    try:
        study.RUN=RUN
        study.summarize()
    finally:study.RUN=previous
    selection=json.loads((RUN/'SELECTION.json').read_text())
    if selection['selected'] is not None:
        candidate=selection['selected'];case=candidate['arm']
        candidate['case']=case;candidate['arm']='shared' if case in ('shared_think','validation_shared') else 'original'
        candidate['family_weights_by_year']=json.loads((RUN/'FAMILY_WEIGHTS.json').read_text())[case]
        write(RUN/'SELECTION.json',selection)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','evaluate','summarize'],required=True)
    args=p.parse_args();{'freeze':freeze,'evaluate':evaluate,'summarize':summarize}[args.stage]()


if __name__=='__main__':main()
