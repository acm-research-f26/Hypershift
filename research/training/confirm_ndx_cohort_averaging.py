"""Confirm the original no-band target-averaging proposal on unused seeds.

This is explicitly separate from the failed stricter all-group improvement gate.
The rule is fixed, not a selection of a small trading-band improvement.
"""
import argparse
from datetime import datetime,timezone
import json

import pandas as pd
import confirm_ndx_portfolio_robustness as confirm
import run_ndx_portfolio_robustness_v2 as study
from run_ndx_retention_replication import sha,write

ROOT=study.ROOT
CONF=confirm.CONF
CODE=ROOT/'research/training/confirm_ndx_cohort_averaging.py'


def freeze():
    plan=study.verify();source=pd.read_csv(study.RUN/'original/METRICS.csv')
    if CONF.exists():raise FileExistsError('Never overwrite confirmation')
    selected=dict(arm='original',allocation='retain5',aggregation='cohort_targets',band=0.)
    for group in plan['groups']:
        row=source[(source.group==group)&(source.allocation=='retain5')&(source.aggregation=='cohort_targets')&
                   (source.band==0)&(source.cost==1)&(source.borrow==.03)].iloc[0]
        selected[group+'_sharpe']=float(row.sharpe)
    candidate=study.RUN/'COHORT_CONFIRMATION_CANDIDATE.json'
    write(candidate,dict(selected=selected,
        scope='Original suggestion: no-band average of five-seed cohort targets. The source strict primary goal failed; this is a separate pooling-effect confirmation, not a claim it passed.',
        source_selection_sha256=sha(study.RUN/'SELECTION.json')))
    expansion=json.loads((ROOT/'config/ndx_seed_expansion_20261020.json').read_text())
    seeds=expansion['new_seeds'][10:15]
    assert len(seeds)==5 and not set(seeds)&set(plan['seed_groups']['fifteen_seed'])
    CONF.mkdir()
    protocol=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),candidate=selected,seeds=seeds,
        expanded_seeds=plan['seed_groups']['fifteen_seed']+seeds,years=plan['years'],epochs=12,fits=40,
        strategies=['baseline_issuer','baseline_matched','candidate'],groups=['unused_five','expanded_twenty'],
        cost_regimes=plan['cost_regimes'],selection_relative=candidate.relative_to(ROOT).as_posix(),
        selection_sha256=sha(candidate),script_sha256=sha(ROOT/'research/training/confirm_ndx_portfolio_robustness.py'),
        wrapper_sha256=sha(CODE),study_protocol_sha256=sha(study.RUN/'PROTOCOL.json'),
        original_strict_goal='Not passed in the540accountsource study; confirmation below does not retroactively change that result',
        goal='New unused-five retention SR>=1.10; all four disjoint five-seed cohortSR>=1.10 and spread<=0.25; expanded20cohort-target averaging improvesSR over matched rank pooling by>=0.10 at base costs and is no worse under2xfees/6%borrow. Report every case and no reselection.',
        aspirational_goal='Separately report all four disjoint cohorts and expanded20SR>1.5; do not hide failure',
        mathematical_control='With exactly5seeds, cohort averaging and rank pooling use the same single cohort and must reproduce each other exactly. New-five checks initialization robustness; expanded20 checks aggregation.',
        limits='Same repeatedly inspected market years. This is unused-initialization confirmation, not a market holdout or prospectivealpha evidence.')
    write(CONF/'PROTOCOL.json',protocol)
    write(CONF/'FREEZE.json',dict(protocol_sha256=sha(CONF/'PROTOCOL.json'),fits_absent=True,accounts_absent=True))
    print('Frozen original no-band pooling confirmation;40fits;18accounts; seeds',seeds,flush=True)


def verify():
    plan=confirm.verify();assert sha(CODE)==plan['wrapper_sha256']
    return plan


def train():
    verify();confirm.train();verify()


def evaluate():
    plan=verify();confirm.evaluate()
    frame=pd.read_csv(CONF/'METRICS.csv')
    def row(group,strategy,cost=1,borrow=.03):
        return frame[(frame.group==group)&(frame.strategy==strategy)&(frame.cost==cost)&(frame.borrow==borrow)].iloc[0]
    for cost,borrow in ((1,.03),(2,.03),(2,.06)):
        a=row('unused_five','candidate',cost,borrow);b=row('unused_five','baseline_matched',cost,borrow)
        assert abs(a.sharpe-b.sharpe)<1e-12
        assert abs(a.terminal_nav-b.terminal_nav)<1e-6
    disjoint=[plan['candidate'][g+'_sharpe'] for g in ('original_five','new_five','fresh_five')]
    disjoint.append(float(row('unused_five','candidate').sharpe))
    delta=float(row('expanded_twenty','candidate').sharpe-row('expanded_twenty','baseline_matched').sharpe)
    stress=float(row('expanded_twenty','candidate',2,.06).sharpe-row('expanded_twenty','baseline_matched',2,.06).sharpe)
    passed=min(disjoint)>=1.10 and max(disjoint)-min(disjoint)<=.25 and delta>=.10 and stress>=0
    result=dict(passed=passed,goal='Separately predefined pooling/seed-consistency confirmation',
        four_disjoint_sharpes=disjoint,disjoint_gap=max(disjoint)-min(disjoint),
        expanded_twenty_candidate_sharpe=float(row('expanded_twenty','candidate').sharpe),
        expanded_twenty_rank_pool_sharpe=float(row('expanded_twenty','baseline_matched').sharpe),
        expanded_twenty_sharpe_improvement=delta,stress_sharpe_improvement=stress,
        all_above_1_5=min(disjoint+[float(row('expanded_twenty','candidate').sharpe)])>1.5,
        original_strict_primary_goal_passed=False,
        limits='Independent seeds, same spent market outcomes; no out-of-time confirmation')
    write(CONF/'POOLING_CONFIRMATION.json',result)
    write(CONF/'POOLING_PUBLICATION_LOCK.json',dict(wrapper_sha256=sha(CODE),
        source_completion_sha256=sha(CONF/'COMPLETE.json'),result_sha256=sha(CONF/'POOLING_CONFIRMATION.json')))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','train','evaluate'],required=True)
    args=p.parse_args();{'freeze':freeze,'train':train,'evaluate':evaluate}[args.stage]()


if __name__=='__main__':main()
