"""Fixed training-rank-prior regularization, frozen before unused-seed outcomes."""
import argparse
from datetime import datetime,timezone
import json

import numpy as np
import pandas as pd

import run_ndx_tabular_alpha as tabular
import run_ndx_portfolio_robustness_v2 as parent
from ndx_rank_prior import training_rank_prior
from ndx_rank_retraining import percentile_predictions
from ndx_portfolio_robustness import average_targets,simulate_banded
from ndx_drawdown_engine import market_inputs
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha,write

ROOT=parent.ROOT
NAME='ndx_prior_robustness_20261030'
RUN=ROOT/'runs'/NAME
CASES=['prior13','prior26','prior52','prior_full']
HALVES=[13,26,52,0]


def freeze():
    old=parent.verify();tabular.verify()
    assert json.loads((tabular.RUN/'SELECTION.json').read_text())['primary_passes']==0
    reserve=parent.RUN/'confirmation'
    assert (reserve/'PROTOCOL.json').exists() and not (reserve/'METRICS.csv').exists()
    p=json.loads((tabular.RUN/'PROTOCOL.json').read_text())
    code=dict(p['source_sha256']);inputs=dict(p['input_sha256'])
    for name in ('research/analysis/run_ndx_prior_robustness.py','research/core/ndx_rank_prior.py','tests/test_ndx_rank_prior.py'):
        code[name]=sha(ROOT/name)
    for path in (tabular.RUN/'SELECTION.json',tabular.RUN/'ALL_METRICS.csv',reserve/'PROTOCOL.json'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    plan=dict(old)
    plan.update(study='Training-rank-prior shrinkage after weak shared-feature predictors',
        architectures=CASES,half_lives=HALVES,allocations=['retain5','issuer5'],
        aggregations=['rank_pool','cohort_targets'],bands=[0.],accounts_per_arm=60,accounts=264,new_fits=0,
        prior='Observed within-date training-return percentile ranks with exp2 decay half-lives13/26/52decisionweeks or fullhistory. Twenty neutral0.5pseudo-observations persecurity. No validation/testlabels enter prior. New/unsupported stocks are neutral.',
        blend='0.5eligible percentile rank of training prior +0.25mean(source THINKseed ranks)+0.25mean(source dynamicseed ranks), same predeclared seeds. This is deliberate stock-specific historical preference regularization; not unseen-stock generalization.',
        controls='24seed-free prior-only accounts, raw mean-rank score scale consistent with old training_mean_rank control. All6full-history controls must reproduce completed source metrics. Neural blends use percentile priors to align rank scales; retention scale differs from raw-score controls.',
        trigger='SharedTHINK,familyselection,ExtraTrees/Ridge all failed strict goal; training_mean_rank controlSR1.123 was much stronger than shared-feature controls. Small4prior fixed grid, no handpicked stocks.',
        selection='Same strict all-five improvement/stress gate; separately report allgroupsSR1.5. Choose maximum min-all-groupSR among passing recipes, no seed filtering.',
        confirmation='Freeze any chosen prior/portfolio rule before unused-five financial outcomes. Reuse the independently training next5neural seeds, and never choose a prior using those outcomes. Same market history remains spent.')
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',dict(config=plan,source_sha256=code,input_sha256=inputs))
    write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'),financial_accounts_absent=True,unused_seed_accounts_absent=True))
    print('Frozen240blended+24seed-free prior accounts before new-seed outcomes',flush=True)


def verify():
    p=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256','input_sha256'):
        for name,expected in p[section].items():assert sha(ROOT/name)==expected,name
    return p['config']


def priors(plan,data,ids,half):
    annual=[];records=[]
    for year in plan['years']:
        train,validation,test=parent.parent.fold(data,year)
        assert data['exit'][train].max()<data['origin'][validation].min()
        score,count=training_rank_prior(data,train,half)
        annual.append(np.broadcast_to(score,(len(test),len(score))).copy())
        records.extend(dict(year=year,security_id=ticker,score=float(score[j]),weighted_observations=float(count[j]))
                       for j,ticker in enumerate(data['panel'].tickers))
    raw=np.concatenate(annual)
    return raw,percentile_predictions(raw,data['mask'][ids]),records


def evaluate():
    plan=verify();data=parent.parent.load_data();ids=np.concatenate([parent.parent.fold(data,y)[2] for y in plan['years']])
    neural=parent.load_scores(plan,data,ids,'original');market=market_inputs(data)
    combined=append_qqq(data,etf_data(data,'QQQ'));eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    controls=[];prior_records=[];control=RUN/'controls';control.mkdir(exist_ok=False)
    source=pd.read_csv(parent.SOURCE/'METRICS.csv')
    for case,half in zip(CASES,HALVES):
        raw,rank,records=priors(plan,data,ids,half)
        prior_records.extend(dict(case=case,**row) for row in records)
        dest=RUN/case;dest.mkdir(exist_ok=False);(dest/'ledgers').mkdir();rows=[];paths={};targets={}
        def pool(index):return .5*rank+.25*neural[0,index].mean(0)+.25*neural[1,index].mean(0)
        for allocation in plan['allocations']:
            for group,members in plan['seed_groups'].items():
                index=[plan['seed_groups']['fifteen_seed'].index(s) for s in members]
                for aggregation in plan['aggregations']:
                    target=(parent.signed_targets(data,ids,pool(index),allocation,market) if aggregation=='rank_pool' else
                        average_targets([parent.signed_targets(data,ids,pool(index[k:k+5]),allocation,market) for k in range(0,len(index),5)]))
                    for setting in plan['cost_regimes']:
                        cost,borrow=setting['cost'],setting['borrow'];key=parent.key_for(case,allocation,aggregation,group,0.,cost,borrow)
                        measured,path,ledger,effective,_=simulate_banded(combined,ids,target,eligible,0.,borrow,cost)
                        error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))));assert error<1e-12
                        ledger.to_csv(dest/'ledgers'/(key+'.csv'),index=False)
                        rows.append(dict(id=key,arm=case,allocation=allocation,aggregation=aggregation,group=group,
                            band=0.,cost=cost,borrow=borrow,max_independent_daily_error=error,**measured))
                        paths[key]=path;targets[key]=effective
                print(f'Prior accounts {case}/{allocation}/{group}: {len(rows)}/60',flush=True)
            target=parent.signed_targets(data,ids,raw,allocation,market)
            for setting in plan['cost_regimes']:
                cost,borrow=setting['cost'],setting['borrow'];key=f'{case}__{allocation}__seed_free__c{cost}__b{int(borrow*100):02d}'
                measured,path,ledger,effective,_=simulate_banded(combined,ids,target,eligible,0.,borrow,cost)
                error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))));assert error<1e-12
                if half==0:
                    match=source[(source.method=='training_mean_rank')&(source.allocation==allocation)&(source.cap==.75)&(source.cost==cost)&(source.borrow==borrow)].iloc[0]
                    assert abs(measured['sharpe']-match.sharpe)<1e-11
                ledger.to_csv(control/(key+'.csv'),index=False)
                controls.append(dict(id=key,case=case,allocation=allocation,group='seed_free',cost=cost,borrow=borrow,
                    max_independent_daily_error=error,**measured))
        assert len(rows)==60
        pd.DataFrame(rows).to_csv(dest/'METRICS.csv',index=False)
        np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'TARGETS.npz',ids=ids,**targets)
        write(dest/'COMPLETE.json',dict(accounts=60,output_sha256={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
    assert len(controls)==24
    pd.DataFrame(controls).to_csv(RUN/'CONTROL_METRICS.csv',index=False)
    pd.DataFrame(prior_records).to_csv(RUN/'TRAINING_PRIORS.csv',index=False)
    verify()


def summarize():
    verify();previous=parent.RUN
    try:parent.RUN=RUN;parent.summarize()
    finally:parent.RUN=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','evaluate','summarize'],required=True)
    args=p.parse_args();{'freeze':freeze,'evaluate':evaluate,'summarize':summarize}[args.stage]()


if __name__=='__main__':main()
