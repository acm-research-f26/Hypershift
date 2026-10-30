"""Final fixed risk-control arm on every one of the twenty existing seeds."""
import argparse
from datetime import datetime,timezone
import json

import numpy as np
import pandas as pd

import run_ndx_prior_robustness as prior
import confirm_ndx_rank_prior as checked
import confirm_ndx_portfolio_robustness as reserve
from ndx_drawdown_engine import allocations,market_inputs
from ndx_beta_risk import risk_targets,trailing_beta
from ndx_portfolio_robustness import average_targets,simulate_banded
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha,write

ROOT=prior.ROOT
NAME='ndx_beta_robustness_20261030'
RUN=ROOT/'runs'/NAME


def freeze():
    prior_plan=prior.verify();confirmation=checked.verify()
    assert json.loads((checked.RUN/'RESULT.json').read_text())['passed']
    old=reserve.verify();p=json.loads((prior.RUN/'PROTOCOL.json').read_text())
    code=dict(p['source_sha256']);inputs=dict(p['input_sha256'])
    for name in ('research/analysis/run_ndx_beta_robustness.py','research/core/ndx_beta_risk.py','tests/test_ndx_beta_risk.py',
                 'research/evaluation/confirm_ndx_rank_prior.py'):
        code[name]=sha(ROOT/name)
    for path in (checked.RUN/'PROTOCOL.json',checked.RUN/'RESULT.json',checked.RUN/'METRICS.csv',
                 reserve.CONF/'FITS_COMPLETE.json',reserve.CONF/'METRICS.csv'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in (reserve.CONF/'base/fits').rglob('loss_predictions.npz'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    groups=dict(prior_plan['seed_groups']);groups['unused_five']=old['seeds'];groups['expanded_twenty']=old['expanded_seeds']
    order=['original_five','new_five','fresh_five','unused_five','ten_seed','fifteen_seed','expanded_twenty']
    plan=dict(study='Trailing beta hedge versus inherited unused-budget hedge',years=old['years'],
        seed_groups={g:groups[g] for g in order},groups=order,
        models=['neural_issuer','prior26_issuer'],policies=['original20','cash15','beta60','beta120'],
        cost_regimes=prior_plan['cost_regimes'],accounts=168,half_life=26,
        scope='All20seeds including the recently confirmed cohort now form development for this new risk arm. No seed omitted. No new fitting.',
        controls='original20 reproduces every existing account; cash15 reduces stock volatility target to15% with the old unused-budget QQQ hedge. Both beta rules retain20%stockvol target and reduce long budget to<=1/(1+positive beta).',
        beta='Current chosen stock-basket beta againstQQQ from preceding60or120close returns only. Use complete basket-return days, require half-window or20days, else beta1. Clip beta[0,3]. Long=min(20%/basketvol20,1/(1+beta)); short=min75%,beta*long,20%/QQQvol20. No leverage or short-proceeds reinvestment. Actual beta can drift and caps can prevent neutrality.',
        aggregation='Neural baseline pools seed rankings then selects5issuers. Prior candidate is the frozen26week prior/neural blend with5issuer portfolios averaged across consecutive5seedcohorts. Portfolio execution,fees,borrow,ADV,cash,fills unchanged.',
        goal='Base-costSR>1.5 in every one of seven groups, four-disjoint-cohort spread<=0.35, and worst stressed-costSR>=1.0. Only nonbaseline risk policies qualify. Report all168cases and every original20parity check.',
        selection='Among qualifying policies maximize minimum all7SR, break ties by meanSR thenmodel/policy lexical order. Freeze chosen rule before any additional seeds are fitted.',
        next_seed_reserve='If target passes, next5unused October20list new_seeds[15:20], no filtering; fixed candidate and reference on these seeds plus expanded25. This remains initialization confirmation, not unseen market data.')
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',dict(config=plan,source_sha256=code,input_sha256=inputs))
    write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'),financial_accounts_absent=True))
    print('Frozen168risk accounts;all20seeds;1.5target unchanged',flush=True)


def verify():
    p=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256','input_sha256'):
        for name,expected in p[section].items():assert sha(ROOT/name)==expected,name
    return p['config']


def targets(plan,data,ids,neural,rank,market):
    output={};details=[];all_seeds=plan['seed_groups']['expanded_twenty']
    for model in plan['models']:
        for group,members in plan['seed_groups'].items():
            index=[all_seeds.index(s) for s in members]
            chunks=[index] if model=='neural_issuer' else [index[j:j+5] for j in range(0,len(index),5)]
            for policy in plan['policies']:
                constituent=[]
                for k,sub in enumerate(chunks):
                    score=(.5*neural[0,sub].mean(0)+.5*neural[1,sub].mean(0) if model=='neural_issuer' else
                        .5*rank+.25*neural[0,sub].mean(0)+.25*neural[1,sub].mean(0))
                    stock=allocations(data,ids,score,'issuer5')
                    desired,detail=risk_targets(data,ids,stock,market,policy);constituent.append(desired)
                    beta=detail['beta']
                    if np.isnan(beta).any():beta,_=trailing_beta(data,ids,stock,market['qqq'],60)
                    for row,i in enumerate(ids):
                        details.append(dict(model=model,group=group,policy=policy,cohort=k,row=row,
                            origin=int(data['origin'][i]),trailing_beta=float(beta[row]),
                            long_target=float(detail['long_target'][row]),short_target=float(detail['short_target'][row]),
                            estimated_net_beta=float(beta[row]*detail['long_target'][row]-detail['short_target'][row])))
                output[model,group,policy]=average_targets(constituent)
    return output,details


def evaluate():
    plan=verify();reserve_plan=reserve.verify();data=prior.parent.parent.load_data()
    ids=np.concatenate([prior.parent.parent.fold(data,y)[2] for y in plan['years']])
    neural=reserve.scores(reserve_plan,data,ids,'original')
    _,rank,_=prior.priors(plan,data,ids,plan['half_life'])
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'))
    desired,details=targets(plan,data,ids,neural,rank,market)
    eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    baseline=pd.read_csv(prior.parent.SOURCE/'METRICS.csv')
    source_prior=pd.read_csv(prior.RUN/'METRICS.csv')
    new_baseline=pd.read_csv(reserve.CONF/'METRICS.csv');new_prior=pd.read_csv(checked.RUN/'METRICS.csv')
    (RUN/'ledgers').mkdir(exist_ok=False);rows=[];paths={};raw={};checks=[]
    for (model,group,policy),w in desired.items():
        raw[f'{model}__{group}__{policy}']=w
        for setting in plan['cost_regimes']:
            cost,borrow=setting['cost'],setting['borrow'];key=f'{model}__{group}__{policy}__c{cost}__b{int(borrow*100):02d}'
            measured,path,ledger,effective,_=simulate_banded(combined,ids,w,eligible,0.,borrow,cost)
            error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))));assert error<1e-12
            if policy=='original20':
                if group in ('unused_five','expanded_twenty'):
                    source=new_baseline if model=='neural_issuer' else new_prior
                    match=source[(source.group==group)&(source.cost==cost)&(source.borrow==borrow)]
                    if model=='neural_issuer':match=match[match.strategy=='baseline_issuer']
                else:
                    source=baseline if model=='neural_issuer' else source_prior
                    match=source[(source.group==group)&(source.cost==cost)&(source.borrow==borrow)&(source.allocation=='issuer5')]
                    match=(match[(match.method=='loss')&(match.cap==.75)] if model=='neural_issuer' else
                           match[(match.arm=='prior26')&(match.aggregation=='cohort_targets')])
                assert len(match)==1
                assert abs(measured['sharpe']-match.iloc[0].sharpe)<1e-11
            ledger.to_csv(RUN/'ledgers'/(key+'.csv'),index=False);paths[key]=path
            rows.append(dict(id=key,model=model,group=group,policy=policy,cost=cost,borrow=borrow,
                             max_independent_daily_error=error,**measured))
            checks.append(dict(id=key,max_independent_daily_error=error,source_parity=policy=='original20'))
        print('Risk accounts',model,group,policy,len(rows),'/168',flush=True)
    assert len(rows)==168
    pd.DataFrame(rows).to_csv(RUN/'METRICS.csv',index=False);pd.DataFrame(details).to_csv(RUN/'DECISIONS.csv',index=False)
    pd.DataFrame(checks).to_csv(RUN/'ACCOUNT_CHECKS.csv',index=False)
    np.savez_compressed(RUN/'PATHS.npz',dates=dates,**paths);np.savez_compressed(RUN/'TARGETS.npz',ids=ids,**raw)
    verify();write(RUN/'COMPLETE.json',dict(accounts=168,output_sha256={p.relative_to(RUN).as_posix():sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}))


def summarize():
    plan=verify();frame=pd.read_csv(RUN/'METRICS.csv');rows=[]
    for (model,policy),block in frame.groupby(['model','policy'],sort=False):
        base=block[(block.cost==1)&(block.borrow==.03)];stress=block[(block.cost==2)&(block.borrow==.06)]
        row=dict(model=model,policy=policy)
        values=[]
        for g in plan['groups']:
            sr=float(base[base.group==g].iloc[0].sharpe);row[g+'_sharpe']=sr;values.append(sr)
        row.update(min_all_sharpe=min(values),mean_all_sharpe=float(np.mean(values)),
                   disjoint_gap=max(values[:4])-min(values[:4]),stress_min=float(stress.sharpe.min()))
        row['goal_passed']=policy!='original20' and min(values)>1.5 and row['disjoint_gap']<=.35 and row['stress_min']>=1.
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values('min_all_sharpe',ascending=False);summary.to_csv(RUN/'CONSISTENCY.csv',index=False)
    passing=summary[summary.goal_passed].sort_values(['min_all_sharpe','mean_all_sharpe','model','policy'],ascending=[False,False,True,True])
    write(RUN/'SELECTION.json',dict(passes=len(passing),selected=None if passing.empty else passing.iloc[0].to_dict()))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','evaluate','summarize'],required=True)
    args=p.parse_args();{'freeze':freeze,'evaluate':evaluate,'summarize':summarize}[args.stage]()
