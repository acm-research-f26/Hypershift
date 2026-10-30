"""Fixed portfolio experiments and controlled shared-THINK retraining.

Use the hypershift launcher; completed source runs are read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

import run_ndx_rank_retraining as trainer
import run_ndx_drawdown_research as parent
from ndx_shared_think import SharedThink, initialize_shared
from ndx_rank_retraining import percentile_predictions
from ndx_portfolio_robustness import average_targets, simulate_banded
from ndx_retention import retention_targets, hedged_targets
from ndx_drawdown_engine import allocations, daily_allocation, risk_inputs, exposure_series, market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq, simulate_signed
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha, write
from training_device import configure_device, device_metadata

ROOT = parent.ROOT
NAME = 'ndx_portfolio_robustness_20261030_v2'
RUN = ROOT/'runs'/NAME
SOURCE = ROOT/'runs'/trainer.DEFAULT
CONFIG = ROOT/'config'/f'{NAME}.json'


def freeze():
    prior = trainer.verify(SOURCE)
    protocol = json.loads((SOURCE/'PROTOCOL.json').read_text())
    code = dict(protocol['source_sha256']); inputs = dict(protocol['input_sha256'])
    for path in [CONFIG, SOURCE/'COMPLETE.json', SOURCE/'FITS_COMPLETE.json',
                 SOURCE/'accounts/TARGETS.npz', SOURCE/'accounts/PATHS.npz', SOURCE/'METRICS.csv']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    # Pin every forecast selected by validation before any new financial scores.
    for path in (SOURCE/'fits').rglob('loss_predictions.npz'):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    names = ['research/analysis/run_ndx_portfolio_robustness_v2.py',
             'research/core/ndx_portfolio_robustness.py', 'research/models/ndx_shared_think.py',
             'research/evaluation/audit_ndx_weak_market_short.py',
             'tests/test_ndx_portfolio_robustness.py']
    for name in names: code[name] = sha(ROOT/name)
    plan = json.loads(CONFIG.read_text()); plan['seed_groups'] = prior['groups']; plan['years'] = prior['years']
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'PROTOCOL.json', dict(config=plan, source_sha256=code, input_sha256=inputs))
    write(RUN/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'), financial_accounts_absent=True, shared_fits_absent=True))
    for name in code:
        target = RUN/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(RUN/'ENVIRONMENT.json', device_metadata(configure_device('cuda')))
    print('Frozen60fits and540accounts; no new financial paths scored', flush=True)


def verify():
    protocol = json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json') == json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[section].items():
            assert sha(ROOT/name) == expected, 'Frozen source/input changed: '+name
    return protocol['config']


def train_year(year):
    plan = verify(); device = configure_device('cuda')
    prepared = trainer.prepare(year, device)
    write(RUN/f'FOLD_{year}.json', {name:prepared[name].tolist() for name in
                                 ('train','validation','selection','audit','test')})
    # Isolated process globals replace only the explicitly declared constructor
    # and initialization; the frozen trainer's optimizer/loop/selectors are used.
    trainer.ExtendedModel = SharedThink
    trainer.neutralize_stock_initialization = initialize_shared
    atomic = trainer.atomic_json
    progress = RUN/'worker_progress'; progress.mkdir(exist_ok=True)
    def log(path, value):
        if path == RUN/'PROGRESS.json': path = progress/f'think_{year}.json'
        atomic(path, value)
    trainer.atomic_json = log
    for seed in plan['seed_groups']['fifteen_seed']:
        trainer.fit(RUN, prepared, 'think', year, seed, plan, device)
    return year


def train():
    plan = verify()
    if (RUN/'FITS_COMPLETE.json').exists(): raise FileExistsError('Shared fits already locked')
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        jobs = [pool.submit(train_year, year) for year in reversed(plan['years'])]
        for future in as_completed(jobs): print('Shared year complete', future.result(), flush=True)
    records = sorted((RUN/'fits').rglob('RECORD.json')); assert len(records) == plan['fits']
    hashes = {}
    for record in records:
        for name, expected in json.loads(record.read_text())['hashes'].items():
            path = record.parent/name; assert sha(path) == expected
            hashes[path.relative_to(RUN).as_posix()] = expected
        hashes[record.relative_to(RUN).as_posix()] = sha(record)
    verify()
    write(RUN/'FITS_COMPLETE.json', dict(fits=len(records), hashes=hashes,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), shared_accounts_absent=not (RUN/'shared').exists()))
    print('All60sharedfits locked', flush=True)


def load_scores(plan, data, ids, arm):
    seeds = plan['seed_groups']['fifteen_seed']; families = []
    if arm == 'shared':
        lock = json.loads((RUN/'FITS_COMPLETE.json').read_text())
        assert lock['fits'] == 60
        for name, expected in lock['hashes'].items(): assert sha(RUN/name) == expected, name
    for family in ('think', 'dynamic'):
        base = RUN if arm == 'shared' and family == 'think' else SOURCE
        values = []
        for seed in seeds:
            annual = []
            for year in plan['years']:
                path = base/'fits'/family/str(year)/str(seed)/'loss_predictions.npz'
                with np.load(path, allow_pickle=False) as saved: annual.append(saved['test'].copy())
            values.append(percentile_predictions(np.concatenate(annual), data['mask'][ids]))
        families.append(np.asarray(values))
    return np.asarray(families)


def signed_targets(data, ids, score, allocation, market):
    w = (allocations(data, ids, score, 'issuer5') if allocation == 'issuer5' else
         retention_targets(score, data['mask'][ids], 5))
    assert np.max(np.abs(w[~data['mask'][ids]]), initial=0.) == 0
    risk = risk_inputs(data, daily_allocation(data, ids, w))
    long = exposure_series(dict(vol='basket',window=20,target=.20),market,risk)[data['origin'][ids]]
    return hedged_targets(w,long,market['qqqvol20'][data['origin'][ids]],.75)


def target_grid(plan, data, ids, scores):
    seeds = plan['seed_groups']['fifteen_seed']; market = market_inputs(data)
    output = {}
    for allocation in plan['allocations']:
        def pooled_score(index):
            return .5*scores[0,index].mean(0)+.5*scores[1,index].mean(0)
        single = [signed_targets(data,ids,pooled_score([j]),allocation,market) for j in range(len(seeds))]
        pooled = {}
        for group, members in plan['seed_groups'].items():
            subset = [seeds.index(seed) for seed in members]
            pooled[group] = signed_targets(data, ids, pooled_score(subset), allocation, market)
        for group, members in plan['seed_groups'].items():
            subset = [seeds.index(seed) for seed in members]
            cohorts = [subset[start:start+5] for start in range(0,len(subset),5)]
            cohort = [signed_targets(data,ids,pooled_score(index),allocation,market) for index in cohorts]
            output[allocation,'rank_pool',group] = pooled[group]
            output[allocation,'seed_targets',group] = average_targets(np.asarray(single)[subset])
            output[allocation,'cohort_targets',group] = average_targets(cohort)
    return output


def key_for(arm, allocation, aggregation, group, band, cost, borrow):
    return f'{arm}__{allocation}__{aggregation}__{group}__band{int(band*1000):03d}__c{cost}__b{int(borrow*100):02d}'


def accounts(arm):
    plan=verify(); dest=RUN/arm; dest.mkdir(exist_ok=True); (dest/'accounts').mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists(): raise FileExistsError('Arm already complete')
    data=parent.load_data(); ids=np.concatenate([parent.fold(data,y)[2] for y in plan['years']])
    scores=load_scores(plan,data,ids,arm); targets=target_grid(plan,data,ids,scores)
    combined=append_qqq(data,etf_data(data,'QQQ'))
    eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    assert len(dates)==1003
    old_metrics=pd.read_csv(SOURCE/'METRICS.csv').set_index('id')
    with np.load(SOURCE/'accounts/PATHS.npz',allow_pickle=False) as saved:
        old_paths={key:saved[key].copy() for key in saved.files}
    with np.load(SOURCE/'accounts/TARGETS.npz',allow_pickle=False) as saved:
        old_targets={key:saved[key].copy() for key in saved.files}
    rows=[]; checks=[]; paths={}; executed={}; raw={}
    for (allocation,aggregation,group), target in targets.items():
        raw[f'{allocation}__{aggregation}__{group}']=target
        for band in plan['bands']:
            for setting in plan['cost_regimes']:
                cost,borrow=setting['cost'],setting['borrow']
                key=key_for(arm,allocation,aggregation,group,band,cost,borrow)
                folder=dest/'accounts'/key; record=folder/'METRICS.json'
                if record.exists():
                    saved=json.loads(record.read_text())
                    for name,expected in saved['hashes'].items(): assert sha(folder/name)==expected
                    measured=saved['metrics']; check=saved['check']
                    with np.load(folder/'PATH.npz',allow_pickle=False) as p:
                        path=p['daily'].copy(); effective=p['effective'].copy()
                else:
                    folder.mkdir(parents=True,exist_ok=True)
                    measured,path,ledger,effective,decisions=simulate_banded(
                        combined,ids,target,eligible,band,borrow,cost)
                    independent=value_replay(combined,ids,effective,borrow,cost)
                    error=float(np.max(np.abs(path-independent))); assert error<1e-12
                    parity=None
                    if band==0:
                        _,original,_=simulate_signed(combined,ids,target,borrow,cost)
                        parity=float(np.max(np.abs(path-original))); assert parity<1e-12
                    if arm=='original' and aggregation=='rank_pool' and band==0:
                        oldkey=f'loss__{group}__{allocation}__cap75__c{cost}__b{int(borrow*100):02d}'
                        np.testing.assert_allclose(target,old_targets[oldkey],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(path,old_paths[oldkey],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(measured['sharpe'],old_metrics.loc[oldkey,'sharpe'],rtol=0,atol=1e-11)
                    assert ledger.date.tolist()==dates.tolist()
                    assert ledger['long'].iloc[-1]==0 and ledger['short'].iloc[-1]==0
                    assert np.max(np.abs(effective).sum(1))<=1+1e-12
                    assert np.max(np.abs(effective[~eligible]),initial=0.)==0
                    ledger.to_csv(folder/'LEDGER.csv',index=False)
                    np.savez_compressed(folder/'PATH.npz',daily=path,effective=effective,
                        decision_current=np.stack([d['current'] for d in decisions]),
                        decision_nav=np.asarray([d['nav'] for d in decisions]),
                        decision_day=np.asarray([d['day'] for d in decisions]))
                    check=dict(max_independent_daily_error=error,zero_band_original_engine_error=parity)
                    write(record,dict(metrics=measured,check=check,
                        hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='METRICS.json'}))
                rows.append(dict(id=key,arm=arm,allocation=allocation,aggregation=aggregation,
                                 group=group,band=band,cost=cost,borrow=borrow,**measured))
                checks.append(dict(id=key,**check));paths[key]=path;executed[key]=effective
        print(f'Accounts {arm}/{allocation}/{aggregation}/{group}: {len(rows)}/270',flush=True)
    assert len(rows)==plan['accounts_per_arm']
    pd.DataFrame(rows).to_csv(dest/'METRICS.csv',index=False)
    pd.DataFrame(checks).to_csv(dest/'ACCOUNT_CHECKS.csv',index=False)
    np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths)
    np.savez_compressed(dest/'TARGETS.npz',ids=ids,security_ids=np.array(combined['panel'].tickers,dtype=str),**executed)
    np.savez_compressed(dest/'RAW_TARGETS.npz',ids=ids,**raw)
    verify()
    write(dest/'COMPLETE.json',dict(accounts=len(rows),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
    print('Completed arm',arm,flush=True)


def summarize():
    plan=verify(); frames=[]
    for arm in plan['architectures']:
        dest=RUN/arm
        complete=json.loads((dest/'COMPLETE.json').read_text())
        for name,expected in complete['output_sha256'].items(): assert sha(dest/name)==expected,name
        frames.append(pd.read_csv(dest/'METRICS.csv'))
    frame=pd.concat(frames,ignore_index=True); old=pd.read_csv(SOURCE/'METRICS.csv')
    rows=[]
    for (arm,allocation,aggregation,band,cost,borrow),block in frame.groupby(
            ['arm','allocation','aggregation','band','cost','borrow'],sort=False):
        row=dict(arm=arm,allocation=allocation,aggregation=aggregation,band=band,cost=cost,borrow=borrow)
        deltas=[]; values=[]
        for group in plan['groups']:
            sr=float(block[block.group==group].iloc[0].sharpe);values.append(sr)
            baseline=old[(old.method=='loss')&(old.allocation=='issuer5')&(old.cap==.75)&
                         (old.cost==cost)&(old.borrow==borrow)&(old.group==group)].iloc[0]
            row[group+'_sharpe']=sr;row[group+'_issuer_delta']=sr-float(baseline.sharpe)
            deltas.append(sr-float(baseline.sharpe))
        disjoint=values[:3]
        row.update(min_all_sharpe=min(values),mean_all_sharpe=float(np.mean(values)),
                   min_disjoint_sharpe=min(disjoint),disjoint_gap=max(disjoint)-min(disjoint),
                   minimum_issuer_delta=min(deltas),all_above_1_5=min(values)>1.5)
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary['primary_goal']=False
    stress_base=old[(old.method=='loss')&(old.allocation=='issuer5')&(old.cap==.75)&
                    (old.cost==2)&(old.borrow==.06)&(old.kind=='ensemble')].sharpe.min()
    for index,row in summary[(summary.cost==1)&(summary.borrow==.03)].iterrows():
        stress=summary[(summary.arm==row.arm)&(summary.allocation==row.allocation)&
                       (summary.aggregation==row.aggregation)&(summary.band==row.band)&
                       (summary.cost==2)&(summary.borrow==.06)].iloc[0]
        passed=(row.minimum_issuer_delta>=.01 and row.min_all_sharpe>1.10 and
                row.disjoint_gap<=.15 and stress.min_all_sharpe>=stress_base)
        summary.loc[index,'primary_goal']=passed
    frame.to_csv(RUN/'METRICS.csv',index=False);summary.to_csv(RUN/'CONSISTENCY.csv',index=False)
    base=summary[(summary.cost==1)&(summary.borrow==.03)].sort_values('min_all_sharpe',ascending=False)
    passing=base[base.primary_goal].sort_values(['min_all_sharpe','band','arm','allocation','aggregation'],
                                             ascending=[False,True,True,True,True])
    selected=None if passing.empty else passing.iloc[0].to_dict()
    write(RUN/'SELECTION.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected,primary_passes=int(base.primary_goal.sum()),
        all_groups_above_1_5=int(base.all_above_1_5.sum()),
        selection_scope='Outcome-selected historical development candidate; confirmation pending'))
    print(base[['arm','allocation','aggregation','band','original_five_sharpe','new_five_sharpe',
                'fresh_five_sharpe','ten_seed_sharpe','fifteen_seed_sharpe','disjoint_gap','primary_goal']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['freeze','train','accounts','summarize'],required=True)
    parser.add_argument('--arm',choices=['original','shared'],default='original')
    args=parser.parse_args()
    if args.stage=='freeze': freeze()
    elif args.stage=='train': train()
    elif args.stage=='accounts': accounts(args.arm)
    else: summarize()


if __name__=='__main__': main()
