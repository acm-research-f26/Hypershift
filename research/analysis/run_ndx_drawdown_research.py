"""Finite exploratory risk-control study on sealed 2022-2025 blend forecasts.

Run from the root: python -m hypershift run run_ndx_drawdown_research.
All variants are frozen before new financial metrics; no 2026 observations.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from pit_research_engine import load_data, fold
from ndx_drawdown_engine import (allocations, daily_allocation, market_inputs,
                                 risk_inputs, exposure_series, simulate)
from research_engine import performance

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'hypershift').is_dir())
RUN = ROOT/'runs/ndx_drawdown_20261026/batch1'
DOC = ROOT/'docs/ndx_drawdown_20261026'
PARENT = ROOT/'runs/ndx_reference_networks_20261024'
GROUPS = dict(original_five=[7,19,42,61,97],
              new_five=[1828801906,1868046758,239154507,287727767,981756941])
GROUPS['ten_seed'] = GROUPS['original_five']+GROUPS['new_five']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def recipes():
    risks = [('full',dict(cadence='weekly'))]
    risks += [(f'constant_{int(100*f)}',dict(constant=f,cadence='weekly')) for f in [.6,.75,.9]]
    for cadence in ['weekly','daily']:
        for length in [100,150,200,250]:
            for off in [0.,.25,.5]:
                risks.append((f'ma{length}_off{int(100*off)}_{cadence}',dict(ma=length,off=off,cadence=cadence)))
        risks.append((f'multitrend_{cadence}',dict(multitrend=True,cadence=cadence)))
        for window in [20,60]:
            for target in [.15,.20,.25,.30]:
                risks.append((f'basketvol{window}_target{int(100*target)}_{cadence}',dict(vol='basket',window=window,target=target,cadence=cadence)))
            for target in [.15,.20,.25]:
                risks.append((f'qqqvol{window}_target{int(100*target)}_{cadence}',dict(vol='qqq',window=window,target=target,cadence=cadence)))
        for target in [.20,.25,.30]:
            risks.append((f'ma200off50_basketvol20_target{int(100*target)}_{cadence}',dict(ma=200,off=.5,vol='basket',window=20,target=target,cadence=cadence)))
        for target in [.20,.25]:
            risks.append((f'ma200off25_basketvol60_target{int(100*target)}_{cadence}',dict(ma=200,off=.25,vol='basket',window=60,target=target,cadence=cadence)))
        for threshold in [.10,.15]:
            for off in [.25,.5]:
                risks.append((f'qqqdd{int(100*threshold)}_off{int(100*off)}_{cadence}',dict(market_dd=threshold,off=off,cadence=cadence)))
    result = []
    for allocation in ['equal5','issuer5','issuer_invvol5','issuer8_soft']:
        for name,risk in risks:
            result.append(dict(id=allocation+'/'+name,allocation=allocation,**risk))
    result.append(dict(id='equal10/full',allocation='equal10',cadence='weekly'))
    return result


def freeze(configs):
    sources = [ROOT/'research/core/ndx_drawdown_engine.py',
               ROOT/'research/analysis/run_ndx_drawdown_research.py',
               ROOT/'tests/test_ndx_drawdown_engine.py']
    inputs = [PARENT/'FINAL_PUBLICATION.json', ROOT/'runs/ndx_extended_hours_20261024/FINAL_PUBLICATION.json']
    for folder in [PARENT,ROOT/'runs/ndx_extended_hours_20261024']:
        seal = json.loads((folder/'FINAL_PUBLICATION.json').read_text())
        for name,expected in seal['artifact_hashes'].items():
            if name.endswith('_predictions.npz') or name in ['metrics.csv','curves.csv']:
                path = folder/name
                if sha(path) != expected:
                    raise ValueError('Sealed forecast/metric changed: '+str(path))
                inputs.append(path)
    manifest = json.loads((ROOT/'data/nasdaq_monthly_pit/assembled/manifest.json').read_text())
    for name,expected in manifest['hashes'].items():
        path = ROOT/manifest['root']/name
        if sha(path) != expected:
            raise ValueError('Reviewed market input changed: '+str(path))
        inputs.append(path)
    plan = dict(study='drawdown controls on frozen THINK+mixer AH+PM + dynamic rank blend',
                status='frozen_before_new_metrics', history='2022-2025 repeatedly inspected exploratory development',
                authorization='User requested creative testing for Sharpe on par with the historical leader and lower drawdown.',
                variants=len(configs), seed_groups=GROUPS, base_cost_accounts=len(configs)*3,
                recipes=configs, targets=dict(sharpe=.9194653069771791,max_drawdown=-.25,minimum_ten_seed_cagr=.239),
                screening='Require original-five and ten-seed Sharpe >=0.9194653, all three groups drawdown >=-25%, new-five Sharpe >= its 0.6330 baseline, and ten-seed CAGR >=23.9%. Report failures and all variants.',
                execution='Same 09:20 weekly AH+PM decision and regular-close fill; share sizing from known preceding close NAV/marks. Daily overlay uses only preceding close risk inputs and fills next regular close; no intraday stops.',
                exposure='Long-only, no leverage, idle cash earns zero as in parent. Weekly rankings fixed between dates. Daily risk orders trigger at >=10 percentage-point change from last target; weekly rebalance always updates.',
                timing='Rolling market/risk windows include only signal-close and earlier data; basket volatility uses current target basket applied to past constituent returns, not subsequent portfolio returns.',
                allocation='Issuer aliases: GOOG/GOOGL, FOX/FOXA, LBTYA/LBTYK. Best share class per issuer, tied best classes split issuer budget. Inverse volatility has 30% cap. Soft top-eight weights exp(4*rank-score), 25% cap.',
                stress='Evaluate at most 12 shortlisted variants plus baseline across all three seed groups at 2x/4x modeled trading costs.',
                uncertainty='Pointwise paired 20-session circular blocks, 10000 draws for selected variants; conditional on fitted paths and post-search selection, not adjusted for the full search.',
                stop='One finite initial grid. Any expansion requires a new disclosed batch, preserving this protocol and every result. No 2026 returns, deployment, trades, or automation.',
                source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
                input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(inputs))})
    target = RUN/'PROTOCOL.json'
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError('Existing frozen batch differs; create a new batch')
    else:
        write(target,plan)
        write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                   protocol_sha256=sha(target),financial_results_absent=not (RUN/'metrics.csv').exists()))
    return plan


def saved_scores(data,ids):
    output = {}
    mask = data['mask'][ids]
    for family,folder in [('think',ROOT/'runs/ndx_extended_hours_20261024/after_hours_premarket'),
                          ('dynamic',PARENT/'dynamic_restricted')]:
        seeds = {}
        for seed in GROUPS['ten_seed']:
            predictions = []
            for year in [2022,2023,2024,2025]:
                with np.load(folder/str(year)/f'{seed}_predictions.npz') as saved:
                    predictions.append(saved['test'].copy())
            predictions = np.concatenate(predictions)
            ranked = np.zeros_like(predictions,dtype=float)
            for row,eligible in enumerate(mask):
                ranked[row,eligible] = pd.Series(predictions[row,eligible]).rank(method='average').to_numpy()/eligible.sum()
            seeds[seed] = ranked
        output[family] = {group:np.mean([seeds[s] for s in group_seeds],axis=0) for group,group_seeds in GROUPS.items()}
    return {group:.5*output['think'][group]+.5*output['dynamic'][group] for group in GROUPS}


def evaluate(configs,data,ids,scores,market):
    metrics, paths, names = [], [], []
    prepared = {}
    for group in GROUPS:
        for allocation in sorted({c['allocation'] for c in configs}):
            weights = allocations(data,ids,scores[group],allocation)
            daily = daily_allocation(data,ids,weights)
            prepared[group,allocation] = (weights,risk_inputs(data,daily))
    old_curves = pd.read_csv(PARENT/'curves.csv')
    for j,config in enumerate(configs):
        for group in GROUPS:
            weights,risk = prepared[group,config['allocation']]
            exposure = exposure_series(config,market,risk)
            m,path,_ = simulate(data,ids,weights,exposure,config['cadence'])
            key = config['id']+'/'+group
            metrics.append(dict(id=key,recipe=config['id'],group=group,cost_multiplier=1,**m))
            paths.append(path);names.append(key)
            if config['id'] in ['equal5/full','equal10/full']:
                k = 5 if config['allocation']=='equal5' else 10
                parent_key = f'dynamic_restricted_blend/{group}/ensemble/{k}/1'
                parent = old_curves[old_curves.id == parent_key]
                np.testing.assert_allclose(path,parent.net,atol=1e-13,rtol=1e-13)
        if (j+1)%20 == 0 or j == 0:
            pd.DataFrame(metrics).to_csv(RUN/'metrics_partial.csv',index=False)
            print(f'Completed {j+1}/{len(configs)} variants, {(j+1)*3} accounts',flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(RUN/'metrics.csv',index=False)
    first = int(data['entry'][ids[0]])
    np.savez_compressed(RUN/'paths.npz',ids=np.array(names),dates=data['panel'].dates[first:].strftime('%Y-%m-%d').to_numpy(dtype=str),returns=np.stack(paths,axis=1))
    return frame,dict(zip(names,paths)),prepared


def summarize(frame, paths):
    base = frame[frame.recipe == 'equal5/full'].set_index('group')
    records = []
    for recipe,group in frame.groupby('recipe',sort=False):
        indexed = group.set_index('group')
        row = dict(recipe=recipe)
        for name in GROUPS:
            for field in ['sharpe','annualized_return','maximum_drawdown','turnover_annual']:
                row[name+'_'+field] = float(indexed.loc[name,field])
        row['worst_group_sharpe'] = float(group.sharpe.min())
        row['worst_group_drawdown'] = float(group.maximum_drawdown.min())
        row['all_group_sr_improve'] = bool(all(indexed.loc[g,'sharpe'] >= base.loc[g,'sharpe'] for g in GROUPS))
        row['target_met'] = bool(indexed.loc['original_five','sharpe'] >= .9194653069771791 and
                                indexed.loc['ten_seed','sharpe'] >= .9194653069771791 and
                                indexed.loc['new_five','sharpe'] >= base.loc['new_five','sharpe'] and
                                row['worst_group_drawdown'] >= -.25 and
                                indexed.loc['ten_seed','annualized_return'] >= .239)
        records.append(row)
    summary = pd.DataFrame(records).sort_values(['target_met','all_group_sr_improve','ten_seed_sharpe'],ascending=False)
    summary.to_csv(RUN/'summary.csv',index=False)
    return summary


def stress_and_detail(configs,summary,data,ids,market,prepared,paths):
    shortlist = list(summary.head(8).recipe)
    for recipe in summary[summary.worst_group_drawdown >= -.25].head(4).recipe:
        if recipe not in shortlist:shortlist.append(recipe)
    shortlist = shortlist[:12]
    if 'equal5/full' not in shortlist:shortlist.append('equal5/full')
    chosen = {c['id']:c for c in configs}
    rows, details, curves = [], [], []
    annual = []
    dates = data['panel'].dates[int(data['entry'][ids[0]]):]
    for recipe in shortlist:
        config = chosen[recipe]
        for group in GROUPS:
            weights,risk = prepared[group,config['allocation']]
            exposure = exposure_series(config,market,risk)
            for cost in [1,2,4]:
                m,path,detail = simulate(data,ids,weights,exposure,config['cadence'],cost,detailed=(cost==1))
                rows.append(dict(recipe=recipe,group=group,cost_multiplier=cost,**m))
                if cost == 1:
                    np.testing.assert_allclose(path,paths[recipe+'/'+group],atol=0,rtol=0)
                    details.append(detail.assign(recipe=recipe,group=group))
                    curves.append(pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'),net=path,recipe=recipe,group=group)))
                    for year in [2022,2023,2024,2025]:
                        annual.append(dict(recipe=recipe,group=group,year=year,**performance(path[dates.year==year])))
    pd.DataFrame(rows).to_csv(RUN/'cost_stress.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(RUN/'selected_ledgers.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(RUN/'selected_curves.csv',index=False)
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    return shortlist


def bootstrap(paths,shortlist):
    keys = [p+'/ten_seed' for p in shortlist]
    matrix = np.stack([paths[k] for k in keys],axis=1)
    n = len(matrix);base_index=keys.index('equal5/full/ten_seed')
    generator = np.random.default_rng(20261026)
    samples = []
    for _ in range(100):
        starts = generator.integers(0,n,(100,int(np.ceil(n/20))))
        indices = ((starts[:,:,None]+np.arange(20))%n).reshape(100,-1)[:,:n]
        sample = matrix[indices]
        sr = np.sqrt(252)*(sample.mean(1)-(1.03**(1/252)-1))/sample.std(1,ddof=1)
        samples.append(sr)
    sr = np.concatenate(samples)
    difference = sr-sr[:,base_index,None]
    rows = []
    for j,recipe in enumerate(shortlist):
        rows.append(dict(recipe=recipe,group='ten_seed',block_sessions=20,replicates=10000,
                         sr_low=float(np.quantile(sr[:,j],.025)),sr_high=float(np.quantile(sr[:,j],.975)),
                         paired_sr_difference_low=float(np.quantile(difference[:,j],.025)),
                         paired_sr_difference_high=float(np.quantile(difference[:,j],.975)),
                         scope='Pointwise conditional post-search diagnostic, not full-search adjusted'))
    pd.DataFrame(rows).to_csv(RUN/'uncertainty.csv',index=False)


def main():
    started = time.monotonic()
    configs = recipes()
    plan = freeze(configs)
    print(f'Frozen {len(configs)} variants / {len(configs)*3} base-cost accounts before new metrics',flush=True)
    data = load_data()
    ids = np.concatenate([fold(data,y)[2] for y in [2022,2023,2024,2025]])
    scores = saved_scores(data,ids)
    market = market_inputs(data)
    frame,paths,prepared = evaluate(configs,data,ids,scores,market)
    summary = summarize(frame,paths)
    print('Grid complete. Targets met:',int(summary.target_met.sum()),flush=True)
    shortlist = stress_and_detail(configs,summary,data,ids,market,prepared,paths)
    print('Shortlist cost stress and annual accounting complete; computing paired uncertainty',flush=True)
    bootstrap(paths,shortlist)
    write(RUN/'COMPLETE.json',dict(status='completed_exploratory_pending_independent_audit',
                                 variants=len(configs),base_accounts=len(frame),shortlist=shortlist,
                                 target_met=int(summary.target_met.sum()),seconds=time.monotonic()-started,
                                 protocol_sha256=sha(RUN/'PROTOCOL.json'),
                                 output_sha256={p.name:sha(p) for p in RUN.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    print(summary.head(12)[['recipe','target_met','original_five_sharpe','new_five_sharpe','ten_seed_sharpe',
                          'ten_seed_annualized_return','worst_group_drawdown']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
