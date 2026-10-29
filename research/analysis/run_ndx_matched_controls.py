"""Predeclared benchmark attribution study on sealed 2022-2025 observations.

Run with: python -m hypershift run run_ndx_matched_controls --stage freeze|run.
No network fits, 2026 observations, prospective activation or live orders.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_ndx_drawdown_research as parent
from ndx_drawdown_engine import allocations, daily_allocation, risk_inputs, exposure_series, simulate, market_inputs
from ndx_matched_controls import momentum_weights, etf_data, paired_inference
from pit_research_engine import etf_backtest

ROOT = parent.ROOT
RUN = ROOT/'runs/ndx_matched_controls_20261027'
DOC = ROOT/'docs/ndx_matched_controls_20261027'
POLICY = ROOT/'config/ndx_prospective_candidate_20261027.json'
RULES = dict(full=dict(cadence='weekly'),
             vol20=dict(cadence='weekly', vol='basket', window=20, target=.20),
             vol25trend=dict(cadence='weekly', vol='basket', window=20, target=.25, ma=200, off=.5))
NEURAL = [('equal5/full', 'equal5', 'full'),
          ('equal5/basketvol20_target20_weekly', 'equal5', 'vol20'),
          ('issuer5/basketvol20_target20_weekly', 'issuer5', 'vol20'),
          ('issuer5/ma200off50_basketvol20_target25_weekly', 'issuer5', 'vol25trend')]
PRIMARY = 'neural/ten_seed/issuer5/vol20'
SECONDARY = 'neural/ten_seed/issuer5/vol25trend'


def account_specs():
    specs = []
    for group in parent.GROUPS:
        for recipe, allocation, rule in NEURAL:
            specs.append(dict(id=f'neural/{group}/{allocation}/{rule}', kind='neural',
                              group=group, recipe=recipe, allocation=allocation, rule=rule))
            if rule != 'full':
                specs.append(dict(id=f'qqq_replay/{group}/{allocation}/{rule}', kind='qqq_replay',
                                  group=group, recipe=recipe, allocation=allocation, rule=rule))
    for rule in RULES:
        specs.append(dict(id='momentum/issuer5/'+rule, kind='momentum', rule=rule))
        specs.append(dict(id='qqq_own/'+rule, kind='qqq_own', rule=rule))
    for ticker in ['SPY', 'QQQ']:
        specs.append(dict(id='buy_hold/'+ticker, kind='buy_hold', ticker=ticker))
    assert len(specs) == 29
    return specs


def contrasts():
    return [(PRIMARY, b) for b in ['buy_hold/SPY', 'buy_hold/QQQ', 'qqq_own/vol20',
                                  'qqq_replay/ten_seed/issuer5/vol20', 'momentum/issuer5/vol20']] + [
        (SECONDARY, b) for b in ['qqq_own/vol25trend', 'qqq_replay/ten_seed/issuer5/vol25trend',
                                 'momentum/issuer5/vol25trend']]


def verify_sealed_inputs():
    plan = json.loads((parent.RUN/'PROTOCOL.json').read_text())
    complete = json.loads((parent.RUN/'COMPLETE.json').read_text())
    locks = dict(plan['source_sha256']) | dict(plan['input_sha256'])
    for relative, expected in locks.items():
        if parent.sha(ROOT/relative) != expected:
            raise ValueError('Parent source/input changed: '+relative)
    for relative, expected in complete['output_sha256'].items():
        if parent.sha(parent.RUN/relative) != expected:
            raise ValueError('Parent output changed: '+relative)
    audit = json.loads((parent.RUN/'AUDIT.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Parent audit has not passed')
    return locks | {str((parent.RUN/name).relative_to(ROOT)).replace('\\', '/'): parent.sha(parent.RUN/name)
                    for name in ['PROTOCOL.json', 'FREEZE.json', 'COMPLETE.json', 'AUDIT.json', 'paths.npz', 'metrics.csv', 'cost_stress.csv']}


def prospective_policy(source_hashes, input_hashes):
    """Freeze the candidate independently of the newly computed controls."""
    training = {str(p.relative_to(ROOT)).replace('\\', '/'): json.loads(p.read_text()) for p in
                [ROOT/'config/ndx_extended_hours_20261024.json', ROOT/'latest/config/ndx_reference_networks_20261024.json']}
    fields = dict(status='policy_preregistered_not_activated',
                  candidate='THINK+mixer AH+PM / restricted dynamic 50:50 ranks; ten seeds; five issuers; 20% basket volatility target',
                  candidate_chosen_from='Prior October 26 historical study, before this matched-control study; retain it regardless of new control results',
                  seeds=parent.GROUPS['ten_seed'], seed_groups=parent.GROUPS,
                  ranking='Within each family average eligible percentile ranks across all ten seeds, then average the two families 50:50. Select five issuers by best share-class score; average-rank/tie budgets as frozen issuer_weights.',
                  exposure=dict(rule=RULES['vol20'], cap=1., leverage=False, cash_return=0.,
                                volatility='Sample SD of prior 20 daily economic constituent returns under the current proposed basket weights, annualized sqrt(252), 5% vol floor; weekly only'),
                  universe='Reviewed as-published monthly NASDAQ-100 membership, original eligibility and liquidity rules, both networks using the same fixed supported stock-ID order',
                  unsupported_id_policy='Fail the entire prospective signal and pause collection before outcomes if an otherwise eligible member has an unsupported ID. Do not silently drop names, append random mixer weights, or score a changed universe. Any repair requires a new policy version before a new confirmation start.',
                  refit='Annual only, first eligible exchange session of each year. Train from max(2019-07-01, Jan 1 of Y-5) through Y-2; validate Y-1, purge outcomes crossing both boundaries. For Y=2026: train 2021-2024, validate 2025. Exact architecture/optimizer/checkpoint/imputation rules below. Refit all ten seeds in both families; no return-based seed/checkpoint selection.',
                  training_protocols=training,
                  execution='Rank every fifth eligible exchange session from the activated first entry. Features through prior close plus AH and current PM bars ending <=09:15 ET; freeze orders at 09:20 ET; execute same session regular close (13:00 ET for an early close). Size shares using preceding-close NAV/marks/20-day ADV. Retain 1% ADV caps, cancellations, sell-first cash scaling, corporate actions and modeled fees.',
                  fees='Per dollar traded: .0007 + .0003*sqrt(participation/.01); participation=abs(notional)/prior-close ADV. $1m reference account. Base/2x/4x costs; taxes excluded. Cash earns zero. No terminal liquidation before the fixed evaluation end.',
                  start='First eligible exchange session strictly after actual activation and after this freeze; no earlier than 2026-10-28. No backfilling missed forecasts or claiming already elapsed dates as prospective.',
                  primary_comparator='QQQ own 20-session/20% volatility target, same five-session calendar, close execution, costs and zero-interest cash',
                  secondary_comparators=['QQQ buy-and-hold', 'SPY buy-and-hold', 'QQQ following candidate target exposure', '12-minus-1 issuer momentum with 20% basket target', 'Frozen full-exposure ten-seed blend'],
                  primary_endpoint='Net annualized arithmetic active return versus primary comparator, two-sided paired 20-session circular-block 95% CI. Report paired Sharpe and CAGR differences separately, not as substitutes for the primary endpoint.',
                  evaluation='One formal read after 252 completed exchange sessions. No return-based early stopping, model replacement, parameter search or interim performance inspection. Operational input/timestamp/missing-signal/cost monitoring is permitted. Five/60-session block sensitivity and all seed diagnostics are secondary.',
                  interpretation='252 sessions may be underpowered; a positive point estimate alone is insufficient. Report failures, gaps and intervals. Prior search uncertainty remains disclosed.',
                  prediction_records='Append-only dated records of complete universe, raw family scores, seed ranks, blend, weights, exposure, order quantities, feature cutoffs, model hashes and pre-decision input hashes. Save and externally timestamp/hash-anchor before 09:20 ET and before execution/outcomes. Reject late or replacement records; missing record invalidates the scheduled signal.',
                  activation_requirements=['Fresh reviewed point-in-time membership, issuer identities, actions and daily/AH/PM feeds',
                                           'Audit feature publication latency, revisions and complete pre-decision timestamps',
                                           'Refit/verify all 20 current-year models with purged validation and supported stock IDs',
                                           'Replay and smoke-test future-date inference/execution plumbing before outcomes',
                                           'Separate append-only prediction store and independent pre-outcome timestamp anchor',
                                           'Record activation timestamp and verified exchange-session calendar before first signal'],
                  activation=dict(activated=False, timestamp_utc=None, first_entry=None, completed_sessions=0),
                  original_gate_sha256=parent.sha(ROOT/'latest/config/ndx_confirmation_gate_20261024.json'),
                  source_sha256=source_hashes, sealed_development_input_sha256=input_hashes,
                  activity='Policy preparation only. No 2026 market observations read, fresh models trained, prospective predictions collected, live orders, or scheduled jobs.')
    if POLICY.exists():
        existing = json.loads(POLICY.read_text())
        comparable = {k:v for k,v in existing.items() if k != 'frozen_timestamp_utc'}
        if comparable != fields:
            raise ValueError('Prospective policy changed; use a new version')
        return existing
    fields['frozen_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    parent.write(POLICY, fields)
    return fields


def freeze():
    locks = verify_sealed_inputs()
    sources = [ROOT/name for name in ['research/core/ndx_matched_controls.py',
               'research/analysis/run_ndx_matched_controls.py', 'research/evaluation/audit_ndx_matched_controls.py',
               'research/reporting/publish_ndx_matched_controls.py', 'tests/test_ndx_matched_controls.py']]
    sources += sorted((ROOT/'research/core').glob('*.py'))
    sources += sorted((ROOT/'research/models').glob('*.py'))
    sources += sorted((ROOT/'research/training').glob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): parent.sha(p) for p in sorted(set(sources))}
    policy = prospective_policy(source_hashes, locks)
    plan = dict(status='frozen_before_matched_control_metrics', created_local_date='2026-10-27',
                historical_period='2022-2025; already inspected exploratory development',
                primary_candidate=PRIMARY, secondary_candidate=SECONDARY, candidate_remains_fixed=True,
                prospective_policy_sha256=parent.sha(POLICY), prospective_frozen_timestamp_utc=policy['frozen_timestamp_utc'],
                specs=account_specs(), rules=RULES, neural_recipes=NEURAL, costs=[1,2,4], accounts=87,
                contrasts=contrasts(), primary_family_size=8,
                momentum='No tuning: close[t-21]/close[t-252]-1, using prior-close t. Five issuers, same issuer/tie method and historical eligibility; missing endpoint history excluded only from momentum and disclosed by date.',
                qqq_own='Same weekly rules and dates, but compute basket volatility on QQQ itself. Full-exposure weekly control included separately from buy-and-hold.',
                qqq_replay='Use the neural basket-derived target exposure verbatim on QQQ. Conditional attribution control; realized drift, beta and risk are not thereby equal.',
                execution='Unchanged frozen stock engine; one-asset QQQ adapter uses same lagged quantities, ADV caps, fees, cash scaling and regular-close fills. Buy-and-hold anchors use sealed parent etf_backtest, which invests available dollars at first fill close; disclose this sizing difference.',
                statistics='10,000 joint circular-block draws at 5/20/60 sessions; primary20. Eight-contrast max-absolute standardized bootstrap simultaneous intervals for annualized active means. SR/CAGR difference intervals are pointwise. Neither adjusts all prior research or refitting uncertainty.',
                stop='Exactly these 29 strategies x 3 costs. Publish all failures and controls. No tuning, new training, 2026 data or prospective activation.',
                source_sha256=source_hashes, input_sha256=locks)
    target = RUN/'PROTOCOL.json'
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError('Frozen matched-control protocol differs; use a new study')
    else:
        parent.write(target, plan)
        parent.write(RUN/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     protocol_sha256=parent.sha(target), results_absent=not (RUN/'metrics.csv').exists()))
    print('Frozen prospective policy and 29 matched strategies / 87 cost accounts', flush=True)
    return plan


def verify_plan():
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    for section in ['source_sha256', 'input_sha256']:
        for relative, expected in plan[section].items():
            if parent.sha(ROOT/relative) != expected:
                raise ValueError('Frozen source/input changed: '+relative)
    if parent.sha(POLICY) != plan['prospective_policy_sha256']:
        raise ValueError('Candidate policy changed')
    if parent.sha(RUN/'PROTOCOL.json') != json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    return plan


def prepared_accounts(data, ids, scores):
    market = market_inputs(data)
    prepared = {}
    for group in parent.GROUPS:
        for recipe, allocation, rule in NEURAL:
            key = f'neural/{group}/{allocation}/{rule}'
            weights = allocations(data, ids, scores[group], allocation)
            risk = risk_inputs(data, daily_allocation(data, ids, weights))
            prepared[key] = (data, weights, exposure_series(RULES[rule], market, risk))
    momentum, support = momentum_weights(data, ids)
    risk = risk_inputs(data, daily_allocation(data, ids, momentum))
    for rule in RULES:
        prepared['momentum/issuer5/'+rule] = (data, momentum, exposure_series(RULES[rule], market, risk))
    qqq = etf_data(data, 'QQQ'); one = np.ones((len(ids), 1))
    risk = risk_inputs(qqq, daily_allocation(qqq, ids, one))
    for rule in RULES:
        prepared['qqq_own/'+rule] = (qqq, one, exposure_series(RULES[rule], market, risk))
    for spec in account_specs():
        if spec['kind'] == 'qqq_replay':
            original = f'neural/{spec["group"]}/{spec["allocation"]}/{spec["rule"]}'
            prepared[spec['id']] = (qqq, one, prepared[original][2].copy())
    return prepared, support


def run():
    started = time.monotonic(); plan = verify_plan()
    if (RUN/'COMPLETE.json').exists():
        raise ValueError('Study already completed; audit/publish existing results instead')
    data = parent.load_data()
    ids = np.concatenate([parent.fold(data, year)[2] for year in [2022,2023,2024,2025]])
    scores = parent.saved_scores(data, ids)
    prepared, support = prepared_accounts(data, ids, scores)
    support.to_csv(RUN/'momentum_support.csv', index=False, lineterminator='\n')
    dates = data['panel'].dates[int(data['entry'][ids[0]]):]
    if len(dates) != 1003 or dates[-1] != pd.Timestamp('2025-12-31'):
        raise ValueError('Unexpected test dates')
    rows, curves, keys, ledgers, annual = [], [], [], [], []
    for spec in plan['specs']:
        for cost in plan['costs']:
            if spec['kind'] == 'buy_hold':
                metrics, frame = etf_backtest(data, spec['ticker'], cost)
                path = frame.net.to_numpy(); detail = None
            else:
                account, weights, exposure = prepared[spec['id']]
                metrics, path, detail = simulate(account, ids, weights, exposure, 'weekly', cost, detailed=True)
                metrics['average_equity_exposure'] = float(detail.equity_exposure.iloc[:-1].mean())
                metrics['average_target_exposure'] = float(detail.target_exposure.iloc[:-1].mean())
                ledgers.append(detail.assign(strategy=spec['id'], cost_multiplier=cost))
            key = spec['id']+f'/cost{cost}'
            rows.append(dict(id=key, strategy=spec['id'], kind=spec['kind'], group=spec.get('group','none'),
                             rule=spec.get('rule','buy_hold'), cost_multiplier=cost, **metrics))
            curves.append(path); keys.append(key)
            for year in [2022,2023,2024,2025]:
                annual.append(dict(strategy=spec['id'], cost_multiplier=cost, year=year,
                                   **parent.performance(path[dates.year==year])))
        print('Completed', spec['id'], flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RUN/'metrics.csv', index=False, lineterminator='\n')
    pd.DataFrame(annual).to_csv(RUN/'annual.csv', index=False, lineterminator='\n')
    pd.concat(ledgers, ignore_index=True).to_csv(RUN/'ledgers.csv', index=False, lineterminator='\n')
    np.savez_compressed(RUN/'paths.npz', ids=np.array(keys), dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str), returns=np.stack(curves, axis=1))
    base = {key.removesuffix('/cost1'): curve for key, curve in zip(keys, curves) if key.endswith('/cost1')}
    used = sorted({name for pair in contrasts() for name in pair})
    matrix = np.column_stack([base[name] for name in used])
    inference = pd.concat([paired_inference(matrix, used, contrasts(), block) for block in [5,20,60]], ignore_index=True)
    inference.to_csv(RUN/'uncertainty.csv', index=False, lineterminator='\n')
    pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), **base}).to_csv(RUN/'comparison_curves.csv', index=False, lineterminator='\n')
    # The entire target schedule, not realized leverage, is shared in replay.
    target_rows = []
    for spec in plan['specs']:
        if spec['kind'] not in ['buy_hold']:
            _, _, ex = prepared[spec['id']]
            for i in ids:
                target_rows.append(dict(strategy=spec['id'], origin=str(data['panel'].dates[data['origin'][i]].date()),
                                        exposure=float(ex[data['origin'][i]])))
    pd.DataFrame(target_rows).to_csv(RUN/'target_exposures.csv', index=False, lineterminator='\n')
    parent.write(RUN/'COMPLETE.json', dict(status='completed_exploratory_pending_audit',
                 strategies=29, accounts=87, observations=1003, seconds=time.monotonic()-started,
                 protocol_sha256=parent.sha(RUN/'PROTOCOL.json'),
                 output_sha256={p.name: parent.sha(p) for p in RUN.iterdir() if p.is_file() and p.name not in ['COMPLETE.json','AUDIT.json']}))
    print(metrics[metrics.cost_multiplier==1][['strategy','sharpe','annualized_return','maximum_drawdown']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['freeze','run'], required=True)
    globals()[parser.parse_args().stage]()
