"""Finite, frozen weak-market QQQ short study; no new fitting or 2026 data.

Run via python -m hypershift run run_ndx_weak_market_short --stage freeze|run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as parent
from run_ndx_matched_controls import verify_sealed_inputs, RULES
from ndx_drawdown_engine import allocations, daily_allocation, market_inputs, risk_inputs, exposure_series
from ndx_matched_controls import etf_data, paired_inference
from ndx_weak_market_short import weak_signals, targets, append_qqq, simulate_signed
from research_engine import performance

ROOT = parent.ROOT
RUN = ROOT/'runs/ndx_weak_market_short_20261027/batch2'
DOC = ROOT/'docs/ndx_weak_market_short_20261027'
MODES = ['cash','switch25','switch50','switch100','hedge25','hedge50']
GATES = ['ma200','ma100','ma200_mom63']


def recipes():
    configs = [dict(id='full/baseline', base='full', gate='none', mode='baseline')]
    for base in ['vol20','vol25trend']:
        configs.append(dict(id=base+'/baseline', base=base, gate='none', mode='baseline'))
        for gate in GATES:
            for mode in MODES:
                configs.append(dict(id=f'{base}/{gate}/{mode}', base=base, gate=gate, mode=mode))
    return configs


def freeze():
    locks = verify_sealed_inputs()
    sources = ['research/core/ndx_weak_market_short.py',
               'research/analysis/run_ndx_weak_market_short.py', 'tests/test_ndx_weak_market_short.py',
               'research/core/ndx_matched_controls.py', 'research/analysis/run_ndx_matched_controls.py']
    policy = 'config/ndx_prospective_candidate_20261027.json'
    locks[policy] = parent.sha(ROOT/policy)
    plan = dict(study='QQQ shorts during lagged weak-market signals',
                historical_period='2022-2025 repeatedly inspected exploratory development',
                frozen_before_new_metrics=True, recipes=recipes(), groups=parent.GROUPS,
                repair='Original attempt stopped on account80: cash-scaled covers left a tiny short with zero cash, then accrued $0.118 borrow against $0.054 cash. Preserve original protocol/sources. Same39 recipes; corrected account reserves 28days borrow on current short value before positive trades. No grid or score change; partial original account results were seen during diagnostics.',
                source_sha256={p:parent.sha(ROOT/p) for p in sources}, input_sha256=locks,
                gates={'ma200':'QQQ prior close below trailing 200-session mean',
                       'ma100':'QQQ prior close below trailing 100-session mean',
                       'ma200_mom63':'MA200 weakness AND prior close below 63 sessions earlier'},
                modes={'cash':'Exit entire stock sleeve while weak; hold zero-interest cash',
                       'switch25/50/100':'While weak replace stock sleeve with short QQQ at min(25/50/100% NAV,20%/QQQ20-session volatility)',
                       'hedge25/50':'Keep baseline longs; short QQQ at min(25/50% NAV,unused baseline budget,20%/QQQ20-session volatility)'},
                bases='Fixed five-issuer 20% basket-vol target (primary) and 25% target/half-exposure MA200 cap (secondary); unchanged full exposure control',
                execution='Parent five-session schedule; prior-close NAV/marks/ADV and risk; AH+PM ranking cutoff09:15/decision09:20; same-session regular-close fills. No same-close signal fills. 1% ADV caps, cancellations, signed corporate actions, cash scaling, final close liquidation.',
                costs=[1,2,4], primary_borrow_rate=.03, borrow_stress=[.01,.10],
                accounts='39 recipes x3groups x3costs=351, plus30short recipes x3groups x2borrow stresses at base costs=180; total531',
                funding='Gross target <=100% NAV. Short proceeds not reinvested, cash interest and short rebate zero. Calendar-day borrow on prior-close short value. Economic adjusted-price short embeds dividend obligation; no additional dividend debit. Gross positions drift between fills; no intraday margin enforcement.',
                missing_short_evidence='No historical locates, recalls, actual borrow fee series, short execution restrictions or broker margin replay. 1/3/10% are assumptions, not observed QQQ quotes.',
                inference='10000 joint circular-block draws, blocks5/20/60;54 ten-seed contrasts:36 conditional rules vs matching baseline plus18 switch-vs-cash contrasts. Simultaneous active-mean intervals adjust only this family; SR/CAGR CIs pointwise. No adjustment for earlier repeated history or refitting.',
                stop='One finite grid; all failures/seed groups/costs/borrow cases disclosed. No expansion from outcomes, new training, 2026 returns, trades, automation or prospective policy changes.')
    target = RUN/'PROTOCOL.json'
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError('Frozen study changed; create new study')
    else:
        parent.write(target,plan)
        parent.write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     protocol_sha256=parent.sha(target),results_absent=not (RUN/'metrics.csv').exists()))
    print('Frozen 39 recipes / 531 accounts before financial evaluation',flush=True)


def verify():
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    assert parent.sha(RUN/'PROTOCOL.json') == json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ['source_sha256','input_sha256']:
        for path,expected in plan[section].items():
            if parent.sha(ROOT/path) != expected:
                raise ValueError('Frozen input/source changed: '+path)
    return plan


def run():
    plan = verify()
    if (RUN/'COMPLETE.json').exists():
        raise ValueError('Completed study; refusing overwrite')
    data = parent.load_data()
    ids = np.concatenate([parent.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    if data['panel'].dates[-1].year != 2025:
        raise ValueError('Unexpected future data')
    scores = parent.saved_scores(data,ids)
    market = market_inputs(data); weak = weak_signals(market['qqq'])
    qqq = etf_data(data,'QQQ'); combined = append_qqq(data,qqq)
    dates = data['panel'].dates[int(data['entry'][ids[0]]):]
    origins = data['origin'][ids]
    prepared = {}
    regime = []
    for gate in GATES:
        for row,day in enumerate(origins):
            entry = int(data['entry'][ids[row]])
            end = min(entry+5,len(data['panel'].dates)-1)
            regime.append(dict(gate=gate,signal_date=str(data['panel'].dates[day].date()),
                               fill_date=str(data['panel'].dates[entry].date()),weak=bool(weak[gate][day]),
                               subsequent_qqq_return=float(qqq['marks'][end,0]/qqq['marks'][entry,0]-1)))
    pd.DataFrame(regime).to_csv(RUN/'regime_diagnostics.csv',index=False)
    for group in parent.GROUPS:
        w = allocations(data,ids,scores[group],'issuer5')
        risk = risk_inputs(data,daily_allocation(data,ids,w))
        for base in ['full','vol20','vol25trend']:
            e = exposure_series(RULES[base],market,risk)
            for spec in [s for s in plan['recipes'] if s['base']==base]:
                gate = weak[spec['gate']] if spec['gate']!='none' else np.zeros(len(e),bool)
                prepared[spec['id'],group] = targets(w,e,origins,gate,market['qqqvol20'],spec['mode'])
    old = pd.read_csv(parent.RUN/'cost_stress.csv')
    with np.load(parent.RUN/'paths.npz') as saved:
        old_paths = dict(zip(saved['ids'].astype(str),saved['returns'].T))
    baseline_map = {'full':'issuer5/full','vol20':'issuer5/basketvol20_target20_weekly',
                    'vol25trend':'issuer5/ma200off50_basketvol20_target25_weekly'}
    metrics, paths, target_rows = [], {}, []
    checks = []
    annual = []
    for j,spec in enumerate(plan['recipes']):
        for group in parent.GROUPS:
            w = prepared[spec['id'],group]
            target_rows.extend(dict(recipe=spec['id'],group=group,signal_date=str(data['panel'].dates[d].date()),
                                    fill_date=str(data['panel'].dates[d+1].date()),long_target=float(w[k,:-1].sum()),short_target=float(-w[k,-1]))
                               for k,d in enumerate(origins))
            settings = [(c,.03) for c in [1,2,4]]
            if spec['mode'] not in ['baseline','cash']:
                settings += [(1,b) for b in [.01,.10]]
            for cost,borrow in settings:
                key = f"{spec['id']}/{group}/c{cost}/b{borrow}"
                m,path,ledger = simulate_signed(combined,ids,w,borrow,cost,
                    detailed=(group=='ten_seed' and spec['gate'] in ['none','ma200'] and cost==1 and borrow==.03))
                metrics.append(dict(id=key,recipe=spec['id'],base=spec['base'],gate=spec['gate'],mode=spec['mode'],
                                    group=group,cost_multiplier=cost,borrow_rate=borrow,**m))
                paths[key] = path
                for year in [2022,2023,2024,2025]:
                    a = performance(path[dates.year==year])
                    annual.append(dict(id=key,year=year,**a))
                if len(ledger):
                    ledger.to_csv(RUN/(spec['id'].replace('/','_')+'_ledger.csv'),index=False)
                if spec['mode']=='baseline':
                    parent_key = baseline_map[spec['base']]+'/'+group
                    if cost==1:
                        err = float(np.max(np.abs(path-old_paths[parent_key])))
                        assert err < 1e-12
                        checks.append(dict(id=key,daily_path_max_error=err))
                    if spec['base']!='full':
                        expected = old[(old.recipe==baseline_map[spec['base']])&(old.group==group)&(old.cost_multiplier==cost)].iloc[0]
                        assert abs(m['terminal_nav']-expected.terminal_nav) < 1e-5
        if j%6==0:
            print(f'Completed {j+1}/39 recipes; {len(metrics)} accounts',flush=True)
    frame = pd.DataFrame(metrics)
    assert len(frame)==531
    frame.to_csv(RUN/'metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    pd.DataFrame(target_rows).to_csv(RUN/'targets.csv',index=False)
    np.savez_compressed(RUN/'paths.npz',dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),
                        ids=np.array(list(paths)),returns=np.stack(list(paths.values()),axis=1))
    base = frame[(frame.cost_multiplier==1)&(frame.borrow_rate==.03)]
    summary = base.pivot(index='recipe',columns='group',values=['sharpe','annualized_return','maximum_drawdown','mean_short'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.sort_values('sharpe_ten_seed',ascending=False).to_csv(RUN/'summary.csv')
    names = [s['id'] for s in plan['recipes']]
    matrix = np.stack([paths[n+'/ten_seed/c1/b0.03'] for n in names],axis=1)
    contrasts = [(s['id'],s['base']+'/baseline') for s in plan['recipes'] if s['mode']!='baseline']
    contrasts += [(s['id'],s['base']+'/'+s['gate']+'/cash') for s in plan['recipes'] if s['mode'].startswith('switch')]
    assert len(contrasts)==54
    inference = []
    for block in [5,20,60]:
        print(f'Paired uncertainty: block {block}, 10000 draws, 54 contrasts',flush=True)
        inference.append(paired_inference(matrix,names,contrasts,block))
    pd.concat(inference).to_csv(RUN/'uncertainty.csv',index=False)
    # Independent arithmetic checks on every saved daily path and selected ledgers.
    for row in metrics:
        independent = performance(paths[row['id']])
        for field,value in independent.items():
            assert abs(value-row[field]) < 1e-10
        assert abs(1e6*np.prod(1+paths[row['id']])-row['terminal_nav']) < 1e-5
    for path in RUN.glob('*_ledger.csv'):
        l = pd.read_csv(path)
        np.testing.assert_allclose(l.nav,l.cash+l.long-l.short,atol=1e-7)
        previous = np.r_[1e6,l.nav.to_numpy()[:-1]]
        np.testing.assert_allclose(l.nav-previous,l.pnl-l.fee-l.borrow,atol=1e-7)
    verify()
    parent.write(RUN/'AUDIT.json',dict(status='passed',accounts=len(metrics),
                 tests='6 synthetic tests; all531 daily paths independently remeasured; selected ledgers reconcile NAV and daily P&L',
                 baseline_parity=checks, max_self_financing_error=max(r['max_accounting_error'] for r in metrics),
                 limitation='This is accounting/parent-parity verification, not an independent vendor-data or broker-execution audit.'))
    DOC.mkdir(parents=True,exist_ok=True)
    for name in ['PROTOCOL.json','FREEZE.json','metrics.csv','annual.csv','targets.csv','summary.csv',
                 'uncertainty.csv','regime_diagnostics.csv','AUDIT.json']:
        shutil.copyfile(RUN/name,DOC/name)
    parent.write(RUN/'COMPLETE.json',dict(status='complete',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 accounts=len(metrics),output_sha256={p.name:parent.sha(p) for p in RUN.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    shutil.copyfile(RUN/'COMPLETE.json',DOC/'COMPLETE.json')
    print(base[base.group=='ten_seed'].sort_values('sharpe',ascending=False)[['recipe','annualized_return','sharpe','maximum_drawdown']].head(12).to_string(index=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage',choices=['freeze','run'],required=True)
    args = parser.parse_args()
    freeze() if args.stage=='freeze' else run()
