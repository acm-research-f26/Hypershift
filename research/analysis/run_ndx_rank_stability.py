"""Fixed validation-ranking and allocation experiment; existing checkpoints only.

Run through ``python -m hypershift run run_ndx_rank_stability --stage
freeze|run|report``. Failed attempts are retained; --name selects a fresh run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import run_ndx_drawdown_research as parent
from run_ndx_retention_replication import verify as verify_parent, sha, write
from ndx_rank_stability import (METHODS, ALLOCATIONS, percentile_cube,
    validation_diagnostics, validation_weights, blended_scores, allocate)
from ndx_drawdown_engine import daily_allocation, risk_inputs, exposure_series, market_inputs
from ndx_retention import hedged_targets
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq, simulate_signed
from audit_ndx_weak_market_short import value_replay
from research_engine import performance, ranking_metrics, target_weights, corr, ranks

ROOT = parent.ROOT
CONFIG = ROOT/'config/ndx_rank_stability_20261029.json'
PREVIOUS = ROOT/'runs/ndx_retention_replication_20261029'
DEFAULT = 'ndx_rank_stability_20261029'
FAMILIES = dict(think=ROOT/'runs/ndx_extended_hours_20261024/after_hours_premarket',
                dynamic=ROOT/'runs/ndx_reference_networks_20261024/dynamic_restricted')


def freeze(run):
    inherited = verify_parent(PREVIOUS)
    inputs = dict(inherited['input_sha256'])
    for path in [CONFIG, PREVIOUS/'PROTOCOL.json', PREVIOUS/'COMPLETE.json',
                 PREVIOUS/'METRICS.csv', PREVIOUS/'PATHS.npz', PREVIOUS/'TARGETS.npz']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    code = dict(inherited['source_sha256'])
    for name in ['research/core/ndx_rank_stability.py',
                 'research/analysis/run_ndx_rank_stability.py',
                 'tests/test_ndx_rank_stability.py']:
        code[name] = sha(ROOT/name)
    protocol = dict(config=json.loads(CONFIG.read_text()), source_sha256=code, input_sha256=inputs)
    run.mkdir(parents=True, exist_ok=False)
    write(run/'PROTOCOL.json', protocol)
    write(run/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(run/'PROTOCOL.json'), new_variant_metrics_absent=True,
        scope='Bounded historical development experiment, not untouched confirmation'))
    print('Frozen 24 recipes, 72 base accounts and 216 total cost/borrow accounts', flush=True)


def verify(run):
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    assert sha(run/'PROTOCOL.json') == json.loads((run/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[section].items():
            if sha(ROOT/name) != expected: raise ValueError('Frozen input changed: '+name)
    return protocol


def agreement(first, second, eligible):
    correlations = []; overlap = []
    for a, b, mask in zip(first, second, eligible):
        correlations.append(corr(ranks(a[mask]), ranks(b[mask])))
        wa = target_weights(a, mask, 5, 'equal')
        wb = target_weights(b, mask, 5, 'equal')
        overlap.append(float(np.minimum(wa, wb).sum()))
    return dict(rank_correlation=float(np.nanmean(correlations)),
                top5_weight_overlap=float(np.mean(overlap)))


def prepare(data, run):
    ids = np.concatenate([parent.fold(data, year)[2] for year in (2022, 2023, 2024, 2025)])
    blocks = {(method, group):[] for method in METHODS for group in parent.GROUPS}
    weights_rows = []; validation_rows = []; agreement_rows = []
    seeds = parent.GROUPS['ten_seed']
    for year in (2022, 2023, 2024, 2025):
        _, va, te = parent.fold(data, year)
        assert data['exit'][va].max() < data['origin'][te].min()
        assert len(va) >= 20
        calibration = {}; cubes = {}
        for family, folder in FAMILIES.items():
            vp = []; tp = []
            for seed in seeds:
                with np.load(folder/str(year)/f'{seed}_predictions.npz', allow_pickle=False) as saved:
                    assert saved['validation'].shape == data['y'][va].shape
                    assert saved['test'].shape == data['y'][te].shape
                    vp.append(saved['validation'].astype(float)); tp.append(saved['test'].astype(float))
            vr = percentile_cube(np.asarray(vp), data['mask'][va])
            tr = percentile_cube(np.asarray(tp), data['mask'][te])
            for group, group_seeds in parent.GROUPS.items():
                index = [seeds.index(seed) for seed in group_seeds]
                vc, tc = vr[index], tr[index]
                diagnostics, ic = validation_diagnostics(vc, data['y'][va], data['mask'][va], data['labelmask'][va])
                cubes[family, group] = (vc, tc)
                for method in METHODS:
                    weights = validation_weights(ic, method)
                    calibration[family, group, method] = weights
                    for j, seed in enumerate(group_seeds):
                        weights_rows.append(dict(year=year, family=family, group=group,
                            method=method, seed=seed, weight=float(weights[j]), **diagnostics[j],
                            validation_dates=len(va),
                            last_validation_outcome=str(data['panel'].dates[data['exit'][va].max()].date()),
                            first_test_decision=str(data['panel'].dates[data['origin'][te].min()].date())))
        validation_scores = {}
        for method in METHODS:
            for group in parent.GROUPS:
                args = [calibration['think', group, method], calibration['dynamic', group, method]]
                penalty = .25 if method == 'stable_consensus' else 0.
                val = blended_scores(cubes['think', group][0], cubes['dynamic', group][0], *args, penalty)
                test = blended_scores(cubes['think', group][1], cubes['dynamic', group][1], *args, penalty)
                blocks[method, group].append(test); validation_scores[method, group] = val
                summary, frame = ranking_metrics(val, data['y'][va], data['mask'][va], data['labelmask'][va], [5, 10])
                quarters = [float(part.mean()) for part in np.array_split(
                    frame.RankIC.fillna(0.).to_numpy(), 4)]
                validation_rows.append(dict(year=year, method=method, group=group, **summary,
                    quarter_rankic_sd=float(np.std(quarters, ddof=1)),
                    **{f'quarter{q+1}_rankic':value for q, value in enumerate(quarters)}))
            agreement_rows.append(dict(year=year, method=method, **agreement(
                validation_scores[method, 'original_five'], validation_scores[method, 'new_five'], data['mask'][va])))
        print(f'{year}: prior-year validation ranking weights and diagnostics complete', flush=True)
    pd.DataFrame(weights_rows).to_csv(run/'VALIDATION_WEIGHTS.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(run/'VALIDATION_METRICS.csv', index=False)
    pd.DataFrame(agreement_rows).to_csv(run/'VALIDATION_AGREEMENT.csv', index=False)
    scores = {key:np.concatenate(value) for key, value in blocks.items()}
    np.savez_compressed(run/'SCORES.npz', ids=ids,
                        **{method+'__'+group:value for (method, group), value in scores.items()})
    return ids, scores


def consistency(metrics):
    rows = []
    for (method, allocation, cap, cost, borrow), block in metrics.groupby(
            ['method', 'allocation', 'cap', 'cost', 'borrow'], sort=False):
        group = block.set_index('group')
        original, new, ten = [float(group.loc[name, 'sharpe']) for name in parent.GROUPS]
        gap = abs(original-new)
        rows.append(dict(method=method, allocation=allocation, cap=cap, cost=cost, borrow=borrow,
            original_sharpe=original, new_sharpe=new, ten_sharpe=ten,
            minimum_half_sharpe=min(original, new), half_sharpe_gap=gap,
            original_cagr=float(group.loc['original_five', 'annualized_return']),
            new_cagr=float(group.loc['new_five', 'annualized_return']),
            ten_cagr=float(group.loc['ten_seed', 'annualized_return']),
            primary_goal=min(original, new, ten) >= 1.70,
            secondary_goal=(original >= 1.50 and new >= 1.00 and ten >= 1.3813185726078634 and gap <= .50)))
    return pd.DataFrame(rows).sort_values(['cost', 'borrow', 'minimum_half_sharpe', 'ten_sharpe'],
                                         ascending=[True, True, False, False])


def replay(run):
    protocol = verify(run); cfg = protocol['config']
    if (run/'ledgers').exists(): raise FileExistsError('Attempt preserved; use a fresh --name')
    (run/'ledgers').mkdir()
    data = parent.load_data(); ids, scores = prepare(data, run)
    assert len(ids) == 201 and data['panel'].dates[-1].year == 2025
    dates = data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    assert len(dates) == 1003
    market = market_inputs(data); combined = append_qqq(data, etf_data(data, 'QQQ'))
    previous_paths = np.load(PREVIOUS/'PATHS.npz', allow_pickle=False)
    previous_targets = np.load(PREVIOUS/'TARGETS.npz', allow_pickle=False)
    np.testing.assert_array_equal(dates, previous_paths['dates'])
    np.testing.assert_array_equal(ids, previous_targets['ids'])
    metrics = []; checks = []; paths = {}; target_archive = {}; annual = []; overlap_rows = []
    baseline_count = 0
    for method in METHODS:
        for allocation in ALLOCATIONS:
            books = {}
            for group in parent.GROUPS:
                w = allocate(scores[method, group], data['mask'][ids], allocation)
                assert np.all(w[~data['mask'][ids]] == 0)
                np.testing.assert_allclose(w.sum(axis=1), 1., rtol=0, atol=1e-12)
                risk = risk_inputs(data, daily_allocation(data, ids, w))
                long = exposure_series(dict(vol='basket', window=20, target=.20), market, risk)[data['origin'][ids]]
                books[group] = (w, long, market['qqqvol20'][data['origin'][ids]])
            overlap_rows.append(dict(method=method, allocation=allocation,
                target_weight_overlap=float(np.minimum(books['original_five'][0], books['new_five'][0]).sum(axis=1).mean()),
                **agreement(scores[method, 'original_five'], scores[method, 'new_five'], data['mask'][ids])))
            for group, book in books.items():
                for cap in cfg['hedge_caps']:
                    desired = hedged_targets(*book, cap)
                    assert np.max(np.abs(desired).sum(axis=1)) <= 1+1e-12
                    for setting in cfg['cost_regimes']:
                        cost, borrow = setting['cost'], setting['borrow']
                        key = f'{method}__{allocation}__{group}__cap{int(cap*100)}__b{int(borrow*100):02d}__c{cost}'
                        measured, path, ledger = simulate_signed(combined, ids, desired, borrow, cost, detailed=True)
                        independent = value_replay(combined, ids, desired, borrow, cost)
                        error = float(np.max(np.abs(path-independent)))
                        assert error < 1e-12
                        baseline_error = 0.
                        if method == 'equal' and allocation == 'retain5':
                            previous = f'{group}__retention__cap{int(cap*100):02d}__b{int(borrow*100):02d}__c{cost}'
                            np.testing.assert_allclose(desired, previous_targets[previous], rtol=0, atol=1e-12)
                            baseline_error = float(np.max(np.abs(path-previous_paths[previous])))
                            assert baseline_error < 1e-10
                            baseline_count += 1
                        for field, value in performance(path).items():
                            np.testing.assert_allclose(value, measured[field], rtol=0, atol=1e-12)
                        assert ledger.date.tolist() == dates.tolist() and ledger.nav.min() > 0
                        assert ledger['long'].iloc[-1] == 0 and ledger['short'].iloc[-1] == 0
                        ledger.to_csv(run/'ledgers'/(key+'.csv'), index=False)
                        metrics.append(dict(id=key, method=method, allocation=allocation, group=group,
                            cap=cap, borrow=borrow, cost=cost, **measured))
                        checks.append(dict(id=key, max_independent_daily_error=error,
                                           max_baseline_daily_error=baseline_error))
                        paths[key] = path; target_archive[key] = desired
                        for year in (2022, 2023, 2024, 2025):
                            value = float(np.prod(1+path[pd.DatetimeIndex(dates).year == year])-1)
                            annual.append(dict(id=key, year=year, calendar_return=value))
            print(f'{method}/{allocation}: {len(metrics)}/216 accounts', flush=True)
            if method == 'equal' and allocation == 'retain5':
                assert baseline_count == 18
                print('All 18 matching retention references reproduced before new allocations', flush=True)
    assert len(metrics) == cfg['total_accounts'] == 216 and baseline_count == 18
    frame = pd.DataFrame(metrics)
    frame.to_csv(run/'METRICS.csv', index=False)
    pd.DataFrame(annual).to_csv(run/'CALENDAR.csv', index=False)
    pd.DataFrame(checks).to_csv(run/'ACCOUNT_CHECKS.csv', index=False)
    pd.DataFrame(overlap_rows).to_csv(run/'TEST_AGREEMENT.csv', index=False)
    table = consistency(frame); table.to_csv(run/'CONSISTENCY.csv', index=False)
    np.savez_compressed(run/'PATHS.npz', dates=dates, **paths)
    np.savez_compressed(run/'TARGETS.npz', ids=ids,
        security_ids=np.asarray(combined['panel'].tickers, dtype=str), **target_archive)
    base = table[(table.cost == 1)&(table.borrow == .03)]
    write(run/'AUDIT.json', dict(status='passed', accounts=216, matching_baselines=18,
        max_independent_daily_error=max(r['max_independent_daily_error'] for r in checks),
        max_baseline_daily_error=max(r['max_baseline_daily_error'] for r in checks),
        chronology='Each validation outcome strictly precedes first annual test decision',
        primary_goal_recipes=int(base.primary_goal.sum()), secondary_goal_recipes=int(base.secondary_goal.sum()),
        limitations='Historical postprocessing of MSE-selected checkpoints; reused validation and repeatedly inspected tests; no epoch-selection experiment, new training, independent vendor or live-execution validation'))
    verify(run)
    write(run/'COMPLETE.json', dict(status='complete', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}))
    print(base[['method','allocation','cap','original_sharpe','new_sharpe','ten_sharpe',
                'half_sharpe_gap','primary_goal','secondary_goal']].head(10).to_string(index=False), flush=True)


def report(run):
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = json.loads((run/'COMPLETE.json').read_text())
    for name, expected in complete['output_sha256'].items(): assert sha(run/name) == expected
    doc = ROOT/'docs'/run.name; doc.mkdir(parents=True, exist_ok=True)
    for path in run.glob('*.csv'): shutil.copyfile(path, doc/path.name)
    for name in ('PROTOCOL.json', 'FREEZE.json', 'AUDIT.json', 'COMPLETE.json'):
        shutil.copyfile(run/name, doc/name)
    table = pd.read_csv(run/'CONSISTENCY.csv')
    base = table[(table.cost == 1)&(table.borrow == .03)].sort_values(
        ['minimum_half_sharpe','ten_sharpe'], ascending=False)
    baseline = base[(base.method == 'equal')&(base.allocation == 'retain5')&(base.cap == .75)].iloc[0]
    best = base.iloc[0]
    audit = json.loads((run/'AUDIT.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = dict(equal='#6b7280', validation_ic='#1d77b4', validation_stable='#df851b', stable_consensus='#159777')
    for method in METHODS:
        block = base[base.method == method]
        axes[0].scatter(block.original_sharpe, block.new_sharpe, color=colors[method], label=method, s=55)
    low = min(0., float(base[['original_sharpe','new_sharpe']].min().min())-.1)
    high = max(1.9, float(base[['original_sharpe','new_sharpe']].max().max())+.1)
    axes[0].plot([low, high], [low, high], '--', color='#a1a1aa', linewidth=1)
    axes[0].axvline(1.7, color='#b45309', linestyle=':'); axes[0].axhline(1.7, color='#b45309', linestyle=':')
    axes[0].scatter([baseline.original_sharpe], [baseline.new_sharpe], marker='*', s=210, color='black', label='Previous retention leader')
    axes[0].set(xlim=(low,high), ylim=(low,high), xlabel='Original-five Sharpe', ylabel='New-five Sharpe',
                title='All 24 fixed recipes; dashed line = equal Sharpe')
    axes[0].legend(fontsize=8, loc='upper left'); axes[0].grid(alpha=.2)
    x = np.arange(3)
    axes[1].bar(x-.19, [baseline.original_sharpe,baseline.new_sharpe,baseline.ten_sharpe], .38, color='#6b7280', label='Previous retention / cap75')
    axes[1].bar(x+.19, [best.original_sharpe,best.new_sharpe,best.ten_sharpe], .38, color='#159777',
                label=f'Highest minimum-half SR:\n{best.method} / {best.allocation} / cap{int(best.cap*100)}')
    axes[1].axhline(1.7, color='#b45309', linestyle=':', label='Primary goal: all groups >=1.70')
    axes[1].set_xticks(x, ['Original five','New five','Ten seed']); axes[1].set_ylabel('Sharpe')
    axes[1].set_title('Consistency-ranked candidate versus baseline'); axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=.2)
    fig.suptitle('Validation ranking stability experiment — 2022–2025 development history\nBase trading costs and 3% borrow; trained checkpoints unchanged')
    fig.tight_layout(rect=(0,0,1,.90)); fig.savefig(doc/'seed_consistency.png', dpi=160); plt.close(fig)
    lines = ['# Validation ranking stability experiment — October 29, 2026', '',
        f'Completed the fixed **24-recipe / 72 base-account / 216 total-account** grid. '
        f'**{audit["primary_goal_recipes"]} recipes met the primary goal** (SR >=1.70 in all three ensembles); '
        f'**{audit["secondary_goal_recipes"]} met the secondary goal** (original >=1.50, new >=1.00, ten >=1.3813, half-gap <=0.50). '
        'These are selected historical development results, not prospective or statistical superiority findings.', '',
        '![Seed consistency comparison](seed_consistency.png)', '',
        '## All base-cost recipes, ordered by the weaker seed half', '',
        '| Method | Allocation | QQQ cap | Original SR | New SR | Ten SR | Half SR gap | Ten CAGR | Primary | Secondary |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in base.itertuples():
        lines.append(f'| {row.method} | {row.allocation} | {row.cap:.0%} | {row.original_sharpe:.3f} | {row.new_sharpe:.3f} | {row.ten_sharpe:.3f} | {row.half_sharpe_gap:.3f} | {row.ten_cagr:.2%} | {row.primary_goal} | {row.secondary_goal} |')
    selected = table[(table.method == best.method)&(table.allocation == best.allocation)&(table.cap == best.cap)]
    lines += ['', '## Costs for the consistency-ranked candidate', '',
        '| Cost multiplier | Annual borrow | Original SR | New SR | Ten SR |', '|---:|---:|---:|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.cost} | {row.borrow:.0%} | {row.original_sharpe:.3f} | {row.new_sharpe:.3f} | {row.ten_sharpe:.3f} |')
    lines += ['', '## What changed', '',
        'Four fixed scoring methods were tested: equal weights; validation mean RankIC weights; validation stable RankIC weights; and stable weights plus a cross-model disagreement penalty. '
        'Stable quality subtracts the standard deviation of four chronological validation-quarter mean RankICs. '
        'All quality weights are shrunk 75% toward equal weights; no seed is discarded and the two model families keep equal weight. '
        'The consensus method subtracts 0.25 times cross-model percentile-rank standard deviation from the average score. '
        'Each annual calibration uses only validation outcomes strictly before the first test decision. '
        'NDCG@5/@10, quarterly RankIC and cross-group ranking/selection agreement are retained.', '',
        'Each method crosses three allocations (five-security retention, ten-security retention, and smoothly weighted top ten) with 50%/75% continuous QQQ caps. '
        'The selected basket retains the existing 20-session/20% volatility target. '
        'Each of the 72 base accounts was also tested at 2x transaction costs, and at 2x costs with 6% borrow. '
        'Share classes count as separate securities. The fixed grid was frozen before the new account results and was not expanded after inspection.', '',
        '## Limits and verification', '',
        'These use saved final-checkpoint forecasts. The original models were selected by validation MSE; this experiment changes ensemble postprocessing and allocation, not checkpoint selection or training. '
        'The same validation folds already contributed to checkpoint selection, so the new validation weights are not independently validated. '
        'The 2022–2025 period has been repeatedly inspected. Ensemble halves share market history and the ten-seed group includes both halves. '
        'A smaller gap alone does not establish success if it results from reducing stronger-group returns. '
        'No new neural fits, 2026 returns, live trading or prospective activation were performed.', '',
        f'All 18 matching prior retention paths reproduced (max daily error {audit["max_baseline_daily_error"]:.3g}); '
        f'all216 signed-account paths matched independent dollar-position replay (max daily error {audit["max_independent_daily_error"]:.3g}). '
        'All targets obey current eligibility and gross budgets, annual calibration chronology is checked, and source/input/output hashes are locked. '
        'Synthetic tests cover masking, tie handling, seed-permutation invariance, shrinkage, consensus penalties and lack of future dependence.', '',
        '[All216accounts](METRICS.csv) · [Consistency and cost comparisons](CONSISTENCY.csv) · '
        '[Validation metrics](VALIDATION_METRICS.csv) · [Validation seed weights](VALIDATION_WEIGHTS.csv) · '
        '[Validation agreement](VALIDATION_AGREEMENT.csv) · [Test selection agreement](TEST_AGREEMENT.csv) · '
        '[Audit](AUDIT.json) · [Protocol](PROTOCOL.json)', '',
        '## Reproduce', '', '```powershell',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_rank_stability --stage freeze --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_rank_stability --stage run --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_rank_stability --stage report --name {run.name}_fresh',
        '```', '', f'Full scores, targets, daily paths and ledgers are retained in `runs/{run.name}/`. Existing outputs refuse overwrite.', '']
    (doc/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    print('Published '+str(doc.relative_to(ROOT)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze','run','report'], required=True)
    parser.add_argument('--name', default=DEFAULT); args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name): parser.error('Invalid run name')
    run = ROOT/'runs'/args.name
    if args.stage == 'freeze': freeze(run)
    elif args.stage == 'run': replay(run)
    else: report(run)


if __name__ == '__main__': main()
