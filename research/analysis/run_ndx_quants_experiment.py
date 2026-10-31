"""Fixed quants.md-inspired overlay experiment on all twenty existing seeds.

Run through the hypershift launcher. Never alters a completed source study.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import run_ndx_beta_robustness as parent
from ndx_drawdown_engine import market_inputs
from ndx_portfolio_robustness import average_targets, simulate_banded
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from audit_ndx_weak_market_short import value_replay
from ndx_quants_overlays import RECIPES, market_ranks, portfolio_targets
from research_engine import performance
from run_ndx_retention_replication import sha, write

ROOT = parent.ROOT
RUN = ROOT/'runs/ndx_quants_overlays_20261031'
DOC = ROOT/'docs/ndx_quants_overlays_20261031'
CORE = 'research/core/ndx_quants_overlays.py'
CODE = 'research/analysis/run_ndx_quants_experiment.py'
TEST = 'tests/test_ndx_quants_overlays.py'
SOURCES = {
    'library': 'https://quants.md/',
    'inverse_vol': 'https://github.com/QuantJourneyOrg/quantjourney-bt/blob/main/strategies/example_weights_05_monthly_inverse_vol.py',
    'trend': 'https://github.com/QuantJourneyOrg/quantjourney-bt/blob/main/strategies/example_weights_06_signal_change_defensive.py',
    'risk_parity': 'https://github.com/QuantJourneyOrg/quantjourney-bt/blob/main/strategies/example_weights_19_risk_parity_multiasset.py',
    'momentum_cost': 'https://github.com/QuantJourneyOrg/quantjourney-bt/blob/main/strategies/example_weights_11_quarterly_te_cost_gate.py',
    'reversal': 'https://github.com/QuantJourneyOrg/quantjourney-bt/blob/main/strategies/example_weights_16_cross_sectional_reversal.py',
}


def freeze():
    plan = parent.verify()
    upstream = json.loads((parent.RUN/'PROTOCOL.json').read_text())
    inputs = dict(upstream['input_sha256'])
    code = dict(upstream['source_sha256'])
    for name in (CORE, CODE, TEST):
        code[name] = sha(ROOT/name)
    for path in (parent.RUN/'PROTOCOL.json', parent.RUN/'METRICS.csv', parent.RUN/'PATHS.npz'):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    # Explicitly pin every reviewed market file read by load_data/ETF adapters.
    for path in (ROOT/'data/nasdaq_monthly_pit/assembled').glob('*.csv'):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    path = ROOT/'data/nasdaq_monthly_pit/assembled/manifest.json'
    inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    plan = dict(years=plan['years'], seed_groups=plan['seed_groups'], groups=plan['groups'],
                seeds=plan['seed_groups']['expanded_twenty'], recipes=RECIPES,
                cost_regimes=[dict(cost=1, borrow=.03), dict(cost=2, borrow=.06)],
                accounts=len(RECIPES)*27*2, new_fits=0, sources=SOURCES,
                baseline='Existing prior26 issuer5 cohort-target strategy, all20seeds equal within fixed5seed cohorts; 20%basketvol20 and75%unused-gross-budgetQQQshort. Must reproduce14source accounts.',
                adaptation='Site examples inspire rules; no vendor backtest is replicated. Same historical NDX universe, weekly cadence, next-close fill, costs, ADV, cash and borrow accounting. No new multi-asset/futures/FX data.',
                sizing='Inverse63session annualized volatility with10%floor. ERC uses63session covariance,50%diagonal shrinkage with10%diagonal volatility floor. Both capped at30%perissuer. Best class/tied class splits preserved. Capping can prevent exact equal risk.',
                signals='25%current eligible-stock percentile momentum126or252 or negative21session return +75%existingprior/neural score. Endpoint must be tradable; unsupported signal is neutral0.5. No skip-month or parameter optimization.',
                trend='QQQ below SMA200 at known signal close scales entire signed sleeve to50%or0. Uses QQQ in place of siteSPY and cash in place of defensive ETFs. Weekly checks; no daily stop.',
                drift='Existing actual-holdings no-trade-band engine,2or5percentagepoints. Bands applied after cohort target averaging. Not the site quarterly tracking-error/annual-turnover implementation.',
                aggregation='Four fixed five-seed portfolios then equal average their targets; ten/fifteen/twenty use first2/3/4cohorts. Single-seed diagnostics use50%prior+25%THINK+25%dynamic perseed, not independent market samples.',
                goal='Nonbaseline improves every7group baseSR by>=0.02, four-disjoint spread<=0.25, and no7group SR loss under2xfees/6%borrow. Separate ambitiousSR>1.5allgroups flag. No seed exclusion.',
                selection='If goal passes choose largest minimum7groupSR thenmean thenname; otherwise report descriptive strongest minimum7groupSR with failure explicit. Single-seed metrics do not select candidate.',
                uncertainty='For descriptive strongest candidate: paired circular20session blocks,2000draws, fixed20261031 RNG, pointwise95%Sharpe-difference intervals. Conditional on searched paths, not search-adjusted evidence.',
                limitations='2022-2025 market history and all20seeds already spent. No seed/market holdout. All14recipes frozen before new scores; all756accounts and failed recipes retained. No reliable prospective Sharpe claim. Do not promote automatically.')
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'PROTOCOL.json', dict(config=plan, source_sha256=code, input_sha256=inputs))
    write(RUN/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                protocol_sha256=sha(RUN/'PROTOCOL.json'), new_metrics_absent=True))
    for name in (CORE, CODE, TEST):
        dest = RUN/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
    print('Frozen', plan['accounts'], 'accounts;14recipes;20seeds;no new fitting', flush=True)


def verify():
    p = json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json') == json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in p[section].items():
            assert sha(ROOT/name) == expected, name
    return p['config']


def evaluate():
    plan = verify(); reserve = parent.reserve.verify()
    data = parent.prior.parent.parent.load_data()
    ids = np.concatenate([parent.prior.parent.parent.fold(data, y)[2] for y in plan['years']])
    neural = parent.reserve.scores(reserve, data, ids, 'original')
    _, rank, _ = parent.prior.priors(plan, data, ids, 26)
    market = market_inputs(data)
    combined = append_qqq(data, etf_data(data, 'QQQ'))
    eligible = np.column_stack((data['mask'][ids], np.ones(len(ids), bool)))
    features = dict(momentum126=market_ranks(data, ids, 126), momentum252=market_ranks(data, ids, 252),
                    reversal21=market_ranks(data, ids, 21, True))
    cohorts = [list(range(j, j+5)) for j in range(0, 20, 5)]
    samples = [(g, [plan['seeds'].index(s) for s in members]) for g, members in plan['seed_groups'].items()]
    samples += [(f'seed_{s}', [j]) for j, s in enumerate(plan['seeds'])]
    source = pd.read_csv(parent.RUN/'METRICS.csv')
    source_paths = np.load(parent.RUN/'PATHS.npz', allow_pickle=False)
    dates = data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    np.testing.assert_array_equal(dates, source_paths['dates'])
    (RUN/'ledgers').mkdir(exist_ok=False)
    rows, periods, checks = [], [], []
    paths, targets = {}, {}
    for recipe in plan['recipes']:
        cache = {}
        def target(indices):
            key = tuple(indices)
            if key not in cache:
                score = .5*rank + .25*neural[0, indices].mean(0) + .25*neural[1, indices].mean(0)
                if recipe.get('signal'):
                    score = .75*score + .25*features[recipe['signal']]
                cache[key] = portfolio_targets(data, ids, score, market, recipe)
            return cache[key]
        for group, members in samples:
            chunks = [members] if len(members) == 1 else [c for c in cohorts if set(c).issubset(members)]
            desired = average_targets([target(c) for c in chunks])
            for regime in plan['cost_regimes']:
                cost, borrow = regime['cost'], regime['borrow']
                key = f"{recipe['name']}__{group}__c{cost}__b{int(borrow*100):02d}"
                measured, path, ledger, effective, _ = simulate_banded(
                    combined, ids, desired, eligible, recipe.get('band', 0.), borrow, cost)
                error = float(np.max(np.abs(path-value_replay(combined, ids, effective, borrow, cost))))
                assert error < 1e-12
                parity = None
                if recipe['name'] == 'baseline' and group in plan['groups']:
                    match = source[(source.model == 'prior26_issuer') & (source.policy == 'original20')
                                   & (source.group == group) & (source.cost == cost) & (source.borrow == borrow)].iloc[0]
                    parity = float(np.max(np.abs(path-source_paths[match.id])))
                    assert parity < 1e-12
                    assert abs(measured['sharpe']-match.sharpe) < 1e-11
                ledger.to_csv(RUN/'ledgers'/(key+'.csv.gz'), index=False, compression='gzip')
                paths[key] = path; targets[key] = effective
                row = dict(id=key, recipe=recipe['name'], group=group,
                           sample_type='single_seed' if group.startswith('seed_') else 'ensemble',
                           cost=cost, borrow=borrow, max_independent_daily_error=error, **measured)
                rows.append(row)
                checks.append(dict(id=key, independent_error=error, baseline_parity_error=parity))
                years = pd.to_datetime(dates).year.to_numpy()
                for label, mask in [(str(y), years == y) for y in plan['years']] + [
                        ('2022-2023', years <= 2023), ('2024-2025', years >= 2024)]:
                    periods.append(dict(id=key, recipe=recipe['name'], group=group, cost=cost, borrow=borrow,
                                        period=label, **performance(path[mask])))
            print('Completed', recipe['name'], group, len(rows), '/', plan['accounts'], flush=True)
        pd.DataFrame(rows).to_csv(RUN/'PROGRESS_METRICS.csv', index=False)
    assert len(rows) == plan['accounts']
    source_paths.close()
    pd.DataFrame(rows).to_csv(RUN/'METRICS.csv', index=False)
    pd.DataFrame(periods).to_csv(RUN/'PERIOD_METRICS.csv', index=False)
    pd.DataFrame(checks).to_csv(RUN/'ACCOUNT_CHECKS.csv', index=False)
    np.savez_compressed(RUN/'PATHS.npz', dates=dates, **paths)
    np.savez_compressed(RUN/'TARGETS.npz', ids=ids, **targets)
    verify()
    write(RUN/'COMPLETE.json', dict(accounts=len(rows), max_independent_error=max(r['max_independent_daily_error'] for r in rows),
        baseline_parity_accounts=sum(r['baseline_parity_error'] is not None for r in checks),
        output_sha256={p.relative_to(RUN).as_posix():sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}))


def summarize():
    plan = verify(); lock = json.loads((RUN/'COMPLETE.json').read_text())
    for name, expected in lock['output_sha256'].items():
        assert sha(RUN/name) == expected, name
    frame = pd.read_csv(RUN/'METRICS.csv')
    base = frame[(frame.cost == 1) & (frame.sample_type == 'ensemble')]
    stress = frame[(frame.cost == 2) & (frame.sample_type == 'ensemble')]
    baseline = base[base.recipe == 'baseline'].set_index('group')
    stressed_baseline = stress[stress.recipe == 'baseline'].set_index('group')
    rows = []
    for name, block in base.groupby('recipe', sort=False):
        b = block.set_index('group').loc[plan['groups']]
        s = stress[stress.recipe == name].set_index('group').loc[plan['groups']]
        singleton = frame[(frame.recipe == name) & (frame.cost == 1) & (frame.sample_type == 'single_seed')]
        sr = b.sharpe.to_numpy(); delta = b.sharpe-baseline.loc[b.index].sharpe
        stress_delta = s.sharpe-stressed_baseline.loc[s.index].sharpe
        row = dict(recipe=name, min_all_sharpe=float(sr.min()), mean_all_sharpe=float(sr.mean()),
                   disjoint_gap=float(np.ptp(sr[:4])), min_delta=float(delta.min()),
                   groups_improved=int((delta >= .02).sum()), stress_min_delta=float(stress_delta.min()),
                   single_seed_min=float(singleton.sharpe.min()), single_seed_median=float(singleton.sharpe.median()),
                   single_seed_max=float(singleton.sharpe.max()), single_seed_std=float(singleton.sharpe.std(ddof=1)),
                   goal_passed=bool(name != 'baseline' and delta.min() >= .02 and np.ptp(sr[:4]) <= .25 and stress_delta.min() >= 0),
                   all_above_1_5=bool(sr.min() > 1.5))
        row.update({g:float(b.loc[g].sharpe) for g in plan['groups']})
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(['min_all_sharpe', 'mean_all_sharpe', 'recipe'], ascending=[False, False, True])
    summary.to_csv(RUN/'CONSISTENCY.csv', index=False)
    passing = summary[summary.goal_passed]
    best = summary[summary.recipe != 'baseline'].iloc[0]
    selected = None if passing.empty else passing.iloc[0].to_dict()
    write(RUN/'SELECTION.json', dict(passes=len(passing), selected=selected, descriptive_best=best.to_dict(),
                                   original_goal='Every7group>=+0.02SR,disjointgap<=0.25,stressnotworse',
                                   untouched_confirmation=False))
    # Conditional pointwise uncertainty for the descriptive strongest candidate.
    with np.load(RUN/'PATHS.npz', allow_pickle=False) as saved:
        rng = np.random.default_rng(20261031); intervals = []
        n = len(saved['dates']); offsets = np.arange(20)
        draws = ((rng.integers(0, n, (2000, (n+19)//20, 1)) + offsets) % n).reshape(2000, -1)[:, :n]
        rf = (1.03**(1/252)-1)
        for g in plan['groups']:
            a = saved[f"{best.recipe}__{g}__c1__b03"]; b = saved[f'baseline__{g}__c1__b03']
            aa, bb = a[draws], b[draws]
            ds = np.sqrt(252)*((aa.mean(1)-rf)/aa.std(1, ddof=1)-(bb.mean(1)-rf)/bb.std(1, ddof=1))
            intervals.append(dict(group=g, recipe=best.recipe, delta=float(performance(a)['sharpe']-performance(b)['sharpe']),
                                  lower95=float(np.quantile(ds, .025)), upper95=float(np.quantile(ds, .975))))
    pd.DataFrame(intervals).to_csv(RUN/'PAIRED_INTERVALS.csv', index=False)
    publish(plan, frame, summary, best, lock)
    print(summary.to_string(index=False), flush=True)


def publish(plan, frame, summary, best, lock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DOC.mkdir(parents=True, exist_ok=True)
    names = ['Original5', 'Second5', 'Third5', 'Fourth5', 'Combined10', 'Combined15', 'Combined20']
    plot = summary.set_index('recipe')
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(7)
    for name, color in [('baseline', '#708090'), (best.recipe, '#1565c0')]:
        ax.plot(x, plot.loc[name, plan['groups']].to_numpy(float), marker='o', label=name, color=color)
    ax.axhline(1.5, ls='--', color='#a33', alpha=.6); ax.set_xticks(x, names)
    ax.set_ylabel('Net Sharpe (3% risk-free hurdle)'); ax.set_title('Quants.md adaptations: fixed seed ensembles, 2022–2025 development')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(DOC/'cohort_sharpe.png', dpi=160); plt.close(fig)
    with np.load(RUN/'PATHS.npz', allow_pickle=False) as saved:
        dates = pd.to_datetime(saved['dates']); fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for name, color in [('baseline', '#708090'), (best.recipe, '#1565c0')]:
            a = saved[f'{name}__expanded_twenty__c1__b03']; wealth = np.r_[1., np.cumprod(1+a)]
            axes[0].plot(dates, 100*wealth[1:], label=name, color=color)
            axes[1].plot(dates, wealth[1:]/np.maximum.accumulate(wealth)[1:]-1, color=color)
        axes[0].set_ylabel('Net wealth ($100 start)'); axes[0].legend(); axes[1].set_ylabel('Drawdown')
        axes[0].set_title('Combined twenty: descriptive comparison after fixed search')
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(DOC/'combined_twenty.png', dpi=160); plt.close(fig)
    for name in ('PROTOCOL.json', 'FREEZE.json', 'METRICS.csv', 'CONSISTENCY.csv', 'PERIOD_METRICS.csv',
                 'ACCOUNT_CHECKS.csv', 'SELECTION.json', 'PAIRED_INTERVALS.csv'):
        shutil.copyfile(RUN/name, DOC/name)
    base = frame[(frame.cost == 1) & (frame.sample_type == 'ensemble')].set_index(['recipe', 'group'])
    lines = ['# Quants.md strategy adaptations — October 31, 2026', '',
        f"**{len(plan['recipes'])} frozen recipes, {len(frame)} accounts, twenty existing seeds, no new training.** "
        f"{int(summary.goal_passed.sum())} recipes pass the declared all-group improvement gate; "
        f"{int(summary.all_above_1_5.sum())} reach Sharpe >1.5 across every ensemble.", '',
        '![Seed ensemble Sharpe](cohort_sharpe.png)', '',
        'The reference is the current prior26 mixture with five issuers per fixed five-seed cohort and equal target averaging. '
        'Every result uses the same 1,003 sessions in 2022–2025, 3% annual risk-free hurdle, zero cash interest, '
        'modeled transaction costs, calendar-day borrow and next-close execution. Twenty singleton models are additional initialization diagnostics. '
        'All market history and these seeds were already inspected before this study; there is no untouched confirmation.', '',
        '## Descriptive strongest worst-ensemble result', '',
        f"`{best.recipe}` has minimum seven-ensemble Sharpe {best.min_all_sharpe:.3f} and four-disjoint-cohort spread {best.disjoint_gap:.3f}. "
        f"Declared gate passed: **{bool(best.goal_passed)}**. This label is chosen after the fixed search; it is not a prospective performance estimate.", '',
        '| Ensemble | Baseline SR | Candidate SR | Change | Candidate CAGR | Candidate max DD |',
        '|---|---:|---:|---:|---:|---:|']
    for g, label in zip(plan['groups'], names):
        a, b = base.loc[(best.recipe, g)], base.loc[('baseline', g)]
        lines.append(f'| {label} | {b.sharpe:.3f} | {a.sharpe:.3f} | {a.sharpe-b.sharpe:+.3f} | {a.annualized_return:.2%} | {a.maximum_drawdown:.2%} |')
    lines += ['', '![Net account paths](combined_twenty.png)', '', '## Every fixed recipe', '',
              '| Recipe | Minimum ensemble SR | Disjoint spread | Groups +0.02 | Worst stress SR change | Singleton min / median / max SR | Pass |',
              '|---|---:|---:|---:|---:|---|---|']
    for r in summary.itertuples():
        lines.append(f'| {r.recipe} | {r.min_all_sharpe:.3f} | {r.disjoint_gap:.3f} | {r.groups_improved}/7 | {r.stress_min_delta:+.3f} | {r.single_seed_min:.3f} / {r.single_seed_median:.3f} / {r.single_seed_max:.3f} | {r.goal_passed} |')
    lines += ['', '## What was adapted', '',
        '- Inverse volatility: 63-session estimates, 10% annualized volatility floor, 30% issuer cap, same selected five companies.',
        '- Capped ERC: 63-session covariance with 50% diagonal shrinkage, then 30% issuer cap. Capping can break exact equal risk contributions.',
        '- Momentum/reversal: 25% current percentile rank of 126/252-session return or negative 21-session return, plus 75% current baseline ranking. Unsupported endpoints receive neutral ranks.',
        '- Trend: QQQ below its 200-session SMA reduces the entire signed sleeve to half or cash at the next scheduled weekly decision. Cash earns zero; the hedge is reduced too.',
        '- Drift bands: retain actual holdings when changes stay within 2 or 5 percentage points, after averaging cohort targets. Eligibility and gross budget take precedence.',
        '- Four combinations were fixed with the individual rules before any new scores. No post-result expansion or seed filtering.', '',
        'These are adaptations to our NDX inputs, not exact replications of the site ETF strategies. '
        'The site uses multi-asset ETF baskets, monthly/quarterly cadences, SPY regimes and different controls. '
        'We did not test futures/FX/pairs or add unavailable defensive-asset data. The drift band is not a full tracking-error/annual-turnover gate.', '',
        '## Robustness and uncertainty', '',
        'The declared gate requires at least +0.02 Sharpe in all seven groups, disjoint spread <=0.25, '
        'and no group losing Sharpe at double trading fees plus 6% annual borrow. The separate >1.5 goal remains explicit. '
        'Singletons, overlapping ten/fifteen/twenty groups and shared market dates are not independent observations.', '',
        '[PERIOD_METRICS.csv](PERIOD_METRICS.csv) retains every year and both two-year halves, using continuous-account returns without resets. '
        '[PAIRED_INTERVALS.csv](PAIRED_INTERVALS.csv) gives conditional pointwise 95% Sharpe-change intervals '
        'from 2,000 paired circular 20-session block draws for the descriptive strongest candidate. '
        'These do not correct for this or the extensive earlier search. Seed consistency does not establish future market consistency.', '',
        f"All {lock['accounts']} accounts pass independent dollar-position replay; maximum daily-return discrepancy {lock['max_independent_error']:.3g}. "
        f"All {lock['baseline_parity_accounts']} reference paths reproduce to <1e-12 daily-return error. "
        'The focused tests cover risk-budget mathematics, issuer caps, baseline estimator equivalence, and future-data invariance. '
        'Compressed ledgers, effective targets, daily paths, code snapshots and input hashes remain in the dated run directory.', '',
        '## Sources and reproduction', '']
    for name, url in SOURCES.items():
        lines.append(f'- [{name}]({url})')
    lines += ['', '```powershell', r'.\.venv\Scripts\python.exe -m hypershift test test_ndx_quants_overlays -v',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_quants_experiment --stage freeze',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_quants_experiment --stage evaluate',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_quants_experiment --stage summarize', '```', '',
        'Freeze/evaluate refuse to overwrite completed work; summarize verifies every sealed output and can be rerun. '
        'To reproduce from scratch use a separate dated run path, retaining this frozen study.', '']
    (DOC/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    write(DOC/'MANIFEST.json', dict(run=RUN.relative_to(ROOT).as_posix(), protocol_sha256=sha(RUN/'PROTOCOL.json'),
        complete_sha256=sha(RUN/'COMPLETE.json'), output_sha256={p.name:sha(p) for p in sorted(DOC.iterdir()) if p.name != 'MANIFEST.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['freeze', 'evaluate', 'summarize'])
    args = parser.parse_args()
    dict(freeze=freeze, evaluate=evaluate, summarize=summarize)[args.stage]()
