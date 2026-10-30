"""Replicate a pinned tom-shlom retention study using feat-rs saved forecasts.

Run from the repository root with ``python -m hypershift run
run_ndx_retention_replication --stage source|freeze|run|report``. A failed run is
preserved; use a new --name and freeze again rather than overwriting results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd

import run_ndx_drawdown_research as parent
from ndx_drawdown_engine import (allocations, daily_allocation, risk_inputs,
                                 exposure_series, market_inputs, simulate)
from ndx_matched_controls import etf_data
from ndx_retention import retention_targets, hedged_targets
from ndx_weak_market_short import append_qqq, simulate_signed
from audit_ndx_weak_market_short import value_replay
from research_engine import performance

ROOT = parent.ROOT
CONFIG = ROOT/'config/ndx_retention_replication_20261029.json'
SOURCE = ROOT/'outputs/retention_source_20261029'
SOURCE_DOC = SOURCE/'docs/historical_sharpe_20261029'
SOURCE_RESULTS = SOURCE/'results'
DEFAULT = 'ndx_retention_replication_20261029'
POLICY = dict(vol='basket', window=20, target=.20)
SOURCE_FILES = [
    'docs/historical_sharpe_20261029/README.md',
    'docs/historical_sharpe_20261029/SELECTION_PROTOCOL.md',
    'docs/historical_sharpe_20261029/HEDGE_PROTOCOL.md',
    'docs/historical_sharpe_20261029/selection_batch1/METRICS.csv',
    'docs/historical_sharpe_20261029/hedge_batch1/METRICS.csv',
    'docs/historical_sharpe_20261029/hedge_batch1/CALENDAR.csv',
    'docs/historical_sharpe_20261029/hedge_batch1/REPORT.md',
    'research/analysis/rishi_historical_selection.py',
    'research/analysis/rishi_historical_hedge.py',
    'data/historical_sharpe_exports_20261029/manifest.json',
    'data/historical_sharpe_exports_20261029/experiment_outputs.zip',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def restore_source():
    """Read Git objects and restore only data needed for this replication."""
    commit = json.loads(CONFIG.read_text())['source_commit']
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        payload = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        path = SOURCE/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError('Conflicting source file: '+str(path))
        else:
            path.write_bytes(payload)
    manifest = json.loads((SOURCE/SOURCE_FILES[-2]).read_text())
    archive = SOURCE/SOURCE_FILES[-1]
    assert sha(archive) == manifest['archive_sha256']
    selected = []
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            wanted = (
                name.startswith('results/HISTORICAL_SHARPE_HEDGE_20261029/')
                and name.endswith(('.npz', '.csv'))
            ) or name == 'results/HISTORICAL_SHARPE_SELECTION_20261029/TARGETS.npz' or (
                name.startswith('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_E_')
                and name.endswith('.npz'))
            if not wanted:
                continue
            path = (SOURCE/name).resolve()
            if not path.is_relative_to(SOURCE.resolve()):
                raise ValueError('Unsafe archive member')
            payload = bundle.read(name); entry = manifest['files'][name]
            assert len(payload) == entry['bytes']
            assert hashlib.sha256(payload).hexdigest() == entry['sha256']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError('Conflicting source member: '+name)
            else:
                path.write_bytes(payload)
            selected.append(name)
    receipt = dict(source_commit=commit, source_branch='tom-shlom',
                   archive_sha256=manifest['archive_sha256'], selected_members=selected,
                   source_files={name:sha(SOURCE/name) for name in SOURCE_FILES})
    receipt_path = SOURCE/'RECEIPT.json'
    if receipt_path.exists():
        assert json.loads(receipt_path.read_text()) == receipt
    else:
        write(receipt_path, receipt)
    print(f'Verified pinned source and {len(selected)} restored data files', flush=True)


def input_paths():
    receipt = json.loads((SOURCE/'RECEIPT.json').read_text())
    paths = [CONFIG, SOURCE/'RECEIPT.json']
    paths += [SOURCE/name for name in SOURCE_FILES+receipt['selected_members']]
    assembled = ROOT/'data/nasdaq_monthly_pit/assembled'
    manifest = json.loads((assembled/'manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        path = ROOT/manifest['root']/name
        if sha(path) != expected:
            raise ValueError('Reviewed input changed: '+str(path))
    paths += [assembled/name for name in [
        'manifest.json', 'economic_close.csv', 'dollar_volume.csv', 'tradable.csv',
        'membership.csv', 'etf_adj_close.csv', 'etf_dollar_volume.csv']]
    for run, family in [
            ('ndx_extended_hours_20261024', 'after_hours_premarket'),
            ('ndx_reference_networks_20261024', 'dynamic_restricted')]:
        base = ROOT/'runs'/run
        seal = json.loads((base/'FINAL_PUBLICATION.json').read_text())
        fits = json.loads((base/'FITS_COMPLETE_LOCK.json').read_text())
        paths += [base/'FINAL_PUBLICATION.json', base/'FITS_COMPLETE_LOCK.json']
        # Historical locks mix Windows absolute paths and relative backslashes.
        sealed_hashes = {}
        for key, value in {**fits['hashes'], **seal['artifact_hashes']}.items():
            normalized = key.replace('\\', '/')
            marker = '/runs/'+run+'/'
            if marker in normalized:
                normalized = normalized.split(marker, 1)[1]
            sealed_hashes[normalized] = value
        for year in (2022, 2023, 2024, 2025):
            for seed in parent.GROUPS['ten_seed']:
                name = f'{family}/{year}/{seed}_predictions.npz'
                path = base/name
                if sha(path) != sealed_hashes[name]:
                    raise ValueError('Sealed forecast changed: '+name)
                paths.append(path)
    paths += [ROOT/'runs/ndx_drawdown_20261026/batch1/paths.npz',
              ROOT/'runs/ndx_model_hedge_20261027/paths.npz',
              ROOT/'docs/ndx_model_hedge_20261027/metrics.csv']
    return sorted(set(paths))


def freeze(run):
    paths = input_paths()
    code = [ROOT/name for name in [
        'research/analysis/run_ndx_retention_replication.py',
        'research/core/ndx_retention.py', 'tests/test_ndx_retention.py',
        'research/analysis/run_ndx_drawdown_research.py',
        'research/core/pit_research_engine.py', 'research/core/research_engine.py',
        'research/core/ndx_drawdown_engine.py', 'research/core/ndx_matched_controls.py',
        'research/core/ndx_weak_market_short.py',
        'research/evaluation/audit_ndx_weak_market_short.py']]
    protocol = dict(config=json.loads(CONFIG.read_text()),
                    source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},
                    input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    run.mkdir(parents=True, exist_ok=False)
    write(run/'PROTOCOL.json', protocol)
    write(run/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha(run/'PROTOCOL.json'), results_absent=True,
          interpretation='Implementation frozen before this replay; source outcomes already known'))
    print('Frozen exact 96-account source grid; no additional parameter search', flush=True)


def verify(run):
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    assert sha(run/'PROTOCOL.json') == json.loads((run/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[section].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Frozen file changed: '+name)
    return protocol


def replay(run):
    protocol = verify(run)
    if (run/'ledgers').exists() or (run/'COMPLETE.json').exists():
        raise FileExistsError('Run already attempted; preserve it and use a new --name')
    (run/'ledgers').mkdir()
    data = parent.load_data()
    ids = np.concatenate([parent.fold(data, year)[2] for year in (2022, 2023, 2024, 2025)])
    dates = data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    assert len(ids) == 201 and len(dates) == 1003 and dates[-1] == '2025-12-31'
    scores = parent.saved_scores(data, ids)
    market = market_inputs(data); combined = append_qqq(data, etf_data(data, 'QQQ'))
    source_folder = SOURCE_RESULTS/'HISTORICAL_SHARPE_HEDGE_20261029'
    metrics_source = pd.read_csv(SOURCE_DOC/'hedge_batch1/METRICS.csv').set_index('id')
    annual_source = pd.read_csv(SOURCE_DOC/'hedge_batch1/CALENDAR.csv').set_index(['id', 'year'])
    source_paths = np.load(source_folder/'PATHS.npz', allow_pickle=False)
    source_targets = np.load(source_folder/'TARGETS.npz', allow_pickle=False)
    selection = np.load(SOURCE_RESULTS/'HISTORICAL_SHARPE_SELECTION_20261029/TARGETS.npz', allow_pickle=False)
    old_long = np.load(ROOT/'runs/ndx_drawdown_20261026/batch1/paths.npz', allow_pickle=False)
    old_signed = np.load(ROOT/'runs/ndx_model_hedge_20261027/paths.npz', allow_pickle=False)
    for saved in (selection, source_targets):
        np.testing.assert_array_equal(ids, saved['ids'])
    np.testing.assert_array_equal(selection['security_ids'], data['panel'].tickers)
    for saved in (source_paths, old_long, old_signed):
        np.testing.assert_array_equal(dates, saved['dates'])
    books = {}; parity = []; score_checks = []
    for group in parent.GROUPS:
        with np.load(SOURCE_RESULTS/f'RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_E_{group}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(ids, saved['ids'])
            error = float(np.max(np.abs(scores[group]-saved['scores'])))
            assert error < 1e-12
            score_checks.append(dict(group=group, max_score_error=error))
        equal = allocations(data, ids, scores[group], 'equal5')
        risk = risk_inputs(data, daily_allocation(data, ids, equal))
        for policy in ('full', 'vol20'):
            exposure = np.ones(len(data['panel'].dates)) if policy == 'full' else exposure_series(POLICY, market, risk)
            _, path, _ = simulate(data, ids, equal, exposure)
            key = f'equal5/{"full" if policy == "full" else "basketvol20_target20_weekly"}/{group}'
            column = np.flatnonzero(old_long['ids'] == key)
            assert len(column) == 1
            error = float(np.max(np.abs(path-old_long['returns'][:, column[0]])))
            assert error < 1e-10
            parity.append(dict(group=group, account=key, max_daily_error=error))
        for rule in ('issuer', 'retention'):
            weights = allocations(data, ids, scores[group], 'issuer5') if rule == 'issuer' else retention_targets(scores[group], data['mask'][ids])
            if rule == 'retention':
                np.testing.assert_allclose(weights, selection[f'{group}__k5__retention__vol20'], rtol=0, atol=1e-12)
            risk = risk_inputs(data, daily_allocation(data, ids, weights))
            long = exposure_series(POLICY, market, risk)[data['origin'][ids]]
            books[group, rule] = (weights, long, market['qqqvol20'][data['origin'][ids]])
        for cap, recipe in ((0., 'baseline'), (.5, 'always/hedge50')):
            desired = hedged_targets(*books[group, 'issuer'], cap)
            _, path, _ = simulate_signed(combined, ids, desired)
            key = f'{recipe}/{group}/c1/b0.03'
            column = np.flatnonzero(old_signed['ids'] == key)
            assert len(column) == 1
            error = float(np.max(np.abs(path-old_signed['returns'][:, column[0]])))
            assert error < 1e-10
            parity.append(dict(group=group, account=key, max_daily_error=error))
    pd.DataFrame(parity).to_csv(run/'BASELINE_PARITY.csv', index=False)
    print('All 12 existing daily-path baselines and all three score ensembles reproduced', flush=True)
    paths = {}; targets = {}; metrics = []; annual = []; checks = []
    cfg = protocol['config']
    for (group, rule), book in books.items():
        for cap in cfg['short_caps']:
            desired = hedged_targets(*book, cap)
            assert np.max(np.abs(desired).sum(axis=1)) <= 1+1e-12
            for borrow in cfg['borrow_rates']:
                for cost in cfg['transaction_cost_multipliers']:
                    key = f'{group}__{rule}__cap{int(cap*100):02d}__b{int(borrow*100):02d}__c{cost}'
                    target_error = float(np.max(np.abs(desired-source_targets[key])))
                    assert target_error < 1e-12
                    measured, path, ledger = simulate_signed(combined, ids, desired, borrow, cost, detailed=True)
                    other = value_replay(combined, ids, desired, borrow, cost)
                    independent_error = float(np.max(np.abs(path-other)))
                    source_error = float(np.max(np.abs(path-source_paths[key])))
                    assert independent_error < 1e-12 and source_error < 1e-12
                    expected = metrics_source.loc[key]
                    for field, value in measured.items():
                        np.testing.assert_allclose(value, expected[field], rtol=1e-12, atol=1e-10, err_msg=key+'/'+field)
                    for field, value in performance(path).items():
                        np.testing.assert_allclose(measured[field], value, rtol=0, atol=1e-12)
                    assert ledger.date.tolist() == dates.tolist() and ledger.nav.min() > 0
                    assert ledger['long'].iloc[-1] == 0 and ledger['short'].iloc[-1] == 0
                    expected_ledger = pd.read_csv(source_folder/(key+'_ledger.csv'))
                    np.testing.assert_array_equal(ledger.date, expected_ledger.date)
                    columns = list(ledger.columns.drop('date'))
                    ledger_error = float(np.max(np.abs(ledger[columns].to_numpy()-expected_ledger[columns].to_numpy())))
                    assert ledger_error < 1e-5
                    ledger.to_csv(run/'ledgers'/(key+'.csv'), index=False)
                    metrics.append(dict(id=key, group=group, rule=rule, cap=cap,
                                        borrow=borrow, cost=cost, **measured))
                    paths[key] = path; targets[key] = desired
                    checks.append(dict(id=key, max_target_error=target_error,
                        max_source_daily_error=source_error, max_independent_daily_error=independent_error,
                        max_source_ledger_dollars_error=ledger_error,
                        source_sharpe_error=abs(measured['sharpe']-expected.sharpe)))
                    for year in (2022, 2023, 2024, 2025):
                        value = float(np.prod(1+path[pd.DatetimeIndex(dates).year == year])-1)
                        assert abs(value-annual_source.loc[(key, year), 'calendar_return']) < 1e-12
                        annual.append(dict(id=key, year=year, calendar_return=value))
        print(f'Reproduced {group}/{rule}: {len(metrics)}/96 accounts', flush=True)
    assert len(metrics) == cfg['accounts'] == 96
    pd.DataFrame(metrics).to_csv(run/'METRICS.csv', index=False)
    pd.DataFrame(annual).to_csv(run/'CALENDAR.csv', index=False)
    pd.DataFrame(checks).to_csv(run/'ACCOUNT_CHECKS.csv', index=False)
    np.savez_compressed(run/'PATHS.npz', dates=dates, **paths)
    np.savez_compressed(run/'TARGETS.npz', ids=ids,
                        security_ids=np.asarray(combined['panel'].tickers, dtype=str), **targets)
    audit = dict(status='passed', accounts=96, baseline_daily_paths=12,
        source_commit=cfg['source_commit'], score_checks=score_checks, calendar_returns=384,
        max_source_daily_error=max(c['max_source_daily_error'] for c in checks),
        max_independent_daily_error=max(c['max_independent_daily_error'] for c in checks),
        max_source_target_error=max(c['max_target_error'] for c in checks),
        max_source_ledger_dollars_error=max(c['max_source_ledger_dollars_error'] for c in checks),
        max_source_sharpe_error=max(c['source_sharpe_error'] for c in checks),
        scope='Saved-export/target/accounting reproduction; no selected-checkpoint regeneration, new training, independent vendor-data or broker-execution validation')
    write(run/'AUDIT.json', audit)
    verify(run)
    write(run/'COMPLETE.json', dict(status='complete', timestamp_utc=datetime.now(timezone.utc).isoformat(),
          output_sha256={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}))
    print(json.dumps(audit, indent=2), flush=True)


def report(run):
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = json.loads((run/'COMPLETE.json').read_text())
    for name, expected in complete['output_sha256'].items():
        assert sha(run/name) == expected
    doc = ROOT/'docs'/run.name
    doc.mkdir(parents=True, exist_ok=True)
    for name in ('PROTOCOL.json', 'FREEZE.json', 'METRICS.csv', 'CALENDAR.csv',
                 'BASELINE_PARITY.csv', 'ACCOUNT_CHECKS.csv', 'AUDIT.json', 'COMPLETE.json'):
        shutil.copyfile(run/name, doc/name)
    metrics = pd.read_csv(run/'METRICS.csv').set_index('id')
    with np.load(run/'PATHS.npz', allow_pickle=False) as saved:
        dates = pd.to_datetime(saved['dates'])
        figure, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
        for column, group in enumerate(parent.GROUPS):
            for rule, cap, label, color in [
                    ('issuer', .5, 'Issuer five / 50% hedge cap', '#727e8b'),
                    ('retention', 0., 'Retention / unhedged', '#d07b21'),
                    ('retention', .5, 'Retention / 50% hedge cap', '#169c8b'),
                    ('retention', .75, 'Retention / 75% hedge cap', '#2857b8')]:
                key = f'{group}__{rule}__cap{int(cap*100):02d}__b03__c1'
                row = metrics.loc[key]; wealth = np.r_[100., 100*np.cumprod(1+saved[key])]
                drawdown = wealth/np.maximum.accumulate(wealth)-1
                axes[0, column].plot(dates, wealth[1:], color=color,
                    label=f'{label}\nSR {row.sharpe:.3f} | CAGR {row.annualized_return:.1%}')
                axes[1, column].plot(dates, 100*drawdown[1:], color=color)
            axes[0, column].set_title(group.replace('_', ' ').title())
            axes[0, column].legend(fontsize=8, loc='upper left')
            axes[0, column].set_ylabel('Growth of $100 (after costs)')
            axes[1, column].set_ylabel('Drawdown (%)')
            for axis in axes[:, column]:
                axis.grid(alpha=.2); axis.tick_params(axis='x', rotation=25)
        figure.suptitle('feat-rs reproduction of tom-shlom retention + continuous QQQ hedge\n2022–2025 development history; same saved models; 3% borrow and base trading costs')
        figure.tight_layout(rect=(0, 0, 1, .93))
        figure.savefig(doc/'retention_comparison.png', dpi=150)
        plt.close(figure)
    lines = ['# Retention strategy replication in feat-rs — October 29, 2026', '',
        'The pinned tom-shlom result was reproduced from feat-rs saved forecasts and reviewed market inputs. '
        'All 96 source accounts, targets, daily paths, detailed ledgers and 384 calendar returns passed comparison. '
        'Twelve existing feat-rs daily-path baselines also reproduced. This is replication of selected, repeatedly inspected 2022–2025 development results, not prospective confirmation.', '',
        '| Group | Selection | Hedge cap | Sharpe | CAGR | Max drawdown | Annual traded/NAV |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in parent.GROUPS:
        for rule, cap in [('issuer', .5), ('retention', 0.), ('retention', .5), ('retention', .75)]:
            row = metrics.loc[f'{group}__{rule}__cap{int(cap*100):02d}__b03__c1']
            lines.append(f'| {group} | {rule} | {cap:.0%} | {row.sharpe:.3f} | {row.annualized_return:.2%} | {row.maximum_drawdown:.2%} | {row.turnover_annual:.2f} |')
    stress = metrics.loc['original_five__retention__cap75__b06__c2']
    audit = json.loads((run/'AUDIT.json').read_text())
    lines += ['', '![Matched retention comparisons](retention_comparison.png)', '',
        '## Exact strategy', '',
        'Each rebalance adds 0.02 to the percentile-derived blend score of every security with positive weight in the previous target. '
        'The five highest adjusted scores receive equal weights, with fractional boundary ties and current eligibility enforced. '
        'State continues across annual fits. Retention uses previous targets, not actual filled holdings. Share classes occupy separate security slots; this is not issuer-five selection.', '',
        'The unchanged 50:50 THINK+mixer AH+PM / restricted dynamic-hypergraph rank blend supplies the scores. '
        'The selected basket receives a 20-session/20% volatility target. The always-on QQQ short is min(cap, unused gross budget, 20%/QQQ trailing volatility). '
        'A 75% cap is a ceiling, not a constant 75% short. Decisions are every five exchange sessions at 09:20 ET after the 09:15 feature cutoff; share sizing and risk use prior-close data and fills use the regular close.', '',
        '## Costs and interpretation', '',
        f'The original-five retention/75% account at 6% borrowing and 2× trading costs has SR {stress.sharpe:.3f}, CAGR {stress.annualized_return:.2%} and drawdown {stress.maximum_drawdown:.2%}. '
        'All 96 accounts are retained, including weaker seed groups and unhedged controls. The grid is the source study’s exact 3 groups × 2 selection rules × 4 caps × 2 borrow rates × 2 cost multipliers; no new tuning was performed. '
        'Borrow is charged on short value by calendar day; cash/rebate is zero. Costs, liquidity, settlement and dividend treatment remain inherited simulation assumptions. Broker locates/recalls, full impact and margin enforcement are not established.', '',
        'The original-five ensemble is substantially stronger than the new-five half. The ten-seed ensemble overlaps both halves. '
        'Retention changes holdings, subsequent basket risk and exposure as well as turnover; this replay alone does not attribute the improvement solely to saved transaction costs. '
        'The published outcomes were known before this replication protocol was frozen. No untouched-period superiority or trained market-timing claim follows.', '',
        '## Verification and provenance', '',
        f'Source commit: `fca66ed0b577444928443847a6ec7f7fe21b102a` on `tom-shlom`. '
        f'Maximum source daily-return error: {audit["max_source_daily_error"]:.3g}; maximum independent dollar-position replay error: {audit["max_independent_daily_error"]:.3g}; '
        f'maximum source target error: {audit["max_source_target_error"]:.3g}; maximum ledger dollar discrepancy: ${audit["max_source_ledger_dollars_error"]:.3g}. '
        'All source archive members used were checked against the committed SHA-256 manifest. Existing local forecasts were checked against their publication seals, and inputs/code were hashed before replay. '
        'This verifies saved forecasts and accounting, not numerical regeneration of trained checkpoints or independent vendor/broker correctness.', '',
        '[All metrics](METRICS.csv) · [Account checks](ACCOUNT_CHECKS.csv) · [Audit](AUDIT.json) · [Protocol and input hashes](PROTOCOL.json) · [Baseline parity](BASELINE_PARITY.csv) · [Annual returns](CALENDAR.csv)', '',
        '## Reproduce', '', 'From the repository root:', '', '```powershell',
        'git fetch origin tom-shlom:refs/remotes/origin/tom-shlom',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_retention_replication --stage source',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_retention_replication --stage freeze --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_retention_replication --stage run --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_retention_replication --stage report --name {run.name}_fresh',
        '```', '', 'Original run outputs refuse overwrite. Full daily paths, targets and ledgers are under '
        f'`runs/{run.name}/`. No model was retrained, 2026 returns scored, capital deployed or prospective policy changed.', '']
    (doc/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    print('Published '+str(doc.relative_to(ROOT)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['source', 'freeze', 'run', 'report'], required=True)
    parser.add_argument('--name', default=DEFAULT)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        parser.error('--name must be a simple directory name')
    run = ROOT/'runs'/args.name
    if args.stage == 'source': restore_source()
    elif args.stage == 'freeze': freeze(run)
    elif args.stage == 'run': replay(run)
    else: report(run)


if __name__ == '__main__':
    main()
