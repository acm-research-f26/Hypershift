"""Importable workers and CLI; run artifacts stay outside the Git checkout."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import concurrent.futures as cf
import csv
import html
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .data import Study, SCOPES
from .execution import replay, COLUMNS
from .signals import fit, build
from .statistics import summarize, daily_rows, scales, risk, bootstrap
from .storage import read, sha, atomic, arrays, identity, utc

SOURCE = Path(__file__).resolve().parent
ORDER = ('S01', 'S02', 'S04', 'S09', 'S03', 'S05', 'S08', 'S10', 'S06', 'S07')


def event(output, kind, **kw):
    record = dict(UTC=utc(), kind=kind, **kw)
    atomic(output / 'progress.json', record)
    with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def source_bindings():
    return {p.name: sha(p) for p in sorted(SOURCE.iterdir()) if p.suffix in ('.py', '.json')}


def freeze(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        saved = read(path)
        assert saved['sources'] == source_bindings(), 'Frozen source changed; use a new run directory'
        assert saved['study_root'] == str(Path(args.study_root).resolve())
        return saved
    catalog = read(SOURCE / 'catalog.json')
    for source in source_bindings():
        destination = output / 'source' / source
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(SOURCE / source, destination)
    study = Study(args.study_root, args.market_root, output)
    audit = study.audit_volume()
    configs = [c for c in catalog['rows'] if audit['passed'] or not c['requires_volume']]
    protocol = dict(UTC=utc(), sources=source_bindings(), catalog_sha256=sha(SOURCE / 'catalog.json'),
                    study_root=str(Path(args.study_root).resolve()), market_root=str(Path(args.market_root).resolve()),
                    configs=configs, costs_bps=catalog['costs_bps'], order=list(ORDER),
                    maximum_core_combinations=660, actual_core_combinations=len(configs) * len(catalog['costs_bps']),
                    assumptions=read(SOURCE / 'assumptions.json'), volume_available=audit['passed'],
                    partitions=read(Path(args.study_root) / 'protocol.json')['cohorts'],
                    training_audit_sha256=sha(Path(args.study_root) / 'lead_lag_twohop/corrected/training_data_audit.json'),
                    prior_later_exposure=True, sealed_future_untouched=True,
                    selection='One fixed configuration per family/style, highest qualifying development fee; diagnostic fallback labeled separately',
                    qualifying='Accounting valid, active and defined SR in five blocks, median block SR >= 1, pooled daily SR >= 1, both development cohort returns positive',
                    risk_warmup='First 20 replay sessions cash; earlier first60 fitting sessions excluded from evaluation; trailing20 unscaled net daily returns',
                    no_new_neural_training=True)
    atomic(path, protocol)
    event(output, 'protocol_frozen', configurations=len(configs), policy_cost_combinations=protocol['actual_core_combinations'])
    return protocol


def trial_key(config, scope, fee, delay, borrow, scaled, protocol_hash):
    return identity(dict(config=config, scope=scope, fee=fee, delay=delay, borrow=borrow,
                         scaled=scaled, protocol=protocol_hash))[:24]


def execute(study, config, scope, fee, base, diagnostics, delay=2, borrow=.03, scaled=True, detailed=False):
    output = study.output
    key = trial_key(config, scope, fee, delay, borrow, scaled, sha(output / 'protocol.json'))
    path = output / 'trials' / (key + '.json')
    daily_path = path.with_suffix('.npz')
    curve_path = output / 'curves' / (key + '.npz')
    existing = read(path) if path.exists() else None
    if existing:
        assert sha(daily_path) == existing['daily_sha256']
        if not detailed or curve_path.exists():
            return existing
    m = study.market(scope)
    scale = np.ones(len(m['meta']['session_ids']))
    if scaled:
        shadow = execute(study, config, scope, fee, base, diagnostics, delay, borrow, False)
        with np.load(shadow['daily_path']) as z:
            scale = scales(z['daily'])
    p = config['parameters']
    began = time.perf_counter()
    path_array, closed, elapsed, failures, breaches = replay(base, m['open'], m['close'], m['sessions'], m['actions'],
                                                          m['sector'], config['style'] == 'long_short', scale,
                                                          fee, borrow, delay, p.get('partial_step', 1.),
                                                          p.get('rebalance', 1), p.get('turnover_day_cap', 0.),
                                                          config['family'] == 'S04')
    daily = daily_rows(path_array, m['sessions'])
    ids = m['meta']['session_ids'][60:] if scope == SCOPES[0] else m['meta']['session_ids']
    metrics = summarize(path_array, m['sessions'], ids, daily)
    metrics.update(average_holding_minutes=float(elapsed / closed) if closed else None,
                   holding_duration_scope='Entire replay, includes earlier fit-period risk warmup',
                   engine_failures=failures, turnover_budget_breaches=breaches)
    metrics['accounting_valid'] &= failures == 0 and breaches == 0
    partitions = read(output / 'protocol.json')['partitions'][scope]
    blocks = {name: summarize(path_array, m['sessions'], block['session_ids'], daily)
              for name, block in partitions.items() if name != 'warmup'}
    for block in blocks.values():
        block['accounting_valid'] &= failures == 0 and breaches == 0
    if existing:
        with np.load(daily_path) as z:
            np.testing.assert_array_equal(daily, z['daily'])
    else:
        arrays(daily_path, daily=daily, scales=scale)
        existing = dict(key=key, UTC=utc(), config=config, scope=scope, cost_bps=fee, delay=delay,
                        borrow_annual=borrow, scaled=scaled, metrics=metrics, blocks=blocks, diagnostics=diagnostics,
                        daily_path=str(daily_path), daily_sha256=sha(daily_path), seconds=time.perf_counter() - began)
        atomic(path, existing)
    if detailed:
        arrays(curve_path, equity=path_array, completed=m['completed'], sessions=m['sessions'])
        atomic(curve_path.with_suffix('.json'), dict(sha256=sha(curve_path), columns=COLUMNS, key=key))
    return existing


def job(payload):
    arguments, family, scope = payload
    args = argparse.Namespace(**arguments)
    output = Path(args.output).resolve()
    protocol = read(output / 'protocol.json')
    study = Study(args.study_root, args.market_root, output)
    with np.load(output / 'fit.npz') as z:
        fitted = {name: z[name] for name in z.files}
    configs = [c for c in protocol['configs'] if c['family'] == family]
    for config in configs:
        if all((output / 'trials' / (trial_key(config, scope, fee, 2, .03, True, sha(output / 'protocol.json')) + '.json')).exists()
               for fee in protocol['costs_bps']):
            continue
        base, diagnostics = build(study, config, scope, fitted)
        for fee in protocol['costs_bps']:
            execute(study, config, scope, fee, base, diagnostics)
        event(output, 'configuration_complete', family=family, config=config['id'], scope=scope, costs=len(protocol['costs_bps']))
    return dict(family=family, scope=scope, configurations=len(configs))


def search(args):
    protocol = freeze(args)
    output = Path(args.output).resolve()
    study = Study(args.study_root, args.market_root, output)
    fit(study, protocol['configs'])
    event(output, 'fit_frozen', sha256=sha(output / 'fit.npz'))
    del study
    families = args.families.split(',') if args.families else ORDER
    tasks = [(vars(args), family, scope) for family in families for scope in SCOPES[:2]]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for future in cf.as_completed(futures):
            event(output, 'family_cohort_complete', **future.result())
    event(output, 'development_complete', families=list(families))


def load_trial(output, config, scope, fee, delay=2, borrow=.03, scaled=True):
    key = trial_key(config, scope, fee, delay, borrow, scaled, sha(output / 'protocol.json'))
    return read(output / 'trials' / (key + '.json'))


def development(output, config, fee):
    rows = [load_trial(output, config, scope, fee) for scope in SCOPES[:2]]
    blocks = [rows[0]['blocks']['development_' + str(i)] for i in (1, 2, 3)]
    blocks += [rows[1]['blocks']['development_' + str(i)] for i in (4, 5)]
    sharpes = [b['Sharpe'] for b in blocks]
    eligible = all(b['accounting_valid'] and b['active'] and b['Sharpe'] is not None for b in blocks)
    daily = []
    for i, row in enumerate(rows):
        with np.load(row['daily_path']) as z:
            daily.extend(z['daily'][60 if i == 0 else 0:, 1])
    pooled = risk(daily)
    median = float(np.median(sharpes)) if None not in sharpes else None
    positive = all(row['metrics']['total_return'] is not None and row['metrics']['total_return'] > 0 for row in rows)
    qualifies = bool(eligible and positive and median >= 1 and pooled['Sharpe'] >= 1)
    return dict(config=config, cost_bps=fee, eligible=eligible, qualifies=qualifies,
                median_block_Sharpe=median, pooled_Sharpe=pooled['Sharpe'], block_Sharpes=sharpes,
                cohort_returns=[row['metrics']['total_return'] for row in rows],
                worst_block_drawdown=max(b['maximum_drawdown'] for b in blocks),
                turnover=sum(row['metrics']['turnover'] for row in rows),
                accounting_valid=all(row['metrics']['accounting_valid'] for row in rows))


def select(args):
    output = Path(args.output).resolve()
    if (output / 'selection.json').exists():
        return read(output / 'selection.json')
    protocol = freeze(args)
    comparison = [development(output, config, fee) for config in protocol['configs'] for fee in protocol['costs_bps']]
    atomic(output / 'development_comparison.json', comparison)
    leaders = []
    for family in ORDER:
        for style in ('long_only', 'long_short'):
            candidates = [r for r in comparison if r['config']['family'] == family and r['config']['style'] == style]
            passing = [r for r in candidates if r['qualifies']]
            if passing:
                chosen = min(passing, key=lambda r: (-r['cost_bps'], r['worst_block_drawdown'], r['turnover'], r['config']['id']))
                leaders.append(dict(**chosen, selection_kind='qualified', c_SR1_development=chosen['cost_bps']))
            else:
                diagnostic = [r for r in candidates if r['eligible'] and r['cost_bps'] == 0]
                if not diagnostic:
                    diagnostic = [r for r in candidates if r['cost_bps'] == 0]
                if diagnostic:
                    chosen = min(diagnostic, key=lambda r: (-(r['median_block_Sharpe'] if r['median_block_Sharpe'] is not None else -1e6),
                                                              r['worst_block_drawdown'], r['turnover'], r['config']['id']))
                    leaders.append(dict(**chosen, selection_kind='diagnostic_target_not_met', c_SR1_development=None))
    selection = dict(UTC=utc(), protocol_sha256=sha(output / 'protocol.json'), leaders=leaders,
                     qualifying_policy_cost_cases=sum(r['qualifies'] for r in comparison),
                     later_opened=False, prior_later_exposure=True, no_later_retuning=True)
    atomic(output / 'selection.json', selection)
    event(output, 'selection_frozen', leaders=len(leaders), qualifying_cases=selection['qualifying_policy_cost_cases'])
    return selection


def validate(args):
    output = Path(args.output).resolve()
    selection = select(args)
    protocol = read(output / 'protocol.json')
    study = Study(args.study_root, args.market_root, output)
    with np.load(output / 'fit.npz') as z:
        fitted = {name: z[name] for name in z.files}
    chosen = {r['config']['id']: r['config'] for r in selection['leaders']}
    for config in list(chosen.values()):
        if config['family'] == 'S03':
            for other in protocol['configs']:
                if other['family'] == 'S03' and other['style'] == config['style'] and other['preset'] == config['preset']:
                    chosen[other['id']] = other
    for config in chosen.values():
        base, diagnostics = build(study, config, SCOPES[2], fitted)
        for fee in protocol['costs_bps']:
            execute(study, config, SCOPES[2], fee, base, diagnostics, detailed=fee in (0., .1, 1.))
        for delay in (3, 5):
            for fee in (0., .1):
                execute(study, config, SCOPES[2], fee, base, diagnostics, delay=delay)
        if config['style'] == 'long_short':
            execute(study, config, SCOPES[2], .1, base, diagnostics, borrow=.1)
            execute(study, config, SCOPES[2], 0., base, diagnostics, borrow=0.)
        event(output, 'later_configuration_complete', family=config['family'], config=config['id'])
    atomic(output / 'validation.json', dict(UTC=utc(), configurations=list(chosen), complete=True,
                                          selection_sha256=sha(output / 'selection.json'), later_retuning=False))


def report(args):
    output = Path(args.output).resolve()
    rows = [read(p) for p in sorted((output / 'trials').glob('*.json'))]
    ledger = []
    for row in rows:
        c = row['config']
        ledger.append(dict(key=row['key'], family=c['family'], name=c['name'], config=c['id'], model=c['model'],
                           style=c['style'], preset=c['preset'], scope=row['scope'], fee_bps=row['cost_bps'],
                           delay=row['delay'], borrow=row['borrow_annual'], scaled=row['scaled'], **row['metrics']))
    fields = list(dict.fromkeys(k for row in ledger for k in row))
    with (output / 'trial_ledger.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(ledger)
    selection = read(output / 'selection.json')
    summary = []
    for leader in selection['leaders']:
        c = leader['config']
        later = [r for r in rows if r['config']['id'] == c['id'] and r['scope'] == SCOPES[2]
                 and r['scaled'] and r['delay'] == 2 and r['borrow_annual'] == .03]
        passing = [r['cost_bps'] for r in later if r['metrics']['accounting_valid'] and
                   r['metrics']['Sharpe'] is not None and r['metrics']['Sharpe'] >= 1 and r['metrics']['total_return'] > 0]
        zero = next(r for r in later if r['cost_bps'] == 0)
        tenth = next(r for r in later if r['cost_bps'] == .1)
        with np.load(zero['daily_path']) as z:
            uncertainty = bootstrap(z['daily'][:, 1])
        stresses = {}
        for delay in (3, 5):
            r = load_trial(output, c, SCOPES[2], .1, delay)
            stresses['delay' + str(delay) + '_cost0p1_Sharpe'] = r['metrics']['Sharpe']
        summary.append(dict(family=c['family'], name=c['name'], config=c['id'], style=c['style'], model=c['model'],
                            preset=c['preset'], selection_kind=leader['selection_kind'],
                            development_c_SR1=leader['c_SR1_development'], later_c_SR1=max(passing) if passing else None,
                            development_block_Sharpes=leader['block_Sharpes'],
                            later_zero=zero['metrics'], later_cost0p1=tenth['metrics'], uncertainty=uncertainty, **stresses))
    atomic(output / 'summary.json', dict(UTC=utc(), families=10, core_configurations=len(read(output / 'protocol.json')['configs']),
                                        trial_records=len(rows), leaders=summary, assumptions=read(SOURCE / 'assumptions.json'),
                                        interpretation='Later fee capacity is descriptive after frozen development selection, not a new selection rule'))
    body = ['<!doctype html><meta charset="utf-8"><title>Bounded strategy round</title>',
            '<style>body{font:16px system-ui;margin:40px;max-width:1500px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}th{background:#eef}</style>',
            '<h1>Ten hypotheses / 66 configurations</h1>',
            '<p>Exploratory frozen-model research. Static retrospective universe; full-fill OHLCV proxies; no quotes, impact or capacity model. Training-only fit: first 60 earlier sessions. Five development blocks, selection frozen before later replay. Fees are one-way basis points.</p>',
            '<p>c_SR1 is the highest tested fee with valid net daily Sharpe at least 1. Development qualification additionally requires median five-block Sharpe at least 1 and positive returns in both cohorts. Diagnostic rows failed development qualification.</p>',
            '<table><tr><th>Family</th><th>Style</th><th>Preset/model</th><th>Selection</th><th>Dev c_SR1</th><th>Later c_SR1</th><th>Later SR 0bp</th><th>Later SR .1bp</th><th>Turnover/day</th><th>DD .1bp</th><th>Delay+1 SR .1bp</th></tr>']
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    for r in summary:
        values = [r['family'], r['style'], str(r['preset']) + '/' + str(r['model']), r['selection_kind'],
                  fmt(r['development_c_SR1']), fmt(r['later_c_SR1']), fmt(r['later_zero']['Sharpe']),
                  fmt(r['later_cost0p1']['Sharpe']), fmt(r['later_cost0p1']['turnover_per_day']),
                  fmt(r['later_cost0p1']['maximum_drawdown']), fmt(r['delay3_cost0p1_Sharpe'])]
        body.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
    body.append('</table><p>Daily Sharpe uses 252 sessions/year; cohorts remain separate. Bootstrap intervals and complete accounting status are in summary.json. Every trial and risk shadow is disclosed in trial_ledger.csv.</p>')
    (output / 'report.html').write_text('\n'.join(body), encoding='utf-8')
    event(output, 'report_complete', trial_records=len(rows), leaders=len(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'search', 'select', 'validate', 'report', 'all'))
    parser.add_argument('--study-root', required=True, help='Existing audited frozen-forecast strategy study')
    parser.add_argument('--market-root', required=True, help='Dataset directory containing metadata.json and 1m/bars.npy')
    parser.add_argument('--output', required=True, help='Fresh private run directory, preferably inside the Drive sync root')
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    parser.add_argument('--families', default='', help='Optional comma-separated families for a partial development run')
    args = parser.parse_args()
    if args.command == 'prepare':
        p = freeze(args)
        fit(Study(args.study_root, args.market_root, args.output), p['configs'])
    if args.command in ('search', 'all'):
        search(args)
    if args.command in ('select', 'all'):
        select(args)
    if args.command in ('validate', 'all'):
        validate(args)
    if args.command in ('report', 'all'):
        report(args)


if __name__ == '__main__':
    main()
