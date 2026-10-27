"""CLI for frozen historical validation and order-free forward replay."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import hashlib
import time
import numpy as np

from experiments.strategy_search.runner import load_trial
from experiments.strategy_search.statistics import daily_rows, scales, risk
from experiments.strategy_search.storage import atomic, arrays, read, sha, identity
from .config import Execution, freeze
from .market import load, order_windows
from .baseline import replay_with_orders
from .engine import replay, PATH_COLUMNS, ORDER_COLUMNS
from .fees import charges
from .analysis import dollar_daily, normalized_summary, gate
from .quotes import assemble, download

SCOPES = ('earlier_OOF', 'initial_validation', 'later_development64')


def emit(output, event, **extra):
    record = dict(event=event, **extra)
    atomic(Path(output) / 'progress.json', record)
    print(json.dumps(record), flush=True)


def audit_reference(reference, output, scopes=SCOPES):
    output = Path(output)
    results = []
    for scope in scopes:
        m, base, diagnostics = load(reference, scope)
        cache = output / 'inputs' / (scope + '.npz')
        arrays(cache, base=base)
        for fee in (0., .25, 1., 2.):
            config = read(Path(reference) / 'selection.json')['leaders']
            c = next(x['config'] for x in config if x['config']['family'] == 'S10' and x['config']['style'] == 'long_short')
            original = load_trial(Path(reference), c, scope, fee, delay=2, borrow=.03, scaled=True)
            if sha(original['daily_path']) != original['daily_sha256']:
                raise ValueError('Changed original daily archive')
            with np.load(original['daily_path']) as z:
                scale = z['scales']
                expected = z['daily']
            path, _, _, failures, breaches, journal = replay_with_orders(base, m['open'], m['close'],
                m['sessions'], m['actions'], m['sector'], True, scale, fee, .03, 2)
            actual = daily_rows(path, m['sessions'])
            np.testing.assert_array_equal(actual, expected)
            assert failures == breaches == 0
            receipt = dict(scope=scope, fee_bps=fee, daily_bitwise_equal=True, reference_trial=original['key'],
                           source_daily_sha256=original['daily_sha256'], orders=len(journal))
            if fee == .25:
                orders = np.asarray(journal, float).reshape(-1, 6)
                keep = np.isin(m['sessions'][orders[:, 0].astype(int)], m['evaluate_ids'])
                orders = orders[keep]
                arrays(output / 'baseline' / (scope + '.npz'), orders=orders, daily=actual, equity=path)
                histograms = []
                for capital in (25_000., 100_000., 1_000_000.):
                    settings = Execution(capital=capital)
                    costs = np.array([charges(row[2] * capital, row[3], settings, fractional=True) for row in orders])
                    notionals = orders[:, 4] * capital
                    total = notionals.sum()
                    histograms.append(dict(capital=capital, orders=len(orders),
                        orders_under_1_dollar=int((notionals < 1).sum()), orders_under_500=int((notionals < .005 * capital).sum()),
                        notional_quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'), map(float, np.quantile(notionals, [0, .25, .5, .75, .95, 1])))),
                        fees_dollars=costs.sum(0).tolist(), total_notional=float(total),
                        fee_bps=float(costs.sum() / total * 10000),
                        interpretation='Diagnostic fee burden on fixed fractional trade tape; no cash-feedback replay; fractional shorts assumed and generally not broker executable'))
                    if capital == 100_000.:
                        with (output / 'baseline' / (scope + '_orders.csv')).open('w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['arrival_UTC', 'ticker', 'shares', 'price_proxy', 'notional', 'reason', 'commission', 'exchange', 'clearing', 'regulatory'])
                            for row, cost in zip(orders, costs):
                                writer.writerow([str(np.datetime64(int(m['starts'][int(row[0])]), 'ns')) + 'Z', m['symbols'][int(row[1])],
                                    row[2] * capital, row[3], row[4] * capital, int(row[5]), *cost])
                receipt['cost_audit'] = histograms
            results.append(receipt)
            emit(output, 'baseline_reproduced', scope=scope, fee=fee, orders=len(journal))
        del m, base
    atomic(output / 'baseline_verification.json', dict(all_passed=True, checks=results))
    return results


def scenarios():
    primary = Execution()
    return [('primary', primary), ('adverse_0p5', replace(primary, adverse_bps=.5)),
            ('adverse_1', replace(primary, adverse_bps=1.)), ('delay3', replace(primary, delay=3)),
            ('delay5', replace(primary, delay=5)), ('account25k', replace(primary, capital=25_000.)),
            ('account1m', replace(primary, capital=1_000_000.)),
            ('commission_free', replace(primary, commission_free=True)), ('borrow10', replace(primary, borrow=.1))]


def execute_one(m, base, settings, name, mode, quotes, output, protocol):
    binding = dict(scope=m['scope'], scenario=name, mode=mode, settings=asdict(settings),
                   protocol_sha256=sha(Path(output) / 'protocol.json'),
                   base_sha256=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest(),
                   share_volume_sha256=hashlib.sha256(np.ascontiguousarray(m['volume']).tobytes()).hexdigest(),
                   quote_manifest_sha256=sha(Path(output) / 'quotes' / m['scope'] / 'manifest.json') if mode == 'quotes' else None)
    key = identity(binding)[:24]
    root = Path(output) / 'trials'
    record = root / (key + '.json')
    archive = root / (key + '.npz')
    if record.exists():
        saved = read(record)
        if saved['binding'] != binding or sha(archive) != saved['archive_sha256']:
            raise ValueError('Changed trial inputs or result')
        return saved
    began = time.perf_counter()
    shadow, _, _ = replay(m, base, np.ones(len(m['meta']['session_ids'])), settings, quotes)
    scale = scales(dollar_daily(shadow, m['sessions'], settings.capital))
    path, orders, failures = replay(m, base, scale, settings, quotes)
    metrics, daily = normalized_summary(m, path, orders, failures, settings.capital)
    blocks = {}
    partitions = read(Path(protocol['binding']['reference']) / 'protocol.json')['partitions'][m['scope']]
    for label, partition in partitions.items():
        if label == 'warmup':
            continue
        block_m = dict(m, evaluate_ids=partition['session_ids'])
        blocks[label], _ = normalized_summary(block_m, path, orders, failures, settings.capital)
    arrays(archive, equity=path, orders=orders, daily=daily, scales=scale, sessions=m['sessions'])
    saved = dict(key=key, binding=binding, scope=m['scope'], scenario=name, mode=mode,
                 metrics=metrics, blocks=blocks, archive=str(archive), archive_sha256=sha(archive),
                 path_columns=list(PATH_COLUMNS), order_columns=list(ORDER_COLUMNS), seconds=time.perf_counter() - began,
                 data_status='research-exposed historical replay', borrow_availability='assumed',
                 interpretation='Quote constrained' if mode == 'quotes' else 'Minute-open proxy; spread absent; diagnostic only')
    atomic(record, saved)
    if name == 'primary':
        with archive.with_suffix('.orders.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['arrival_UTC', 'ticker', *ORDER_COLUMNS[2:]])
            for order in orders:
                i, j = map(int, order[:2])
                writer.writerow([str(np.datetime64(int(m['starts'][i]), 'ns')) + 'Z', m['symbols'][j], *order[2:]])
    emit(output, 'trial_complete', scope=m['scope'], scenario=name, mode=mode,
         Sharpe=metrics['Sharpe'], total_return=metrics['total_return'], valid=metrics['accounting_valid'], seconds=saved['seconds'])
    return saved


def replay_all(args, protocol):
    rows = []
    for scope in args.scopes.split(','):
        m, base, _ = load(args.reference, scope)
        windows = order_windows(m, base)
        if args.download_quotes:
            quote = download(m, windows, args.output, args.env_file, workers=args.workers)
        elif args.mode == 'quotes':
            quote = assemble(m, windows, Path(args.output) / 'quotes' / scope)
        else:
            quote = None
        for name, settings in scenarios():
            rows.append(execute_one(m, base, settings, name, args.mode, quote, args.output, protocol))
    atomic(Path(args.output) / ('replay_' + args.mode + '.json'), dict(rows=[r['key'] for r in rows],
                decision=gate(rows, protocol) if args.mode == 'quotes' else {'status': 'diagnostic_only_spread_absent'}))
    return rows


def inventory(reference, output, archive):
    p = Path(archive) / 'manifest.json'
    j = read(p)
    record = dict(archive=str(Path(archive).resolve()), manifest_sha256=sha(p),
        request={k: v for k, v in j['request'].items() if k != 'symbols'}, symbols=len(j['request']['symbols']),
        summary=j['summary'], historical_bid_ask='Separate SIP snapshot cache',
        additional_history='No extra dates opened without an explicit unsealed-range manifest; existing three audited cohorts evaluated',
        static_universe=True, sealed_future_opened=False)
    atomic(Path(output) / 'data_inventory.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'replay', 'report', 'shadow'))
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--scopes', default=','.join(SCOPES))
    parser.add_argument('--mode', choices=('quotes', 'bar_proxy'), default='quotes')
    parser.add_argument('--download-quotes', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=12)
    parser.add_argument('--session', help='Prospective YYYY-MM-DD exchange session, after close +20 minutes')
    args = parser.parse_args()
    protocol = freeze(args.reference, args.output)
    if args.command == 'audit':
        if args.archive:
            inventory(args.reference, args.output, args.archive)
        audit_reference(args.reference, args.output, tuple(args.scopes.split(',')))
    elif args.command == 'replay':
        replay_all(args, protocol)
    elif args.command == 'report':
        from .report import report
        report(args.output)
    elif args.command == 'shadow':
        from .shadow import initialize, collect, replay_forward
        initialize(args.reference, args.output, protocol)
        if args.session:
            emit(args.output, 'forward_collection', **collect(args.reference, args.output, protocol, args.env_file, args.session))
        emit(args.output, 'forward_status', **replay_forward(args.reference, args.output, protocol, args.env_file))


if __name__ == '__main__':
    main()
