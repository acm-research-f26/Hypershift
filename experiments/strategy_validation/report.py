"""Standalone report and plots; missing evidence remains inconclusive."""
import csv
import html
from pathlib import Path
import numpy as np

from experiments.strategy_search.storage import atomic, read, sha
from .analysis import gate


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    protocol = read(output / 'protocol.json')
    rows = [read(p) for p in sorted((output / 'trials').glob('*.json'))]
    decision = gate(rows, protocol)
    ledger = []
    for r in rows:
        if sha(r['archive']) != r['archive_sha256']:
            raise ValueError('Changed trial archive')
        ledger.append(dict(scope=r['scope'], scenario=r['scenario'], mode=r['mode'],
                           **{k: r['metrics'].get(k) for k in ('Sharpe', 'total_return', 'maximum_drawdown',
                               'accounting_valid', 'quote_coverage', 'executed_orders', 'turnover_dollars', 'explicit_cost_bps')}))
    if ledger:
        with (output / 'trial_ledger.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    reference = Path(protocol['binding']['reference'])
    ref = read(reference / 'protocol.json')
    metadata = read(Path(ref['market_root']) / 'metadata.json')
    study_root = Path(ref['study_root'])
    attribution = []
    for r in rows:
        with np.load(r['archive']) as z:
            orders, path, sessions = z['orders'], z['equity'], z['sessions']
        evaluated = [sid for name, block in ref['partitions'][r['scope']].items() if name != 'warmup' for sid in block['session_ids']]
        omask = np.isin(sessions[orders[:, 0].astype(int)], evaluated) if len(orders) else np.zeros(0, bool)
        orders = orders[omask]
        sectors = np.load(study_root / 'cache' / r['scope'] / 'sector.npy')
        for stock in np.unique(orders[:, 1].astype(int)):
            stock_orders = orders[orders[:, 1] == stock]
            pnl = float(np.nansum(-stock_orders[:, 5] * stock_orders[:, 7]) - stock_orders[:, 10:14].sum())
            attribution.append(dict(scope=r['scope'], scenario=r['scenario'], mode=r['mode'], ticker=metadata['symbols'][stock],
                                    sector=int(sectors[stock]), trading_PnL_excluding_borrow_and_residual_inventory=pnl,
                                    explicit_fees=float(stock_orders[:, 10:14].sum()), filled_orders=int((stock_orders[:, 5] != 0).sum())))
    if attribution:
        with (output / 'stock_attribution.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(attribution[0])); writer.writeheader(); writer.writerows(attribution)
        sector_rows = {}
        for row in attribution:
            key = tuple(row[k] for k in ('scope', 'scenario', 'mode', 'sector'))
            item = sector_rows.setdefault(key, dict(zip(('scope', 'scenario', 'mode', 'sector'), key), trading_PnL_excluding_borrow=0., explicit_fees=0.))
            item['trading_PnL_excluding_borrow'] += row['trading_PnL_excluding_borrow_and_residual_inventory']
            item['explicit_fees'] += row['explicit_fees']
        with (output / 'sector_attribution.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(sector_rows.values())))); writer.writeheader(); writer.writerows(sector_rows.values())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    scopes = ['earlier_OOF', 'initial_validation', 'later_development64']
    labels = ['Earlier development', 'Initial development', 'Later replay']
    for mode, color in [('bar_proxy', '#94a3b8'), ('quotes', '#2563eb')]:
        primary = {r['scope']: r for r in rows if r['scenario'] == 'primary' and r['mode'] == mode}
        x = np.arange(3) + (-.17 if mode == 'bar_proxy' else .17)
        for ax, key, factor, title in [(axes[0], 'total_return', 100, 'Net return (%)'), (axes[1], 'Sharpe', 1, 'Annualized daily Sharpe')]:
            y = [primary[s]['metrics'][key] * factor if s in primary and primary[s]['metrics'][key] is not None else np.nan for s in scopes]
            bars = ax.bar(x, y, width=.32, color=color, label='Minute-open diagnostic' if mode == 'bar_proxy' else 'SIP quote constrained')
            for bar, scope in zip(bars, scopes):
                if scope in primary and not primary[scope]['metrics']['accounting_valid']:
                    bar.set_hatch('//')
            ax.set_title(title); ax.axhline(0, c='#64748b', lw=.8)
            ax.set_xticks(range(3), labels, rotation=15)
    axes[1].axhline(1, ls=':', color='#16a34a')
    cost_rows = [r for r in rows if r['scenario'] == 'primary' and r['mode'] == 'quotes']
    cost_by = {r['scope']: r for r in cost_rows}
    bottom = np.zeros(3)
    for key, color in [('commission', '#2563eb'), ('exchange', '#f97316'), ('clearing', '#a855f7'), ('regulatory', '#64748b')]:
        y = np.array([cost_by[s]['metrics']['costs_dollars'][key] / max(cost_by[s]['metrics']['turnover_dollars'], 1e-30) * 10000 if s in cost_by else np.nan for s in scopes])
        axes[2].bar(range(3), y, bottom=bottom, label=key, color=color); bottom += np.nan_to_num(y)
    axes[2].set(title='Explicit fees per traded dollar (bp)', xticks=range(3), xticklabels=labels)
    axes[2].tick_params(axis='x', rotation=15)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle('Closing momentum: one fixed whole-share implementation at $100,000\nHatched bars fail accounting/constraint checks; route fees are an explicit taker scenario', fontsize=14)
    fig.savefig(output / 'execution_validation.png', dpi=160)
    plt.close(fig)
    baseline = read(output / 'baseline_verification.json') if (output / 'baseline_verification.json').exists() else None
    summary = dict(decision=decision, trials=len(rows), baseline=baseline, protocol_sha256=sha(output / 'protocol.json'),
                   prospective_status=read(output / 'shadow' / 'protocol.json') if (output / 'shadow' / 'protocol.json').exists() else None,
                   limitations=['Retrospective universe and taxonomy', 'Historical borrow availability assumed',
                                'Top-of-book fill simulation and taker fee scenario; no proven live capacity',
                                'Historical periods already research-exposed', 'No additional sealed outcomes opened'])
    atomic(output / 'summary.json', summary)
    content = ['<!doctype html><meta charset="utf-8"><title>Closing momentum execution validation</title>',
               '<style>body{font:16px system-ui;max-width:1400px;margin:35px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}</style>',
               '<h1>Closing momentum execution validation</h1>',
               '<p><strong>' + html.escape(decision['status']) + '</strong>: ' + html.escape(decision['reason']) + '</p>',
               '<p>Frozen S10 signals. $100,000 primary; IBKR tiered commission scenario. Baseline fractional execution and one whole-share adaptation are distinct. Historical data is research-exposed.</p>',
               '<img src="execution_validation.png">', '<table><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in (ledger[0] if ledger else [])) + '</tr>']
    for row in ledger:
        content.append('<tr>' + ''.join('<td>' + html.escape(f'{v:.5g}' if isinstance(v, float) else str(v)) + '</td>' for v in row.values()) + '</tr>')
    content.append('</table><p>' + '; '.join(summary['limitations']) + '</p>')
    (output / 'report.html').write_text('\n'.join(content), encoding='utf-8')
    return summary
