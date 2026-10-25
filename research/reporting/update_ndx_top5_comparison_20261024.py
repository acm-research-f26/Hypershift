"""Consolidate the audited fixed top-five follow-up without altering prior rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
BEFORE = ROOT/'.cache/strategy_top5_20261024_previous'
STUDY = ROOT/'docs/ndx_extended_hours_top5_20261024'
DOC = ROOT/'docs/strategy_comparison_20261024'
BOOK = ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')
def read(p): return pd.read_csv(p, dtype={'seed': str})


def assemble():
    publication = json.loads((ROOT/'runs/ndx_extended_hours_top5_20261024/PUBLICATION.json').read_text())
    for name, h in publication['artifact_hashes'].items(): assert sha(STUDY/name) == h, name
    assert json.loads((STUDY/'AUDIT.json').read_text())['status'] == 'passed'
    previous = read(BEFORE/'all_strategies.csv'); catalog = read(BEFORE/'chart_catalog.csv')
    metrics = read(STUDY/'metrics.csv'); bounds = read(STUDY/'sharpe_intervals.csv')
    curve = read(STUDY/'curves.csv'); additions = []; details = []; new_charts = []
    for row in metrics.to_dict('records'):
        view, group, seed, k, cost = [row[n] for n in ['view', 'group', 'seed', 'k', 'cost_multiplier']]
        if k == 0:
            canonical = f'original/{view}/10' if cost == 1 else f'extended_detail/{view}/ten_seed/ensemble/{cost}'
        elif k == 10:
            if group == 'individual': canonical = f'extended_detail/{view}/individual/{seed}/1'
            elif cost != 1: canonical = f'extended_detail/{view}/{group}/ensemble/{cost}'
            elif view == 'daily' and group == 'original_five': canonical = 'original/think_mix/10'
            else: canonical = f'extended/{view}/{group}'
        elif cost == 1 and seed == 'ensemble':
            canonical = 'original/think_mix/5' if view == 'daily' and group == 'original_five' else f'holdings5/{view}/{group}'
        else: canonical = 'top5_detail/'+row['id']
        ci = bounds[(bounds.id == row['id'])&(bounds.block_sessions == 20)].iloc[0]
        title = ('Daily + AH + PM' if view == 'after_hours_premarket' else 'Daily') if k else view
        label = ('THINK + mixer / '+title+' / '+group.replace('_', ' ')+' / top '+str(k)) if seed == 'ensemble' and k else title+' / seed '+seed+' / top '+str(k) if k else view
        item = dict(id=canonical, source_id=row['id'], strategy=label, view=view, group=group, seed=seed, k=k,
            cost_multiplier=cost, sharpe=row['sharpe'], cagr=row['annualized_return'], rankic=None if pd.isna(row['rankic']) else row['rankic'],
            max_drawdown=row['maximum_drawdown'], sr_low=float(ci.sr_low), sr_high=float(ci.sr_high))
        details.append(item)
        if canonical.startswith('holdings5/'):
            additions.append({n: item[n] for n in ['id', 'strategy', 'k', 'sharpe', 'cagr', 'rankic', 'max_drawdown', 'sr_low', 'sr_high']} | dict(study='AH+PM holding counts', policy='full', section='Portfolio variants'))
        if canonical not in set(catalog.id): new_charts.append(item)
    assert len(additions) == 5 and len(details) == 62 and len(new_charts) == 27
    combined = pd.concat([previous, pd.DataFrame(additions)], ignore_index=True)
    assert len(combined) == 280 and combined.id.is_unique
    return combined, additions, details, new_charts, curve, catalog


def prepare():
    combined, additions, details, new_charts, curve, catalog = assemble()
    combined.to_csv(OUT/'all_strategies.csv', index=False)
    pd.DataFrame(details).to_csv(OUT/'top5_all_accounts.csv', index=False)
    payload = json.loads((BEFORE/'workbook_data.json').read_text())
    payload['sheets'][2]['rows'].extend(additions); payload['top5_accounts'] = details
    dump(OUT/'top5_workbook_data.json', dict(additions=additions, details=details,
        input_workbook=str(BEFORE/'workbook.xlsx'), input_sha256=sha(BEFORE/'workbook.xlsx')))
    payload['methodology'].extend([
        ['AH+PM holding counts', 'Saved forecasts tested at five and ten holdings, same execution and costs. The original-five AH+PM ensemble returned 14.51% CAGR at top five versus 22.39% at top ten. No new training.'],
        ['Top-five follow-up detail', 'AH PM holding counts contains all 62 accounts, including daily controls, seed groups, individual seeds and costs. Five new distinct base-cost ensembles are appended to Portfolio variants. See the master charts and audited top-five report.'],
    ])
    dump(OUT/'workbook_data.json', payload)
    prior_ci = read(BEFORE/'sharpe_block_sensitivity.csv'); new_ci = read(STUDY/'sharpe_intervals.csv'); extra = []
    for row in additions:
        source = next(d['source_id'] for d in details if d['id'] == row['id'])
        for c in new_ci[new_ci.id == source].to_dict('records'): extra.append(dict(id=row['id'], block_sessions=c['block_sessions'], sr_low=c['sr_low'], sr_high=c['sr_high']))
    pd.concat([prior_ci, pd.DataFrame(extra)], ignore_index=True).to_csv(OUT/'sharpe_block_sensitivity.csv', index=False)
    with np.load(BEFORE/'analysis_paths.npz') as f:
        ids = f['ids'].copy(); dates = f['dates'].copy(); returns = f['returns'].copy()
    columns = []
    for row in additions:
        source = next(d['source_id'] for d in details if d['id'] == row['id']); p = curve[curve.id == source]
        np.testing.assert_array_equal(p.date, dates); columns.append(p.net.to_numpy())
    np.savez_compressed(OUT/'analysis_paths.npz', dates=dates, ids=np.r_[ids, np.array([r['id'] for r in additions])], returns=np.column_stack([returns, *columns]))
    print('Prepared five new main strategies and all 62 follow-up accounts', flush=True)


def charts():
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    combined, additions, details, new_charts, curve, catalog = assemble()
    source = read(STUDY/'metrics.csv'); paths = {name: p for name, p in curve.groupby('id', sort=False)}
    benchmark = {name: paths[f'{name}/benchmark/ensemble/0/1'].net.to_numpy() for name in ['SPY', 'QQQ']}
    catalog['figure_path'] = 'docs/strategy_comparison_20261024/figures/'+catalog.figure
    entries = []; images = []
    for start in range(0, 27, 9):
        items = new_charts[start:start+9]; fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
        filename = f'top5_followup_{start//9+1:02d}.png'
        for ax, row in zip(axes.flat, items):
            p = paths[row['source_id']]; x = pd.to_datetime(p.date)
            ax.plot(x, np.cumprod(1+p.net.to_numpy()), color='#164a70', lw=1.5)
            for name, color in [('SPY', '#cc5555'), ('QQQ', '#8873aa')]: ax.plot(x, np.cumprod(1+benchmark[name]), color=color, lw=.9, alpha=.8)
            ax.set_title(textwrap.fill(row['strategy']+f" / {row['cost_multiplier']}x fees", 52)+f"\nCAGR {row['cagr']:.2%}   SR {row['sharpe']:.3f}   DD {row['max_drawdown']:.1%}", fontsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.grid(alpha=.18); ax.tick_params(labelsize=8)
            entries.append(dict(id=row['id'], strategy=row['strategy'], figure=filename, section='AH+PM holding-count follow-up', figure_path='docs/strategy_comparison_20261024/figures/'+filename))
        fig.suptitle('Five-holding follow-up — blue: strategy; red: SPY; purple: QQQ (base costs)', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .965)); fig.savefig(DOC/'figures'/filename, dpi=140); plt.close(fig); images.append(filename)
    updated_catalog = pd.concat([catalog, pd.DataFrame(entries)], ignore_index=True)
    assert len(updated_catalog) == 356 and updated_catalog.id.is_unique
    updated_catalog.to_csv(DOC/'chart_catalog.csv', index=False)
    markdown = (BEFORE/'charts.md').read_text(encoding='utf-8')
    section = ['## New test: after-hours + premarket with five holdings', '',
        '**The requested original-five ensemble performed worse with five holdings:** 14.51% CAGR, 0.485 Sharpe and −43.13% drawdown versus 22.39%, 0.700 and −35.01% at ten holdings. The fixed ten-seed ensemble improved slightly; the new-five group deteriorated. No forecasts were retrained.', '',
        '[Audited five-versus-ten report](docs/ndx_extended_hours_top5_20261024/REPORT.md) · [All 62 follow-up accounts](docs/ndx_extended_hours_top5_20261024/metrics.csv) · [Paired uncertainty](docs/ndx_extended_hours_top5_20261024/paired_intervals.csv)', '',
        'The workbook now contains **280 distinct base-cost strategies**, plus the complete follow-up in **AH PM holding counts**. The five new main rows are appended to **Portfolio variants**. This supersedes the earlier statement that AH+PM had only been tested at ten holdings.', '']
    for group in ['original_five', 'ten_seed', 'new_five']: section += [f'![Five versus ten: {group}](docs/ndx_extended_hours_top5_20261024/figures/{group}.png)', '']
    markdown = markdown.replace('## Best historical results', '\n'.join(section)+'\n## Best historical results', 1)
    markdown = markdown.replace('275 distinct base-cost strategies; 72 Models, 31 Defensive and 172 Portfolio variants.', '280 distinct base-cost strategies; 72 Models, 31 Defensive and 177 Portfolio variants.')
    markdown = markdown.replace('one chart-panel entry for each of 275 main strategies and 54 additional extended-hours diagnostic accounts. Twelve reference accounts are aliases already represented in the main gallery.', '356 chart-panel entries covering 280 main strategies and 76 additional diagnostic accounts. Repeated reference accounts are represented once.')
    markdown += '\n### All additional five-holding follow-up accounts\n\n'
    markdown += '\n\n'.join(f'![Five-holding follow-up](docs/strategy_comparison_20261024/figures/{name})' for name in images)+'\n'
    (ROOT/'charts.md').write_text(markdown, encoding='utf-8')
    comparison = (OUT/'COMPARISON.md').read_text(encoding='utf-8')
    marker = '## Five-holding after-hours follow-up'
    if marker in comparison: comparison = comparison.split(marker)[0].rstrip()+'\n'
    comparison += '\n'+marker+'\n\n'
    comparison += 'Five new base-cost ensemble portfolios are included in the workbook and main CSV. All 62 matched accounts are in top5_all_accounts.csv. See [the audited report](../../../docs/ndx_extended_hours_top5_20261024/REPORT.md).\n\n'
    comparison += '| Strategy | CAGR | Sharpe | Max drawdown |\n| --- | ---: | ---: | ---: |\n'
    for row in additions: comparison += f"| {row['strategy']} | {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} |\n"
    (OUT/'COMPARISON.md').write_text(comparison, encoding='utf-8')
    print('Updated master charts: 280 main strategies and 76 additional diagnostics', flush=True)


def cells(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}; result = {}
    with zipfile.ZipFile(path) as z:
        strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        names = [n.attrib['name'] for n in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', ns)]
        for i, name in enumerate(names, 1):
            values = {}
            for c in ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).findall('s:sheetData/s:row/s:c', ns):
                v = c.find('s:v', ns); t = c.attrib.get('t')
                value = strings[int(v.text)] if t == 's' else ''.join(c.find('s:is', ns).itertext()) if t == 'inlineStr' else float(v.text) if v is not None and t not in ['str', 'e'] else v.text if v is not None else None
                values[c.attrib['r']] = value
            result[name] = values
    return result


def audit():
    combined, additions, details, new_charts, curve, catalog = assemble()
    before = cells(BEFORE/'workbook.xlsx'); after = cells(BOOK)
    for sheet, values in before.items():
        for address, value in values.items(): assert after[sheet].get(address) == value, (sheet, address)
    original_rows = len(json.loads((BEFORE/'workbook_data.json').read_text())['sheets'][2]['rows'])
    for n, row in enumerate(additions, original_rows+6):
        expected = [row['strategy'], row['study'], row['sharpe'], row['cagr'], row['rankic'], row['max_drawdown'], row['sr_low'], row['sr_high']]
        for col, value in zip('ABCDEFGH', expected):
            actual = after['Portfolio variants'][f'{col}{n}']
            if isinstance(value, str): assert actual == value
            else: np.testing.assert_allclose(actual, value, atol=1e-12, rtol=1e-12)
    for n, row in enumerate(details, 6):
        expected = [row['strategy'], row['group'], row['k'], row['cost_multiplier'], row['seed'], row['sharpe'], row['cagr'], 'n.a.' if row['rankic'] is None else row['rankic'], row['max_drawdown'], row['sr_low'], row['sr_high']]
        for col, value in zip('ABCDEFGHIJK', expected):
            actual = after['AH PM holding counts'][f'{col}{n}']
            if isinstance(value, str): assert actual == value, (n, col)
            else: np.testing.assert_allclose(actual, value, atol=1e-12, rtol=1e-12)
    current = read(OUT/'all_strategies.csv'); assert len(current) == 280
    with np.load(OUT/'analysis_paths.npz') as f:
        assert len(f['ids']) == 280 and f['returns'].shape == (1003, 280)
    markdown = (ROOT/'charts.md').read_text(encoding='utf-8')
    for p in re.findall(r'\]\(([^)]+)\)', markdown): assert (ROOT/p).exists(), p
    assert len(read(DOC/'chart_catalog.csv')) == 356
    proof = dict(status='passed', study='ndx_extended_hours_top5_20261024', main_strategies=280,
        workbook_sheets=len(after), prior_workbook_cells_preserved=True, added_main_rows_verified=5,
        followup_account_rows_verified=62, workbook_comparison_rows_verified=412, chart_catalog_entries=356,
        inherited_account_audit=json.loads((STUDY/'AUDIT.json').read_text()),
        analysis_source_sha256=sha(__file__), workbook_builder_sha256=sha(OUT/'build_workbook.mjs'),
        output_sha256={str(p.relative_to(ROOT)): sha(p) for p in [BOOK, OUT/'all_strategies.csv', ROOT/'charts.md', DOC/'chart_catalog.csv']})
    dump(OUT/'AUDIT.json', proof); dump(DOC/'AUDIT.json', proof)
    record = json.loads((BEFORE/'ANALYSIS.json').read_text()); record.update(status='completed_audited', strategies=280,
        workbook=str(BOOK.relative_to(ROOT)), workbook_sheets=6, top5_followup_accounts=62, additional_diagnostic_accounts=76,
        sections={'Models':72, 'Defensive':31, 'Portfolio variants':177}, chart_panels=356, workbook_comparison_rows=412,
        top5_followup_source=str(Path(__file__).name), top5_followup_source_sha256=sha(__file__))
    record.setdefault('source_hashes', {}).update({str(p.relative_to(ROOT)): sha(p) for p in [STUDY/'metrics.csv', STUDY/'AUDIT.json', STUDY/'PUBLICATION.json'] if p.exists()})
    dump(OUT/'ANALYSIS.json', record)
    manifest = {p.name: sha(p) for p in (DOC/'figures').glob('*.png')}; dump(DOC/'FIGURE_MANIFEST.json', manifest)
    print(json.dumps({k:v for k,v in proof.items() if k not in ['output_sha256', 'inherited_account_audit']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'charts', 'audit'])
    globals()[parser.parse_args().stage]()
