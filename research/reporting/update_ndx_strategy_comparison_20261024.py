"""Extend the audited comparison and chart every active strategy; no model fitting.

Run with the bundled document Python. The prior workbook is preserved in .cache;
the existing Artifact Tool builder imports it and appends only authorized content.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
PRIOR = ROOT/'.cache/strategy_comparison_20261024_previous'
DOC = ROOT/'docs/strategy_comparison_20261024'
EXT = ROOT/'docs/ndx_extended_hours_20261024'
BOOK = ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'
VIEWS = dict(daily='Daily', after_hours='Daily + AH', after_hours_premarket='Daily + AH + PM',
             SPY='SPY', QQQ='QQQ', monthly_basket='Monthly Nasdaq-100 basket')
GROUPS = dict(ten_seed='ten seeds', original_five='original five seeds', new_five='new five seeds')
SHORTLIST = [
    ('Highest CAGR across base-cost strategies', 'original/stockmixer/5'),
    ('Highest Sharpe across base-cost strategies', 'signed/stockmixer/5/long75_short25'),
    ('Highest CAGR among standard top-ten ensembles', 'extended/after_hours_premarket/original_five'),
    ('Original daily reference', 'original/think_mix/10'),
    ('Highest defensive-policy Sharpe', 'ndx_broad_20261020/reference_original_think_mix/trend_half'),
    ('New extended-hours primary ten-seed blend', 'extended/after_hours_premarket/ten_seed'),
]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return pd.read_csv(p, dtype={'seed': str})
def dump(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')


def metrics(a):
    a = list(map(float, a)); n = len(a); wealth = peak = 1.; dd = 0.
    for r in a:
        wealth *= 1+r; peak = max(peak, wealth); dd = min(dd, wealth/peak-1)
    return dict(sharpe=math.sqrt(252)*(math.fsum(a)/n-math.expm1(math.log1p(.03)/252))/statistics.stdev(a),
                cagr=math.expm1(math.fsum(map(math.log1p, a))*252/n), max_drawdown=dd)


def check_metrics(a, row):
    for name, value in metrics(a).items():
        np.testing.assert_allclose(value, row[name], atol=1e-10, rtol=1e-10)


def load():
    old = json.loads((PRIOR/'ANALYSIS.json').read_text())
    audit = json.loads((PRIOR/'AUDIT.json').read_text())
    assert old['status'] == 'completed_audited' and audit['status'] == 'passed'
    for p, h in audit['output_sha256'].items(): assert sha(PRIOR/p) == h, p
    for p, h in old['source_hashes'].items(): assert sha(ROOT/p) == h, p
    publication = json.loads((ROOT/'runs/ndx_extended_hours_20261024/FINAL_PUBLICATION.json').read_text())
    for p, h in publication['artifact_hashes'].items(): assert sha(EXT/p) == h, p
    assert json.loads((EXT/'AUDIT.json').read_text())['status'] == 'passed'
    rows = read(PRIOR/'all_strategies.csv')
    with np.load(PRIOR/'analysis_paths.npz') as f:
        dates = f['dates'].copy(); paths = dict(zip(f['ids'], f['returns'].T))
    assert len(rows) == 267 and len(dates) == 1003
    for row in rows.to_dict('records'): check_metrics(paths[row['id']], row)
    m, c, rank, quality, intervals = [read(EXT/p) for p in [
        'metrics.csv', 'curves.csv', 'ranking.csv', 'prediction_quality.csv', 'sharpe_intervals.csv']]
    detail = []; detail_paths = {}; additions = []; aliases = {}
    for row in m.to_dict('records'):
        view, group, seed, cost = [row[k] for k in ['view', 'group', 'seed', 'cost_multiplier']]
        key = f"extended_detail/{view}/{group}/{seed}/{cost}"
        p = c[(c['view'] == view) & (c.group == group) & (c.seed == seed) & (c.cost_multiplier == cost)]
        np.testing.assert_array_equal(p.date.to_numpy(), dates)
        values = p.net.to_numpy(); actual = metrics(values)
        for field, original in [('sharpe', 'sharpe'), ('cagr', 'annualized_return'), ('max_drawdown', 'maximum_drawdown')]:
            np.testing.assert_allclose(actual[field], row[original], atol=1e-10, rtol=1e-10)
        ci = intervals[(intervals['view'] == view) & (intervals.group == group) & (intervals.seed == seed)
                       & (intervals.cost_multiplier == cost) & (intervals.block_sessions == 20)].iloc[0]
        ic = None
        if view in ['daily', 'after_hours', 'after_hours_premarket']:
            source = quality[(quality['view'] == view) & (quality.seed == seed)] if group == 'individual' else rank[(rank['view'] == view) & (rank.group == group)]
            assert len(source) == 1; ic = float(source.iloc[0].rankic)
        strategy = VIEWS[view] if view in ['SPY', 'QQQ', 'monthly_basket'] else 'THINK + mixer / '+VIEWS[view]+' / '+(f'seed {seed}' if group == 'individual' else GROUPS[group])
        role = 'Individual-seed diagnostic' if group == 'individual' else 'Cost sensitivity' if cost != 1 else 'Benchmark' if ic is None else 'Primary ten-seed ensemble' if group == 'ten_seed' else 'Seed-subgroup diagnostic'
        item = dict(id=key, strategy=strategy, view=view, group=group, seed=seed, cost_multiplier=cost,
                    role=role, rankic=ic, sr_low=float(ci.sr_low), sr_high=float(ci.sr_high), **actual)
        detail.append(item); detail_paths[key] = values
        if cost == 1 and group != 'individual':
            if view in ['SPY', 'QQQ', 'monthly_basket']: alias = f'original/{view}/10'
            elif view == 'daily' and group == 'original_five': alias = 'original/think_mix/10'
            else:
                alias = f'extended/{view}/{group}'
                additions.append(dict(id=alias, study='Extended-hours information', strategy=strategy,
                                      policy='full', k=10, section='Models', rankic=ic, **actual,
                                      sr_low=float(ci.sr_low), sr_high=float(ci.sr_high)))
                paths[alias] = values
            aliases[key] = alias
            np.testing.assert_allclose(paths[alias], values, atol=1e-14, rtol=0)
    assert len(additions) == 8 and len(detail) == 66 and len(aliases) == 12
    combined = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    assert len(combined) == 275 and not combined.id.duplicated().any()
    # Materialize common-date resamples independently of the optimized source helper.
    candidates = pd.DataFrame(additions).id.tolist()
    matrix = np.column_stack([paths[k] for k in candidates]); n = len(matrix); rng = np.random.default_rng(20261040)
    draws = []
    for _ in range(100):
        starts = rng.integers(0, n, (100, int(np.ceil(n/20))))
        indices = ((starts[:, :, None]+np.arange(20)) % n).reshape(100, -1)[:, :n]
        sample = matrix[indices]
        draws.append(np.sqrt(252)*(sample.mean(1)-((1.03)**(1/252)-1))/sample.std(1, ddof=1))
    bounds = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    for j, key in enumerate(candidates):
        row = combined[combined.id == key].iloc[0]
        np.testing.assert_allclose(bounds[:, j], [row.sr_low, row.sr_high], atol=1e-11, rtol=0)
    return combined, paths, dates, detail, detail_paths, aliases, additions, old


def prepare():
    DOC.mkdir(parents=True, exist_ok=True)
    rows, paths, dates, detail, detail_paths, aliases, additions, old = load()
    rows.to_csv(OUT/'all_strategies.csv', index=False)
    pd.DataFrame(detail).to_csv(OUT/'extended_hours_all_accounts.csv', index=False)
    payload = json.loads((PRIOR/'workbook_data.json').read_text())
    payload['sheets'][0]['rows'].extend(additions)
    payload['extended_hours'] = detail
    payload['updated_date'] = '2026-10-24'
    payload['methodology'][1][1] = 'Net returns at base modeled costs in the first three tabs. The Extended-hours tab also discloses individual seeds and 2x/4x fee stresses.'
    payload['methodology'][12][1] = 'Main tabs omit reference aliases, individual seeds, fee stresses and seed-bootstrap accounts. All 66 extended-hours accounts are separately disclosed. Original-five daily and the three base-cost benchmarks reuse existing main-table rows.'
    payload['methodology'][17][1] += ' Extended-hours and premarket information study (October 24) is included.'
    payload['methodology'].extend([
        ['Extended-hours clock', 'AH: prior regular close through 20:00 ET. PM: next exchange session 04:00–09:15 ET. All views decide at 09:20 ET and fill at that session’s regular close.'],
        ['Extended-hours seeds', 'Primary ten seeds are fixed before metrics. Original-five and new-five ensembles and all individual seeds are diagnostics. The original five daily forecasts reproduce the earlier model.'],
        ['Extended-hours limits', 'About 50% of member sessions have at least four AH bars and 46% have at least four PM bars. Missing activity is flagged. PM prior-close price gaps were excluded to avoid raw corporate-action discontinuities.'],
        ['Current leaders', 'Highest base-cost CAGR: StockMixer top 5. Highest base-cost Sharpe: StockMixer top 5, 75% long / 25% short. Highest standard top-10 ensemble CAGR: original-five THINK + mixer with AH + PM. These are historical rankings, not forward selection rules.'],
        ['Seed evidence', 'AH + PM improved 5/10 individual seeds. Its original-five CAGR rose from 19.60% to 22.39%, but the new-five group fell from 15.51% to 13.69%. Primary paired intervals include zero.'],
        ['Sources and charts', 'See root charts.md for every main strategy and all additional extended-hours seed/cost accounts, source report links and paired uncertainty. Updated analysis: update_ndx_strategy_comparison_20261024.py.'],
    ])
    dump(OUT/'workbook_data.json', payload)
    sensitivity = read(PRIOR/'sharpe_block_sensitivity.csv')
    ext_ci = read(EXT/'sharpe_intervals.csv'); extra = []
    for item in additions:
        _, view, group = item['id'].split('/')
        part = ext_ci[(ext_ci['view'] == view) & (ext_ci.group == group) & (ext_ci.cost_multiplier == 1)]
        for r in part.to_dict('records'): extra.append(dict(id=item['id'], block_sessions=r['block_sessions'], sr_low=r['sr_low'], sr_high=r['sr_high']))
    pd.concat([sensitivity, pd.DataFrame(extra)], ignore_index=True).to_csv(OUT/'sharpe_block_sensitivity.csv', index=False)
    np.savez_compressed(OUT/'analysis_paths.npz', dates=dates, ids=rows.id.to_numpy(dtype=str), returns=np.column_stack([paths[k] for k in rows.id]))
    sources = dict(old['source_hashes'])
    for name in ['metrics.csv', 'curves.csv', 'ranking.csv', 'prediction_quality.csv', 'sharpe_intervals.csv', 'AUDIT.json', 'ten_seed_intervals.csv', 'REPORT.md']:
        p = EXT/name; sources[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'ANALYSIS.json', dict(status='updated_pending_workbook_audit', updated_local_date='2026-10-24',
        strategies=275, daily_observations=1003, extended_hours_accounts=66, additional_detail_accounts=54,
        source_hashes=sources, analysis_source_sha256=sha(__file__), prior_audit_sha256=sha(PRIOR/'AUDIT.json'),
        prior_workbook_sha256=sha(PRIOR/'NASDAQ100_strategy_comparison.xlsx'), source_policy='Existing 267 strategies preserved; eight new base-cost ensembles appended; all 66 extended-hours accounts disclosed separately.'))
    print('Prepared 275 main strategies, 66 extended-hours detail rows, eight independent 10,000-sample SR checks', flush=True)


def table(frame, criteria=False):
    out = ['| '+('Historical distinction | ' if criteria else '')+'Strategy | Sharpe | CAGR | RankIC | Max drawdown | 95% SR interval |',
           '| '+('--- | ' if criteria else '')+'--- | ---: | ---: | ---: | ---: | --- |']
    for row in frame:
        ic = 'n.a.' if row['rankic'] is None or pd.isna(row['rankic']) else f"{row['rankic']:.4f}"
        out.append('| '+(row['criterion']+' | ' if criteria else '')+f"{row['strategy']} | {row['sharpe']:.3f} | {row['cagr']:.2%} | {ic} | {row['max_drawdown']:.2%} | [{row['sr_low']:.3f}, {row['sr_high']:.3f}] |")
    return out


def charts():
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    rows, paths, dates, detail, detail_paths, aliases, additions, old = load()
    figures = DOC/'figures'; figures.mkdir(parents=True, exist_ok=True)
    x = pd.to_datetime(dates); spy = paths['original/SPY/10']; qqq = paths['original/QQQ/10']
    main = rows.set_index('id'); leaders = [dict(main.loc[key], id=key, criterion=criterion) for criterion, key in SHORTLIST]
    assert main.loc['original/stockmixer/5', 'cagr'] == rows.cagr.max()
    assert main.loc['signed/stockmixer/5/long75_short25', 'sharpe'] == rows.sharpe.max()
    assert main.loc['extended/after_hours_premarket/original_five', 'cagr'] == rows[rows.section == 'Models'].cagr.max()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for item in leaders[:4]+[dict(main.loc['original/SPY/10'], id='original/SPY/10'), dict(main.loc['original/QQQ/10'], id='original/QQQ/10')]:
        a = paths[item['id']]; w = np.r_[1., np.cumprod(1+a)]; drawdown = (w/np.maximum.accumulate(w)-1)[1:]
        line, = axes[0].plot(x, w[1:], label=f"{item['strategy']} ({item['cagr']:.1%} CAGR)", lw=1.4)
        axes[1].plot(x, drawdown*100, color=line.get_color(), lw=1)
    axes[0].set(title='Historical leaders and benchmarks, 2022–2025', ylabel='Net wealth per initial $1')
    axes[0].legend(loc='upper left', fontsize=8); axes[1].set(ylabel='Drawdown (%)')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures/'historical_leaders.png', dpi=150); plt.close(fig)
    pages = []; catalog = []
    def gallery(items, source_paths, heading, prefix):
        for start in range(0, len(items), 9):
            selected = items[start:start+9]; grid_rows = math.ceil(len(selected)/3)
            fig, axes = plt.subplots(grid_rows, 3, figsize=(18, 4.3*grid_rows+.4), sharex=True, squeeze=False)
            for ax, item in zip(axes.flat, selected):
                for label, a, color, width in [(item['strategy'], source_paths[item['id']], '#164a70', 1.6), ('SPY', spy, '#cc5555', .9), ('QQQ', qqq, '#8873aa', .9)]:
                    ax.plot(x, np.cumprod(1+a), color=color, lw=width, alpha=1 if width>1 else .8)
                label = item['strategy']+(f" / {item['cost_multiplier']}x fees" if 'cost_multiplier' in item else '')
                ax.set_title(textwrap.fill(label, 52)+f"\nCAGR {item['cagr']:.2%}   SR {item['sharpe']:.3f}   DD {item['max_drawdown']:.1%}", fontsize=9)
                ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.grid(alpha=.18); ax.tick_params(labelsize=8)
            for ax in list(axes.flat)[len(selected):]: ax.set_visible(False)
            filename = f'{prefix}_{start//9+1:02d}.png'
            fig.suptitle(heading+' — blue: strategy, red: SPY, purple: QQQ; net wealth per $1', fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .965)); fig.savefig(figures/filename, dpi=140); plt.close(fig)
            pages.append((heading, filename, len(selected)))
            for item in selected: catalog.append(dict(id=item['id'], strategy=item['strategy'], figure=filename, section=heading))
        print('Charted', heading, len(items), flush=True)
    for section in ['Models', 'Defensive', 'Portfolio variants']:
        for study, part in rows[rows.section == section].groupby('study', sort=False):
            prefix = re.sub('[^a-z0-9]+', '_', (section+' '+study).lower()).strip('_')
            gallery(part.to_dict('records'), paths, section+' / '+study, prefix)
    diagnostics = [item for item in detail if item['id'] not in aliases]
    for role in ['Individual-seed diagnostic', 'Cost sensitivity']:
        part = [item for item in diagnostics if item['role'] == role]
        gallery(part, detail_paths, 'Extended-hours / '+role, role.lower().replace('-', '_').replace(' ', '_'))
    assert len(catalog) == 329 and len({r['id'] for r in catalog}) == 329
    pd.DataFrame(catalog).to_csv(DOC/'chart_catalog.csv', index=False)
    intro = ['# NASDAQ-100 master charts', '', 'Updated October 24, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. All charts use the same 1,003 sessions.', '',
        '## Best historical results', '', 'These distinctions rank the completed base-cost strategy results after observing this history. They are not a validation-selected shortlist or evidence of reliable future superiority.', '']
    intro += table(leaders, True)
    intro += ['', 'StockMixer top five has the highest CAGR across the consolidated base-cost strategies. The 75% long / 25% short StockMixer version has the highest Sharpe, with 3% annual short borrowing and assumed borrow availability. Neither short availability nor concentration risk should be confused with an improvement in stock predictions.', '',
        'Among standard full-exposure top-ten ensembles, the original-five THINK + mixer with after-hours and premarket leads at 22.39% CAGR. Its new-five counterpart returned 13.69% versus 15.51% for daily inputs. The primary fixed ten-seed comparison was 15.36% versus 14.54%, improving only 5/10 individual seeds. The paired intervals include zero. The original daily five-seed ensemble remains the established reference, although its 25-seed expansion returned 12.86%. No model has established reliable future superiority over SPY/QQQ.', '',
        'The SR intervals use a fixed 3% annual hurdle and 10,000 circular-block resamples, with 20 sessions primary. They condition on fitted paths, omit the full research search and retraining uncertainty, and do not measure the probability of beating a benchmark. These years have been repeatedly inspected.', '',
        '![Historical leaders](docs/strategy_comparison_20261024/figures/historical_leaders.png)', '',
        '## Files and scope', '',
        '- [Updated comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx): 275 distinct base-cost strategies; 72 Models, 31 Defensive and 172 Portfolio variants. Benchmarks repeat in the latter tabs for convenience.',
        '- [Every extended-hours account](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/extended_hours_all_accounts.csv): all 66 accounts, including individual seeds and 2x/4x fees, also in the new workbook tab.',
        '- [Chart catalog](docs/strategy_comparison_20261024/chart_catalog.csv): one chart-panel entry for each of 275 main strategies and 54 additional extended-hours diagnostic accounts. Twelve reference accounts are aliases already represented in the main gallery.',
        '- [All strategy metrics](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/all_strategies.csv) and [5/20/60-session SR sensitivity](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/sharpe_block_sensitivity.csv).', '',
        'Defensive uses the fixed QQQ/MA200 half-exposure rule. Portfolio variants change holding count or long/cash/short allocation. All main rows are ensemble or declared selection procedures; individual extended-hours seeds are diagnostics. Failed and negative strategies are retained.', '',
        'For fee-stress charts the red/purple comparators remain the base-cost SPY/QQQ references; the strategy fee multiplier is stated in each panel. No new models were fit for this consolidation. Retired 2026-selected survivor universes and unrelated/deferred NYSE studies are excluded from this active-model ranking.', '',
        '## Source reports and uncertainty', '',
        '- [Original monthly study](docs/nasdaq_monthly_pit/CHARTS.md)',
        '- [Original holding-count/loss comparison](docs/nasdaq_monthly_pit/topk_pseudohuber/metrics.csv)',
        '- [Original long/short allocations](docs/nasdaq_monthly_pit/long_short/primary_metrics.csv)',
        '- [Alpaca regular-session features](docs/nasdaq_alpaca_20261019/REPORT.md)',
        '- [Ranking and residual objectives](docs/ndx_ranking_20261020/REPORT.md)',
        '- [Broad features and model families](docs/ndx_broad_20261020/REPORT.md)',
        '- [Frozen-core corrections](docs/ndx_corrections_20261020/REPORT.md)',
        '- [25-seed expansion](docs/ndx_seed_expansion_20261020/REPORT.md)',
        '- [After-hours and premarket experiment](docs/ndx_extended_hours_20261024/REPORT.md)', '',
        '![Paired information-increment uncertainty](docs/ndx_extended_hours_20261024/figures/information_intervals.png)', '',
        '## Complete model and strategy gallery', '', 'Each panel compares one strategy with SPY and QQQ. Titles report net CAGR, Sharpe and maximum drawdown. All declared main strategies appear, including equal realized paths from distinct selection rules. Numeric results and unique IDs are searchable in the linked catalog.', '']
    previous = None
    for heading, filename, count in pages:
        if heading != previous: intro += ['### '+heading, '']; previous = heading
        intro += [f'![{heading}, {count} strategies](docs/strategy_comparison_20261024/figures/{filename})', '']
    (ROOT/'charts.md').write_text('\n'.join(intro), encoding='utf-8')
    # Human-readable companion keeps all detail without modifying frozen source reports.
    lines = ['# NASDAQ-100 strategy comparison, updated October 24, 2026', '', 'All financial metrics use the same 2022–2025 dates. Main strategies are at base modeled costs. See [master charts](../../../charts.md).', '']
    for section in ['Models', 'Defensive', 'Portfolio variants']:
        lines += ['## '+section, '']+table(rows[rows.section == section].to_dict('records'))+['']
    lines += ['## Extended-hours individual seeds and fee stresses', '', 'All 66 accounts, including reference aliases, are in extended_hours_all_accounts.csv and the workbook Extended-hours tab.', '']
    (OUT/'COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Created master charts.md with', len(catalog), 'panels in', len(pages), 'gallery figures', flush=True)


def audit():
    payload = json.loads((OUT/'workbook_data.json').read_text()); prior = json.loads((PRIOR/'workbook_data.json').read_text())
    for j in range(3): assert payload['sheets'][j]['rows'][:len(prior['sheets'][j]['rows'])] == prior['sheets'][j]['rows']
    expected = {}
    def values(r): return [r['strategy'], r['study'], r['sharpe'], r['cagr'], 'n.a.' if r['rankic'] is None else r['rankic'], r['max_drawdown'], r['sr_low'], r['sr_high']]
    for section in payload['sheets']: expected[section['name']] = [values(r) for r in section['rows']]
    expected['Extended-hours'] = [[r['strategy'], r['role'], r['seed'], r['cost_multiplier'], r['sharpe'], r['cagr'], 'n.a.' if r['rankic'] is None else r['rankic'], r['max_drawdown'], r['sr_low'], r['sr_high']] for r in payload['extended_hours']]
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    def contents(path):
        result = {}
        with zipfile.ZipFile(path) as z:
            shared = []
            if 'xl/sharedStrings.xml' in z.namelist(): shared = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
            names = [s.attrib['name'] for s in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', ns)]
            for j, name in enumerate(names, 1):
                xml = ET.fromstring(z.read(f'xl/worksheets/sheet{j}.xml')); cells = {}
                for c in xml.findall('s:sheetData/s:row/s:c', ns):
                    v = c.find('s:v', ns); t = c.attrib.get('t')
                    value = shared[int(v.text)] if t == 's' else ''.join(c.find('s:is', ns).itertext()) if t == 'inlineStr' else v.text if t in ['str', 'e'] and v is not None else float(v.text) if v is not None else None
                    cells[c.attrib['r']] = (value, t, c.findtext('s:f', None, ns))
                result[name] = (cells, xml)
        return result
    before = contents(PRIOR/'NASDAQ100_strategy_comparison.xlsx'); after = contents(BOOK)
    assert list(after) == ['Models', 'Defensive', 'Portfolio variants', 'Method', 'Extended-hours']
    # Compare effective cell formats rather than volatile exported style indices.
    def native_styles(path):
        with zipfile.ZipFile(path) as z:
            styles = ET.fromstring(z.read('xl/styles.xml'))
            collections = {tag: list(styles.find('s:'+tag, ns)) for tag in ['fonts', 'fills', 'borders', 'cellXfs']}
            custom = {int(e.attrib['numFmtId']): e.attrib['formatCode'] for e in styles.findall('s:numFmts/s:numFmt', ns)}
            def signature(c):
                xf = collections['cellXfs'][int(c.attrib.get('s', 0))]
                result = {key: value for key, value in xf.attrib.items() if key not in ['fontId', 'fillId', 'borderId', 'numFmtId', 'xfId']}
                for attr, group in [('fontId', 'fonts'), ('fillId', 'fills'), ('borderId', 'borders')]:
                    result[attr] = ET.tostring(collections[group][int(xf.attrib.get(attr, 0))])
                number = int(xf.attrib.get('numFmtId', 0)); result['number'] = custom.get(number, number)
                result['alignment'] = [ET.tostring(e) for e in xf]
                return result
            answer = {}
            for index in [1, 2, 3]:
                xml = ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'))
                answer[index] = {c.attrib['r']: signature(c) for c in xml.findall('s:sheetData/s:row/s:c', ns)}
            return answer
    old_styles = native_styles(PRIOR/'NASDAQ100_strategy_comparison.xlsx')
    new_styles = native_styles(BOOK)
    for index in [1, 2, 3]:
        for address, style in old_styles[index].items(): assert style == new_styles[index][address], ('style', index, address)
    with zipfile.ZipFile(PRIOR/'NASDAQ100_strategy_comparison.xlsx') as old_zip, zipfile.ZipFile(BOOK) as new_zip:
        for index in [2, 3]: assert old_zip.read(f'xl/tables/table{index}.xml') == new_zip.read(f'xl/tables/table{index}.xml')
    verified = 0
    for name, matrix in expected.items():
        cells, xml = after[name]
        for rownum, row in enumerate(matrix, 6):
            for col, value in enumerate(row):
                actual, kind, formula = cells[f'{chr(65+col)}{rownum}']
                if isinstance(value, str): assert actual == value, (name, rownum, col, actual, value)
                else:
                    assert kind in [None, 'n']; np.testing.assert_allclose(actual, value, atol=1e-12, rtol=1e-12)
            verified += 1
        assert xml.find('s:sheetViews/s:sheetView/s:pane', ns) is not None
    for name in ['Defensive', 'Portfolio variants']:
        assert before[name][0] == after[name][0], name
    for address, item in before['Models'][0].items():
        if address != 'A3': assert item == after['Models'][0][address], address
    for j, row in enumerate(payload['methodology'], 6):
        for col, value in zip('AB', row): assert after['Method'][0][f'{col}{j}'][0] == value
    rows = read(OUT/'all_strategies.csv'); detail = read(OUT/'extended_hours_all_accounts.csv')
    assert len(rows) == 275 and len(detail) == 66 and verified == 345
    receipt = json.loads((OUT/'ANALYSIS.json').read_text())
    for p, h in receipt['source_hashes'].items(): assert sha(ROOT/p) == h, p
    catalog = read(DOC/'chart_catalog.csv'); assert len(catalog) == 329 and catalog.id.is_unique
    assert set(rows.id).issubset(set(catalog.id))
    from PIL import Image
    figure_hashes = {}
    for p in (DOC/'figures').glob('*.png'):
        assert p.stat().st_size > 1000
        with Image.open(p) as im:
            assert im.width >= 1800 and im.height >= 600
            im.verify()
        figure_hashes[p.name] = sha(p)
    markdown = (ROOT/'charts.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', markdown): assert (ROOT/target).exists(), target
    assert not any(v[1] == 'e' for cells, xml in after.values() for v in cells.values())
    receipt.update(status='completed_audited', analysis_source_sha256=sha(__file__), workbook=str(BOOK.relative_to(ROOT)), workbook_sheets=5,
                   workbook_comparison_rows=verified, chart_panels=329)
    dump(OUT/'ANALYSIS.json', receipt)
    proof = dict(status='passed', updated_local_date='2026-10-24', main_strategies=275, extended_hours_accounts=66,
        workbook_comparison_rows_verified=verified, prior_main_rows_preserved=271, unchanged_allocation_tabs=2,
        source_hashes_verified=len(receipt['source_hashes']), independent_new_sharpe_bootstraps=8,
        bootstrap_replicates=10000, block_sessions=20, scalar_financial_accounts_checked=333,
        chart_panels=329, chart_figures=len(list((DOC/'figures').glob('*.png'))),
        prior_cell_formats_preserved=True, allocation_tables_preserved=True,
        analysis_source_sha256=sha(__file__), workbook_builder_sha256=sha(OUT/'build_workbook.mjs'),
        output_sha256={p.name: sha(p) for p in [BOOK, OUT/'all_strategies.csv', OUT/'extended_hours_all_accounts.csv', OUT/'COMPARISON.md', ROOT/'charts.md']})
    dump(OUT/'AUDIT.json', proof); dump(DOC/'AUDIT.json', proof)
    dump(DOC/'FIGURE_MANIFEST.json', figure_hashes)
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'charts', 'audit'])
    globals()[parser.parse_args().stage]()
