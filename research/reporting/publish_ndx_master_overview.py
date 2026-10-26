"""Plot the complete saved comparison with the historical leader emphasized.

Presentation-only follow-up: validate all return paths against published metrics,
then write a supplementary figure and provenance record. No fits or backtests.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p/'hypershift').is_dir())
SOURCE = ROOT/'research/reporting/publish_ndx_master_overview.py'
DATA = ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
DOC = ROOT/'docs/strategy_comparison_20261024'
OUTPUT = DOC/'overview_20261026'
BEST = 'reference/dynamic_restricted_blend/original_five/top5'
HIGHLIGHTS = [
    (BEST, 'Best model | THINK+mixer + dynamic HG', '#086a9f', '-', 3.2),
    ('original/SPY/10', 'SPY', '#c44832', '--', 2.6),
    ('original/QQQ/10', 'QQQ', '#7544a0', '-', 2.6),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_paths():
    rows = pd.read_csv(DATA/'all_strategies.csv').set_index('id')
    assert rows.index.is_unique
    with np.load(DATA/'analysis_paths.npz') as saved:
        dates = pd.DatetimeIndex(saved['dates'])
        ids = saved['ids'].tolist()
        returns = saved['returns'].copy()
    assert len(rows) == len(ids) == 346 and len(dates) == 1003
    assert len(set(ids)) == len(ids) and set(rows.index) == set(ids)
    assert dates.is_unique and dates.is_monotonic_increasing
    assert str(dates[0].date()) == '2022-01-03' and str(dates[-1].date()) == '2025-12-31'
    assert returns.shape == (len(dates), len(ids))
    assert np.isfinite(returns).all() and (returns > -1).all()
    ordered = rows.loc[ids]
    wealth = np.vstack([np.ones(len(ids)), np.cumprod(1+returns, axis=0)])
    cagr = wealth[-1]**(252/len(dates))-1
    vol = returns.std(axis=0, ddof=1)*np.sqrt(252)
    sharpe = (returns.mean(axis=0)-(1.03**(1/252)-1))*252/vol
    drawdown = (wealth/np.maximum.accumulate(wealth, axis=0)-1).min(axis=0)
    for values, column in [(cagr, 'cagr'), (sharpe, 'sharpe'), (drawdown, 'max_drawdown')]:
        np.testing.assert_allclose(values, ordered[column], atol=1e-10, rtol=1e-10)
    assert rows.sharpe.idxmax() == rows.cagr.idxmax() == BEST
    # Existing audit is a dated publication snapshot; preserve it and its figures.
    audit = json.loads((DOC/'AUDIT.json').read_text())
    assert audit['status'] == 'passed'
    resave = json.loads((ROOT/'docs/audit/workbook_resave_20261025.json').read_text())
    for relative, expected in audit['output_sha256'].items():
        relative = relative.replace('\\', '/')
        if relative == 'charts.md':
            continue
        if relative == 'docs/NASDAQ100_strategy_comparison.xlsx':
            assert resave['status'] == 'passed_semantic_equivalence'
            assert resave['baseline_workbook_sha256'] == expected
            expected = resave['current_workbook_sha256']
        assert sha(ROOT/relative) == expected, relative
    original_figures = json.loads((DOC/'FIGURE_MANIFEST.json').read_text())
    for name, expected in original_figures.items():
        assert sha(DOC/'figures'/name) == expected, name
    x = dates.insert(0, pd.Timestamp('2021-12-31'))
    return rows, ids, x, wealth*100, audit


def main():
    rows, ids, dates, wealth, audit = load_verified_paths()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 8.8), facecolor='#fafbfc')
    ax = fig.add_axes([.075, .18, .72, .59], facecolor='white')
    selected = {key for key, *_ in HIGHLIGHTS}
    background = [j for j, key in enumerate(ids) if key not in selected]
    ax.plot(dates, wealth[:, background], color='#939da8', alpha=.22, lw=.65, zorder=1)
    handles = []
    for key, label, color, style, width in HIGHLIGHTS:
        values = wealth[:, ids.index(key)]
        line, = ax.plot(dates, values, color=color, ls=style, lw=width, zorder=4)
        ax.scatter(dates[-1], values[-1], color=color, s=28, zorder=5, clip_on=False)
        r = rows.loc[key]
        handles.append(Line2D([], [], color=color, ls=style, lw=width,
                              label=f'{label}\nCAGR {r.cagr:.2%}  |  Sharpe {r.sharpe:.3f}'))
        end_label = 'BEST MODEL' if key == BEST else label
        offset = {'BEST MODEL': 0, 'SPY': -22, 'QQQ': 18}[end_label]
        ax.annotate(f'{end_label}\n${values[-1]:.2f}', xy=(dates[-1], values[-1]),
                    xytext=(16, offset), textcoords='offset points', va='center',
                    color=color, fontsize=11, weight='bold', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=color, lw=1), zorder=6)
    handles.append(Line2D([], [], color='#939da8', lw=1.2,
                          label='343 other strategies\nAll base-cost comparison paths'))
    legend = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.07, .87),
                        ncol=4, frameon=False, fontsize=10.5, handlelength=2.5,
                        columnspacing=2.4, labelspacing=1)
    for label in legend.get_texts()[:3]:
        label.set_weight('bold')
    fig.text(.075, .95, 'Our best historical model, in the full comparison',
             fontsize=23, weight='bold', color='#182938')
    fig.text(.075, .906, '346 strategies  |  Growth of $100 after modeled costs  |  January 2022 - December 2025',
             fontsize=12, color='#4c5d6b')
    ax.set_ylabel('Portfolio value per initial $100', fontsize=11)
    ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#c9d0d7')
    ax.tick_params(colors='#4c5d6b', labelsize=10)
    fig.text(.075, .102, 'Highlighted model: THINK+mixer AH+PM + restricted dynamic hypergraph; 50:50 ranks; original five seeds; top five holdings.',
             fontsize=10, color='#334a5c')
    fig.text(.075, .069, 'Original-five leader: Sharpe 0.919, CAGR 32.94%. Ten-seed version: Sharpe 0.896, CAGR 31.87%.',
             fontsize=10, color='#334a5c')
    fig.text(.075, .035, 'Retrospective development comparison. Five holdings was secondary; ten holdings was primary. Future performance remains unconfirmed.',
             fontsize=9.5, color='#606b75')
    path = OUTPUT/'master_all_strategies.png'
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest = dict(
        status='verified_presentation_update', presentation_date='2026-10-26',
        strategies=len(ids), background_strategies=len(background), daily_observations=1003,
        highlighted_ids=[key for key, *_ in HIGHLIGHTS],
        coverage='All 346 main base-cost strategies; additional seed/cost diagnostics remain in the gallery.',
        checks=['All 346 CAGR, Sharpe and drawdown values reproduced from saved daily returns',
                'Complete unique strategy-ID and common-date coverage',
                'Historical leader agrees with both highest CAGR and highest Sharpe',
                'Prior numerical outputs and original figures retain their hashes; workbook matches its documented semantic-equivalence resave'],
        prior_publication_charts_sha256=audit['output_sha256']['charts.md'],
        note='Supplementary presentation record supersedes the prior charts.md hash only; original audit remains a dated snapshot.',
        input_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):sha(p) for p in
                      [DATA/'all_strategies.csv', DATA/'analysis_paths.npz', DOC/'AUDIT.json',
                       ROOT/'docs/audit/workbook_resave_20261025.json', SOURCE]},
        output_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):sha(p) for p in [path, ROOT/'charts.md']},
    )
    (OUTPUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(f'Verified {len(ids)} paths and original publication assets; wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
