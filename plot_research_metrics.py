"""Plot completed, locked experiments; never train or select using test outcomes."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from report_architecture_research import LABELS

DATASETS = ['modern12', 'NYSE', 'NASDAQ']
COLUMNS = ['Modern12\n2022–26 · 5 folds', 'NYSE sample\n2015–17 · 3 folds',
           'NASDAQ sample\n2015–17 · 3 folds']
NOTE = ('Validation-selected portfolios; modeled trading costs included. Each test fold starts from cash.\n'
        'Fixed survivor cohorts; older price-series data have incomplete corporate-action/dividend validation.')


def matrix(frame, metric, models):
    return frame.pivot(index='model', columns='dataset', values=metric).reindex(
        index=models, columns=DATASETS).to_numpy(dtype=float)


def heatmap(ax, values, rows, columns, title, fmt='.2f', diverging=False,
            limits=None, cmap='YlGnBu', show_rows=True):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f'No finite data for {title}')
    lo, hi = limits if limits else (float(finite.min()), float(finite.max()))
    norm = None
    if diverging:
        extent = max(abs(lo), abs(hi), 1e-6)
        norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
        cmap = 'RdBu'
    image = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm,
                      **({} if norm else {'vmin': lo, 'vmax': hi}))
    ax.set_xticks(range(len(columns)), columns, fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=8)
    ax.set_yticks(range(len(rows)), [LABELS[r] for r in rows] if show_rows else [])
    ax.set_title(title, fontsize=13, fontweight='bold', pad=45)
    for y in range(len(rows)):
        for x in range(len(columns)):
            val = values[y, x]
            if np.isfinite(val):
                rgba = image.cmap(image.norm(val))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(x, y, format(val, fmt), ha='center', va='center',
                        color='white' if luminance < .48 else '#17212b', fontsize=10)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save(fig, path, subtitle, note=None):
    fig.text(.02, .025, subtitle + '\n' + (NOTE if note is None else note), fontsize=9, color='#485260', va='bottom')
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def historical_comparison(root, historical_root, models):
    """Compare saved annual accounts without rerunning training or policy selection."""
    palette = list(plt.get_cmap('tab20').colors)
    colors = {kind: palette[i] for i, kind in enumerate(models)}
    colors.update(market_proxy='#111111', buy_hold='#777777', equal_universe='#217a45')
    inventory = []
    markdown = ['## Saved 2017 results beside the modern tests', '',
                'Each row compares **2017 | 2024 | 2025**, with all 15 model/strategy/benchmark kinds, '
                'consistent colors and a shared vertical scale within each cohort. Curves compound the saved '
                'daily returns after modeled costs; every annual account starts with $1. Models were fitted '
                'separately for each year and retain their original validation-selected portfolio policies. '
                'No retraining or test-based selection was performed for this comparison.', '',
                '**Read this as a historical comparison, not a controlled change of test year.** '
                'NYSE uses SPY in all panels; NASDAQ uses **QQQ in 2017 and SPY in 2024–2025**. '
                'The older stock prices have unverified dividend/corporate-action conventions; modern data '
                'use adjusted-price total-return approximations. Both use fixed historical candidate cohorts, '
                'with different available/eligible stocks and survivorship limitations. The modern evaluator '
                'also fixes eligible-universe ensemble ranking and masking of missing confidence weights. '
                'Date ranges below the panels show the actual saved trading windows; 2017 ends in early December.', '']
    for cohort in ['NYSE', 'NASDAQ']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
        for ax, year in zip(axes, [2017, 2024, 2025]):
            source_root = historical_root if year == 2017 else root
            dataset = cohort if year == 2017 else cohort + '_recent'
            metrics = pd.read_csv(source_root / 'all_metrics.csv')
            selected = metrics[(metrics.dataset == dataset) & (metrics.year == year) &
                               (metrics.seed == 'ensemble') & metrics.selected_policy &
                               (metrics.cost_multiplier == 1)].set_index('model')
            assert set(selected.index) == set(models) and selected.index.is_unique
            reference_dates = None
            benchmark = 'QQQ' if cohort == 'NASDAQ' and year == 2017 else 'SPY'
            for kind in models:
                path = source_root / f'{dataset}_{year}' / f'{kind}_daily.csv'
                daily = pd.read_csv(path)
                dates = pd.to_datetime(daily.date)
                assert dates.is_monotonic_increasing and dates.is_unique
                assert (dates.dt.year == year).all()
                if reference_dates is None:
                    reference_dates = dates
                else:
                    pd.testing.assert_series_equal(dates, reference_dates)
                returns = daily.net.to_numpy(dtype=float)
                assert np.isfinite(returns).all() and (returns > -1).all()
                equity = np.cumprod(1 + returns)
                np.testing.assert_allclose(equity[-1] - 1, selected.loc[kind, 'net_cumulative_return'],
                                           rtol=1e-8, atol=1e-10)
                ax.plot(dates, equity, label=('Market ETF (named in each panel)' if kind == 'market_proxy'
                                             else LABELS[kind]), color=colors[kind],
                        linestyle='--' if kind in ['buy_hold', 'equal_universe'] else '-',
                        linewidth=2.3 if kind == 'market_proxy' else 1.25,
                        alpha=1 if kind == 'market_proxy' else .9)
                inventory.append(dict(dataset=dataset, year=year, model=kind, benchmark=benchmark,
                                      source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      start=str(dates.iloc[0].date()), end=str(dates.iloc[-1].date()),
                                      observations=len(daily), final_equity=float(equity[-1]),
                                      k=int(selected.loc[kind, 'k']), method=selected.loc[kind, 'method']))
            assert len(ax.lines) == 15
            ax.axhline(1, color='#bbbbbb', linewidth=.7, zorder=0)
            ax.set_title(f'{year}  |  Benchmark: {benchmark}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}\n{len(dates)} trading observations', fontsize=10)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Net portfolio value per $1 initial capital')
        fig.suptitle(f'{cohort} historical cohort: saved 2017 vs modern 2024–2025', fontsize=19, y=.96)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .07), ncol=5, fontsize=9,
                   frameon=False, columnspacing=1.6)
        fig.text(.04, .025, 'Separate annual fits and accounts; original validation-selected policies; modeled costs included. '
                 'Data coverage and evaluator differ across old and modern runs.\n'
                 'Fixed survivor cohorts; historical dividend conventions unverified. '
                 'Black line: SPY, except NASDAQ 2017 uses QQQ. See CHARTS.md for comparison limitations.',
                 fontsize=9, color='#485260')
        fig.subplots_adjust(left=.06, right=.985, top=.86, bottom=.30, wspace=.08)
        filename = f'{cohort}_2017_vs_2024_2025.png'
        fig.savefig(root / 'figures' / filename, dpi=170, facecolor='white')
        plt.close(fig)
        markdown.extend([f'### {cohort}: 2017, 2024 and 2025', '',
                         f'![All 15 kinds, {cohort}: 2017 beside 2024 and 2025](figures/{filename})', ''])
    (root / 'historical_comparison_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    return markdown


def main():
    global DATASETS, COLUMNS, NOTE
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/architecture_research'))
    parser.add_argument('--historical-run', type=Path, default=Path('runs/architecture_research'))
    args = parser.parse_args()
    root = args.run
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    comparison = pd.read_csv(root / 'architecture_comparison.csv')
    metrics = pd.read_csv(root / 'all_metrics.csv')
    primary = metrics[(metrics.seed == 'ensemble') & metrics.selected_policy &
                      (metrics.cost_multiplier == 1)].copy()
    recent = all(str(name).endswith('_recent') for name in primary.dataset.unique())
    DATASETS = list(primary.dataset.unique())
    COLUMNS = []
    for name in DATASETS:
        years = sorted(primary[primary.dataset==name].year.unique())
        label = name.replace('_recent', ' cohort') if recent else ('Modern12' if name=='modern12' else name+' sample')
        COLUMNS.append(f'{label}\n{years[0]}–{str(years[-1])[-2:]} · {len(years)} folds')
    if recent:
        LABELS['market_proxy'] = 'SPY (S&P 500 ETF)'
        NOTE = ('Validation-selected portfolios; modeled costs included. SPY benchmark on identical dates.\n'
                'Fixed historical cohorts; public-data availability/survivorship bias remains. Annual folds start from cash.')
    assert not primary.duplicated(['dataset', 'year', 'model']).any()
    assert len(comparison) == len(LABELS) * len(DATASETS)
    assert len(primary) == len(LABELS) * len(primary[['dataset', 'year']].drop_duplicates())
    models = list(LABELS)
    rankers = models[:12]
    medians = primary.groupby(['dataset', 'model']).median(numeric_only=True).reset_index()
    means = primary.groupby(['dataset', 'model']).mean(numeric_only=True).reset_index()
    np.testing.assert_allclose(matrix(comparison, 'median_net_sharpe', models),
                               matrix(medians, 'net_sharpe', models))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(left=.34, right=.97, top=.84, bottom=.15)
    heatmap(ax, matrix(comparison, 'median_net_sharpe', models), models, COLUMNS,
            'Median out-of-sample net Sharpe', diverging=True)
    ax.axhline(11.5, color='#344454', linewidth=2)
    save(fig, out / 'sharpe_comparison.png',
         'Median of annual-fold Sharpes, not a pooled Sharpe. Fixed 3% annual hurdle. Model order is predeclared.')

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.subplots_adjust(left=.27, right=.97, top=.83, bottom=.15, wspace=.2)
    for ax, metric, title in zip(axes, ['median_gross_sharpe', 'median_net_sharpe'],
                               ['Before trading costs', 'After trading costs']):
        heatmap(ax, matrix(comparison, metric, models), models, COLUMNS,
                title, diverging=True, limits=(-2, 2), show_rows=ax is axes[0])
    fig.suptitle('Sharpe before and after modeled costs', fontsize=17, y=.98)
    save(fig, out / 'cost_comparison.png',
         'Median annual-fold Sharpe; 3% hurdle. Gross uses matched actual holdings with fees/borrow excluded.')

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    fig.subplots_adjust(left=.27, right=.97, top=.87, bottom=.12, wspace=.2, hspace=.48)
    for i, (ax, metric) in enumerate(zip(axes.flat, ['IC', 'RankIC', 'ICIR', 'RankICIR'])):
        heatmap(ax, matrix(means, metric, rankers), rankers, COLUMNS, metric,
                fmt='.3f', diverging=True, show_rows=i % 2 == 0)
    fig.suptitle('Ranking quality across the full eligible universe', fontsize=17, y=.98)
    save(fig, out / 'ranking_quality.png',
         'Unweighted means across test folds. ICIR/RankICIR are unannualized. Each panel has its own color scale.')

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 8), squeeze=False)
    fig.subplots_adjust(left=.25, right=.98, top=.80, bottom=.17, wspace=.2)
    for i, (ax, dataset) in enumerate(zip(axes[0], DATASETS)):
        ks = [3, 5, 8] if dataset == 'modern12' else [5, 10, 20]
        vals = means[means.dataset == dataset].set_index('model').reindex(rankers)[
            [f'NDCG@{k}' for k in ks]].to_numpy()
        heatmap(ax, vals, rankers, [f'K = {k}' for k in ks], COLUMNS[i].replace('\n', ' · '),
                fmt='.3f', limits=(0, 1), show_rows=i == 0)
    fig.suptitle('Top-of-ranking quality: NDCG at several K values', fontsize=17, y=.97)
    save(fig, out / 'ndcg_comparison.png',
         'Unweighted fold means; linear percentile return relevance. Higher is better; different K values are not interchangeable.')

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.subplots_adjust(left=.27, right=.97, top=.83, bottom=.15, wspace=.2)
    heatmap(axes[0], -100 * matrix(comparison, 'median_drawdown', models), models, COLUMNS,
            'Maximum drawdown (%)', fmt='.1f', cmap='YlOrRd')
    heatmap(axes[1], matrix(comparison, 'median_turnover', models), models, COLUMNS,
            'Annual turnover (× NAV)', fmt='.1f', cmap='YlOrRd', show_rows=False)
    fig.suptitle('Portfolio risk and trading intensity', fontsize=17, y=.98)
    save(fig, out / 'risk_turnover.png',
         'Median across test folds. Drawdown shown as positive loss magnitude. Turnover counts buys plus sells; lower is better.')

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.subplots_adjust(left=.27, right=.97, top=.88, bottom=.11, wspace=.2, hspace=.43)
    specifications = [('net_annualized_return', 'Annualized net return (%)', True),
                      ('net_cumulative_return', 'Cumulative net return per fold (%)', True),
                      ('net_annualized_volatility', 'Annualized net volatility (%)', False),
                      ('profitable_periods', 'Profitable 5-session periods (%)', False)]
    for i, (ax, (metric, title, diverging)) in enumerate(zip(axes.flat, specifications)):
        heatmap(ax, 100 * matrix(medians, metric, models), models, COLUMNS, title,
                fmt='.1f', diverging=diverging, show_rows=i % 2 == 0)
    fig.suptitle('Returns, volatility and win rate', fontsize=17, y=.98)
    save(fig, out / 'return_metrics.png',
         'Median across test folds. Cumulative returns cover each fold, not the full multi-year span.'+
         (' 2026 is partial-year.' if not recent else ' Two annual folds per cohort.'))

    entries = [('Median net Sharpe', 'sharpe_comparison'),
               ('Before/after cost Sharpe', 'cost_comparison'),
               ('IC, RankIC, ICIR and RankICIR', 'ranking_quality'),
               ('NDCG at each K', 'ndcg_comparison'),
               ('Drawdown and turnover', 'risk_turnover'),
               ('Returns, volatility and win rate', 'return_metrics')]
    if recent:
        groups = [(name, year) for name in DATASETS for year in sorted(primary[primary.dataset==name].year.unique())]
        values = np.array([primary[(primary.dataset==name)&(primary.year==year)].set_index('model').reindex(models).net_sharpe.to_numpy()
                           for name, year in groups]).T
        columns = [f'{name.replace("_recent", "")}\n{year}' for name, year in groups]
        for filename, vals, title in [('yearly_sharpe', values, 'Net Sharpe in each held-out year'),
                                     ('sharpe_vs_spy', values-values[-1:, :], 'Net Sharpe difference versus SPY')]:
            fig, ax = plt.subplots(figsize=(12, 9))
            fig.subplots_adjust(left=.32, right=.97, top=.84, bottom=.15)
            heatmap(ax, vals, models, columns, title, diverging=True)
            ax.axhline(11.5, color='#344454', linewidth=2)
            save(fig, out/f'{filename}.png',
                 'Each cell is a separate annual test. '+
                 ('Positive differences favor the model; this is not a significance test.' if filename=='sharpe_vs_spy' else
                  'Sharpe uses a fixed 3% annual hurdle; higher is better.'))
            entries.insert(0, (title, filename))
        random_path = root/'random_topk_controls.csv'
        if random_path.exists():
            random = pd.read_csv(random_path)
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.subplots_adjust(left=.09, right=.97, top=.85, bottom=.21)
            samples = [random[(random.dataset==name)&(random.year==year)&(random.k==10)].net_sharpe.to_numpy()
                       for name, year in groups]
            box = ax.boxplot(samples, positions=np.arange(len(groups)), widths=.4,
                             whis=(5,95), showfliers=False, patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor('#d4dfe8')
            for kind, color, marker in [('think','#dc6e30','o'),('think_mix','#206cb0','D'),('market_proxy','#23865c','s')]:
                values = []
                for name, year in groups:
                    sub = metrics[(metrics.dataset==name)&(metrics.year==year)&(metrics.model==kind)&
                                  (metrics.seed=='ensemble')&(metrics.cost_multiplier==1)]
                    if kind!='market_proxy':
                        sub = sub[(sub.method=='equal')&(sub.k==10)]
                    values.append(sub.net_sharpe.iloc[0])
                ax.scatter(np.arange(len(groups)), values, label=LABELS[kind], color=color,
                           marker=marker, s=70, zorder=3)
            ax.axhline(0, color='#888888', linewidth=.8)
            ax.set_xticks(np.arange(len(groups)), columns)
            ax.set_ylabel('Net Sharpe (3% hurdle)')
            ax.set_title('Can ranking beat random top-10 portfolios?', fontsize=17, pad=18)
            ax.grid(axis='y', alpha=.2)
            ax.legend(loc='best', fontsize=9)
            save(fig, out/'random_controls.png',
                 'Boxes: 50 random equal-weight top-10 portfolios, median/IQR and 5th–95th percentile whiskers. Not confidence intervals.',
                 note='THINK/mixer points also use fixed equal weights and K=10; SPY uses matching dates. Costs included.\n'
                      'Public-data availability/survivorship bias remains. These controls do not select a model.')
            entries.append(('Random top-10 controls', 'random_controls'))
    lines = ['# Result graphs', '',
             'These charts describe the completed locked run; no models were retrained or reselected to produce them.', '',
             'The heatmaps retain predeclared model order. Portfolio policies were selected on validation data. '
             'Sharpe uses a fixed 3% annual hurdle. Fold medians are not pooled multi-year performance.', '',
             ('Public-vendor availability leaves substantial survivorship bias in the recent cohorts. ' if recent else
              'All cohorts have survivor/selection bias; old cohorts have incomplete corporate-action/dividend validation. ')+
             'StockMixer/MASTER labels denote local adaptations. See [the report](REPORT.md) for full caveats and uncertainty.', '',
             ('![Both chronological test years](test_equity.png)' if recent else '![Final chronological test equity curves](final_equity.png)'), '',
             ('Equity curves show separate 2024 and 2025 test portfolios. ' if recent else
              'Equity curves show only the final test fold: Modern12 in 2026, NYSE/NASDAQ in 2017. ')+
             'Each panel includes the architecture chosen on validation data, THINK and benchmarks. '
             'No statistically established robust improvement is claimed.', '']
    for title, filename in entries:
        lines.extend([f'## {title}', '', f'![{title}](figures/{filename}.png)', ''])
    if recent:
        equity = ['## Every tested model and benchmark: line charts', '',
                  '**All 15 kinds are plotted:** THINK; THINK + ranking loss, feature gate, stock mixer and volatility task; '
                  'MLP; LSTM; StockMixer-inspired MLP; MASTER-inspired Transformer; ridge; momentum; reversal; '
                  'equal-weight buy-and-hold; equal-weight rebalanced universe; and SPY.', '',
                  'Each model uses its validation-selected K/weighting in each year, with the fixed three-seed ensemble '
                  'for neural models. Every line includes modeled costs. These are separate annual accounts, not a '
                  'stitched two-year live portfolio. The black line is SPY.', '',
                  '![All models across both cohorts and test years](figures/all_models_equity.png)', '',
                  '### Full-size panels', '']
        for dataset in DATASETS:
            for year in sorted(primary[primary.dataset==dataset].year.unique()):
                equity.extend([f'![All 15 models: {dataset}, {year}](figures/all_models_{dataset}_{year}.png)', ''])
        equity.extend(['### Focused comparisons', '',
                       'These supplemental views make overlapping lines easier to distinguish; the complete charts above retain all models.', '',
                       '![THINK component ablations](figures/think_ablations_equity.png)', '',
                       '![Alternative models and simple strategies](figures/alternative_models_equity.png)', ''])
        lines[2:2] = equity
        lines[2:2] = historical_comparison(root, args.historical_run, models)
    (root / 'CHARTS.md').write_text('\n'.join(lines), encoding='utf-8')
    if recent:
        images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', '\n'.join(lines))
        assert all((root / path).is_file() for path in images)
        (root / 'graph_audit.json').write_text(json.dumps(dict(
            embedded_images=len(images), models_per_cohort_year=len(models), cohort_years=4,
            historical_comparison_panels=6, historical_comparison_curves=90,
            status='All image paths exist; all tested kinds included; historical curve endpoints match saved metrics'
        ), indent=2), encoding='utf-8')
    print(f'Wrote {len(entries)} metric charts and {root / "CHARTS.md"}; verified {len(primary)} primary model/fold rows.')


if __name__ == '__main__':
    main()
