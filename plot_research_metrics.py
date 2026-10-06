"""Plot completed, locked experiments; never train or select using test outcomes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def save(fig, path, subtitle):
    fig.text(.02, .025, subtitle + '\n' + NOTE, fontsize=9, color='#485260', va='bottom')
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def main():
    global DATASETS, COLUMNS, NOTE
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/architecture_research'))
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
                 'Each cell is a separate annual test. Positive differences favor the model; this is not a significance test.')
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
                 'Boxes: 50 random equal-weight top-10 portfolios, median/IQR and 5th–95th percentile whiskers. Not confidence intervals.')
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
    (root / 'CHARTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {len(entries)} metric charts and {root / "CHARTS.md"}; verified {len(primary)} primary model/fold rows.')


if __name__ == '__main__':
    main()
