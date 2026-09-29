"""Generate a readable report and scientific plots from a completed experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/direction'))
    args = parser.parse_args()
    root = args.run
    summary = pd.read_csv(root / 'summary.csv', index_col='model')
    metrics = pd.read_csv(root / 'metrics.csv')
    experiment = json.loads((root / 'experiment.json').read_text())
    geometry = json.loads((root / 'geometry.json').read_text())
    details = json.loads((root / 'metrics.json').read_text())
    order = ['always_up', 'train_prior', 'stock_prior', 'logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic']
    labels = {'always_up': 'Always up / equal-weight basket', 'train_prior': 'Pooled training up-rate',
              'stock_prior': 'Per-stock training up-rate', 'logistic': 'Logistic regression',
              'mlp': 'Independent neural network', 'gcn': 'Ordinary GNN',
              'hypergraph': 'Euclidean hypergraph', 'hyperbolic': 'Hyperbolic hypergraph'}
    chosen_seed = experiment['arguments']['seeds'][0]  # Fixed by argument order, never selected on test results.
    horizon = experiment['arguments']['horizon']
    up_fraction = details['always_up_0']['classification']['actual_up_fraction']
    learned = ['logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic']
    no_edge = ((summary.loc[learned, 'roc_auc_mean'] < .5).all() and
               (summary.loc[learned, 'net_annualized_sharpe_rf0_mean'] < summary.loc['always_up', 'net_annualized_sharpe_rf0_mean']).all())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    plotted = [m for m in order if m != 'train_prior']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ['roc_auc', 'net_annualized_sharpe_rf0'],
                                  ['Direction discrimination (AUC)', 'Net annualized Sharpe: delayed monthly baskets']):
        means = summary.loc[plotted, metric + '_mean']
        errors = summary.loc[plotted, metric + '_std'].fillna(0)
        ax.barh(np.arange(len(plotted)), means, xerr=errors, capsize=3,
                color=['#778899' if m in ['always_up', 'stock_prior'] else '#307d91' for m in plotted])
        ax.set_yticks(np.arange(len(plotted)), [labels[m] for m in plotted])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.axvline(.5 if metric == 'roc_auc' else summary.loc['always_up', metric + '_mean'],
                   color='#b34839', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_xlim(0, .58)
    fig.suptitle('Monthly direction experiment | five training seeds; bars show mean +/- seed SD', fontsize=13)
    fig.savefig(root / 'comparison.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for kind in ['always_up', 'logistic', 'mlp', 'hypergraph', 'hyperbolic']:
        seed = 0 if kind == 'always_up' else chosen_seed
        trades = pd.read_csv(root / f'{kind}_{seed}_trades.csv')
        dates = [pd.Timestamp(experiment['splits']['test']['first_origin'])] + list(pd.to_datetime(trades.exit_date))
        ax.plot(dates, [1] + trades.net_nav.tolist(), marker='.', label=labels[kind])
    ax.set_title(f'Net basket-exit wealth, starting at $1 | seed {chosen_seed} fixed in advance')
    ax.set_ylabel('Simulated wealth after costs')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(root / 'wealth.png', dpi=160)
    plt.close(fig)

    p = pd.read_csv(root / 'predictions.csv')
    sample = p[(p.model == 'hyperbolic') & (p.seed == chosen_seed)].copy()
    sample['probability_bin'] = pd.cut(sample.probability_up, np.linspace(0, 1, 11), include_lowest=True)
    calibration = sample.groupby('probability_bin', observed=True).agg(
        average_probability=('probability_up', 'mean'), observed_up_fraction=('actual_up', 'mean'), count=('actual_up', 'size'))
    calibration.to_csv(root / 'hyperbolic_calibration.csv')
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.scatter(calibration.average_probability, calibration.observed_up_fraction,
               s=np.sqrt(calibration['count']) * 10, color='#307d91')
    for _, row in calibration.iterrows():
        ax.annotate(f'n={int(row["count"])}', (row.average_probability, row.observed_up_fraction), xytext=(5, 7), textcoords='offset points')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed up fraction',
           title=f'Hyperbolic model: descriptive calibration, seed {chosen_seed}')
    ax.legend()
    fig.savefig(root / 'calibration.png', dpi=160)
    plt.close(fig)

    conclusion = ('The classifiers did not establish a useful predictive or trading advantage in this experiment. '
             'All five learned model families have mean AUC below 0.50 and lower mean net Sharpe than the equal-weight baseline.', '',
             ) if no_edge else ('Inspect discrimination, probability quality and trading results separately; no single metric establishes a robust advantage.', '')
    lines = [f'# {horizon}-session up/down experiment', '', *conclusion,
             'These are real historical adjusted-close observations from the existing Yahoo Finance snapshot, '
             'not synthetic prices. This is a THINK-inspired teaching experiment, not a reproduction of its benchmark.', '',
             '## Evaluation window', '',
             f"- Horizon: {experiment['arguments']['horizon']} trading sessions; {len(experiment['tickers'])} stocks.",
             f"- Training: {experiment['splits']['train']['first_origin']} to {experiment['splits']['train']['last_origin']} forecast origins.",
             f"- Validation: {experiment['splits']['validation']['first_origin']} to {experiment['splits']['validation']['last_origin']}.",
             f"- Test: {experiment['splits']['test']['first_origin']} to {experiment['splits']['test']['last_origin']}; targets end {experiment['splits']['test']['last_target']}.",
             f"- {experiment['splits']['test']['origins']} test origins, {len(sample):,} stock/date labels; overlapping horizons and correlated stocks.",
             f"- Seeds fixed as {experiment['arguments']['seeds']}. Mean +/- sample SD measures training randomness, not statistical confidence.", '',
             '## Model comparison', '',
             '| Model | Accuracy | Macro F1 | Balanced accuracy | AUC | NDCG@3 | Net Sharpe |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in order:
        vals = []
        for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'roc_auc', 'ndcg_at_k', 'net_annualized_sharpe_rf0']:
            mean, std = summary.loc[name, key + '_mean'], summary.loc[name, key + '_std']
            vals.append(f'{mean:.3f}' + (f' +/- {std:.3f}' if np.isfinite(std) else ''))
        lines.append('| ' + labels[name] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '![Model comparison](comparison.png)', '',
              f'Always-up accuracy is {up_fraction:.2%}, equal to the observed up fraction. Its balanced accuracy '
              'and AUC are both 0.50. Higher macro F1 than always-up does not establish an edge: the learned '
              'models must also be assessed against a 0.50 balanced-accuracy reference. Pooled training-rate Brier score is '
              f"{summary.loc['train_prior', 'brier_mean']:.4f}, versus {summary.loc['hyperbolic', 'brier_mean']:.4f} for the hyperbolic model (lower is better).", '',
              'NDCG@3 uses positive realized percentage return as linear relevance, log2 discounts, and expected '
              'DCG over tied scores. All-negative dates are excluded and counted in metrics.json. The paper does '
              'not specify enough details to identify this as its exact NDCG convention. Probability of rising '
              'and expected profit are different ranking targets.', '',
              'The always-up and pooled-rate trading scores are tied, so their allocations average over ties: '
              'they hold all 12 stocks equally. Their trading results are an equal-weight market-exposure baseline, '
              'not evidence that a constant classifier predicts the market.', '',
              '## Trading simulation', '',
              'Signals use close t, execution occurs at close t+1, and exit at close t+1+h. '
              'A new signal arrives every h+1 sessions. Each basket fully liquidates; there is one cash session '
              'between baskets. We allocate equally to the three highest scores, with fractional allocation '
              'across ties. This strategy always holds a basket during each holding period; it does not use the '
              'classification threshold to time cash or take short positions.', '',
              f"There are {int(metrics.periods.iloc[0])} completed baskets. Entry and exit each cost {experiment['arguments']['cost_bps']:g} basis points "
              '(10 basis points = 0.10%). Net return per cycle is (1-cost)^2*(1+gross return)-1. '
              'Sharpe is mean net cycle return / sample standard deviation, with zero risk-free return; '
              'annualization multiplies by sqrt(252/(h+1)). Raw and annualized values are both saved. '
              'Adjusted closes approximate corporate-action-adjusted performance; they are not quoted executable '
              'prices. Spread, liquidity, taxes and additional slippage are absent.', '',
              '| Model | Net cumulative return, mean | Daily-close maximum drawdown, mean |',
              '|---|---:|---:|']
    for name in ['always_up', 'logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic']:
        lines.append(f"| {labels[name]} | {summary.loc[name, 'net_total_return_mean']:.2%} | {summary.loc[name, 'net_daily_close_max_drawdown_mean']:.2%} |")
    lines += ['', '![Wealth at basket exits](wealth.png)', '',
              f'The wealth plot uses seed {chosen_seed}, chosen by the experiment argument order before inspecting results. '
              'It joins basket exits; intraperiod drawdown is separately calculated from daily closing valuations.', '',
              '## Geometry', '',
              f"The training-only correlation construction produced {len(geometry['hyperedges'])} distinct hyperedges. "
              f"For s=1, delta is {geometry['s1']['global_delta']}, diameter is {geometry['s1']['components'][0]['diameter']}, "
              f"and normalized graph delta is {geometry['s1']['components'][0]['relative_delta']}. "
              f"For s=2, there are {len(geometry['s2']['components'])} components; no finite global delta is reported. "
              f"The normalized delta of our standardized training-return trajectory metric is {geometry['training_trajectory_metric']['relative_delta']:.4f}.", '',
              'These diagnostics depend on the chosen groups and distance definition. They do not measure forecasting '
              'skill. A clique also has zero vertex-metric delta, so a low value alone does not demonstrate an '
              'economically meaningful hierarchy.', '',
              '## Historical predictions to inspect', '',
              f'The following are the first chronological predictions for seed {chosen_seed}, not selected successes. '
              'The complete prediction/outcome pairs are in predictions.csv.', '',
              '| As of | Outcome date | Stock | P(up) | Threshold | Predicted | Actual return | Actual |',
              '|---|---|---|---:|---:|---|---:|---|']
    for _, row in sample.head(12).iterrows():
        lines.append(f"| {row.as_of} | {row.target_date} | {row.ticker} | {row.probability_up:.5f} | {row.threshold:.3f} | {'Up' if row.predicted_up else 'Down/flat'} | {row.actual_return_pp:+.2f}% | {'Up' if row.actual_up else 'Down/flat'} |")
    lines += ['', '## Probability calibration', '', '![Calibration](calibration.png)', '',
              'This describes observed frequencies on the overlapping test samples. It is not a fitted calibration '
              'correction or an independent confidence interval. A score of 0.60 is a model estimate, not a '
              'demonstrated 60% real-world success rate.', '',
              '## Limits and next experiment', '',
              'The earlier regression experiments already used this historical test window, so these results are '
              'exploratory even though this fit never learns from its test labels. Only 12 selected surviving stocks '
              'are included. Monthly labels overlap heavily. The nonoverlap_classification fields use every h-th '
              f"origin, yielding only {len(sample.iloc[::len(experiment['tickers']) * horizon])} sampled origins here; cross-stock dependence still remains. The trading "
              'window is short, and five seeds do not replace independent market periods.', '',
              'Do not flip predictions after seeing AUC below 0.50 and call that a discovered strategy: doing so '
              'would use test outcomes to design the strategy. A sensible next test would freeze the model and '
              'rules, add point-in-time data and useful covariates, then run rolling historical training and a '
              'genuinely new holdout. See ../../DIRECTION_TUTORIAL.md for architecture explanations and alternatives.', '',
              '## Files', '',
              '- metrics.json: confusion matrices, fixed-0.5 and validation-threshold metrics, nonoverlap metrics, and raw Sharpe.',
              '- metrics.csv / summary.csv: per-seed and aggregate results.',
              '- predictions.csv: every test probability with actual direction and return.',
              '- *_trades.csv: each entry, exit, allocation and realized basket return.',
              '- geometry.json: training-only hyperedges and exact 12-node geometry diagnostics.',
              '- *.pt: reloadable classifier checkpoints; *_training.csv: loss history.',
              '- *_latest.csv: predictions at the last CSV date, whose future outcomes are not present.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(root / 'REPORT.md')


if __name__ == '__main__':
    main()
