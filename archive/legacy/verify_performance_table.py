"""Independently rescore saved local predictions; does not reproduce THINK.

No training/evaluation functions from the original experiment are imported.
Historical outcomes and portfolio returns are rebuilt from the price snapshot.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def dcg_reference(gains, scores, k):
    # Each member of a score tie receives the average discount of its ranks.
    result = 0.0
    for score in np.unique(scores):
        tied = scores == score
        start = int(np.sum(scores > score))
        ranks = np.arange(start, start + tied.sum())
        discounts = np.where(ranks < k, 1 / np.log2(ranks + 2), 0)
        result += gains[tied].sum() * discounts.mean()
    return float(result)


def main():
    root = Path('runs/direction')
    output = Path('runs/performance_verification')
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / 'experiment.json').read_text())
    args = metadata['arguments']
    h, k, cost = args['horizon'], args['top_k'], args['cost_bps'] / 10000
    price_path = Path('data/prices.csv')
    assert hashlib.sha256(price_path.read_bytes()).hexdigest() == metadata['data_sha256']
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    predictions = pd.read_csv(root / 'predictions.csv', parse_dates=['as_of', 'target_date'])
    saved = pd.read_csv(root / 'metrics.csv').set_index(['model', 'seed'])
    records, checks = [], []
    for (model, seed), group in predictions.groupby(['model', 'seed'], sort=True):
        p = group.pivot(index='as_of', columns='ticker', values='probability_up').reindex(columns=prices.columns)
        dates = p.index
        assert p.notna().all().all() and ((p >= 0) & (p <= 1)).all().all()
        positions = prices.index.get_indexer(dates)
        assert (positions >= 0).all()
        actual = (prices.iloc[positions+h].to_numpy() / prices.iloc[positions].to_numpy() - 1) * 100
        stored_actual = group.pivot(index='as_of', columns='ticker', values='actual_return_pp').reindex(columns=prices.columns).to_numpy()
        np.testing.assert_allclose(actual, stored_actual, atol=2e-5, rtol=2e-6)
        stored_targets = group.groupby('as_of')['target_date']
        assert stored_targets.nunique().eq(1).all()
        np.testing.assert_array_equal(stored_targets.first().to_numpy(), prices.index[positions+h].to_numpy())
        label = actual > 0
        stored_labels = group.pivot(index='as_of', columns='ticker', values='actual_up').reindex(columns=prices.columns).to_numpy()
        np.testing.assert_array_equal(label, stored_labels)
        assert group['threshold'].nunique() == 1
        predicted = p.to_numpy() >= group['threshold'].iloc[0]
        stored_predicted = group.pivot(index='as_of', columns='ticker', values='predicted_up').reindex(columns=prices.columns).to_numpy()
        np.testing.assert_array_equal(predicted, stored_predicted)
        f1 = []
        for cls in [False, True]:
            actual_class, predicted_class = label == cls, predicted == cls
            denominator = actual_class.sum() + predicted_class.sum()
            f1.append(2 * (actual_class & predicted_class).sum() / denominator if denominator else 0)
        ndcg_values = []
        for gains, scores in zip(np.maximum(actual, 0), p.to_numpy()):
            ideal = dcg_reference(gains, gains, k)
            if ideal > 0:
                ndcg_values.append(dcg_reference(gains, scores, k) / ideal)
        portfolio_returns = []
        for i in range(0, len(dates), h+1):
            entry, exit_ = positions[i]+1, positions[i]+1+h
            if exit_ >= len(prices):
                continue
            scores = p.iloc[i].to_numpy()
            # Average the top-k allocation across all permutations of tied stocks.
            weights = np.zeros(len(scores))
            for j, score in enumerate(scores):
                better, tied = np.sum(scores > score), np.sum(scores == score)
                weights[j] = np.clip(k-better, 0, tied) / tied / k
            assert np.isclose(weights.sum(), 1)
            gross = float(weights @ (prices.iloc[exit_].to_numpy() / prices.iloc[entry].to_numpy() - 1))
            portfolio_returns.append((1-cost)**2 * (1+gross) - 1)
        returns = np.asarray(portfolio_returns)
        sr = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252/(h+1)))
        row = {'model': model, 'seed': int(seed), 'macro_f1': float(np.mean(f1)),
               'ndcg_at_k': float(np.mean(ndcg_values)), 'net_annualized_sharpe_rf0': sr,
               'accuracy': float((predicted == label).mean()), 'periods': len(returns),
               'net_total_return': float(np.prod(1+returns)-1)}
        old = saved.loc[(model,seed)]
        for metric in ['macro_f1','ndcg_at_k','net_annualized_sharpe_rf0','accuracy','net_total_return','periods']:
            error = abs(row[metric] - float(old[metric]))
            assert error < 1e-6, (model, seed, metric, row[metric], old[metric])
            checks.append({'model':model, 'seed':int(seed), 'metric':metric, 'absolute_difference':error})
        records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(output / 'independent_metrics.csv', index=False)
    metrics = ['macro_f1','ndcg_at_k','net_annualized_sharpe_rf0']
    summary = table.groupby('model')[metrics].agg(['mean','std'])
    summary.to_csv(output / 'independent_summary.csv')
    verification = {'scope':'Rescoring existing local predictions, not THINK replication or independent retraining.',
                    'model_seed_groups':len(records), 'metric_checks':len(checks),
                    'max_absolute_difference':max(c['absolute_difference'] for c in checks),
                    'price_sha256':metadata['data_sha256'],
                    'prediction_sha256':hashlib.sha256((root/'predictions.csv').read_bytes()).hexdigest(),
                    'checks':checks}
    (output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    labels = {'always_up':'Always up / equal-weight basket', 'train_prior':'Pooled training up-rate',
              'stock_prior':'Per-stock training up-rate','logistic':'Logistic regression',
              'mlp':'Independent neural network','gcn':'Ordinary GNN',
              'hypergraph':'Euclidean hypergraph','hyperbolic':'Hyperbolic hypergraph (teaching model)'}
    lines = ['# Performance table: meaning and independent scoring check', '',
             '**THINK Table II is not independently verified.** This report supplies a similar table for our existing experiment, '
             'with scores freshly recomputed from saved predictions and the original historical price snapshot. '
             'It does not relabel those models as THINK or use their scores to confirm or refute its benchmark.', '',
             '## What the screenshot measures', '',
             'The metric varies by column. The geometry rows describe the dataset, whereas the model rows assess prediction or trading outcomes. '
             'The task/metric mapping and 25-run count come from [THINK, Section IV and Table II](https://tylersnetwork.github.io/papers/icdm22-think.pdf).', '',
             '| Columns | Metric | Interpretation |', '|---|---|---|',
             '| DTT, CPox, WMill, Risk (CSE) | MSE, lower is better | Average squared numerical prediction error. Scaling of the target determines the units. |',
             '| NYSE, TSE | SR, higher is better | Mean excess portfolio return divided by its standard deviation. |',
             '| NYSE, TSE | NDCG, higher is better | Reward for putting more relevant stocks higher in a ranked list, relative to the ideal ranking. |',
             '| Clf (NASDAQ) | F1, higher is better | Balance of precision and recall for movement classification; the task includes up/down/neutral. |', '',
             'F1 is not accuracy: for one class, F1 = 2TP/(2TP+FP+FN). Multiclass F1 also requires an averaging rule '
             '(macro, weighted or micro). [F1 reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). '
             'NDCG depends on the relevance labels, gain function, rank cutoff and tie handling. '
             '[NDCG reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).', '',
             'MSE = mean((prediction-actual)^2); it is not a percentage and values across differently scaled targets are not comparable. '
             'Sharpe is not a percentage return; annualization, risk-free rate, costs and execution timing must be specified.', '',
             'For example, the screenshot\'s 1.18 +/- 4e-3 means 1.18 +/- 0.004, not 1.18 percent. The caption identifies the mean '
             'over 25 runs and a Wilcoxon signed-rank significance test at p<0.01 for asterisks. It does not clearly define '
             'whether the +/- quantity is standard deviation, standard error or another uncertainty measure. '
             'We cannot reconstruct paired significance tests from means and error bars alone.', '',
             '## Independently rescored local experiment', '',
             f"{len(records)} model/seed groups; {len(checks)} checks against saved metrics passed. Maximum absolute difference: "
             f"{verification['max_absolute_difference']:.3g}. The audit imports none of the original evaluation functions.", '',
             'Setup: 12 US stocks, 21-session horizon, binary up versus down/flat, 377 test origins from 2024-05-31 through '
             '2025-12-01, with outcomes ending 2025-12-31. Learned models have five seeds; deterministic baselines have one result. '
             'The original training used earlier data; this audit does not retrain the models.', '',
             '| Model | Runs | Macro F1 (up) | NDCG@3 (up) | Net annualized Sharpe (up) |',
             '|---|---:|---:|---:|---:|']
    for name,label in labels.items():
        cells = []
        for metric in metrics:
            mean, std = summary.loc[name,(metric,'mean')], summary.loc[name,(metric,'std')]
            cells.append(f'{mean:.3f}' if pd.isna(std) else f'{mean:.3f} +/- {std:.3f}')
        lines.append(f"| {label} | {int((table.model==name).sum())} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Here +/- is explicitly the sample standard deviation across the five seeds, not a confidence interval. '
              'There are no significance stars. The five fits share a market period and do not represent five independent market histories.', '',
              'The independent checker rebuilt 21-session actual returns and labels from prices, checked target-date alignment, '
              'recalculated macro F1 from class counts, recalculated tie-averaged NDCG, and rebuilt portfolio returns from scores and prices. '
              'NDCG uses positive realized percentage return as linear gain and excludes dates with no positive gain. '
              'Trading buys the top three scores with fractional allocation over ties, enters one session after the signal, holds 21 sessions, '
              'and charges 10 basis points on both entry and exit. There are 18 completed, nonoverlapping baskets. '
              'Annualization uses sqrt(252/22) and risk-free return is zero.', '',
              '**These are different tasks and model implementations from THINK.** Our hyperbolic model\'s F1 must not be compared '
              'numerically with the paper\'s three-class NASDAQ F1 as a replication result. The local experiment also has a short trading '
              'window, overlapping classification labels, a selected surviving-stock universe, and a previously examined test period. '
              'Rechecking arithmetic does not cure those research limitations or establish predictive validity.', '',
              '## What can be verified about the paper now?', '',
              '| Table II task | Available inputs | Remaining obstacle to numerical verification |',
              '|---|---|---|',
              '| CPox | Public graph and series recovered | Exact preprocessing, splits, hypergraph merging and model settings not recovered. |',
              '| DTT | Both public Twitter variants recovered | Variant, snapshot treatment, training/test setup and processed groups unresolved. |',
              '| WMill | Official source mirror recovered | Weight treatment, preprocessing, split and processed groups unresolved. |',
              '| NYSE | Matching-size source stock panel and relation files recovered | Exact graph, ranking loss, top-k, execution, costs, NDCG and Sharpe conventions unresolved. |',
              '| NASDAQ Clf | Matching-size source stock panel recovered | Neutral-label thresholds, F1 averaging, splits and exact trained model unresolved. |',
              '| TSE | Some source loading code available | Matching full processed panel and groups unavailable. |',
              '| Risk/CSE | Related source references available | Matching processed panel and exact risk targets unavailable. |', '',
              'The [THINK code repository](https://github.com/shivamag125/ICDM22-THINK) is empty as inspected on 2026-09-28. '
              'The authors\' per-run predictions, scores and checkpoints have not been obtained. None of their Table II cells or '
              'significance claims is marked verified by this local rescoring.', '',
              'A new independent implementation remains possible: freeze a documented reconstruction, implement the full THINK '
              'architecture and comparison models, use chronological train/validation/test splits, and train each for 25 seeds. '
              'Save every forecast and portfolio return, explicitly define all metric conventions, and evaluate paired differences '
              'with a specified multiple-comparison policy. Such an experiment tests the published method under declared assumptions; '
              'matching the original table requires resolving the missing inputs and protocol.', '',
              '## Reproduce this scoring audit', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe verify_performance_table.py', '```', '',
              'Results: `runs/performance_verification/independent_metrics.csv`, `independent_summary.csv` and `verification.json`. '
              'Original training results remain unchanged. See `runs/direction/REPORT.md` for the original experiment and '
              '`THINK_AUDIT.md` for the separate geometry audit.', '']
    Path('PERFORMANCE_TABLE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k!='checks'},indent=2))


if __name__ == '__main__':
    main()
