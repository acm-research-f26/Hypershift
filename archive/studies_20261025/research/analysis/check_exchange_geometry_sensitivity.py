"""Equal-stock-count sensitivity checks for completed exact geometry diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_exchange_hyperbolicity import geo, save_json

ROOT = Path('runs/hyperbolicity_20261010_exact')
rows = []
rng = np.random.default_rng(20261010)
for year in range(2022, 2026):
    for representation in ['model_inputs', 'per_stock_standardized', 'returns_only']:
        loaded = {}
        for market, dataset in [('NYSE', 'NYSE_top100_20261010'), ('NASDAQ', 'NASDAQ100_20261010')]:
            source = np.load(ROOT/f'{dataset}_{year}_{representation}_distances.npz')
            loaded[market] = source['distances']
            labels = source['labels'].tolist()
            if market == 'NASDAQ' and 'GOOGL' in labels:
                ids = [i for i, label in enumerate(labels) if label != 'GOOGL']
                result = geo.delta_metric(source['distances'][np.ix_(ids, ids)])
                rows.append(dict(test_year=year, representation=representation, check='one_security_per_company',
                                 repetition=0, market=market, nodes=len(ids), delta_rel=result['relative_lower_bound']))
        for repetition in range(40):
            for market, distances in loaded.items():
                ids = rng.choice(len(distances), 80, replace=False)
                result = geo.delta_metric(distances[np.ix_(ids, ids)])
                assert result['exact']
                rows.append(dict(test_year=year, representation=representation, check='equal_count_80',
                                 repetition=repetition, market=market, nodes=80, delta_rel=result['relative_lower_bound']))
    print('Matched-count sensitivity complete for', year, flush=True)
frame = pd.DataFrame(rows)
frame.to_csv(ROOT/'stock_count_sensitivity.csv', index=False)
equal = frame[frame.check.eq('equal_count_80')]
pairs = equal.pivot(index=['test_year', 'representation', 'repetition'], columns='market', values='delta_rel')
pairs['NASDAQ_minus_NYSE'] = pairs.NASDAQ-pairs.NYSE
pairs.to_csv(ROOT/'stock_count_paired_differences.csv')
summary = pairs.groupby(['test_year', 'representation']).NASDAQ_minus_NYSE.agg(
    median='median', mean='mean', p10=lambda x:x.quantile(.1), p90=lambda x:x.quantile(.9),
    fraction_NASDAQ_lower=lambda x:float((x<0).mean()))
summary.to_csv(ROOT/'stock_count_sensitivity_summary.csv')
save_json(ROOT/'stock_count_sensitivity_method.json', {
    'stocks_per_market': 80, 'repetitions': 40, 'seed': 20261010,
    'method': 'Random without-replacement stock subsets of the same precomputed Euclidean metric. Exhaustive four-point maximum and subset diameter recomputed each time.',
    'interpretation': 'Descriptive sensitivity to which stocks are included; percentiles are not confidence intervals or a causal significance test.',
    'one_security_check': 'Remove GOOGL, retaining GOOG; re-evaluate complete Nasdaq distances exactly.'})
print(summary.to_string(), flush=True)
