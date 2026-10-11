"""Exact THINK-equation geometry diagnostics; does not fit forecasting models.

Uses frozen training hyperedges and training-only feature trajectories. The
paper's equations are reused, not its industry/Wikidata graph construction.
"""
import argparse
import hashlib
import importlib.util
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_engine import load_panel, prepare_panel, fold_indices, training_groups
from run_robust_research import scale_inputs

spec = importlib.util.spec_from_file_location('geometry_audit', Path(__file__).parent/'archive/legacy/hyperbolicity_audit.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)


def save_json(path, obj):
    def numeric(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=numeric), encoding='utf-8')


def verified_math():
    d = geo.euclidean_distances(np.random.default_rng(91).normal(size=(7, 4)))
    expected = 0.
    for w, x, y, z in permutations(range(7), 4):
        gp = lambda a, b: (d[w, a] + d[w, b] - d[a, b])/2
        expected = max(expected, min(gp(x, y), gp(y, z)) - gp(x, z))
    actual = geo.delta_metric(d)
    np.testing.assert_allclose(actual['delta'], expected, atol=1e-12)
    np.testing.assert_allclose(geo.delta_metric(17*d)['relative_lower_bound'], actual['relative_lower_bound'])
    assert geo.delta_metric(np.abs(np.arange(7)[:, None]-np.arange(7)[None, :]))['delta'] == 0
    assert geo.delta_metric(np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]))['delta'] == 1
    h = geo.group_adjacency(3, [{0, 1}, {0, 1}, {1, 2}], 2)
    assert h[0, 1] and not h[1, 2]
    return {'four_point_equals_equations_1_2_18_19': True, 'tree_zero': True,
            'cycle_four_delta_one': True, 'scale_invariance': True, 's_overlap_rule': True}


def feature_audit(vectors, labels):
    distances = geo.euclidean_distances(vectors)
    result = geo.delta_metric(distances)
    assert result['exact'] and result['quadruples_evaluated'] == result['possible_quadruples']
    if result['witness']:
        result['witness']['labels'] = [labels[i] for i in result['witness']['vertices']]
    result['labels'] = labels
    result['vector_dimensions'] = vectors.shape[1]
    return result, distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('runs/hyperbolicity_20261010'))
    parser.add_argument('--include-original', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out/'math_verification.json', verified_math())
    datasets = [('NYSE_top100_20261010', 'NYSE', 'large_company', Path('runs/largecap_20261010/five_session')),
                ('NASDAQ100_20261010', 'NASDAQ', 'large_company', Path('runs/largecap_20261010/five_session'))]
    if args.include_original:
        datasets += [(f'{m}_recent', m, 'original_sample', Path('runs/robust_2022_2025_final')) for m in ['NYSE', 'NASDAQ']]
    rows, diagnostics = [], []
    for dataset, market, cohort, run_root in datasets:
        data = prepare_panel(load_panel(dataset), horizon=5)
        names = data['panel'].tickers
        for year in range(2022, 2026):
            folder = run_root/f'{dataset}_{year}'
            preprocessing = np.load(folder/'preprocessing.npz')
            train, val, test = fold_indices(data, year)
            for key, value in [('train', train), ('validation', val), ('test', test)]:
                np.testing.assert_array_equal(value, preprocessing[key])
            scaled, mean, std = scale_inputs(data, train)
            np.testing.assert_allclose(mean, preprocessing['mean'], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(std, preprocessing['std'], rtol=1e-6, atol=1e-7)
            checkpoint = torch.load(folder/'think_7.pt', map_location='cpu', weights_only=False)
            groups = [set(g) for g in checkpoint['groups']]
            assert groups == training_groups(data, train)
            base = {'dataset': dataset, 'market': market, 'cohort': cohort, 'test_year': year,
                    'train_signal_start': str(data['panel'].dates[data['origin'][train[0]]].date()),
                    'train_signal_end': str(data['panel'].dates[data['origin'][train[-1]]].date()),
                    'training_samples': len(train)}
            connected = geo.graph_audit(geo.group_adjacency(len(names), groups, s=1), names)
            largest = connected['components'][0]
            record = dict(base, metric='hypergraph_s1', nodes=connected['nodes'],
                          component_count=connected['component_count'], largest_component_nodes=largest['n'],
                          delta_hg=connected['global_delta'], largest_component_delta_hg=largest['delta'],
                          max_component_delta_hg=connected['max_component_lower_bound'],
                          delta_rel=np.nan, diameter=largest['diameter'],
                          graph_relative_delta=largest['relative_lower_bound'], exact=True)
            rows.append(record)
            save_json(args.out/f'{dataset}_{year}_graph_s1.json', dict(base, groups=[sorted(g) for g in groups], **connected))
            # Higher-overlap walks are a declared sensitivity, not the fitted graph.
            s2 = geo.graph_audit(geo.group_adjacency(len(names), groups, s=2), names)
            save_json(args.out/f'{dataset}_{year}_graph_s2.json', dict(base, **s2))
            # One point per stock; every training input window is flattened in time order.
            # Keep only names eligible at ALL training origins, so no missing values or
            # cash/missing-history placeholders determine the primary feature metric.
            selected = data['mask'][train].all(0)
            labels = [name for name, keep in zip(names, selected) if keep]
            raw = data['x'][train][:, :, selected, :].transpose(2, 0, 1, 3)
            transformed = scaled[train][:, :, selected, :].transpose(2, 0, 1, 3)
            standardized = (raw-raw.mean((1, 2), keepdims=True))/np.maximum(raw.std((1, 2), keepdims=True), 1e-8)
            representations = {
                'model_inputs': transformed.reshape(len(labels), -1),
                'per_stock_standardized': standardized.reshape(len(labels), -1),
                'returns_only': transformed[:, :, :, 0].reshape(len(labels), -1)}
            for representation, vectors in representations.items():
                result, distances = feature_audit(vectors, labels)
                info = dict(base, representation=representation,
                            excluded_names=[name for name, keep in zip(names, selected) if not keep], **result)
                save_json(args.out/f'{dataset}_{year}_{representation}.json', info)
                np.savez_compressed(args.out/f'{dataset}_{year}_{representation}_distances.npz', distances=distances, labels=labels)
                rows.append(dict(base, metric=representation, nodes=result['n'], delta_rel=result['relative_lower_bound'],
                                 euclidean_delta=result['delta'], diameter=result['diameter'], exact=result['exact'],
                                 quadruples_evaluated=result['quadruples_evaluated']))
                diagnostics.append(info)
            print(dataset, year, 'graph:', record['delta_hg'], 'largest:', largest['n'], largest['delta'],
                  'features:', len(labels), diagnostics[-3]['relative_lower_bound'], flush=True)
    pd.DataFrame(rows).to_csv(args.out/'geometry_summary.csv', index=False)
    save_json(args.out/'methodology.json', {
        'paper': 'https://tylersnetwork.github.io/papers/icdm22-think.pdf',
        'formulas': {'gromov_product': '(d(w,x)+d(w,y)-d(x,y))/2',
                     'exact_delta': 'max over all unordered four-stock subsets of (largest pair-sum - middle pair-sum)/2',
                     'delta_rel': '2*euclidean_delta / max Euclidean pairwise distance'},
        's': 1, 'graph': 'Exact saved training-only up-to-four positive-correlation-neighbor hyperedges; duplicates retained as trained.',
        'graph_disconnection': 'Global delta_hg undefined for disconnected finite-distance metric; report largest component and all component values, never replace infinities.',
        'temporal_points': 'One stock is one point, flattened 16-observation x 5-feature windows at saved five-session training origins. Overlapping windows have their actual model-input weighting.',
        'feature_processing': 'Primary: saved training-fitted pooled scaler, clip +/-8 and multiply 0.2. Only consistently eligible stocks; no missing-value imputation. Sensitivities: per-stock standardization before flattening, and return channel only.',
        'validation_test_use': 'No held-out feature observations or outcomes enter geometry calculation. Previously computed performance is compared descriptively only.',
        'interpretation': 'These are frozen cohort/fold properties, not exchange-wide constants. Equations match THINK; graph construction and feature tensors are local adaptations, not a reproduction of Table I.',
        'selection_bias': '2026 membership lists applied retrospectively to 2018-2025, as in completed retraining.',
        'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('archive/legacy/hyperbolicity_audit.py'), Path('data/largecap_20261010/adj_close.csv'), Path('data/largecap_20261010/volume.csv')]}})


if __name__ == '__main__':
    main()
