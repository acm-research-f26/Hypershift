"""Checks for geometry, evaluation semantics, leakage, and saved inference."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from hypergraph import (DirectionNet, correlation_hyperedges, exp0, log0,
                        poincare_distance, einstein_midpoint, four_point_delta, hypergraph_geometry)
from direction_metrics import classification, select_threshold, ndcg, topk_weights, trading_backtest
from stock_gnn import prepare, synthetic_prices
from train_direction import predict_direction


class DirectionTests(unittest.TestCase):
    def test_known_hyperbolicity_metrics(self):
        line = np.abs(np.arange(5)[:, None] - np.arange(5)[None])
        self.assertEqual(four_point_delta(line)['delta'], 0)
        clique = np.ones((5, 5)) - np.eye(5)
        self.assertEqual(four_point_delta(clique)['delta'], 0)
        circle = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
        self.assertEqual(four_point_delta(circle)['delta'], 1)

    def test_disconnected_and_s_walk(self):
        h = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertTrue(hypergraph_geometry(h, 1)['connected'])
        result = hypergraph_geometry(h, 2)
        self.assertIsNone(result['global_delta'])
        self.assertEqual(len(result['components']), 4)

    def test_roundtrip_midpoint_distance_and_gradients(self):
        torch.manual_seed(3)
        v = (torch.randn(2, 5, 8, dtype=torch.float64) * .12).requires_grad_()
        x = exp0(v)
        torch.testing.assert_close(log0(x), v)
        identity = torch.eye(5, dtype=torch.float64)
        torch.testing.assert_close(einstein_midpoint(x, identity), x)
        torch.testing.assert_close(poincare_distance(x, x), torch.zeros(2, 5, dtype=torch.float64), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(poincare_distance(x[:, 0], x[:, 1]), poincare_distance(x[:, 1], x[:, 0]))
        poincare_distance(x[:, 0], x[:, 1]).sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        zero = torch.zeros(1, 3, requires_grad=True)
        log0(exp0(zero)).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_model_gradients_and_permutation(self):
        h = correlation_hyperedges(np.random.default_rng(1).normal(size=(80, 5)), 3)
        x = torch.randn(4, 5, 23)
        permutation = torch.tensor([4, 1, 3, 0, 2])
        for kind in ['logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic']:
            model = DirectionNet(23, 8, kind, h, np.eye(5)).eval()
            before = model(x)
            before.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            model.incidence = model.incidence[permutation]
            model.adjacency = model.adjacency[permutation][:, permutation]
            torch.testing.assert_close(model(x[:, permutation]), before[:, permutation])

    def test_training_hyperedges_ignore_future_prices(self):
        prices = synthetic_prices(300, 6)
        data = prepare(prices, 20, 3, 21)
        cutoff = data.dates[data.splits['train']][-1]
        def graph(p):
            return correlation_hyperedges(p.pct_change(fill_method=None).loc[:cutoff].iloc[1:].to_numpy(), 3)
        old = graph(prices)
        prices.loc[prices.index > cutoff] *= np.arange(1, (prices.index > cutoff).sum() + 1)[:, None]
        np.testing.assert_array_equal(old, graph(prices))

    def test_classification_detects_always_up(self):
        y = np.array([1, 1, 1, 0, 0])
        result = classification(y, np.ones(5))
        self.assertEqual(result['accuracy'], .6)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['roc_auc'], .5)
        self.assertEqual(result['macro_f1'], .375)
        perfect = classification(y, y)
        self.assertEqual(perfect['mcc'], 1)
        self.assertEqual(perfect['macro_f1'], 1)
        self.assertEqual(perfect['roc_auc'], 1)

    def test_tied_ranking_and_all_negative_dates(self):
        actual = np.array([[3., 2., 1., -1.], [-1, -2, -3, -4]])
        self.assertEqual(ndcg(actual, actual, 2)['ndcg_at_k'], 1)
        result = ndcg(actual, np.zeros_like(actual), 2)
        expected = 1.5 * (1 + 1 / np.log2(3)) / (3 + 2 / np.log2(3))
        self.assertAlmostEqual(result['ndcg_at_k'], expected)
        self.assertEqual(result['dates_excluded'], 1)
        np.testing.assert_allclose(topk_weights(np.ones(4), 2), [.25] * 4)
        np.testing.assert_allclose(topk_weights(np.array([3, 2, 2, 1]), 2), [.5, .25, .25, 0])

    def test_backtest_delay_cost_and_nonoverlap(self):
        dates = pd.bdate_range('2020-01-01', periods=9)
        prices = pd.DataFrame({'A': [100, 200, 200, 220, 220, 220, 220, 220, 220],
                               'B': [100] * 9}, index=dates)
        result, rows = trading_backtest(prices, dates[:6], np.tile([1, 0], (6, 1)), 2, 1, 10)
        # First signal sees 100; execution is 200, so gain is 10%, not 120%.
        self.assertAlmostEqual(rows[0]['gross_return'], .1)
        self.assertAlmostEqual(rows[0]['net_return'], .999 ** 2 * 1.1 - 1)
        self.assertEqual(rows[0]['entry_date'], str(dates[1].date()))
        self.assertLess(rows[0]['exit_date'], rows[1]['entry_date'])
        self.assertEqual(result['periods'], 2)

    def test_threshold_is_selected_from_fixed_grid(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.4, .5, .6, .7])
        threshold = select_threshold(y, p)
        self.assertGreater(threshold, .5)
        self.assertEqual(classification(y, p, threshold)['macro_f1'], 1)

    def test_checkpoint_restores_direction_in_column_order(self):
        prices = synthetic_prices(250, 6)
        data = prepare(prices, 20, 3, 21)
        h = correlation_hyperedges(prices.pct_change().iloc[1:100].to_numpy(), 3)
        model = DirectionNet(23, 8, 'hyperbolic', h, data.adjacency).eval()
        checkpoint = {'state_dict': model.state_dict(), 'kind': 'hyperbolic', 'hidden': 8,
                      'incidence': torch.tensor(h), 'adjacency': torch.tensor(data.adjacency),
                      'mean': torch.tensor(data.mean), 'scale': torch.tensor(data.scale),
                      'lookback': 20, 'horizon': 21, 'tickers': prices.columns.tolist(), 'threshold': .55}
        # Last sample's information set ends exactly at its origin.
        truncated = prices.loc[:data.dates[-1]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=True)
        result = predict_direction(restored, truncated[truncated.columns[::-1]])
        with torch.no_grad():
            expected = model(torch.tensor(data.x[-1:])).sigmoid()[0].numpy()
        np.testing.assert_allclose(result.probability_up, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
