"""Focused checks for time leakage, label alignment, and graph semantics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock_gnn import (StockGCN, correlation_graph, load_prices, make_samples,
                       metrics, predict_latest, prepare, synthetic_prices)


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_prices(days=220, stocks=6)

    def test_labels_and_lags_are_aligned(self):
        x, y, returns, origins = make_samples(self.prices, 20)
        t = origins[0]
        np.testing.assert_allclose(x[0, :, :20], returns.iloc[:20].to_numpy().T, rtol=1e-6)
        expected = (self.prices.iloc[t + 2] / self.prices.iloc[t + 1] - 1) * 100
        np.testing.assert_allclose(y[0], expected, rtol=1e-6)
        self.assertEqual(returns.index[t], self.prices.index[20])

    def test_future_changes_cannot_change_training_pipeline(self):
        first = prepare(self.prices, 20, 2)
        train = first.splits["train"]
        changed = self.prices.copy()
        cutoff = first.target_dates[train][-1]
        # Dramatically alter every price AFTER the last training label.
        mask = changed.index > cutoff
        changed.loc[mask] *= np.linspace(1, 5, mask.sum())[:, None]
        second = prepare(changed, 20, 2)
        for name in ("mean", "scale", "weights", "adjacency"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        np.testing.assert_array_equal(first.x[train], second.x[train])
        np.testing.assert_array_equal(first.y[train], second.y[train])
        self.assertFalse(np.allclose(first.y[first.splits["test"]], second.y[second.splits["test"]]))

    def test_boundary_labels_precede_next_forecast_block(self):
        data = prepare(self.prices, 20, 2)
        for left, right in (("train", "validation"), ("validation", "test")):
            self.assertLess(data.target_dates[data.splits[left]][-1], data.dates[data.splits[right]][0])

    def test_feature_window_never_uses_next_price(self):
        first, labels, _, _ = make_samples(self.prices)
        changed = self.prices.copy()
        changed.iloc[21] *= 2
        second, new_labels, _, _ = make_samples(changed)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertFalse(np.allclose(labels[0], new_labels[0]))

    def test_graph_normalization_and_constant_stock(self):
        t = np.arange(10, dtype=float)
        _, a = correlation_graph(np.column_stack([t, t, np.zeros(10)]), k=1)
        np.testing.assert_allclose(a, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]], atol=1e-6)

    def test_identity_graph_is_independent_per_stock(self):
        model = StockGCN(23, 8, torch.eye(6)).eval()
        x = torch.randn(4, 6, 23)
        with torch.no_grad():
            before = model(x)
            x[:, 0] += 10
            after = model(x)
        torch.testing.assert_close(before[:, 1:], after[:, 1:])

    def test_model_is_equivariant_to_consistent_node_reordering(self):
        data = prepare(self.prices, 20, 2)
        model = StockGCN(23, 8, torch.tensor(data.adjacency)).eval()
        permutation = torch.tensor([5, 3, 1, 2, 0, 4])
        x = torch.tensor(data.x[:3])
        with torch.no_grad():
            before = model(x)
            model.adjacency = model.adjacency[permutation][:, permutation]
            after = model(x[:, permutation])
        torch.testing.assert_close(before[:, permutation], after)

    def test_saved_checkpoint_restores_forecast_and_column_order(self):
        data = prepare(self.prices, 20, 2)
        model = StockGCN(23, 8, torch.tensor(data.adjacency)).eval()
        checkpoint = {"state_dict": model.state_dict(), "tickers": self.prices.columns.tolist(),
                      "lookback": 20, "hidden": 8, "mean": torch.tensor(data.mean),
                      "scale": torch.tensor(data.scale)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=True)
        original = predict_latest(checkpoint, self.prices)
        reordered = predict_latest(restored, self.prices.iloc[:, ::-1])
        np.testing.assert_array_equal(original, reordered)
        # The last supervised input is the same as latest prediction without the final row.
        with torch.no_grad():
            expected = model(torch.tensor(data.x[-1:]))[0].numpy()
        np.testing.assert_allclose(predict_latest(restored, self.prices.iloc[:-1]), expected, atol=1e-6)

    def test_loader_rejects_missing_prices_and_unsorted_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            changed = self.prices.copy()
            changed.iloc[2, 1] = np.nan
            changed.to_csv(path)
            with self.assertRaises(ValueError):
                load_prices(path)
            self.prices.iloc[::-1].to_csv(path)
            with self.assertRaises(ValueError):
                load_prices(path)

    def test_zero_forecast_has_zero_relative_r2_and_undefined_rank(self):
        actual = np.array([[1., -2., 0.], [2., 1., -1.]])
        result = metrics(actual, np.zeros_like(actual))
        self.assertEqual(result["r2_vs_zero"], 0)
        self.assertIsNone(result["mean_daily_rank_ic"])
        self.assertAlmostEqual(result["sign_accuracy"], 1 / 6)

    def test_loader_rejects_duplicate_tickers_before_pandas_renames_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("Date,AAPL,AAPL\n2020-01-02,100,101\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_prices(path)

    def test_weekly_target_is_compounded_five_session_return(self):
        # 10% every session: five-session return is 61.051%, not 50% or 10%.
        prices = pd.DataFrame(np.tile((100 * 1.1 ** np.arange(40))[:, None], (1, 2)),
                              index=pd.bdate_range("2020-01-01", periods=40), columns=["A", "B"])
        x, y, returns, origins = make_samples(prices, 20, horizon=5)
        self.assertEqual(len(x), 15)
        np.testing.assert_allclose(y, 61.051, atol=1e-5)
        self.assertEqual(returns.index[origins[0] + 5], prices.index[25])
        np.testing.assert_allclose(x[0, :, :20], 10, atol=1e-5)

    def test_horizons_preserve_calendar_boundaries_and_purge_labels(self):
        daily = prepare(self.prices, 20, 2)
        for horizon in (5, 21):
            with self.subTest(horizon=horizon):
                data = prepare(self.prices, 20, 2, horizon)
                for left, right in (("train", "validation"), ("validation", "test")):
                    self.assertLess(data.target_dates[data.splits[left]][-1],
                                    data.dates[data.splits[right]][0])
                    self.assertEqual(data.dates[data.splits[right]][0],
                                     daily.dates[daily.splits[right]][0])
                    self.assertEqual(data.target_dates[data.splits[left]][-1],
                                     daily.target_dates[daily.splits[left]][-1])
                    self.assertEqual(data.splits[right].start - data.splits[left].stop, horizon)
                self.assertEqual(data.target_dates[-1], self.prices.index[-1])

    def test_weekly_future_cannot_change_training_and_latest_input_alignment(self):
        first = prepare(self.prices, 20, 2, horizon=5)
        train = first.splits["train"]
        changed = self.prices.copy()
        mask = changed.index > first.target_dates[train][-1]
        changed.loc[mask] *= np.linspace(1, 5, mask.sum())[:, None]
        second = prepare(changed, 20, 2, horizon=5)
        for name in ("mean", "scale", "weights", "adjacency"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        np.testing.assert_array_equal(first.x[train], second.x[train])
        np.testing.assert_array_equal(first.y[train], second.y[train])
        model = StockGCN(23, 8, torch.tensor(first.adjacency)).eval()
        checkpoint = {"state_dict": model.state_dict(), "tickers": self.prices.columns.tolist(),
                      "lookback": 20, "hidden": 8, "mean": torch.tensor(first.mean),
                      "scale": torch.tensor(first.scale), "horizon_sessions": 5}
        with torch.no_grad():
            expected = model(torch.tensor(first.x[-1:]))[0].numpy()
        np.testing.assert_allclose(predict_latest(checkpoint, self.prices.iloc[:-5]), expected, atol=1e-6)

    def test_nonoverlap_weekly_targets_share_no_return_intervals(self):
        data = prepare(self.prices, 20, 2, horizon=5)
        ids = np.arange(data.splits["test"].start, len(data.y), 5)
        self.assertGreater(len(ids), 1)
        # Touching endpoint prices are okay; no daily return is counted twice.
        self.assertTrue((data.target_dates[ids[:-1]] <= data.dates[ids[1:]]).all())

    def test_invalid_or_unavailable_horizons_fail_clearly(self):
        for horizon in (0, -1, 1.5, 500):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                make_samples(self.prices, 20, horizon=horizon)

    def test_monthly_target_is_exact_21_session_total_return(self):
        prices = pd.DataFrame(np.tile((100 * 1.01 ** np.arange(80))[:, None], (1, 2)),
                              index=pd.bdate_range("2020-01-01", periods=80), columns=["A", "B"])
        x, y, returns, origins = make_samples(prices, 20, horizon=21)
        self.assertEqual(len(x), 39)
        np.testing.assert_allclose(y, 100 * (1.01 ** 21 - 1), atol=1e-5)
        self.assertEqual(returns.index[origins[0] + 21], prices.index[41])
        # A future endpoint changes the target but not the information seen now.
        changed = prices.copy()
        changed.iloc[41] *= 2
        changed_x, changed_y, _, _ = make_samples(changed, 20, horizon=21)
        np.testing.assert_array_equal(x[0], changed_x[0])
        self.assertFalse(np.allclose(y[0], changed_y[0]))

    def test_monthly_future_is_excluded_and_checkpoint_uses_origin_features(self):
        data = prepare(self.prices, 20, 2, horizon=21)
        train = data.splits["train"]
        changed = self.prices.copy()
        mask = changed.index > data.target_dates[train][-1]
        changed.loc[mask] *= np.linspace(1, 5, mask.sum())[:, None]
        other = prepare(changed, 20, 2, horizon=21)
        for name in ("mean", "scale", "weights", "adjacency"):
            np.testing.assert_array_equal(getattr(data, name), getattr(other, name))
        np.testing.assert_array_equal(data.x[train], other.x[train])
        np.testing.assert_array_equal(data.y[train], other.y[train])
        model = StockGCN(23, 8, torch.tensor(data.adjacency)).eval()
        checkpoint = {"state_dict": model.state_dict(), "tickers": self.prices.columns.tolist(),
                      "lookback": 20, "hidden": 8, "mean": torch.tensor(data.mean),
                      "scale": torch.tensor(data.scale), "horizon_sessions": 21}
        with torch.no_grad():
            expected = model(torch.tensor(data.x[-1:]))[0].numpy()
        np.testing.assert_allclose(predict_latest(checkpoint, self.prices.iloc[:-21]), expected, atol=1e-6)
        ids = np.arange(data.splits["test"].start, len(data.y), 21)
        self.assertTrue((data.target_dates[ids[:-1]] <= data.dates[ids[1:]]).all())


if __name__ == "__main__":
    unittest.main()
