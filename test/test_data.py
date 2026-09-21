"""Leakage and shape tests for real-stock graph preparation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import data.data as data_module


TICKERS = ("AAA", "BBB", "CCC", "DDD")


def _market_frames(days: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-03", periods=days)
    time = np.arange(days, dtype=float)
    common = 0.002 * np.sin(time / 5.0) + 0.001 * np.cos(time / 11.0)

    returns = np.column_stack(
        [
            common + 0.0015 * np.sin(time / (3.0 + index)) + 0.0002 * index
            for index in range(len(TICKERS))
        ]
    )
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=TICKERS,
    )
    volume = pd.DataFrame(
        {
            ticker: 1_000_000.0
            + 1_000.0 * time
            + 25_000.0 * np.sin(time / (4.0 + index))
            for index, ticker in enumerate(TICKERS)
        },
        index=dates,
    )
    return close, volume


def test_compute_feature_frames_uses_only_backward_looking_values() -> None:
    close, volume = _market_frames(30)
    features, log_returns = data_module.compute_feature_frames(
        close,
        volume,
        volatility_window=3,
    )

    assert list(features.columns.names) == ["ticker", "feature"]
    assert set(features.columns.get_level_values("feature")) == {
        "log_return",
        "log_volume_change",
        "rolling_volatility",
    }
    assert np.isfinite(features.to_numpy()).all()
    first_date = features.index[0]
    previous_date = close.index[close.index.get_loc(first_date) - 1]
    expected_return = np.log(close.loc[first_date, "AAA"] / close.loc[previous_date, "AAA"])
    assert features.loc[first_date, ("AAA", "log_return")] == pytest.approx(
        expected_return
    )
    assert log_returns.loc[first_date, "AAA"] == pytest.approx(expected_return)


def test_top_k_correlation_graph_is_reciprocal_valid_and_loop_free() -> None:
    close, _ = _market_frames(60)
    returns = np.log(close).diff().dropna()

    edge_index = data_module.build_correlation_edge_index(returns, neighbors=1)

    assert edge_index.dtype == torch.long
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert int(edge_index.min()) >= 0
    assert int(edge_index.max()) < len(TICKERS)
    edges = set(map(tuple, edge_index.t().tolist()))
    assert all(source != target for source, target in edges)
    assert all((target, source) in edges for source, target in edges)
    assert all(any(source == node for source, _ in edges) for node in range(len(TICKERS)))


def test_prepare_graph_datasets_shapes_order_and_training_scaler() -> None:
    close, volume = _market_frames()
    lookback = 6
    splits = data_module.prepare_graph_datasets(
        close,
        volume,
        lookback=lookback,
        volatility_window=3,
        neighbors=1,
    )

    assert splits.train and splits.validation and splits.test
    assert max(splits.train_dates) < min(splits.validation_dates)
    assert max(splits.validation_dates) < min(splits.test_dates)
    assert splits.tickers == TICKERS

    for sample in (*splits.train, *splits.validation, *splits.test):
        assert sample.x.shape == (len(TICKERS), lookback * 3)
        assert sample.y.shape == (len(TICKERS), 1)
        assert torch.isfinite(sample.x).all()
        assert torch.isfinite(sample.y).all()
        torch.testing.assert_close(sample.edge_index, splits.edge_index)

    stacked = torch.stack(
        [sample.x.reshape(len(TICKERS), lookback, 3) for sample in splits.train]
    )
    torch.testing.assert_close(
        stacked.mean(dim=(0, 1, 2)),
        torch.zeros(3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        stacked.std(dim=(0, 1, 2), unbiased=False),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )

    first_date = splits.train_dates[0]
    expected_target = np.log(close).diff().loc[first_date, list(TICKERS)].to_numpy()
    np.testing.assert_allclose(
        splits.train[0].y.squeeze(1).numpy(),
        expected_target,
        rtol=1e-5,
        atol=1e-7,
    )


def test_future_changes_cannot_affect_training_features_scaler_or_graph() -> None:
    close, volume = _market_frames()
    original = data_module.prepare_graph_datasets(
        close,
        volume,
        lookback=6,
        volatility_window=3,
        neighbors=1,
    )
    cutoff = original.train_dates[-1]
    future_mask = close.index > cutoff
    changed_close = close.copy()
    changed_volume = volume.copy()
    changed_close.loc[future_mask] *= np.exp(
        np.linspace(0.01, 0.25, int(future_mask.sum()))[:, None]
    )
    changed_volume.loc[future_mask] *= 3.0

    changed = data_module.prepare_graph_datasets(
        changed_close,
        changed_volume,
        lookback=6,
        volatility_window=3,
        neighbors=1,
    )

    torch.testing.assert_close(changed.edge_index, original.edge_index)
    torch.testing.assert_close(changed.feature_mean, original.feature_mean)
    torch.testing.assert_close(changed.feature_std, original.feature_std)
    assert changed.train_dates == original.train_dates
    for before, after in zip(original.train, changed.train, strict=True):
        torch.testing.assert_close(after.x, before.x)
        torch.testing.assert_close(after.y, before.y)


def test_download_adapter_passes_required_arguments_without_real_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = pd.MultiIndex.from_product([TICKERS, ("Close", "Volume")])
    values = np.empty((len(dates), len(columns)), dtype=float)
    for column, (_, field) in enumerate(columns):
        values[:, column] = 100.0 + column if field == "Close" else 1_000_000.0 + column
    raw = pd.DataFrame(values, index=dates, columns=columns)
    recorded: dict[str, object] = {}

    def fake_download(tickers: object, **kwargs: object) -> pd.DataFrame:
        recorded["tickers"] = tickers
        recorded.update(kwargs)
        return raw

    monkeypatch.setattr(
        data_module,
        "yf",
        SimpleNamespace(download=fake_download),
        raising=False,
    )
    close, volume = data_module.download_stock_data(
        TICKERS,
        start="2021-01-01",
        end="2026-01-01",
    )

    assert list(close.columns) == list(TICKERS)
    assert list(volume.columns) == list(TICKERS)
    assert recorded["start"] == "2021-01-01"
    assert recorded["end"] == "2026-01-01"
    assert recorded["interval"] == "1d"
    assert recorded["auto_adjust"] is True
    assert recorded["group_by"] == "ticker"
    assert recorded["progress"] is False
    assert recorded["threads"] is False


def test_zero_return_baseline_is_target_mean_square() -> None:
    close, volume = _market_frames()
    splits = data_module.prepare_graph_datasets(
        close,
        volume,
        lookback=6,
        volatility_window=3,
        neighbors=1,
    )
    targets = torch.cat([sample.y.reshape(-1) for sample in splits.test])

    baseline = data_module.zero_return_baseline_mse(splits.test)

    assert baseline == pytest.approx(float(targets.square().mean()))
