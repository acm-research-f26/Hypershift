"""Opt-in integration test for a research-scale hyperbolicity calculation.

The dataset is deterministic and market-like rather than downloaded. This keeps
the test reproducible and prevents network availability or revised vendor data
from changing its result.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from hyperbolicity.dataset import summarize_dataset


N_SECTORS = 25
STOCKS_PER_SECTOR = 10
N_STOCKS = N_SECTORS * STOCKS_PER_SECTOR
N_PRICE_ROWS = 504
N_STYLE_GROUPS = STOCKS_PER_SECTOR


def _market_like_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create two years of correlated prices and overlapping memberships."""
    rng = np.random.default_rng(20260928)
    tickers = [f"STOCK_{index:03d}" for index in range(N_STOCKS)]
    sector_ids = np.repeat(np.arange(N_SECTORS), STOCKS_PER_SECTOR)
    style_ids = np.tile(np.arange(N_STYLE_GROUPS), N_SECTORS)

    n_return_rows = N_PRICE_ROWS - 1
    market_returns = rng.normal(0.0002, 0.0080, size=(n_return_rows, 1))
    sector_returns = rng.normal(
        0.0,
        0.0060,
        size=(n_return_rows, N_SECTORS),
    )
    style_returns = rng.normal(
        0.0,
        0.0030,
        size=(n_return_rows, N_STYLE_GROUPS),
    )
    idiosyncratic_returns = rng.normal(
        0.0,
        0.0080,
        size=(n_return_rows, N_STOCKS),
    )

    log_returns = (
        market_returns
        + 0.60 * sector_returns[:, sector_ids]
        + 0.30 * style_returns[:, style_ids]
        + idiosyncratic_returns
    )
    initial_prices = rng.uniform(25.0, 250.0, size=N_STOCKS)
    log_prices = np.vstack(
        [
            np.log(initial_prices),
            np.log(initial_prices) + np.cumsum(log_returns, axis=0),
        ]
    )
    close = pd.DataFrame(
        np.exp(log_prices),
        index=pd.date_range("2024-01-02", periods=N_PRICE_ROWS, freq="B"),
        columns=tickers,
    )

    memberships = np.zeros(
        (N_STOCKS, N_SECTORS + N_STYLE_GROUPS),
        dtype=np.int64,
    )
    stock_indices = np.arange(N_STOCKS)
    memberships[stock_indices, sector_ids] = 1
    memberships[stock_indices, N_SECTORS + style_ids] = 1
    incidence = pd.DataFrame(
        memberships,
        index=tickers,
        columns=[
            *(f"sector_{index:02d}" for index in range(N_SECTORS)),
            *(f"style_{index:02d}" for index in range(N_STYLE_GROUPS)),
        ],
    )
    return close, incidence


@pytest.mark.skipif(
    os.environ.get("RUN_HYPERSHIFT_LONG_TESTS") != "1",
    reason="Set RUN_HYPERSHIFT_LONG_TESTS=1 to run the 250-stock integration test.",
)
def test_relative_hyperbolicity_on_large_market_like_dataset() -> None:
    close, incidence = _market_like_fixture()

    result = summarize_dataset(
        close,
        incidence,
        dataset="synthetic_250_stock_factor_market",
        s=1,
        method="sampled",
        n_samples=100_000,
        seed=20260928,
    )

    assert len(result) == 1
    row = result.iloc[0]

    print(
        "\nLarge synthetic-market hyperbolicity result"
        f"\n  delta_hg       = {row['delta_hg']:.6f}"
        f"\n  delta_features = {row['delta_features']:.6f}"
        f"\n  feature diameter = {row['diameter_features']:.6f}"
        f"\n  delta_rel      = {row['delta_rel']:.6f}"
    )

    assert row["dataset"] == "synthetic_250_stock_factor_market"
    assert row["n_nodes"] == N_STOCKS
    assert row["n_price_rows"] == N_PRICE_ROWS
    assert row["n_timesteps"] == N_PRICE_ROWS - 1
    assert row["method"] == "sampled_lower_bound"
    assert row["n_samples"] == 100_000
    assert row["seed"] == 20260928

    assert np.isfinite(row["delta_hg"])
    assert np.isfinite(row["delta_features"])
    assert np.isfinite(row["delta_rel"])
    assert row["diameter_hg"] == pytest.approx(2.0)
    assert row["diameter_features"] > 0.0
    assert row["delta_hg"] == pytest.approx(1.0)
    assert row["delta_features"] > 0.0
    assert row["delta_rel"] == pytest.approx(
        2.0 * row["delta_features"] / row["diameter_features"]
    )
