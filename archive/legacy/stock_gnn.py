"""A small, inspectable stock-return GCN. Read TUTORIAL.md alongside this file.

Run: python stock_gnn.py --csv data/prices.csv --horizon 21 --out runs/monthly
All return values and predictions use percentage points: 1.0 means +1%.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


# 1. Data: an offline teaching dataset, or a strict adjusted-close CSV.
def synthetic_prices(days: int = 1500, stocks: int = 12, seed: int = 7) -> pd.DataFrame:
    """Invent market/sector factors with persistence; these are NOT market data."""
    if days < 100 or stocks < 2:
        raise ValueError("Use at least 100 dates and 2 stocks.")
    rng = np.random.default_rng(seed)
    sectors = np.arange(stocks) % 3
    market = np.zeros(days - 1)
    sector = np.zeros((days - 1, 3))
    for t in range(1, days - 1):
        market[t] = 0.15 * market[t - 1] + rng.normal(0, 0.005)
        sector[t] = 0.55 * sector[t - 1] + rng.normal(0, 0.006, 3)
    returns = 0.0002 + market[:, None] + sector[:, sectors]
    returns += rng.normal(0, 0.008, (days - 1, stocks))
    prices = np.vstack([np.ones(stocks), np.cumprod(1 + returns, axis=0)]) * 100
    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2018-01-01", periods=days, name="Date"),
        columns=[f"SYN{i:02d}" for i in range(stocks)],
    )


def load_prices(path: str | Path) -> pd.DataFrame:
    """Require aligned observations; never silently fill or drop missing prices."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])[1:]
    if len(set(header)) != len(header) or any(not ticker.strip() for ticker in header):
        raise ValueError("CSV stock column names must be nonempty and unique.")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("The first CSV column must contain valid dates.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and in increasing order.")
    if len(prices.columns) < 2 or prices.columns.has_duplicates:
        raise ValueError("Provide at least two unique stock columns.")
    try:
        prices = prices.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Every price must be numeric.") from exc
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, positive, and complete; fix missing data explicitly.")
    return prices


def feature_names(lookback: int) -> list[str]:
    """Names in the exact input-column order, for inspection of saved models."""
    return [f"daily_return_lag_{lag}" for lag in range(lookback - 1, -1, -1)] + [
        "mean_daily_return_5_sessions", "mean_daily_return_20_sessions",
        "daily_return_std_20_sessions"]


def window_features(window: np.ndarray) -> np.ndarray:
    """[lookback, stocks] -> [stocks, lookback + 3], oldest lag first."""
    return np.concatenate(
        [window.T, window[-5:].mean(0)[:, None], window[-20:].mean(0)[:, None],
         window[-20:].std(0, ddof=0)[:, None]], axis=1,
    )


def make_samples(prices: pd.DataFrame, lookback: int = 20, horizon: int = 1):
    if lookback < 20:
        raise ValueError("lookback must be >= 20 for the 20-session features.")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of trading sessions.")
    returns = prices.pct_change(fill_method=None).iloc[1:] * 100.0
    values = returns.to_numpy()
    if len(prices) < lookback + horizon + 1:
        raise ValueError("Not enough prices for the lookback and the full target horizon.")
    # t indexes daily returns; its corresponding price is at t+1.
    # Predict the cumulative, compounded return over h sessions, not the average
    # daily return, the sum of returns, or just the return on the last future day.
    origins = np.arange(lookback - 1, len(values) - horizon)
    x = np.stack([window_features(values[t - lookback + 1:t + 1]) for t in origins])
    price_values = prices.to_numpy()
    y = 100.0 * (price_values[origins + 1 + horizon] / price_values[origins + 1] - 1)
    return x.astype(np.float32), y.astype(np.float32), returns, origins


# 2. Chronological split and training-only preprocessing.
def chronological_splits(samples: int, horizon: int = 1) -> dict[str, slice]:
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if samples < 100:
        raise ValueError("Need at least 100 usable samples after lookback and horizon.")
    # Anchor boundaries to the SAME calendar positions as the daily experiment.
    # Longer horizons remove samples at the dataset tail, not shift test start.
    daily_samples = samples + horizon - 1
    train_boundary, val_boundary = int(0.6 * daily_samples), int(0.8 * daily_samples)
    # Purge h origins at each boundary: every earlier label must be known strictly
    # before the next block's first forecast. h=1 reproduces the original split.
    splits = {"train": slice(0, train_boundary - horizon + 1),
              "validation": slice(train_boundary + 1, val_boundary - horizon + 1),
              "test": slice(val_boundary + 1, samples)}
    if any(part.stop <= part.start for part in splits.values()):
        raise ValueError("Horizon is too long for these split sizes; use more historical data.")
    return splits


def fit_scaler(train_x: np.ndarray):
    # One mean/std per feature; pool training dates and stocks, retain feature axis.
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    scale = train_x.std(axis=(0, 1), keepdims=True)
    return mean, np.maximum(scale, 1e-6)


def correlation_graph(train_returns: np.ndarray, k: int = 3):
    """Positive top-k return correlations, symmetric union, then GCN normalization."""
    n = train_returns.shape[1]
    if not 0 <= k < n:
        raise ValueError(f"neighbors must be between 0 and {n - 1}.")
    # Compute correlation without NaNs for a constant-return stock.
    centered = train_returns - train_returns.mean(0)
    norm = np.sqrt((centered ** 2).sum(0))
    z = centered / np.where(norm > 1e-12, norm, 1.0)
    corr = np.clip(z.T @ z, 0, 1)
    np.fill_diagonal(corr, 0)
    weights = np.zeros((n, n), dtype=np.float64)
    if k:
        for i in range(n):
            neighbors = np.argsort(-corr[i], kind="stable")[:k]
            weights[i, neighbors] = corr[i, neighbors]
    weights = np.maximum(weights, weights.T)
    with_self = weights + np.eye(n)
    inverse_sqrt_degree = 1 / np.sqrt(with_self.sum(axis=1))
    normalized = inverse_sqrt_degree[:, None] * with_self * inverse_sqrt_degree[None, :]
    return weights.astype(np.float32), normalized.astype(np.float32)


@dataclass
class Prepared:
    x: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    splits: dict[str, slice]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    adjacency: np.ndarray


def prepare(prices: pd.DataFrame, lookback: int, k: int, horizon: int = 1) -> Prepared:
    x, y, returns, origins = make_samples(prices, lookback, horizon)
    splits = chronological_splits(len(x), horizon)
    mean, scale = fit_scaler(x[splits["train"]])
    last_train_origin = origins[splits["train"].stop - 1]
    weights, adjacency = correlation_graph(returns.iloc[:last_train_origin + 1].to_numpy(), k)
    return Prepared((x - mean) / scale, y, returns.index[origins],
                    returns.index[origins + horizon], splits, mean, scale, weights, adjacency)


# 3. The entire graph neural network: two shared linear maps and two aggregations.
class StockGCN(nn.Module):
    def __init__(self, features: int, hidden: int, adjacency: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.first = nn.Linear(features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.second = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch dates, stocks, features]; adjacency: [stocks, stocks].
        h = torch.relu(self.first(self.adjacency @ x))
        h = self.dropout(h)
        return self.second(self.adjacency @ h).squeeze(-1)  # [batch dates, stocks]


# 4. Training: backpropagation on training data; early stopping on validation only.
def train_model(data: Prepared, adjacency: np.ndarray, args):
    torch.manual_seed(args.seed)
    model = StockGCN(data.x.shape[-1], args.hidden, torch.from_numpy(adjacency))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_function = nn.MSELoss()
    x, y = torch.from_numpy(data.x), torch.from_numpy(data.y)
    train, val = data.splits["train"], data.splits["validation"]
    generator = torch.Generator().manual_seed(args.seed)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train.stop, generator=generator)
        total = 0.0
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(x[val]), y[val]).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Training diverged. Check the data and lower the learning rate.")
        history.append({"epoch": epoch, "train_mse_pp2": total / train.stop,
                        "validation_mse_pp2": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x).numpy()
    return model, predictions, history, best_epoch


# 5. Baselines and metrics: judge forecasts before considering a trading strategy.
def ridge_predictions(x: np.ndarray, y: np.ndarray, train: slice, alpha: float = 1.0):
    design = np.concatenate([x.astype(np.float64), np.ones((*x.shape[:2], 1))], axis=-1)
    a, b = design[train].reshape(-1, design.shape[-1]), y[train].reshape(-1)
    penalty = np.eye(a.shape[1]) * alpha
    penalty[-1, -1] = 0  # Do not penalize the intercept.
    coefficients = np.linalg.solve(a.T @ a + penalty, a.T @ b)
    return design @ coefficients


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mse = float(np.mean((actual - predicted) ** 2))
    zero_mse = float(np.mean(actual ** 2))
    correlations = []
    for observed, forecast in zip(actual, predicted):
        a = pd.Series(observed).rank().to_numpy()
        p = pd.Series(forecast).rank().to_numpy()
        if np.std(a) > 0 and np.std(p) > 0:
            correlations.append(float(np.corrcoef(a, p)[0, 1]))
    return {
        "rmse_pp": float(np.sqrt(mse)),
        "mae_pp": float(np.mean(np.abs(actual - predicted))),
        "r2_vs_zero": 1 - mse / zero_mse if zero_mse > 0 else None,
        "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "mean_daily_rank_ic": float(np.mean(correlations)) if correlations else None,
        "rank_ic_days": len(correlations),
    }


def predict_latest(checkpoint: dict, prices: pd.DataFrame) -> pd.Series:
    """Use saved training preprocessing and graph on the last observed close."""
    if set(prices.columns) != set(checkpoint["tickers"]):
        raise ValueError("Prediction CSV must contain exactly the checkpoint's stock universe.")
    prices = prices.loc[:, checkpoint["tickers"]]  # Node order must match adjacency.
    lookback = checkpoint["lookback"]
    if len(prices) < lookback + 1:
        raise ValueError(f"Prediction needs at least {lookback + 1} complete price rows.")
    window = prices.pct_change(fill_method=None).iloc[-lookback:].to_numpy() * 100.0
    x = torch.tensor(window_features(window)[None], dtype=torch.float32)
    x = (x - checkpoint["mean"]) / checkpoint["scale"]
    model = StockGCN(x.shape[-1], checkpoint["hidden"], checkpoint["state_dict"]["adjacency"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        forecast = model(x)[0].numpy()
    return pd.Series(forecast, index=checkpoint["tickers"], name="predicted_return_pp")


# 6. A reproducible experiment with inspectable output files.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Wide adjusted-close CSV; omit for synthetic data.")
    parser.add_argument("--out", type=Path, default=Path("runs/monthly_demo"))
    parser.add_argument("--horizon", type=int, default=21,
                        help="Forecast total return over this many trading sessions (default: 21, about one month).")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.hidden, args.batch_size, args.horizon) < 1 or not 0 < args.lr < 1:
        parser.error("epochs, patience, hidden, batch-size, horizon must be positive; require 0 < lr < 1.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty. Choose a new --out to preserve old experiments.")
    torch.set_num_threads(1)  # Small matrices: avoids excessive CPU thread overhead.
    torch.use_deterministic_algorithms(True)
    prices = load_prices(args.csv) if args.csv else synthetic_prices(seed=args.seed)
    data = prepare(prices, args.lookback, args.neighbors, args.horizon)
    train = data.splits["train"]
    forecasts = {
        "zero": np.zeros_like(data.y),
        "train_mean": np.broadcast_to(data.y[train].mean(0), data.y.shape),
        "ridge": ridge_predictions(data.x, data.y, train),
    }
    source = "user_csv" if args.csv else "SYNTHETIC_TEACHING_DATA"
    provenance = None
    if args.csv and args.csv.with_suffix(".metadata.json").exists():
        provenance = json.loads(args.csv.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if provenance["sha256"] != hashlib.sha256(args.csv.read_bytes()).hexdigest():
            raise ValueError("Price CSV differs from its metadata checksum. Resolve the source mismatch.")
        source = provenance["provider"]
    print(f"Source: {source}; horizon={args.horizon} sessions; X={data.x.shape}, y={data.y.shape}")
    print(f"Graph: {int(np.count_nonzero(data.weights) // 2)} undirected edges (excluding self-loops)")
    args.out.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out / "prices.csv", index_label="Date")
    report = {"source": source, "arguments": {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}, "versions": {"python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__)},
              "prices_sha256": hashlib.sha256((args.out / "prices.csv").read_bytes()).hexdigest(),
              "return_units": "percentage_points", "splits": {}, "best_epochs": {}, "metrics": {}}
    report["data_provenance"] = provenance
    report["horizon_sessions"] = args.horizon
    report["feature_names"] = feature_names(args.lookback)
    report["forecast_definition"] = "100 * (adjusted_close[t+horizon] / adjusted_close[t] - 1)"
    report["graph_definition"] = "Positive top-k daily-return correlations on eligible training history; symmetric union and self-loops."
    report["split_policy"] = "Fixed daily-calendar boundaries; purge horizon origins before each later block."
    for name, part in data.splits.items():
        report["splits"][name] = {"samples": len(data.x[part]),
            "first_origin": str(data.dates[part][0].date()),
            "last_origin": str(data.dates[part][-1].date()),
            "last_target": str(data.target_dates[part][-1].date())}
    for name, adjacency in [("mlp", np.eye(len(prices.columns), dtype=np.float32)),
                             ("gnn", data.adjacency)]:
        model, forecasts[name], history, epoch = train_model(data, adjacency, args)
        report["best_epochs"][name] = epoch
        pd.DataFrame(history).to_csv(args.out / f"{name}_history.csv", index=False)
        print(f"{name}: best validation epoch {epoch}; stopped at {len(history)}")
        if name == "gnn":
            checkpoint = {"state_dict": model.state_dict(), "mean": torch.from_numpy(data.mean),
                          "scale": torch.from_numpy(data.scale), "tickers": prices.columns.tolist(),
                          "hidden": args.hidden, "lookback": args.lookback, "source": source,
                          "horizon_sessions": args.horizon,
                          "feature_names": feature_names(args.lookback),
                          "train_last_target": report["splits"]["train"]["last_target"],
                          "validation_last_target": report["splits"]["validation"]["last_target"]}
            torch.save(checkpoint, args.out / "gnn.pt")
            latest = predict_latest(checkpoint, prices).rename_axis("ticker").reset_index()
            latest.insert(0, "as_of", str(prices.index[-1].date()))
            latest.insert(1, "horizon_sessions", args.horizon)
            latest.to_csv(args.out / "latest_forecast.csv", index=False)
    for split in ("validation", "test"):
        part = data.splits[split]
        report["metrics"][split] = {name: metrics(data.y[part], prediction[part])
                                      for name, prediction in forecasts.items()}
    # Prespecified check: every h-th origin from the first test origin. These
    # target intervals do not overlap, though markets can still be dependent.
    nonoverlap = np.arange(data.splits["test"].start, len(data.y), args.horizon)
    report["metrics"]["test_nonoverlap"] = {
        name: metrics(data.y[nonoverlap], prediction[nonoverlap])
        for name, prediction in forecasts.items()}
    report["nonoverlap_origin_count"] = len(nonoverlap)
    part = data.splits["test"]
    n = len(prices.columns)
    predictions = pd.DataFrame({
        "as_of": np.repeat(data.dates[part].strftime("%Y-%m-%d"), n),
        "target_date": np.repeat(data.target_dates[part].strftime("%Y-%m-%d"), n),
        "horizon_sessions": args.horizon,
        "ticker": np.tile(prices.columns, len(data.y[part])), "actual_return_pp": data.y[part].ravel(),
        **{f"{name}_return_pp": prediction[part].ravel() for name, prediction in forecasts.items()},
    })
    # Price columns explicitly refer to the endpoint of the forecast horizon.
    origin_prices = prices.loc[data.dates[part]].to_numpy().ravel()
    predictions["as_of_adjusted_close"] = origin_prices
    predictions["actual_target_adjusted_close"] = prices.loc[data.target_dates[part]].to_numpy().ravel()
    predictions["gnn_predicted_target_adjusted_close"] = origin_prices * (1 + predictions["gnn_return_pp"] / 100)
    predictions.to_csv(args.out / "test_predictions.csv", index=False)
    predictions[predictions.as_of.isin(data.dates[nonoverlap].strftime("%Y-%m-%d"))].to_csv(
        args.out / "test_predictions_nonoverlap.csv", index=False)
    for name, matrix in [("graph_weights", data.weights), ("graph_normalized", data.adjacency)]:
        pd.DataFrame(matrix, index=prices.columns, columns=prices.columns).to_csv(args.out / f"{name}.csv")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("\nHeld-out test metrics (RMSE/MAE in percentage points):")
    print(pd.DataFrame(report["metrics"]["test"]).T.round(4).to_string())
    print(f"\nSaved outputs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
