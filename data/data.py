from typing import Sequence
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from torch_geometric.data import Data

FEATURE_NAMES = (
    "log_return",
    "log_volume_change",
    "rolling_volatility"
)

# calls yf.download
def download_stock_data(tickers: Sequence[str], *, start:str="2021-01-01", end:str="2026-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = tuple(str(ticker).strip().upper() for ticker in tickers)
    if len(symbols) < 2: # Need atleast two graph nodes for a GNN
        raise ValueError("Atleast 2 tickers are required to form a graph.") 
    if any(not symbol for symbol in symbols):
        raise ValueError("Ticker symbols cannot be empty.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Ticker symbols must be unique.")
    raw = yf.download(
        interval="1d", 
        auto_adjust=True, 
        group_by="ticker", 
        progress=False, 
        start=start, 
        end=end, 
        tickers=list(symbols),
        threads=False
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no market data.")
    close = _extract_field(raw, "Close", symbols)
    vol = _extract_field(raw, "Volume", symbols)
    close.index = pd.to_datetime(close.index)
    vol.index = pd.to_datetime(vol.index)
    close = close.sort_index()
    vol = vol.sort_index()

    if close.index.has_duplicates:
        raise ValueError("Downloaded data contains duplicate dates.")
    if not close.index.equals(vol.index):
        raise ValueError("Close and volume dates do not match.")
    
    complete_rows = (
        close.notna().all(axis=1)
        & vol.notna().all(axis=1)
    )

    close = close.loc[complete_rows]
    vol = vol.loc[complete_rows]

    if close.empty:
        raise ValueError("No complete rows remain after removing missing data.")
    if not np.isfinite(close.to_numpy()).all():
        raise ValueError("Close prices must be finite.")
    if not np.isfinite(vol.to_numpy()).all():
        raise ValueError("Volume values must be finite.")
    if (close.to_numpy() <= 0).any():
        raise ValueError("Close prices must be strictly positive.")
    if (vol.to_numpy() < 0).any():
        raise ValueError("Volume cannot be negative.")
    return close, vol

# Computes Log Return, Log Volume Change, and Rolling Volatility
def compute_feature_frames(close: pd.DataFrame, volume: pd.DataFrame, *, volatility_window: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2.")
    if close.empty or volume.empty:
        raise ValueError("Close and volume cannot be empty.")
    if not close.index.equals(volume.index):
        raise ValueError("Close and volume must have identical dates.")
    if list(close.columns) != list(volume.columns):
        raise ValueError("Close and volume must have identical ticker columns.")
    if close.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates.")
    if not close.index.is_monotonic_increasing:
        raise ValueError("Market data must be ordered by date.")
    if not np.isfinite(close.to_numpy()).all():
        raise ValueError("Close prices must be finite.")
    if not np.isfinite(volume.to_numpy()).all():
        raise ValueError("Volume must be finite.")
    if (close.to_numpy() <= 0).any():
        raise ValueError("Close prices must be positive.")
    if (volume.to_numpy() < 0).any():
        raise ValueError("Volume cannot be negative.")
    log_close = pd.DataFrame(
        np.log(close.to_numpy(dtype=np.float64)),
        index=close.index,
        columns=close.columns,
    )

    log_volume = pd.DataFrame(
        np.log1p(volume.to_numpy(dtype=np.float64)),
        index=volume.index,
        columns=volume.columns,
    )

    log_returns = log_close.diff()  # $\text{log}C_t - \text{log}C_{t-1} = \text{log}(\frac{C_{t}}{C_{t-1}})$
    log_vol_change = log_volume.diff() # $q_t = \text{log}(1 + V_t) - \text{log}(1 + V_{t-1})$

    # for each date and ticker, rolling_vol calculates the standard deviation of its most recent returns
    # $\sigma_{t} = \sqrt{\frac{1}{w}\sum_{k=0}^{w-1} {(r_{t-k} - \overline{r}_t)^2}}$
    
    rolling_vol = (log_returns.rolling(window=volatility_window, min_periods=volatility_window).std(ddof=0))
    features = pd.concat(
        {
            "log_return": log_returns,
            "log_volume_change": log_vol_change,
            "rolling_volatility": rolling_vol
        },
        axis=1
    )
    features = features.swaplevel(0,1,axis=1)
    ordered_cols = pd.MultiIndex.from_product([close.columns, FEATURE_NAMES], names = ["ticker", "feature"])
    features = features.reindex(columns=ordered_cols)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(how="any")
    if features.empty:
        raise ValueError("Not enough history remains after feature construction.")
    return features, log_returns

# returns: pd.DataFrame $\in R^{T \times N}$
# Where T is the number of training dates and N is the number of stocks
def build_correlation_edge_index(returns: pd.DataFrame, *, neighbors: int=2) -> torch.Tensor:
    num_nodes = returns.shape[1]
    if num_nodes < 2:
        raise ValueError("At least two return series are required.")
    if not 1 <= neighbors < num_nodes:
        raise ValueError("neighbors must be between 1 and num_nodes - 1.")
    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must contain only finite values.")

    #Absolute Pairwise Pearson Correlations
    # $\rho_{ij} = \lvert\frac{\text{cov}(r_i, r_j)}{\sigma_i \sigma_j}\rvert$
    # i.e. the absolute value of the covariance of two variables divided by the product of their standard deviations
    # Yields an NxN matrix. 
    correlations = (returns.astype(float).corr().abs().to_numpy(copy=True))
    if not np.isfinite(correlations).all():
        raise ValueError("Cannot construct a graph from undefined correlations.")
    np.fill_diagonal(correlations, -np.inf) # To avoid selecting the same stock as that stock's neighbor

    undirected_pairs: set[tuple[int, int]] = set()

    # Selecting each node's strongest neighbors
    for src in range(num_nodes):
        ranked_targets = np.argsort(-correlations[src], kind="stable")[:neighbors]
        # Makes the union of undirected pairs
        for target in ranked_targets:
            # make (0,1) equivalent to (1,0) so they can be eliminated by set
            left, right = sorted((src, int(target))) 
            undirected_pairs.add((left, right))

    directed_edges: list[tuple[int, int]] = []
    for left, right in sorted(undirected_pairs):
        directed_edges.append((left, right))
        directed_edges.append((right, left))
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    return edge_index

# Container for relevant asset info
@dataclass(frozen=True)
class GraphDatasetSplits:
    train: list[Data]
    validation: list[Data]
    test: list[Data]

    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]

    tickers: tuple[str, ...]
    edge_index: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

# Calculates daily features, 
# creates lookback window & split by time (not shuffling before splitting), 
# fits feature normalization, 
# builds the stock graph, 
# creates PyG samples, and return the results
# i.e. it is the entire data pipeline combined
def prepare_graph_datasets(close: pd.DataFrame, volume: pd.DataFrame, *, lookback: int =20, volatility_window: int = 5, neighbors: int = 2) -> GraphDatasetSplits:
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    # Calculate daily features
    features, log_returns = compute_feature_frames(close, volume, volatility_window=volatility_window)

    tickers = tuple(str(column) for column in close.columns)
    num_nodes = len(tickers)
    num_features = len(FEATURE_NAMES)

    raw_inputs: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    sample_dates: list[pd.Timestamp] = []

    # Create lookback window
    for target_position in range(lookback, len(features)):
        # $X_t = [F_{t-L}, \ldots, F_{t-1}] \in \mathbb{R}^{N \times L \times F}$
        window = features.iloc[target_position - lookback : target_position].to_numpy(dtype=np.float64)
        window = window.reshape(lookback, num_nodes, num_features)
        node_window = window.transpose(1,0,2)

        # $y_t = r_t \in \mathbb{R}^{N \times 1} \text{date } t \text{ is excluded from } X_t$
        target_date = features.index[target_position]
        target = (
            log_returns.loc[target_date, list(tickers)]
            .to_numpy(dtype=np.float64, copy=True)
            .reshape(num_nodes, 1)
        )
        raw_inputs.append(node_window)
        raw_targets.append(target)
        sample_dates.append(pd.Timestamp(target_date))

    num_samples = len(raw_inputs)
    train_count = int(num_samples * 0.70)
    validation_count = int(num_samples * 0.15)
    test_count = num_samples - train_count - validation_count

    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("Not enough samples for nonempty 70/15/15 splits.")

    train_end = train_count
    validation_end = train_count + validation_count

    all_inputs = np.stack(raw_inputs)
    training_inputs = all_inputs[:train_end]

    # $\mu_{f} = \frac{1}{SNL}\sum_{s=1}^{S}\sum_{n=1}^{N}\sum_{l=1}^{L}X_{s,n,l,f}$

    feature_mean = training_inputs.mean(axis=(0, 1, 2))

    # $\sigma_f = \sqrt{\frac{1}{SNL}\sum_{s,n,l}(X_{s,n,l,f} - \mu_f)^2}$

    feature_std = training_inputs.std(axis=(0, 1, 2), ddof=0)

    if not np.isfinite(feature_mean).all():
        raise ValueError("Feature means must be finite.")
    if not np.isfinite(feature_std).all() or (feature_std <= 0).any():
        raise ValueError("Training features must have positive finite standard deviations.")

    # $\hat{X}_{s,n,l,f} = \frac{X_{s,n,l,f} - \mu_f}{\sigma_f}$

    scaled_inputs = (all_inputs - feature_mean) / feature_std

    training_cutoff = sample_dates[train_end - 1]
    training_returns = (log_returns.loc[:training_cutoff, list(tickers)].dropna(how="any"))
    edge_index = build_correlation_edge_index(training_returns, neighbors=neighbors)

    def make_samples(start: int, stop: int) -> list[Data]:
        samples: list[Data] = []
        for sample_index in range(start, stop):
            # $X_t \in \mathbb{R}^{N \times L \times F} \mapsto \mathbb{R}^{N \times (LF)}$
            flattened_input = scaled_inputs[sample_index].reshape(num_nodes, lookback * num_features)
            samples.append(
                Data(
                    x=torch.as_tensor(flattened_input, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.as_tensor(raw_targets[sample_index], dtype=torch.float32),
                )
            )
        return samples

    train = make_samples(0, train_end)
    validation = make_samples(train_end, validation_end)
    test = make_samples(validation_end, num_samples)

    return GraphDatasetSplits(
        train=train,
        validation=validation,
        test=test,
        train_dates=tuple(sample_dates[:train_end]),
        validation_dates=tuple(sample_dates[train_end:validation_end]),
        test_dates=tuple(sample_dates[validation_end:]),
        tickers=tickers,
        edge_index=edge_index,
        feature_mean=torch.as_tensor(feature_mean, dtype=torch.float32),
        feature_std=torch.as_tensor(feature_std, dtype=torch.float32),
    )

# The zero-return predictor has $\hat{y}_i=0$ 
# $\operatorname{MSE}_0=\frac{1}{K}\sum_{i=1}^{K}(y_i-0)^2$
# $=\frac{1}{K}\sum_{i=1}^{K}y_i^2$

def zero_return_baseline_mse(samples: Sequence[Data]) -> float:
    # Reject empty sample sequence
    if not samples:
        raise ValueError("At least one sample is required.")

    total_squared_error = 0.0
    target_count = 0
    for sample in samples:
        targets = sample.y

        if not isinstance(targets, torch.Tensor):
            raise ValueError("Every sample must contain a tensor target.")
        if targets.numel() == 0: #numel() = # of elements
            raise ValueError("Every sample must contain at least one target.")
        if not torch.isfinite(targets).all():
            raise ValueError("Targets must contain only finite values.")

        total_squared_error += float(targets.detach().square().sum().item())
        target_count += targets.numel()

    return total_squared_error / target_count

# Internal yfinance adapter
# Extracts "Close" and "Volume" fields, standarizes the ticker names, and return standardly ordered
def _extract_field(raw: pd.DataFrame, field: str, tickers: tuple[str,...]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected YF to return MultiIndex cols.")
    field_lvl = None
    for lvl in range(raw.columns.nlevels):
        values = raw.columns.get_level_values(lvl)

        if field in values:
            field_lvl = lvl
            break
    if field_lvl is None: 
        raise ValueError(f"Downloaded data does not contain {field!r} field.")
    extracted = raw.xs(field, axis=1, level=field_lvl).copy()
    if not isinstance(extracted, pd.DataFrame):
        raise TypeError(f"Expected {field!r} extraction to produce a DataFrame.")
    frame = extracted.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    missing = [ticker 
               for ticker in tickers if ticker not in frame.columns]
    if missing: raise ValueError(f"Missing ticker(s): {missing}")
    return frame.loc[:,list(tickers)].astype(float)
