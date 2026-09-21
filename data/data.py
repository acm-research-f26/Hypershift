from typing import Sequence
import yfinance as yf
import pandas as pd
import numpy as np

FEATURE_NAMES = (
    "log_return",
    "log_volume_change",
    "rolling_volatility"
)

def download_stock_data(tickers: Sequence[str], *, start:str="2021-01-01", end:str="2026-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = tuple(str(ticker).strip().upper() for ticker in tickers)
    if len(symbols) < 2: # Need a rolling window of more than one day for volatility
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
        threads=True
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
