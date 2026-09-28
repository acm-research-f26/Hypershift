"""Fresh S&P 500 panels (yfinance / Alpaca) in RSR-compatible MarketData form. Decision nodes D12, D13, D17."""
from __future__ import annotations

import io
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hypershift.data.hypergraph import Hypergraph, canonical
from hypershift.data.rsr import MarketData

NY = "America/New_York"
UNIVERSE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
COLS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


def load_universe() -> pd.DataFrame:
    try:
        df = pd.read_csv(UNIVERSE_URL)
    except Exception:
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        df = pd.read_html(io.StringIO(html))[0]
    df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "GICS Sub-Industry": "subindustry"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "sector", "subindustry"]].drop_duplicates("ticker").reset_index(drop=True)


def fetch_yf(tickers, interval, start=None, end=None, period=None) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 50):
        chunk = list(tickers[i:i + 50])
        raw = yf.download(chunk, interval=interval, start=start, end=end, period=period, group_by="ticker",
                          auto_adjust=True, prepost=False, threads=True, progress=False)
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna(how="all").copy()
            if d.empty:
                continue
            d.columns = [str(c).lower() for c in d.columns]
            d.index.name = "timestamp"
            d = d.reset_index()
            d["ticker"] = t
            frames.append(d[COLS])
    return pd.concat(frames, ignore_index=True)


def fetch_alpaca(tickers, start, end, feed: str = "sip") -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    for i in range(0, len(tickers), 50):
        chunk = [t.replace("-", ".") for t in tickers[i:i + 50]]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                               start=pd.Timestamp(start, tz=NY), end=pd.Timestamp(end, tz=NY),
                               adjustment=Adjustment.ALL, feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        df = client.get_stock_bars(req).df
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"symbol": "ticker"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        frames.append(to_rth_hourly(df[COLS]))          # resample per chunk to keep memory small
        print(f"alpaca: {i + len(chunk)}/{len(tickers)} tickers")
    return pd.concat(frames, ignore_index=True)


def to_rth_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(minutes >= 570) & (minutes < 960)]                          # 09:30 <= t < 16:00
    g = df.set_index("timestamp").groupby("ticker").resample("60min", origin="start_day", offset="30min")
    # per-column reducers: pandas 3 turns a dict .agg() on a grouped resample into a cross-product MultiIndex
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out = out.dropna(subset=["close"]).reset_index()
    return out[COLS]


def to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df["day"] = df["timestamp"].dt.normalize()
    out = (df.sort_values("timestamp").groupby(["ticker", "day"])
             .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                  volume=("volume", "sum"))
             .reset_index().rename(columns={"day": "timestamp"}))
    return out[COLS]


def build_panel(bars: pd.DataFrame, train_frac=0.6, val_frac=0.2, max_missing=0.05, min_coverage=0.5,
                tickers=None, split_at=None, ma_windows=(5, 10, 20, 30)) -> MarketData:
    df = bars.copy()
    ts_raw = pd.to_datetime(df["timestamp"])
    # yfinance daily bars can be tz-naive dates: treat them as NY dates, not UTC midnight
    df["timestamp"] = ts_raw.dt.tz_localize(NY) if ts_raw.dt.tz is None else ts_raw.dt.tz_convert(NY)
    close = df.pivot_table(index="timestamp", columns="ticker", values="close", aggfunc="last").sort_index()
    close = close[close.notna().mean(axis=1) >= min_coverage]
    if tickers is not None:
        close = close.reindex(columns=list(tickers)).dropna(axis=1, how="all")
    else:
        close = close.loc[:, close.isna().mean(axis=0) <= max_missing]
    mask = close.notna().to_numpy().T.astype(np.float32)
    filled = close.ffill().bfill()
    c = filled.to_numpy().T.astype(np.float64)
    ts = pd.DatetimeIndex(close.index)
    T = c.shape[1]
    if split_at is None:
        vi, ti = int(T * train_frac), int(T * (train_frac + val_frac))
    else:
        vi, ti = (int(ts.searchsorted(pd.Timestamp(x))) for x in split_at)
    mas = [filled.rolling(w, min_periods=1).mean().to_numpy().T for w in ma_windows]
    scale = c[:, :vi].max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)[:, None]
    feats = np.stack(mas + [c], axis=2) / scale[:, :, None]
    gt = np.zeros_like(c)
    gt[:, 1:] = (c[:, 1:] / c[:, :-1] - 1) * mask[:, 1:] * mask[:, :-1]
    return MarketData(list(close.columns), feats.astype(np.float32), mask, gt.astype(np.float32),
                      (c / scale).astype(np.float32), vi, ti, timestamps=ts.as_unit("ns").asi8.copy())


def daily_horizon(h: MarketData) -> MarketData:
    """Hourly inputs, but predict close(next day) / close(today) - 1 from the last bar of each day."""
    day = pd.to_datetime(h.timestamps, utc=True).tz_convert(NY).normalize().as_unit("ns").asi8
    last = np.nonzero(np.r_[day[1:] != day[:-1], True])[0]
    gt = h.gt.copy()
    for e, e2 in zip(last[:-1], last[1:]):
        gt[:, e + 1] = (h.base_price[:, e2] / h.base_price[:, e] - 1) * h.mask[:, e] * h.mask[:, e2]
    return replace(h, gt=gt.astype(np.float32), eligible_ends=last[:-1].copy())


def save_panel(data: MarketData, panel_dir, universe: pd.DataFrame | None = None) -> None:
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_dir / "panel.npz", features=data.features, mask=data.mask, gt=data.gt, base_price=data.base_price,
        valid_index=np.array(data.valid_index), test_index=np.array(data.test_index),
        timestamps=data.timestamps if data.timestamps is not None else np.array([], dtype=np.int64),
        has_eligible=np.array(data.eligible_ends is not None),
        eligible_ends=data.eligible_ends if data.eligible_ends is not None else np.array([], dtype=np.int64),
    )
    (panel_dir / "tickers.txt").write_text("\n".join(data.tickers) + "\n")
    if universe is not None:
        universe[universe.ticker.isin(data.tickers)].to_csv(panel_dir / "universe.csv", index=False)


def load_panel(panel_dir) -> MarketData:
    panel_dir = Path(panel_dir)
    z = np.load(panel_dir / "panel.npz")
    return MarketData(
        tickers=(panel_dir / "tickers.txt").read_text().split(),
        features=z["features"], mask=z["mask"], gt=z["gt"], base_price=z["base_price"],
        valid_index=int(z["valid_index"]), test_index=int(z["test_index"]),
        timestamps=z["timestamps"] if len(z["timestamps"]) else None,
        eligible_ends=z["eligible_ends"] if bool(z["has_eligible"]) else None,
    )


def gics_hypergraph(panel_dir, level: str) -> Hypergraph:
    panel_dir = Path(panel_dir)
    tickers = (panel_dir / "tickers.txt").read_text().split()
    uni = pd.read_csv(panel_dir / "universe.csv").set_index("ticker")
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        if t in uni.index:
            groups.setdefault(str(uni.loc[t, level]), []).append(i)
    return Hypergraph(len(tickers), canonical(groups.values()))
