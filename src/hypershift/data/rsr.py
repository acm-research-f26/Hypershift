"""Loader for the RSR (Feng et al. 2019) NYSE/NASDAQ daily dataset.

Each file NYSE_<T>_1.csv has rows: date_index, ma5, ma10, ma20, ma30, close, all divided by the
stock's full-series max close ("paper" normalization). -1234 marks a missing day.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MISSING = -1234.0
FILL = 1.1  # value the original code writes into missing cells
SPLITS = {"NYSE": (756, 1008), "NASDAQ": (756, 1008)}


@dataclass
class MarketData:
    tickers: list[str]
    features: np.ndarray    # [N, T, C] float32
    mask: np.ndarray        # [N, T] float32, 1 = price observed on day t
    gt: np.ndarray          # [N, T] float32, return from t-1 to t (0 if either day missing)
    base_price: np.ndarray  # [N, T] float32, normalized close at t
    valid_index: int
    test_index: int
    timestamps: np.ndarray | None = None
    eligible_ends: np.ndarray | None = None  # allowed "last input step" indices; None = all

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_steps(self) -> int:
        return self.features.shape[1]

    def subset(self, idx: np.ndarray) -> "MarketData":
        idx = np.asarray(idx)
        return replace(
            self,
            tickers=[self.tickers[i] for i in idx],
            features=self.features[idx],
            mask=self.mask[idx],
            gt=self.gt[idx],
            base_price=self.base_price[idx],
        )


def read_ticker_file(path: Path) -> list[str]:
    return [ln.strip().split("\t")[0] for ln in Path(path).read_text().splitlines() if ln.strip()]


def parse_eod(raw: np.ndarray, drop_last: bool):
    if drop_last:
        raw = raw[:-1]
    miss = np.abs(raw[:, -1] - MISSING) < 1e-8
    close = raw[:, -1]
    gt = np.zeros(len(raw), dtype=np.float32)
    ok = (~miss[1:]) & (~miss[:-1])
    gt[1:][ok] = (close[1:][ok] - close[:-1][ok]) / close[:-1][ok]
    feats = raw[:, 1:].copy()
    feats[np.abs(feats - MISSING) < 1e-8] = FILL
    base = feats[:, -1].copy()
    return feats.astype(np.float32), (~miss).astype(np.float32), gt, base.astype(np.float32)


def renormalize_train(data: MarketData) -> MarketData:
    """Divide each stock by its max observed close over the training period (removes look-ahead)."""
    close = np.where(data.mask > 0, data.features[:, :, -1], 0.0)
    scale = close[:, : data.valid_index].max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    feats = data.features / scale[:, None, None]
    base = data.base_price / scale[:, None]
    miss = data.mask < 0.5
    feats[miss] = FILL
    base[miss] = FILL
    return replace(data, features=feats.astype(np.float32), base_price=base.astype(np.float32))


def load_rsr(root: Path | str, market: str, norm: str = "train") -> MarketData:
    root = Path(root)
    tickers = read_ticker_file(root / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    parts = [
        parse_eod(
            np.loadtxt(root / "2013-01-01" / f"{market}_{t}_1.csv", delimiter=",", dtype=np.float64),
            drop_last=(market == "NASDAQ"),
        )
        for t in tickers
    ]
    vi, ti = SPLITS[market]
    data = MarketData(
        tickers=tickers,
        features=np.stack([p[0] for p in parts]),
        mask=np.stack([p[1] for p in parts]),
        gt=np.stack([p[2] for p in parts]),
        base_price=np.stack([p[3] for p in parts]),
        valid_index=vi,
        test_index=ti,
    )
    if norm == "train":
        return renormalize_train(data)
    if norm != "paper":
        raise ValueError(f"unknown norm {norm!r}")
    return data
