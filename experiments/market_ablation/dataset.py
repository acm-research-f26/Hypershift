"""One-pass monthly archive preparation and lazy common-support forecasting batches.

All session minutes remain in history. Only training origins are subsampled.
No price adjustment or future value enters a predictor.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from data.archive import BAR_COLUMNS, load_symbol_bars
from data.calendars import load_calendar, session_schedule
from .config import FEATURE_NAMES, FEATURE_PACKS
from .storage import atomic_json, read_json, event, reserve


def grids(config):
    schedule = session_schedule(load_calendar("XNYS"), config.start, config.end)
    result = {}
    for interval in config.intervals:
        minutes = {"1m": 1, "15m": 15, "1h": 60, "1d": 1440}[interval]
        starts, ends, sessions, expected = [], [], [], []
        for sid, (_, row) in enumerate(schedule.iterrows()):
            begin, end = row.session_open.value, row.session_close.value
            points = np.arange(begin, end, minutes * 60_000_000_000, dtype=np.int64)
            stop = np.minimum(points + minutes * 60_000_000_000, end)
            starts.extend(points)
            ends.extend(stop)
            sessions.extend([sid] * len(points))
            expected.extend((stop - points) // 60_000_000_000)
        result[interval] = {"starts": np.asarray(starts, np.int64), "times": np.asarray(ends, np.int64),
                            "sessions": np.asarray(sessions, np.int32), "expected": np.asarray(expected, np.int16),
                            "session_dates": schedule.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)}
    return result


def aggregate(raw, grid, selection):
    """Bins use observed minute starts, including the exchange's shortened last bar."""
    starts, ends = grid["starts"][selection], grid["times"][selection]
    out = np.zeros((len(starts), len(BAR_COLUMNS)), np.float32)
    if not len(starts) or raw.empty:
        return out, np.zeros(len(starts), bool)
    ts = raw.index.as_unit("ns").asi8
    positions = np.searchsorted(starts, ts, side="right") - 1
    keep = (positions >= 0) & (positions < len(starts))
    indices = np.flatnonzero(keep)
    indices = indices[ts[indices] < ends[positions[indices]]]
    if not len(indices):
        return out, np.zeros(len(starts), bool)
    bins = positions[indices]
    values = raw.loc[:, list(BAR_COLUMNS)].to_numpy(np.float64)[indices]
    counts = np.bincount(bins, minlength=len(starts))
    first = np.r_[True, bins[1:] != bins[:-1]]
    last = np.r_[bins[1:] != bins[:-1], True]
    out[bins[first], 0] = values[first, 0]
    out[bins[last], 3] = values[last, 3]
    high, low = np.full(len(starts), -np.inf), np.full(len(starts), np.inf)
    np.maximum.at(high, bins, values[:, 1])
    np.minimum.at(low, bins, values[:, 2])
    out[:, 1] = np.where(counts, high, 0)
    out[:, 2] = np.where(counts, low, 0)
    for column in (4, 5):
        out[:, column] = np.bincount(bins, weights=values[:, column], minlength=len(starts))
    volume = np.bincount(bins, weights=values[:, 4], minlength=len(starts))
    weighted = np.bincount(bins, weights=values[:, 4] * values[:, 6], minlength=len(starts))
    out[:, 6] = np.divide(weighted, volume, out=np.zeros_like(weighted), where=volume > 0)
    valid = (counts == grid["expected"][selection]) & (out[:, :4] > 0).all(axis=1)
    return out, valid


def feature_block(bars, valid, grid, begin, interval):
    """Compute features with past observations only; missing features carry masks."""
    n = len(bars)
    x, mask = np.zeros((n, len(FEATURE_NAMES)), np.float32), np.zeros((n, len(FEATURE_NAMES)), bool)
    close, volume = bars[:, 3], bars[:, 4]
    previous = np.r_[0., close[:-1]]
    return_valid = valid & np.r_[False, valid[:-1]] & (previous > 0)
    x[:, 0] = np.log(np.divide(close, previous, out=np.ones(n), where=return_valid))
    mask[:, 0] = return_valid
    x[:, 1] = np.log1p(volume)
    mask[:, 1] = valid
    returns = pd.Series(np.where(return_valid, x[:, 0], np.nan))
    rv = returns.rolling(20, min_periods=2).std(ddof=0)
    x[:, 2] = rv.fillna(0)
    mask[:, 2] = rv.notna().to_numpy() & valid
    x[:, 3] = np.divide(bars[:, 1] - bars[:, 2], close, out=np.zeros(n), where=valid)
    x[:, 4] = np.log(np.divide(close, bars[:, 0], out=np.ones(n), where=valid))
    x[:, 5] = np.log(np.divide(bars[:, 0], previous, out=np.ones(n), where=return_valid))
    mask[:, 3:5] = valid[:, None]
    mask[:, 5] = return_valid
    trailing = pd.Series(np.where(valid, volume, np.nan)).rolling(20, min_periods=2).mean()
    x[:, 6] = np.divide(volume, trailing.fillna(0), out=np.zeros(n), where=trailing.fillna(0).to_numpy() > 0)
    mask[:, 6] = valid & trailing.notna().to_numpy()
    starts = pd.to_datetime(grid["starts"][begin:begin + n], utc=True).tz_convert("America/New_York")
    angle = (starts.hour * 60 + starts.minute - 570) / 390 * 2 * np.pi
    x[:, 7], x[:, 8] = np.sin(angle), np.cos(angle)
    x[:, 9] = grid["expected"][begin:begin + n] < {"1m": 1, "15m": 15, "1h": 60, "1d": 390}[interval]
    mask[:, 7:] = True
    return np.where(mask, x, 0), mask


def prepare(config):
    source = json.loads((Path(config.archive) / "manifest.json").read_text())
    symbols = source["request"]["symbols"]
    if len(symbols) != 250 or source["request"]["adjustment"] != "raw":
        raise ValueError("Expected the complete raw 250-stock archive")
    identity = sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    prepared = config.root / "dataset"
    metadata = read_json(prepared / "metadata.json")
    if metadata and metadata["source_sha256"] != identity:
        raise ValueError("Source archive changed; use a new output directory")
    grid_by_interval = grids(config)
    estimated = sum(len(g["times"]) * 250 * (7 * 4 + 10 * 4 + 10 + 1) for g in grid_by_interval.values())
    if not metadata:
        reserve(config, estimated)
    arrays = {}
    for interval, grid in grid_by_interval.items():
        directory = prepared / interval
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in grid.items():
            path = directory / f"{name}.npy"
            if not path.exists():
                np.save(path, values)
        t = len(grid["times"])
        arrays[interval] = {}
        for name, shape, dtype in (("bars", (t, 250, 7), np.float32), ("valid", (t, 250), bool),
                                   ("features", (t, 250, 10), np.float32), ("feature_mask", (t, 250, 10), bool)):
            path = directory / f"{name}.npy"
            arrays[interval][name] = open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape)
    state = read_json(prepared / "preparation.json", {"completed": []})
    completed = set(state["completed"])
    atomic_json(prepared / "metadata.json", {"symbols": symbols, "source_sha256": identity, "features": FEATURE_NAMES,
                "intervals": list(grid_by_interval), "adjustment": "raw", "coverage_rule": "all_expected_minutes",
                "target": "next_bar_log_return", "universe_protocol": "retrospective_2026_universe"})
    months = pd.date_range(pd.Timestamp(config.start).replace(day=1), config.end, freq="MS", inclusive="left")
    for month in months:
        lo, hi = max(pd.Timestamp(config.start), month), min(pd.Timestamp(config.end), month + pd.offsets.MonthBegin(1))
        for stock, symbol in enumerate(symbols):
            key = f"{month:%Y-%m}/{symbol}"
            if key in completed:
                continue
            reserve(config)
            raw = load_symbol_bars(config.archive, symbol, lo.tz_localize("UTC"), hi.tz_localize("UTC"), expected_timeframe="1Min")
            for interval, grid in grid_by_interval.items():
                start = np.searchsorted(grid["starts"], lo.tz_localize("UTC").value)
                end = np.searchsorted(grid["starts"], hi.tz_localize("UTC").value)
                if end == start:
                    continue
                out, valid = aggregate(raw, grid, slice(start, end))
                a = arrays[interval]
                a["bars"][start:end, stock] = out
                a["valid"][start:end, stock] = valid
                previous = max(0, start - 21)
                x, masks = feature_block(a["bars"][previous:end, stock], a["valid"][previous:end, stock], grid, previous, interval)
                a["features"][start:end, stock] = x[start - previous:]
                a["feature_mask"][start:end, stock] = masks[start - previous:]
                for values in a.values():
                    values.flush()
            completed.add(key)
            if stock % 25 == 0 or stock == 249:
                atomic_json(prepared / "preparation.json", {"completed": sorted(completed), "complete": False})
                event(config, "prepare", month=f"{month:%Y-%m}", symbol=symbol, partitions=len(completed), total=len(months) * len(symbols))
        atomic_json(prepared / "preparation.json", {"completed": sorted(completed), "complete": False})
    atomic_json(prepared / "preparation.json", {"completed": sorted(completed), "complete": True})
    event(config, "prepared", intervals={k: len(v["times"]) for k, v in grid_by_interval.items()}, stocks=len(symbols))


class MarketDataset:
    def __init__(self, config, interval):
        self.config, self.interval = config, interval
        self.root = config.root / "dataset" / interval
        self.symbols = tuple(read_json(config.root / "dataset/metadata.json")["symbols"])
        for name in ("bars", "valid", "features", "feature_mask", "times", "starts", "sessions", "expected", "session_dates"):
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        self.origins = np.arange(config.lookback - 1, len(self.times) - 1)
        if interval != "1d":
            self.origins = self.origins[self.sessions[self.origins] == self.sessions[self.origins + 1]]

    def split(self, fold):
        target = self.times[self.origins + 1]
        cutoff, validation, test = [pd.Timestamp(fold[k]).value for k in ("fit_cutoff", "validation_end", "test_end")]
        return {"train": self.origins[target <= cutoff], "validation": self.origins[(self.times[self.origins] >= cutoff) & (target <= validation)],
                "test": self.origins[(self.times[self.origins] >= validation) & (target <= test)]}

    def moments(self, fold):
        path = self.config.root / "normalization" / self.interval / f"fold-{fold['id']}.npz"
        if path.exists():
            with np.load(path) as saved:
                if "price_target_scale" in saved.files:
                    return {k: saved[k] for k in saved.files}
        from .storage import atomic_npz
        stop = np.searchsorted(self.times, pd.Timestamp(fold["fit_cutoff"]).value, side="right")
        sums, squares, counts = [np.zeros((250, 10), np.float64) for _ in range(3)]
        for start in range(0, stop, 4096):
            x = np.asarray(self.features[start:min(stop, start + 4096)], np.float64)
            m = self.feature_mask[start:min(stop, start + 4096)]
            sums += np.where(m, x, 0).sum(axis=0)
            squares += np.where(m, x*x, 0).sum(axis=0)
            counts += m.sum(axis=0)
        mean = sums / np.maximum(counts, 1)
        std = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean**2, 0))
        std = np.where(std > 1e-8, std, 1)
        # One common target scale keeps price cross-sectional normalization out of the loss.
        target_scale = max(float(np.sqrt(squares[:, 0].sum() / max(counts[:, 0].sum(), 1))), 1e-6)
        price_sum, price_count = np.zeros(250), np.zeros(250)
        price_squared_sum=0.
        for start in range(self.config.lookback-1, stop-1, 4096):
            end = min(stop-1,start+4096)
            origins = np.arange(start,end)
            positions = origins[:,None]+np.arange(1-self.config.lookback,1)[None,:]
            active = self.valid[origins] & (self.feature_mask[positions,:,0].sum(axis=1)>=2)
            eligible = active & self.valid[origins+1]
            if self.interval != "1d":
                eligible &= (self.sessions[origins]==self.sessions[origins+1])[:,None]
            change=np.where(eligible,self.bars[origins+1,:,3].astype(np.float64)-self.bars[origins,:,3],0)
            price_sum += np.abs(change).sum(axis=0)
            price_squared_sum+=np.square(change).sum()
            price_count += eligible.sum(axis=0)
        value = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "target_scale": np.array(target_scale, np.float32),
                 "price_target_scale":np.array(max(np.sqrt(price_squared_sum/max(price_count.sum(),1)),1e-6),np.float32),
                 "train_price_naive_MAE":np.divide(price_sum,price_count,out=np.zeros(250),where=price_count>0).astype(np.float32)}
        atomic_npz(path, **value)
        return value

    def batch(self, origins, variant, moments):
        origins = np.asarray(origins, np.int64)
        positions = origins[:, None] + np.arange(1 - self.config.lookback, 1)[None, :]
        channels = list(FEATURE_PACKS[variant.features])
        # Gather only selected channels. R is the main track: loading all ten
        # channels first needlessly multiplies CPU memory traffic by ten.
        x = np.stack([self.features[positions,:,channel] for channel in channels],axis=-1)
        m = np.stack([self.feature_mask[positions,:,channel] for channel in channels],axis=-1)
        mean, std = moments["mean"][:, channels], moments["std"][:, channels]
        x = np.where(m, (x - mean) / std, 0)
        x = np.concatenate([x, m.astype(np.float32)], axis=-1).transpose(0, 2, 1, 3).reshape(len(origins), 250, -1)
        # The eligibility rule is identical for R, RV, richer features, and PH context.
        history_count = self.feature_mask[positions, :, 0].sum(axis=1)
        active = self.valid[origins] & (history_count >= 2)
        target_mask = active & self.valid[origins + 1]
        current, future = self.bars[origins, :, 3], self.bars[origins + 1, :, 3]
        target = np.log(np.divide(future, current, out=np.ones_like(future), where=target_mask)) / moments["target_scale"]
        return x.astype(np.float32), active, target.astype(np.float32), target_mask

    def tuning_origins(self, train):
        # Even quantiles of chronological support span months and intraday phases.
        if len(train) <= self.config.tuning_origin_cap:
            return train
        return train[np.linspace(0, len(train) - 1, self.config.tuning_origin_cap, dtype=np.int64)]

    def epoch_origins(self, train, epoch, tuning=False):
        if tuning:
            return self.tuning_origins(train)
        stride = self.config.minute_training_stride if self.interval == "1m" else 1
        return train[train % stride == epoch % stride]
