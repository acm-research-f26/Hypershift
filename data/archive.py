import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BAR_COLUMNS = (
    "open", "high", "low", "close",
    "volume", "trade_count", "vwap",
)

def load_symbol_bars(archive_path, symbol, start, end, *, expected_timeframe, verify_checksum=True):
    root = Path(archive_path)

    # 1. Validate timezone-aware boundaries and normalize the window to UTC.
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if (pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None):
        raise ValueError("start and end must be timezone-aware timestamps")

    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")

    if start >= end:
        raise ValueError("start must precede end")

    # 2. Confirm the archive supports the requested symbol, interval, and dates.
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported archive schema")

    request = manifest["request"]

    if request["timeframe"] != expected_timeframe:
        raise ValueError("Archive timeframe does not match the experiment")

    if (not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,19}", symbol) or symbol not in request["symbols"]):
        raise ValueError("Symbol is not in this archive's universe")

    if (start < pd.Timestamp(request["start_inclusive"]) or end > pd.Timestamp(request["end_exclusive"])):
        raise ValueError("Requested window is outside the archive range")

    # 3. Identify only the monthly partitions overlapping the requested window.
    first_month = start.normalize().replace(day=1)
    months = pd.date_range(first_month, end, freq="MS", inclusive="left")

    frames = []

    for month in months:
        relative = f"bars/{symbol}/{month:%Y-%m}.csv.gz"
        # 4. Verify partition completion, coverage, integrity, and bar values.
        entry = manifest["partitions"].get(relative)

        if entry is None or entry["status"] != "complete":
            raise ValueError(f"Partition is not complete: {relative}")

        needed_start = max(start, month)
        needed_end = min(end, month + pd.offsets.MonthBegin(1))

        if (pd.Timestamp(entry["start_inclusive"]) > needed_start or pd.Timestamp(entry["end_exclusive"]) < needed_end):
            raise ValueError(f"Partition does not cover the window: {relative}")

        path = root / relative

        if verify_checksum:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()

            if digest != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {relative}")

        frame = pd.read_csv(path)

        if set(frame.columns) != {"timestamp_utc", *BAR_COLUMNS}:
            raise ValueError(f"Unexpected bar columns: {relative}")

        if len(frame) != entry["rows"]:
            raise ValueError(f"Row count mismatch: {relative}")

        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp_utc"), utc=True, errors="raise"), name="timestamp_utc")

        frame = frame.loc[:, list(BAR_COLUMNS)].apply(pd.to_numeric, errors="raise")

        if (frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
            raise ValueError(f"Invalid timestamp ordering: {relative}")

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Nonfinite bar values: {relative}"
            )

        outside_partition = (frame.index < pd.Timestamp(entry["start_inclusive"])) | (frame.index >= pd.Timestamp(entry["end_exclusive"]))

        if outside_partition.any():
            raise ValueError(f"Bar is outside its partition: {relative}")

        # 5. Keep [start, end) observations for the final chronological result.
        selected = frame.loc[(frame.index >= start) & (frame.index < end)]
        frames.append(selected)

    result = pd.concat(frames)

    if (not result.index.is_unique or not result.index.is_monotonic_increasing):
        raise ValueError("Overlapping or out-of-order partitions")

    return result
