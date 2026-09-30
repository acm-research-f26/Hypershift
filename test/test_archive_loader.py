"""Offline contract for the learner's first archive adapter.

These tests skip until data/archive.py is implemented. Synthetic archive files
exercise the same manifest and gzip format as the downloader, with no network.
"""

import hashlib
import importlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest


try:
    archive_module = importlib.import_module("data.archive")
except ModuleNotFoundError as exc:
    if exc.name != "data.archive":
        raise
    pytest.skip("Implement data/archive.py from the archive-loader lesson", allow_module_level=True)

load_symbol_bars = archive_module.load_symbol_bars
COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


@pytest.fixture
def synthetic_archive():
    parent = Path(__file__).resolve().parents[1] / "local-data" / "scratch" / "loader-tests"
    root = parent / uuid.uuid4().hex
    root.mkdir(parents=True)
    request = {
        "symbols": ["AAA"], "timeframe": "1Min",
        "start_inclusive": "2024-01-01T00:00:00Z",
        "end_exclusive": "2024-03-01T00:00:00Z",
    }
    manifest = {"schema_version": 1, "request": request, "partitions": {}}
    try:
        yield root, manifest
    finally:
        if root.resolve().parent != parent.resolve():
            raise RuntimeError("Unsafe test cleanup path")
        shutil.rmtree(root)


def write_partition(archive, month, timestamps, *, status="complete"):
    root, manifest = archive
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    relative = f"bars/AAA/{month}.csv.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[stamp, 100, 102, 99, 101, 1200, 20, 100.5] for stamp in timestamps]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, compression="gzip")
    manifest["partitions"][relative] = {
        "status": status, "rows": len(rows),
        "start_inclusive": start.isoformat(), "end_exclusive": end.isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_half_open_window_crosses_months_preserves_gaps_and_values(synthetic_archive):
    root, _ = synthetic_archive
    write_partition(synthetic_archive, "2024-01", ["2024-01-31T23:57:00Z", "2024-01-31T23:59:00Z"])
    write_partition(synthetic_archive, "2024-02", ["2024-02-01T00:00:00Z", "2024-02-01T00:01:00Z"])
    result = load_symbol_bars(root, "AAA", "2024-01-31T23:57:00Z", "2024-02-01T00:01:00Z", expected_timeframe="1Min")
    assert list(result.index) == list(pd.to_datetime(["2024-01-31T23:57:00Z", "2024-01-31T23:59:00Z", "2024-02-01T00:00:00Z"]))
    assert str(result.index.tz) == "UTC"
    assert list(result.columns) == COLUMNS[1:]
    assert result.iloc[0].to_dict() == dict(zip(COLUMNS[1:], [100, 102, 99, 101, 1200, 20, 100.5]))


def test_end_at_month_boundary_does_not_require_next_month(synthetic_archive):
    root, _ = synthetic_archive
    write_partition(synthetic_archive, "2024-01", ["2024-01-31T23:59:00Z"])
    result = load_symbol_bars(root, "AAA", "2024-01-31T18:59:00-05:00", "2024-02-01T00:00:00Z", expected_timeframe="1Min")
    assert len(result) == 1


@pytest.mark.parametrize("overrides", [
    {"symbol": "BBB"}, {"expected_timeframe": "15Min"},
    {"start": "2024-01-01"}, {"end": "2024-02-01"},
    {"start": "2023-12-31T00:00:00Z"},
    {"end": "2024-04-01T00:00:00Z"},
    {"end": "2024-01-01T00:00:00Z"},
])
def test_invalid_requests_fail_explicitly(synthetic_archive, overrides):
    root, _ = synthetic_archive
    write_partition(synthetic_archive, "2024-01", [])
    arguments = dict(symbol="AAA", start="2024-01-01T00:00:00Z", end="2024-02-01T00:00:00Z", expected_timeframe="1Min")
    arguments.update(overrides)
    with pytest.raises(ValueError):
        load_symbol_bars(root, **arguments)


@pytest.mark.parametrize("damage", ["missing_entry", "incomplete", "missing_file", "checksum", "duplicate", "nonfinite", "short_coverage"])
def test_incomplete_or_corrupt_partition_is_not_silently_dropped(synthetic_archive, damage):
    root, manifest = synthetic_archive
    stamps = ["2024-01-02T14:30:00Z"] * (2 if damage == "duplicate" else 1)
    path = write_partition(synthetic_archive, "2024-01", stamps)
    entry = manifest["partitions"]["bars/AAA/2024-01.csv.gz"]
    if damage == "missing_entry":
        manifest["partitions"].clear()
    elif damage == "incomplete":
        entry["status"] = "error"
    elif damage == "missing_file":
        path.unlink()
    elif damage == "checksum":
        entry["sha256"] = "0" * 64
    elif damage == "nonfinite":
        frame = pd.read_csv(path, dtype={"close": float})
        frame.loc[0, "close"] = float("inf")
        frame.to_csv(path, index=False, compression="gzip")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "short_coverage":
        entry["end_exclusive"] = "2024-01-15T00:00:00Z"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    expected_error = FileNotFoundError if damage == "missing_file" else ValueError
    with pytest.raises(expected_error):
        load_symbol_bars(root, "AAA", "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", expected_timeframe="1Min")


def test_completed_empty_partition_is_a_valid_empty_frame(synthetic_archive):
    root, _ = synthetic_archive
    write_partition(synthetic_archive, "2024-01", [])
    result = load_symbol_bars(root, "AAA", "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", expected_timeframe="1Min")
    assert result.empty
    assert str(result.index.tz) == "UTC"
    assert list(result.columns) == COLUMNS[1:]
