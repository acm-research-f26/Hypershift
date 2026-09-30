"""Synthetic-manifest integration tests for roadmap item 4.

No network or actual market archive is needed. This checkpoint skips only until
data/adapters.py exists; a skip is not an implementation pass.
"""

import hashlib
import importlib
import json
import shutil
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from data.archive import BAR_COLUMNS
from data.calendars import load_calendar, session_schedule
from experiments.config import ComponentConfig, DatasetGroup


try:
    adapters = importlib.import_module("data.adapters")
except ModuleNotFoundError as exc:
    if exc.name != "data.adapters":
        raise
    pytest.skip("Implement data/adapters.py for roadmap item 4", allow_module_level=True)


WINDOW = ("2024-01-02T14:30:00Z", "2024-01-02T21:00:00Z")


def minute_frame(index, base):
    prices = base + np.arange(len(index), dtype=float)
    return pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices + 0.5, "volume": 10 * (np.arange(len(index)) + 1),
        "trade_count": np.ones(len(index)), "vwap": prices + 0.25,
    }, index=pd.DatetimeIndex(index, name="timestamp_utc"))


def save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_partition(root, manifest, symbol, month, bars):
    relative = f"bars/{symbol}/{month}.csv.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.rename_axis("timestamp_utc").reset_index().to_csv(path, index=False, compression="gzip")
    start = pd.Timestamp(month + "-01", tz="UTC")
    manifest["partitions"][relative] = {
        "status": "complete", "rows": len(bars),
        "start_inclusive": start.isoformat(),
        "end_exclusive": (start + pd.offsets.MonthBegin(1)).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    save_manifest(root, manifest)
    return path


@pytest.fixture
def scratch_root():
    parent = Path(__file__).resolve().parents[1] / "local-data/scratch/panel-tests"
    root = parent / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().parent != parent.resolve():
            raise RuntimeError("Unsafe panel-test cleanup path")
        shutil.rmtree(root)


@pytest.fixture
def archive(scratch_root):
    manifest = {
        "schema_version": 1,
        "request": {
            "symbols": ["AAA", "BBB"], "timeframe": "1Min", "feed": "sip", "adjustment": "raw",
            "start_inclusive": "2024-01-01T00:00:00Z", "end_exclusive": "2025-01-01T00:00:00Z",
        }, "partitions": {},
    }
    full_index = pd.date_range(*WINDOW, freq="min", inclusive="left")
    write_partition(scratch_root, manifest, "AAA", "2024-01", minute_frame(full_index, 100))
    sparse_index = pd.to_datetime([
        "2024-01-02T14:31Z", "2024-01-02T14:34Z", "2024-01-02T14:35Z"])
    write_partition(scratch_root, manifest, "BBB", "2024-01", minute_frame(sparse_index, 200))
    return scratch_root, manifest


@pytest.fixture
def group(archive):
    root, _ = archive
    return DatasetGroup(
        name="synthetic_2min", domain="finance",
        source=ComponentConfig("alpaca_archive", {
            "path": str(root), "native_interval": "1min", "feed": "sip",
            "adjustment": "raw", "timestamp_role": "interval_start"}),
        nodes=ComponentConfig("universe_csv", {"expected_count": 250}),
        sampling=ComponentConfig("market_ohlcv", {
            "output_interval": "2min", "calendar": "XNYS", "timezone": "America/New_York",
            "session": "regular", "intraday_anchor": "session_open",
            "output_timestamp_role": "interval_end", "missing_policy": "mask",
            "minimum_coverage": 1.0, "partial_bar_policy": "keep_and_flag"}))


def with_sampling(group, **overrides):
    return replace(group, sampling=ComponentConfig(group.sampling.name, {**group.sampling.params, **overrides}))


def test_real_loader_and_resampler_produce_known_values_in_requested_stock_order(archive, group):
    panel = adapters.load_alpaca_panel(group, WINDOW, ("BBB", "AAA"))
    assert panel.node_ids == ("BBB", "AAA")
    assert panel.field_names == BAR_COLUMNS
    assert panel.values.shape == panel.observation_mask.shape == panel.eligible_mask.shape == (195, 2, 7)
    expected = pd.date_range("2024-01-02 14:32Z", "2024-01-02 21:00Z", freq="2min", name="bar_end")
    assert_index_equal(panel.timestamps, expected, exact=False)
    assert str(panel.timestamps.tz) == "UTC"
    assert panel.values[0, 1, :6] == pytest.approx([100, 102, 99, 101.5, 30, 2])
    assert panel.values[0, 1, 6] == pytest.approx((100.25 * 10 + 101.25 * 20) / 30)
    assert panel.values[-1, 1, 3] == 489.5  # Includes the final source minute.
    np.testing.assert_array_equal(panel.expected_minutes, np.full(195, 2))
    assert panel.observed_minutes[:3, 0].tolist() == [1, 0, 2]
    assert panel.observed_minutes[:, 1].tolist() == [2] * 195
    assert panel.session_ids.unique().tolist() == [pd.Timestamp("2024-01-02")]
    assert not panel.is_partial.any()
    assert panel.units == {"open": "USD", "high": "USD", "low": "USD", "close": "USD",
                           "vwap": "USD", "volume": "shares", "trade_count": "trades"}


def test_gaps_and_ineligible_aggregates_remain_visible_without_forward_filling(group):
    panel = adapters.load_alpaca_panel(group, WINDOW, ("BBB", "AAA"))
    assert panel.values[0, 0, 3] == 200.5
    assert panel.observation_mask[0, 0].all()
    assert not panel.eligible_mask[0, 0].any()
    assert np.isnan(panel.values[1, 0]).all()
    assert not panel.observation_mask[1, 0].any()
    assert not panel.eligible_mask[1, 0].any()
    assert panel.eligible_mask[2, 0].all()


def test_relaxed_coverage_changes_only_policy_mask(group):
    strict = adapters.load_alpaca_panel(group, WINDOW, ("BBB",))
    relaxed = adapters.load_alpaca_panel(with_sampling(group, minimum_coverage=0.5), WINDOW, ("BBB",))
    np.testing.assert_allclose(strict.values, relaxed.values, equal_nan=True)
    np.testing.assert_array_equal(strict.observation_mask, relaxed.observation_mask)
    assert relaxed.eligible_mask[0, 0].all()
    assert not relaxed.eligible_mask[1, 0].any()


def test_zero_volume_masks_vwap_without_masking_observed_closing_price(archive, group):
    root, manifest = archive
    path = root / "bars/AAA/2024-01.csv.gz"
    bars = pd.read_csv(path)
    bars.index = pd.to_datetime(bars.pop("timestamp_utc"), utc=True)
    bars.iloc[:2, bars.columns.get_loc("volume")] = 0
    write_partition(root, manifest, "AAA", "2024-01", bars)
    panel = adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
    assert np.isnan(panel.values[0, 0, 6])
    assert not panel.observation_mask[0, 0, 6]
    assert not panel.eligible_mask[0, 0, 6]
    assert panel.eligible_mask[0, 0, 3]


def test_provenance_references_exact_manifest_and_used_partitions_and_preserves_config(archive, group):
    root, manifest = archive
    original = deepcopy(group)
    panel = adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
    provenance = panel.provenance
    assert group == original
    assert provenance["manifest_sha256"] == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    assert provenance["source_request"] == manifest["request"]
    assert provenance["partitions"] == {"bars/AAA/2024-01.csv.gz": manifest["partitions"]["bars/AAA/2024-01.csv.gz"]}
    assert provenance["native_interval"] == "1min"
    assert provenance["minimum_coverage"] == 1.0
    assert provenance["calendar_name"] == "XNYS"
    assert provenance["calendar_provider_version"]
    assert provenance["timestamp_role"] == "interval_end"
    assert_index_equal(panel.availability_times, panel.timestamps)


def test_declared_availability_delay_preserves_bar_end_timestamps(group):
    panel = adapters.load_alpaca_panel(with_sampling(group, availability_delay="30s"), WINDOW, ("AAA",))
    assert_index_equal(panel.availability_times, panel.timestamps + pd.Timedelta("30s"))
    assert "publication times unavailable" in panel.provenance["availability_assumption"]


def test_each_raw_symbol_is_loaded_once_over_only_the_requested_window(group, monkeypatch):
    calls = []
    original = adapters.load_symbol_bars
    def spy(path, node, start, end, **kwargs):
        calls.append((node, start, end, kwargs))
        return original(path, node, start, end, **kwargs)
    monkeypatch.setattr(adapters, "load_symbol_bars", spy)
    adapters.load_alpaca_panel(group, WINDOW, ("BBB", "AAA"))
    assert [c[0] for c in calls] == ["BBB", "AAA"]
    assert all((c[1], c[2]) == tuple(map(pd.Timestamp, WINDOW)) for c in calls)
    assert all(c[3] == {"expected_timeframe": "1Min"} for c in calls)


@pytest.mark.parametrize("damage", ["incomplete", "checksum", "missing_file", "missing_entry"])
def test_missing_or_corrupt_selected_stock_partition_fails_the_whole_panel(archive, group, damage):
    root, manifest = archive
    relative = "bars/BBB/2024-01.csv.gz"
    if damage == "incomplete":
        manifest["partitions"][relative]["status"] = "pending"
    elif damage == "checksum":
        manifest["partitions"][relative]["sha256"] = "0" * 64
    elif damage == "missing_file":
        (root / relative).unlink()
    else:
        del manifest["partitions"][relative]
    save_manifest(root, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        adapters.load_alpaca_panel(group, WINDOW, ("AAA", "BBB"))


@pytest.mark.parametrize("damage", ["configured_15min", "manifest_15min", "feed", "adjustment", "timestamp_role"])
def test_source_identity_is_validated_instead_of_inferred_from_timestamp_spacing(archive, group, damage):
    root, manifest = archive
    source = dict(group.source.params)
    if damage == "configured_15min":
        source["native_interval"] = "15min"
    elif damage == "manifest_15min":
        manifest["request"]["timeframe"] = "15Min"
    elif damage in {"feed", "adjustment"}:
        manifest["request"][damage] = "other"
    else:
        source["timestamp_role"] = "interval_end"
    save_manifest(root, manifest)
    group = replace(group, source=ComponentConfig("alpaca_archive", source))
    with pytest.raises(ValueError):
        adapters.load_alpaca_panel(group, WINDOW, ("AAA",))


@pytest.mark.parametrize("nodes", [(), ("AAA", "AAA"), ("MISSING",), "AAA"])
def test_bad_node_selection_cannot_drop_or_replace_stocks(group, nodes):
    with pytest.raises(ValueError):
        adapters.load_alpaca_panel(group, WINDOW, nodes)


@pytest.mark.parametrize("window", [
    ("2024-01-02", WINDOW[1]), (WINDOW[0], WINDOW[0]),
    ("2024-01-02T14:31Z", WINDOW[1]), (WINDOW[0], "2024-01-02T20:59Z"),
    ("2024-01-06T00:00Z", "2024-01-07T00:00Z"),
])
def test_ambiguous_empty_or_truncated_session_windows_fail_clearly(group, window):
    with pytest.raises(ValueError):
        adapters.load_alpaca_panel(group, window, ("AAA",))


@pytest.mark.parametrize("parameters", [
    {"output_interval": "1week"}, {"output_interval": "30s"},
    {"session": "extended"}, {"intraday_anchor": "midnight"},
    {"missing_policy": "forward_fill"}, {"availability_delay": "-1s"},
    {"aggregation": {"close": "mean"}},
])
def test_unsupported_sampling_choices_do_not_silently_fall_back(group, parameters):
    with pytest.raises(ValueError):
        adapters.load_alpaca_panel(with_sampling(group, **parameters), WINDOW, ("AAA",))


def test_holiday_and_early_close_use_calendar_grid_and_preserve_short_final_bin(archive, group):
    root, manifest = archive
    schedule = session_schedule(load_calendar("XNYS"), "2024-11-27", "2024-11-30")
    frames = [minute_frame(pd.date_range(s.session_open, s.session_close, freq="min", inclusive="left"),
                           100 + i * 1000) for i, s in enumerate(schedule.itertuples())]
    write_partition(root, manifest, "AAA", "2024-11", pd.concat(frames))
    hourly = with_sampling(group, output_interval="1h")
    window = ("2024-11-27T05:00Z", "2024-11-30T05:00Z")
    panel = adapters.load_alpaca_panel(hourly, window, ("AAA",))
    assert panel.values.shape[0] == 11
    assert panel.session_ids.unique().tolist() == [pd.Timestamp("2024-11-27"), pd.Timestamp("2024-11-29")]
    assert panel.expected_minutes.tolist() == [60] * 6 + [30] + [60] * 3 + [30]
    assert panel.timestamps[-1] == pd.Timestamp("2024-11-29T18:00Z")
    assert panel.bar_starts[-1] == pd.Timestamp("2024-11-29T17:30Z")
    assert panel.is_partial.tolist() == [False] * 6 + [True] + [False] * 3 + [True]
    assert panel.eligible_mask.all()
    excluded = adapters.load_alpaca_panel(with_sampling(hourly, partial_bar_policy="exclude"), window, ("AAA",))
    np.testing.assert_allclose(excluded.values, panel.values)
    assert not excluded.eligible_mask[excluded.is_partial].any()
    assert excluded.eligible_mask[~excluded.is_partial].all()


def test_no_observed_minutes_still_returns_full_grid_and_all_missing_masks(archive, group):
    root, manifest = archive
    empty = minute_frame(pd.DatetimeIndex([], tz="UTC"), 100)
    write_partition(root, manifest, "AAA", "2024-01", empty)
    panel = adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
    assert panel.values.shape == (195, 1, 7)
    assert np.isnan(panel.values).all()
    assert not panel.observation_mask.any() and not panel.eligible_mask.any()
    assert not panel.observed_minutes.any()


def test_month_boundary_endpoint_does_not_require_next_month_partition(archive, group):
    root, manifest = archive
    index = pd.date_range("2024-01-31T14:30Z", "2024-01-31T21:00Z", freq="min", inclusive="left")
    write_partition(root, manifest, "AAA", "2024-01", minute_frame(index, 100))
    panel = adapters.load_alpaca_panel(group, ("2024-01-31T00:00Z", "2024-02-01T00:00Z"), ("AAA",))
    assert panel.eligible_mask.all()
    assert set(panel.provenance["partitions"]) == {"bars/AAA/2024-01.csv.gz"}


def test_future_rows_do_not_change_values_from_an_earlier_requested_window(archive, group):
    root, manifest = archive
    before = adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
    original = minute_frame(pd.date_range(*WINDOW, freq="min", inclusive="left"), 100)
    future = minute_frame(pd.date_range("2024-01-03T14:30Z", periods=4, freq="min"), 1_000_000)
    write_partition(root, manifest, "AAA", "2024-01", pd.concat([original, future]))
    after = adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
    np.testing.assert_array_equal(before.values, after.values)
    np.testing.assert_array_equal(before.eligible_mask, after.eligible_mask)
    assert before.provenance["manifest_sha256"] != after.provenance["manifest_sha256"]


def test_changed_selected_manifest_metadata_cannot_be_saved_as_original_provenance(archive, group, monkeypatch):
    root, manifest = archive
    original = adapters.load_symbol_bars
    def load_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest["partitions"]["bars/AAA/2024-01.csv.gz"]["sha256"] = "0" * 64
        save_manifest(root, manifest)
        return result
    monkeypatch.setattr(adapters, "load_symbol_bars", load_then_change)
    with pytest.raises(RuntimeError):
        adapters.load_alpaca_panel(group, WINDOW, ("AAA",))
