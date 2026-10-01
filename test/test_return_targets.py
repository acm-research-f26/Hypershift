"""Segment 2: forward return labels, masks, and label availability."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from data.types import ObservationPanel


try:
    targets = importlib.import_module("features.targets")
except ModuleNotFoundError as exc:
    if exc.name != "features.targets":
        raise
    pytest.skip("Add features/targets.py with make_return_targets", allow_module_level=True)


def make_panel(close, *, timestamps=None, sessions=None):
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    t, n = close.shape
    times = (pd.date_range("2024-01-02T14:45Z", periods=t, freq="15min", name="bar_end")
             if timestamps is None else pd.DatetimeIndex(timestamps, name="bar_end"))
    values = close[:, :, None]
    observed = np.isfinite(values)
    return ObservationPanel(
        timestamps=times, node_ids=tuple(f"STOCK{i}" for i in range(n)), field_names=("close",),
        values=values, observation_mask=observed, eligible_mask=observed.copy(),
        session_ids=pd.DatetimeIndex(["2024-01-02"] * t if sessions is None else sessions),
        bar_starts=times - pd.Timedelta("15min"), availability_times=times + pd.Timedelta("30s"),
        expected_minutes=np.full(t, 15), observed_minutes=np.full((t, n), 15),
        is_partial=np.zeros(t, dtype=bool), units={"close": "USD"}, provenance={},
    )


def test_next_bar_targets_align_with_current_rows_and_preserve_inputs():
    panel = make_panel([[100, 200], [110, 180], [99, 198]])
    original = panel.values.copy()
    labels = targets.make_return_targets(panel)
    assert labels.values.shape == labels.mask.shape == (3, 2)
    np.testing.assert_allclose(labels.values[:2], [[np.log(1.1), np.log(0.9)], [np.log(0.9), np.log(1.1)]])
    assert labels.mask[:2].all() and labels.mask.dtype == np.dtype(bool)
    assert np.isnan(labels.values[-1]).all() and not labels.mask[-1].any()
    assert labels.units == "log_return"
    np.testing.assert_array_equal(panel.values, original)


def test_target_bounds_and_availability_describe_the_future_endpoint():
    panel = make_panel([100, 110, 99])
    labels = targets.make_return_targets(panel)
    assert_index_equal(labels.start_times, panel.timestamps)
    assert_index_equal(labels.end_times[:-1], panel.timestamps[1:], check_names=False)
    assert_index_equal(labels.availability_times[:-1], panel.availability_times[1:], check_names=False)
    assert pd.isna(labels.end_times[-1]) and pd.isna(labels.availability_times[-1])


def test_target_is_not_known_until_both_endpoint_prices_are_available():
    panel = make_panel([100, 110, 99])
    available = pd.DatetimeIndex(["2024-01-02T15:10Z", "2024-01-02T15:00:30Z", "2024-01-02T15:15:30Z"])
    labels = targets.make_return_targets(replace(panel, availability_times=available))
    assert labels.availability_times[0] == pd.Timestamp("2024-01-02T15:10Z")
    assert labels.availability_times[1] == pd.Timestamp("2024-01-02T15:15:30Z")


@pytest.mark.parametrize("damage", ["ineligible", "missing", "nonpositive"])
def test_bad_endpoint_masks_adjacent_labels_for_only_the_affected_stock(damage):
    panel = make_panel([[100, 200], [110, 180], [99, 198]])
    if damage == "ineligible":
        mask = panel.eligible_mask.copy()
        mask[1, 0, 0] = False
        panel = replace(panel, eligible_mask=mask)
    else:
        values = panel.values.copy()
        values[1, 0, 0] = np.nan if damage == "missing" else 0
        panel = replace(panel, values=values)
    labels = targets.make_return_targets(panel)
    assert not labels.mask[:, 0].any() and np.isnan(labels.values[:, 0]).all()
    assert labels.mask[:2, 1].all()


def test_session_boundary_never_becomes_an_intraday_target():
    panel = make_panel(
        [100, 110, 120, 132],
        timestamps=["2024-01-02T20:45Z", "2024-01-02T21:00Z", "2024-01-03T14:45Z", "2024-01-03T15:00Z"],
        sessions=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
    )
    labels = targets.make_return_targets(panel)
    np.testing.assert_array_equal(labels.mask[:, 0], [True, False, True, False])
    np.testing.assert_allclose(labels.values[[0, 2], 0], np.log(1.1))
    assert np.isnan(labels.values[[1, 3], 0]).all()


def test_missing_bin_in_time_axis_does_not_change_one_step_target_meaning():
    panel = make_panel([100, 110, 121], timestamps=["2024-01-02T14:45Z", "2024-01-02T15:15Z", "2024-01-02T15:30Z"])
    labels = targets.make_return_targets(panel)
    np.testing.assert_array_equal(labels.mask[:, 0], [False, True, False])
    assert np.isnan(labels.values[0, 0])
    assert labels.values[1, 0] == pytest.approx(np.log(1.1))


@pytest.mark.parametrize("t", [0, 1])
def test_panel_without_future_pair_keeps_shapes_and_missing_endpoint_metadata(t):
    panel = make_panel(np.full((t, 2), 100.0))
    labels = targets.make_return_targets(panel)
    assert labels.values.shape == labels.mask.shape == (t, 2)
    assert len(labels.start_times) == len(labels.end_times) == len(labels.availability_times) == t
    assert labels.end_times.isna().all() and labels.availability_times.isna().all()
    assert np.isnan(labels.values).all() and not labels.mask.any()


def test_changing_future_endpoint_changes_only_its_corresponding_target_window():
    panel = make_panel([100, 110, 99])
    before = targets.make_return_targets(panel)
    values = panel.values.copy()
    values[-1, 0, 0] = 121
    after = targets.make_return_targets(replace(panel, values=values))
    assert before.values[0, 0] == after.values[0, 0]
    assert before.values[1, 0] != after.values[1, 0]
    assert not after.mask[-1].any()


def test_equivalent_timezones_are_normalized_to_utc_metadata():
    panel = make_panel([100, 110, 99])
    shifted = replace(panel, timestamps=panel.timestamps.tz_convert("America/New_York"),
                      availability_times=panel.availability_times.tz_convert("America/New_York"))
    labels = targets.make_return_targets(shifted)
    assert_index_equal(labels.start_times, panel.timestamps)
    assert str(labels.end_times.tz) == str(labels.availability_times.tz) == "UTC"


@pytest.mark.parametrize("damage", ["wrong_length", "naive", "missing", "before_bar_end"])
def test_invalid_availability_metadata_fails_explicitly(damage):
    panel = make_panel([100, 110, 99])
    if damage == "wrong_length":
        available = panel.availability_times[:-1]
    elif damage == "naive":
        available = panel.availability_times.tz_localize(None)
    elif damage == "missing":
        available = pd.DatetimeIndex([panel.availability_times[0], pd.NaT, panel.availability_times[2]])
    else:
        available = panel.timestamps - pd.Timedelta("1s")
    with pytest.raises(ValueError):
        targets.make_return_targets(replace(panel, availability_times=available))
