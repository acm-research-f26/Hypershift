"""Segment 3a: candidate forecast indices use chronology and grid metadata."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.types import ObservationPanel


try:
    windows = importlib.import_module("data.windows")
except ModuleNotFoundError as exc:
    if exc.name != "data.windows":
        raise
    pytest.skip("Add data/windows.py with iter_forecast_origins", allow_module_level=True)


def make_panel(t=6, *, timestamps=None, sessions=None, starts=None):
    times = (pd.date_range("2024-01-02T14:45Z", periods=t, freq="15min")
             if timestamps is None else pd.DatetimeIndex(timestamps))
    t = len(times)
    values = (100 + np.arange(t, dtype=float))[:, None, None]
    flags = np.ones(values.shape, dtype=bool)
    return ObservationPanel(
        timestamps=times, node_ids=("AAA",), field_names=("close",),
        values=values, observation_mask=flags, eligible_mask=flags.copy(),
        session_ids=pd.DatetimeIndex(["2024-01-02"] * t if sessions is None else sessions),
        bar_starts=times - pd.Timedelta("15min") if starts is None else pd.DatetimeIndex(starts),
        availability_times=times + pd.Timedelta("30s"), expected_minutes=np.full(t, 15),
        observed_minutes=np.full((t, 1), 15), is_partial=np.zeros(t, dtype=bool),
        units={"close": "USD"}, provenance={},
    )


def test_known_forecast_rows_reserve_history_and_future_endpoint():
    panel = make_panel()
    origins = list(windows.iter_forecast_origins(panel, lookback=3))
    assert origins == [2, 3, 4]
    # At t=4, the feature slice is rows 2,3,4; its future endpoint is row 5.
    t = origins[-1]
    assert list(range(t - 3 + 1, t + 1)) == [2, 3, 4]
    assert t + 1 == 5


def test_single_bar_history_can_begin_at_first_row_but_never_at_final_row():
    assert list(windows.iter_forecast_origins(make_panel(), lookback=1)) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("t,lookback", [(0, 1), (1, 1), (6, 6), (6, 7)])
def test_insufficient_history_or_no_future_row_produces_no_candidates(t, lookback):
    assert list(windows.iter_forecast_origins(make_panel(t), lookback)) == []


def test_history_and_target_cannot_cross_session_boundary():
    times = pd.date_range("2024-01-02T14:45Z", periods=4, freq="15min").append(
        pd.date_range("2024-01-03T14:45Z", periods=4, freq="15min"))
    panel = make_panel(timestamps=times, sessions=["2024-01-02"] * 4 + ["2024-01-03"] * 4)
    assert list(windows.iter_forecast_origins(panel, 2)) == [1, 2, 5, 6]


def test_time_grid_gap_invalidates_windows_that_jump_it_in_history_or_target():
    panel = make_panel(timestamps=["2024-01-02T14:45Z", "2024-01-02T15:00Z",
                                  "2024-01-02T15:30Z", "2024-01-02T15:45Z", "2024-01-02T16:00Z"])
    assert list(windows.iter_forecast_origins(panel, 2)) == [3]


def test_contiguous_short_final_bar_is_a_valid_future_endpoint():
    panel = make_panel(
        timestamps=["2024-01-02T19:30Z", "2024-01-02T20:30Z", "2024-01-02T21:00Z"],
        starts=["2024-01-02T18:30Z", "2024-01-02T19:30Z", "2024-01-02T20:30Z"],
    )
    panel = replace(panel, expected_minutes=np.array([60, 60, 30]), is_partial=np.array([False, False, True]))
    assert list(windows.iter_forecast_origins(panel, 2)) == [1]


def test_candidate_selection_does_not_inspect_prices_or_stock_eligibility():
    panel = make_panel()
    original = list(windows.iter_forecast_origins(panel, 3))
    changed = replace(panel, values=np.full(panel.values.shape, np.nan),
                      eligible_mask=np.zeros(panel.values.shape, dtype=bool))
    assert list(windows.iter_forecast_origins(changed, 3)) == original


def test_equivalent_timezones_preserve_candidates():
    panel = make_panel()
    changed = replace(panel, timestamps=panel.timestamps.tz_convert("America/New_York"))
    assert list(windows.iter_forecast_origins(changed, 3)) == [2, 3, 4]


def test_numpy_integer_lookback_is_supported():
    assert list(windows.iter_forecast_origins(make_panel(), np.int64(3))) == [2, 3, 4]


@pytest.mark.parametrize("lookback", [0, -1, 2.5, True, None])
def test_invalid_lookback_is_rejected(lookback):
    with pytest.raises(ValueError):
        list(windows.iter_forecast_origins(make_panel(), lookback))


@pytest.mark.parametrize("damage", ["unsorted", "duplicate", "naive", "missing_time",
                                  "wrong_start_length", "wrong_session_length", "missing_session", "naive_start"])
def test_invalid_chronology_or_temporal_metadata_fails_explicitly(damage):
    panel = make_panel()
    if damage == "unsorted":
        panel = replace(panel, timestamps=panel.timestamps[::-1])
    elif damage == "duplicate":
        panel = replace(panel, timestamps=panel.timestamps.take([0, 0, 2, 3, 4, 5]))
    elif damage == "naive":
        panel = replace(panel, timestamps=panel.timestamps.tz_localize(None))
    elif damage == "missing_time":
        panel = replace(panel, timestamps=pd.DatetimeIndex([pd.NaT, *panel.timestamps[1:]]))
    elif damage == "wrong_start_length":
        panel = replace(panel, bar_starts=panel.bar_starts[:-1])
    elif damage == "wrong_session_length":
        panel = replace(panel, session_ids=panel.session_ids[:-1])
    elif damage == "missing_session":
        panel = replace(panel, session_ids=pd.DatetimeIndex([pd.NaT, *panel.session_ids[1:]]))
    else:
        panel = replace(panel, bar_starts=panel.bar_starts.tz_localize(None))
    with pytest.raises(ValueError):
        list(windows.iter_forecast_origins(panel, 3))
