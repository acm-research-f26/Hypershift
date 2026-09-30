"""Contract tests for a session grid, independent of bar data and calendars.

The learner supplies data/resampling.py. Until then, only this module skips;
missing dependencies inside an implemented module must still fail collection.
"""

import importlib

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_integer_dtype
from pandas.testing import assert_frame_equal


try:
    resampling_module = importlib.import_module("data.resampling")
except ModuleNotFoundError as exc:
    if exc.name != "data.resampling":
        raise
    pytest.skip(
        "Implement data/resampling.py from the session-bin lesson",
        allow_module_level=True,
    )

make_session_bins = resampling_module.make_session_bins


def regular_session(interval):
    return make_session_bins(
        session_open=pd.Timestamp("2026-09-30 09:30", tz="America/New_York"),
        session_close=pd.Timestamp("2026-09-30 16:00", tz="America/New_York"),
        output_interval=interval,
    )


def test_fifteen_minute_grid_covers_regular_session_and_has_typed_metadata():
    result = regular_session("15min")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "bar_end"
    assert str(result.index.tz) == "UTC"
    assert list(result.columns) == ["bar_start", "expected_minutes", "is_partial"]
    assert str(result["bar_start"].dt.tz) == "UTC"
    assert is_integer_dtype(result["expected_minutes"])
    assert is_bool_dtype(result["is_partial"])

    expected_ends = pd.DatetimeIndex(
        [
            "2026-09-30 13:45", "2026-09-30 14:00",
            "2026-09-30 14:15", "2026-09-30 14:30",
            "2026-09-30 14:45", "2026-09-30 15:00",
            "2026-09-30 15:15", "2026-09-30 15:30",
            "2026-09-30 15:45", "2026-09-30 16:00",
            "2026-09-30 16:15", "2026-09-30 16:30",
            "2026-09-30 16:45", "2026-09-30 17:00",
            "2026-09-30 17:15", "2026-09-30 17:30",
            "2026-09-30 17:45", "2026-09-30 18:00",
            "2026-09-30 18:15", "2026-09-30 18:30",
            "2026-09-30 18:45", "2026-09-30 19:00",
            "2026-09-30 19:15", "2026-09-30 19:30",
            "2026-09-30 19:45", "2026-09-30 20:00",
        ],
        tz="UTC",
        name="bar_end",
    )
    pd.testing.assert_index_equal(result.index, expected_ends)
    assert result["bar_start"].iloc[0] == pd.Timestamp("2026-09-30 13:30Z")
    assert result["bar_start"].iloc[1:].tolist() == expected_ends[:-1].tolist()
    assert result["expected_minutes"].tolist() == [15] * 26
    assert result["expected_minutes"].sum() == 390
    assert not result["is_partial"].any()


def test_hourly_bins_start_at_session_open_and_clip_last_bin_to_close():
    result = regular_session(pd.Timedelta(hours=1))

    assert result.index.tolist() == [
        pd.Timestamp("2026-09-30 14:30Z"),
        pd.Timestamp("2026-09-30 15:30Z"),
        pd.Timestamp("2026-09-30 16:30Z"),
        pd.Timestamp("2026-09-30 17:30Z"),
        pd.Timestamp("2026-09-30 18:30Z"),
        pd.Timestamp("2026-09-30 19:30Z"),
        pd.Timestamp("2026-09-30 20:00Z"),
    ]
    assert result["bar_start"].iloc[0] == pd.Timestamp("2026-09-30 13:30Z")
    assert result["bar_start"].iloc[1:].tolist() == result.index[:-1].tolist()
    assert result["expected_minutes"].tolist() == [60, 60, 60, 60, 60, 60, 30]
    assert result["is_partial"].tolist() == [False] * 6 + [True]


def test_supplied_early_close_is_respected_without_inventing_later_bins():
    result = make_session_bins(
        session_open=pd.Timestamp("2026-11-27 09:30", tz="America/New_York"),
        session_close=pd.Timestamp("2026-11-27 13:00", tz="America/New_York"),
        output_interval="1h",
    )

    assert result.index.tolist() == [
        pd.Timestamp("2026-11-27 15:30Z"),
        pd.Timestamp("2026-11-27 16:30Z"),
        pd.Timestamp("2026-11-27 17:30Z"),
        pd.Timestamp("2026-11-27 18:00Z"),
    ]
    assert result["bar_start"].iloc[0] == pd.Timestamp("2026-11-27 14:30Z")
    assert result["expected_minutes"].tolist() == [60, 60, 60, 30]
    assert result["is_partial"].tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    ("session_open", "session_close"),
    [
        ("2026-09-30T09:30:00-04:00", "2026-09-30T16:00:00-04:00"),
        ("2026-09-30T13:30:00Z", "2026-09-30T20:00:00Z"),
        ("2026-09-30T15:30:00+02:00", "2026-09-30T20:00:00Z"),
    ],
)
def test_equivalent_aware_boundaries_produce_identical_utc_grids(session_open, session_close):
    result = make_session_bins(
        session_open=pd.Timestamp(session_open),
        session_close=pd.Timestamp(session_close),
        output_interval="60min",
    )
    assert_frame_equal(result, regular_session("1h"))


def test_interval_longer_than_session_produces_one_partial_bin():
    result = regular_session("1D")

    assert result.index.tolist() == [pd.Timestamp("2026-09-30 20:00Z")]
    assert result["bar_start"].tolist() == [pd.Timestamp("2026-09-30 13:30Z")]
    assert result["expected_minutes"].tolist() == [390]
    assert result["is_partial"].tolist() == [True]


def test_interval_equal_to_session_produces_one_complete_bin():
    result = regular_session("390min")

    assert result.index.tolist() == [pd.Timestamp("2026-09-30 20:00Z")]
    assert result["expected_minutes"].tolist() == [390]
    assert result["is_partial"].tolist() == [False]


@pytest.mark.parametrize(
    "output_interval",
    ["0min", "-15min", "90s", "1min 1ns", "not a duration", pd.NaT, None],
)
def test_invalid_or_non_whole_minute_intervals_raise_value_error(output_interval):
    with pytest.raises(ValueError):
        regular_session(output_interval)


@pytest.mark.parametrize(
    ("session_open", "session_close"),
    [
        (pd.Timestamp("2026-09-30 13:30"), pd.Timestamp("2026-09-30 20:00Z")),
        (pd.Timestamp("2026-09-30 13:30Z"), pd.Timestamp("2026-09-30 20:00")),
        (pd.NaT, pd.Timestamp("2026-09-30 20:00Z")),
        (pd.Timestamp("2026-09-30 13:30Z"), pd.NaT),
        (pd.Timestamp("2026-09-30 20:00Z"), pd.Timestamp("2026-09-30 20:00Z")),
        (pd.Timestamp("2026-09-30 20:01Z"), pd.Timestamp("2026-09-30 20:00Z")),
        (pd.Timestamp("2026-09-30 13:30:01Z"), pd.Timestamp("2026-09-30 20:00Z")),
        (pd.Timestamp("2026-09-30 13:30Z"), pd.Timestamp("2026-09-30 20:00:00.000000001Z")),
        ("not a timestamp", pd.Timestamp("2026-09-30 20:00Z")),
    ],
)
def test_invalid_or_unaligned_session_boundaries_raise_value_error(session_open, session_close):
    with pytest.raises(ValueError):
        make_session_bins(
            session_open=session_open,
            session_close=session_close,
            output_interval="15min",
        )
