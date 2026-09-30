"""Behavioral contract for the exchange-calendar adapter."""

import pandas as pd
import pytest

from data.calendars import (
    SCHEDULE_COLUMNS,
    load_calendar,
    session_schedule,
    validate_schedule,
)


@pytest.fixture(scope="module")
def xnys():
    return load_calendar("XNYS")


def test_load_calendar_records_provider_identity(xnys):
    assert xnys.name == "XNYS"
    assert xnys.provider == "exchange_calendars"
    assert xnys.provider_version


@pytest.mark.parametrize("name", ["", "XHKG", None])
def test_load_calendar_rejects_unsupported_or_invalid_names(name):
    with pytest.raises(ValueError):
        load_calendar(name)


def test_schedule_uses_half_open_date_range_and_project_schema(xnys):
    result = session_schedule(xnys, "2026-09-30", "2026-10-01")

    assert result.index.equals(
        pd.DatetimeIndex(["2026-09-30"], name="session_id")
    )
    assert list(result.columns) == list(SCHEDULE_COLUMNS)
    assert result.loc["2026-09-30", "session_open"] == pd.Timestamp(
        "2026-09-30 13:30Z"
    )
    assert result.loc["2026-09-30", "session_close"] == pd.Timestamp(
        "2026-09-30 20:00Z"
    )
    assert result.attrs["range_semantics"] == "[start, end)"
    assert result.attrs["calendar_provider_version"] == xnys.provider_version


def test_schedule_excludes_holidays_and_weekends(xnys):
    result = session_schedule(xnys, "2026-09-04", "2026-09-09")

    assert result.index.tolist() == [
        pd.Timestamp("2026-09-04"),
        pd.Timestamp("2026-09-08"),
    ]


def test_schedule_preserves_early_close(xnys):
    result = session_schedule(xnys, "2026-11-27", "2026-11-28")

    row = result.iloc[0]
    assert row["session_open"] == pd.Timestamp("2026-11-27 14:30Z")
    assert row["session_close"] == pd.Timestamp("2026-11-27 18:00Z")
    assert row["session_close"] - row["session_open"] == pd.Timedelta(hours=3.5)


def test_schedule_reflects_daylight_saving_shift_in_utc(xnys):
    result = session_schedule(xnys, "2026-03-06", "2026-03-10")

    assert result["session_open"].tolist() == [
        pd.Timestamp("2026-03-06 14:30Z"),
        pd.Timestamp("2026-03-09 13:30Z"),
    ]
    assert result["session_close"].tolist() == [
        pd.Timestamp("2026-03-06 21:00Z"),
        pd.Timestamp("2026-03-09 20:00Z"),
    ]


def test_range_with_no_sessions_returns_valid_empty_schedule(xnys):
    result = session_schedule(xnys, "2026-09-05", "2026-09-08")

    assert result.empty
    assert list(result.columns) == list(SCHEDULE_COLUMNS)
    validate_schedule(result)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-09-30", "2026-09-30"),
        ("2026-10-01", "2026-09-30"),
        ("2026-09-30 12:00", "2026-10-01"),
        (pd.Timestamp("2026-09-30", tz="UTC"), "2026-10-01"),
        ("not-a-date", "2026-10-01"),
    ],
)
def test_schedule_rejects_ambiguous_or_reversed_ranges(xnys, start, end):
    with pytest.raises(ValueError):
        session_schedule(xnys, start, end)


def test_validate_schedule_rejects_intraday_breaks(xnys):
    schedule = session_schedule(xnys, "2026-09-30", "2026-10-01")
    schedule.loc[:, "break_start"] = pd.Timestamp("2026-09-30 16:00Z")
    schedule.loc[:, "break_end"] = pd.Timestamp("2026-09-30 17:00Z")

    with pytest.raises(NotImplementedError, match="Intraday-break"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_overlapping_sessions(xnys):
    schedule = session_schedule(xnys, "2026-09-29", "2026-10-01")
    schedule.iloc[1, schedule.columns.get_loc("session_open")] = schedule.iloc[0][
        "session_close"
    ] - pd.Timedelta(minutes=1)

    with pytest.raises(ValueError, match="overlap"):
        validate_schedule(schedule)
