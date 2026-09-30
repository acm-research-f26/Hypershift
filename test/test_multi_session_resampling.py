"""Contract tests for calendar-aware resampling across market sessions."""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_integer_dtype
from pandas.testing import assert_frame_equal

from data.archive import BAR_COLUMNS
from data.calendars import load_calendar, session_schedule
from data import resampling


if not hasattr(resampling, "resample_sessions"):
    pytest.skip(
        "Implement data.resampling.resample_sessions",
        allow_module_level=True,
    )


def make_bars(schedule, *, base_by_session=None):
    """Create one complete minute-bar sequence for every supplied session."""

    base_by_session = base_by_session or {}
    frames = []

    for session_id, session in schedule.iterrows():
        index = pd.date_range(
            session["session_open"],
            session["session_close"],
            freq="min",
            inclusive="left",
            name="timestamp_utc",
        )
        base = float(base_by_session.get(session_id, 100.0))
        offsets = np.arange(len(index), dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "open": base + offsets,
                    "high": base + offsets + 2.0,
                    "low": base + offsets - 2.0,
                    "close": base + offsets + 1.0,
                    "volume": np.ones(len(index)),
                    "trade_count": np.ones(len(index)),
                    "vwap": base + offsets + 0.5,
                },
                index=index,
            )
        )

    if frames:
        return pd.concat(frames)

    return pd.DataFrame(
        {
            column: pd.Series(dtype="float64")
            for column in BAR_COLUMNS
        },
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp_utc"),
    )


@pytest.fixture(scope="module")
def xnys():
    return load_calendar("XNYS")


def test_regular_and_early_close_sessions_are_resampled_separately(xnys):
    schedule = session_schedule(xnys, "2026-11-25", "2026-11-28")
    first_id, second_id = schedule.index
    bars = make_bars(
        schedule,
        base_by_session={first_id: 100.0, second_id: 1_000.0},
    )
    original = bars.copy(deep=True)

    result = resampling.resample_sessions(bars, schedule, "1h")

    assert_frame_equal(bars, original)
    assert result.index.name == "bar_end"
    assert str(result.index.tz) == "UTC"
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.groupby("session_id", sort=False).size().to_dict() == {
        first_id: 7,
        second_id: 4,
    }

    regular = result.loc[result["session_id"] == first_id]
    early_close = result.loc[result["session_id"] == second_id]

    assert regular["expected_minutes"].tolist() == [60] * 6 + [30]
    assert early_close["expected_minutes"].tolist() == [60] * 3 + [30]
    assert regular["is_partial"].tolist() == [False] * 6 + [True]
    assert early_close["is_partial"].tolist() == [False] * 3 + [True]
    assert result["is_complete"].all()

    assert regular["open"].iloc[0] == 100.0
    assert early_close["open"].iloc[0] == 1_000.0
    assert regular.index[-1] == schedule.loc[first_id, "session_close"]
    assert early_close.index[-1] == schedule.loc[second_id, "session_close"]


def test_outside_session_observations_are_not_aggregated(xnys):
    schedule = session_schedule(xnys, "2026-09-30", "2026-10-01")
    bars = make_bars(schedule)
    session = schedule.iloc[0]
    outside = pd.DataFrame(
        999_999.0,
        columns=list(BAR_COLUMNS),
        index=pd.DatetimeIndex(
            [
                session["session_open"] - pd.Timedelta(minutes=1),
                session["session_close"],
            ],
            name="timestamp_utc",
        ),
    )
    combined = pd.concat([bars, outside]).sort_index()

    result = resampling.resample_sessions(combined, schedule, "390min")

    assert len(result) == 1
    assert result["open"].iloc[0] == bars["open"].iloc[0]
    assert result["close"].iloc[0] == bars["close"].iloc[-1]
    assert result["high"].iloc[0] < 999_999.0
    assert result["observed_minutes"].iloc[0] == 390


def test_missing_minutes_remain_local_to_their_session(xnys):
    schedule = session_schedule(xnys, "2026-09-29", "2026-10-01")
    bars = make_bars(schedule)
    second_session = schedule.iloc[1]
    missing_timestamp = second_session["session_open"] + pd.Timedelta(minutes=5)
    bars = bars.drop(index=missing_timestamp)

    result = resampling.resample_sessions(bars, schedule, "15min")
    first_id, second_id = schedule.index
    first = result.loc[result["session_id"] == first_id]
    second = result.loc[result["session_id"] == second_id]

    assert first["is_complete"].all()
    assert not second["is_complete"].iloc[0]
    assert second["observed_minutes"].iloc[0] == 14
    assert second["coverage_fraction"].iloc[0] == pytest.approx(14 / 15)
    assert second["is_complete"].iloc[1:].all()


def test_no_session_range_returns_typed_empty_result_with_provenance(xnys):
    schedule = session_schedule(xnys, "2026-09-05", "2026-09-08")
    bars = make_bars(schedule)

    result = resampling.resample_sessions(bars, schedule, "15min")

    assert result.empty
    assert result.index.name == "bar_end"
    assert str(result.index.tz) == "UTC"
    assert list(result.columns) == [
        "session_id",
        "bar_start",
        "expected_minutes",
        "is_partial",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "observed_minutes",
        "vwap",
        "coverage_fraction",
        "is_complete",
    ]
    assert is_integer_dtype(result["expected_minutes"])
    assert is_integer_dtype(result["observed_minutes"])
    assert is_bool_dtype(result["is_partial"])
    assert is_bool_dtype(result["is_complete"])
    assert result.attrs["calendar_name"] == "XNYS"
    assert result.attrs["output_interval"] == str(pd.Timedelta("15min"))


@pytest.mark.parametrize("output_interval", [None, "0min", "-1min", "90s"])
def test_invalid_interval_is_rejected_even_when_schedule_is_empty(
    xnys, output_interval
):
    schedule = session_schedule(xnys, "2026-09-05", "2026-09-08")
    bars = make_bars(schedule)

    with pytest.raises(ValueError):
        resampling.resample_sessions(bars, schedule, output_interval)

