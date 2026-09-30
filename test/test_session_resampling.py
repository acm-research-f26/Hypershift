"""Behavioral contract for the learner's single-session minute-bar resampler."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data import resampling


FIELDS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
OPEN = pd.Timestamp("2026-09-30 13:30Z")
CLOSE = OPEN + pd.Timedelta(minutes=5)


@pytest.fixture
def run():
    if not hasattr(resampling, "resample_session"):
        pytest.skip("Implement resample_session in data/resampling.py")

    def call(bars, **overrides):
        options = dict(session_open=OPEN, session_close=CLOSE, output_interval="2min")
        options.update(overrides)
        return resampling.resample_session(bars, **options)

    return call


@pytest.fixture
def bars():
    return pd.DataFrame(
        [
            [10, 12, 9, 11, 2, 1, 10],
            [11, 15, 10, 14, 6, 3, 12],
            [20, 22, 19, 21, 3, 2, 20],
            [21, 25, 18, 24, 1, 1, 24],
            [30, 32, 29, 31, 4, 2, 30],
        ],
        columns=FIELDS,
        index=pd.date_range(OPEN, periods=5, freq="min", name="timestamp_utc"),
        dtype=float,
    )


def test_hand_calculated_ohlcv_and_volume_weighted_vwap(run, bars):
    result = run(bars)
    expected = pd.DataFrame(
        [[10, 15, 9, 14, 8, 4, 11.5], [20, 25, 18, 24, 4, 3, 21],
         [30, 32, 29, 31, 4, 2, 30]],
        columns=FIELDS,
        index=pd.DatetimeIndex([OPEN + pd.Timedelta(minutes=n) for n in (2, 4, 5)],
                               name="bar_end"),
        dtype=float,
    )
    assert_frame_equal(result[FIELDS], expected, check_freq=False)
    assert result["bar_start"].tolist() == [OPEN + pd.Timedelta(minutes=n) for n in (0, 2, 4)]
    assert result["expected_minutes"].tolist() == [2, 2, 1]
    assert result["observed_minutes"].tolist() == [2, 2, 1]
    assert result["coverage_fraction"].tolist() == [1, 1, 1]
    assert result["is_complete"].tolist() == [True, True, True]
    assert result["is_partial"].tolist() == [False, False, True]


def test_session_is_half_open_and_internal_boundary_starts_next_bin(run, bars):
    outside = pd.DataFrame(999.0, columns=FIELDS,
                           index=pd.DatetimeIndex([OPEN - pd.Timedelta(minutes=1), CLOSE]))
    result = run(pd.concat([bars, outside]).sort_index())
    assert_frame_equal(result, run(bars))
    assert result["open"].tolist() == [10, 20, 30]
    assert result["close"].tolist() == [14, 24, 31]


def test_missing_observations_preserve_empty_bins_and_flag_coverage(run, bars):
    result = run(bars.iloc[[0, 4]])
    assert len(result) == 3
    assert result["observed_minutes"].tolist() == [1, 0, 1]
    assert result["coverage_fraction"].tolist() == [0.5, 0, 1]
    assert result["is_complete"].tolist() == [False, False, True]
    assert result["is_partial"].tolist() == [False, False, True]
    assert result.loc[result.index[1], FIELDS].isna().all()
    assert result.iloc[0]["close"] == 11


@pytest.mark.parametrize("outside_only", [False, True])
def test_no_session_observations_still_returns_expected_grid(run, bars, outside_only):
    sample = bars.copy() if outside_only else bars.iloc[:0].copy()
    if outside_only:
        sample.index = sample.index + pd.Timedelta(days=1)
    result = run(sample)
    assert len(result) == 3
    assert result[FIELDS].isna().all().all()
    assert result["observed_minutes"].tolist() == [0, 0, 0]
    assert result["coverage_fraction"].tolist() == [0, 0, 0]
    assert not result["is_complete"].any()


def test_zero_observed_volume_has_missing_vwap_but_complete_coverage(run, bars):
    bars.loc[:, "volume"] = 0
    result = run(bars)
    assert result["vwap"].isna().all()
    assert result["volume"].tolist() == [0, 0, 0]
    assert result["close"].tolist() == [14, 24, 31]
    assert result["is_complete"].all()


def test_future_bar_changes_do_not_change_an_already_ended_bin(run, bars):
    before = run(bars)
    changed = bars.copy()
    changed.iloc[2:] *= 10
    assert_frame_equal(run(changed).iloc[:1], before.iloc[:1])
    assert_frame_equal(run(bars.iloc[:2]).iloc[:1], before.iloc[:1])


def test_input_is_not_mutated(run, bars):
    before = bars.copy(deep=True)
    run(bars)
    assert_frame_equal(bars, before)


def test_equivalent_timezones_give_identical_utc_result(run, bars):
    localized = bars.tz_convert("America/New_York")
    result = run(localized, session_open=OPEN.tz_convert("America/New_York"),
                 session_close=CLOSE.tz_convert("America/New_York"))
    assert_frame_equal(result, run(bars))
    assert str(result.index.tz) == "UTC"


def test_one_minute_interval_preserves_observed_values_and_shifts_label(run, bars):
    result = run(bars, output_interval="1min")
    expected = bars.copy()
    expected.index = (expected.index + pd.Timedelta(minutes=1)).rename("bar_end")
    assert_frame_equal(result[FIELDS], expected, check_freq=False)
    assert not result["is_partial"].any()


def test_interval_longer_than_session_produces_one_complete_partial_bar(run, bars):
    result = run(bars, output_interval="1h")
    assert result.index.tolist() == [CLOSE]
    assert result["expected_minutes"].tolist() == [5]
    assert result["observed_minutes"].tolist() == [5]
    assert result["is_partial"].tolist() == [True]
    assert result["is_complete"].tolist() == [True]
    assert result.iloc[0]["volume"] == 16
    assert result.iloc[0]["vwap"] == pytest.approx(296 / 16)


@pytest.mark.parametrize("problem", ["naive", "duplicate", "unsorted", "nat", "seconds", "not_datetime"])
def test_invalid_timestamps_are_rejected(run, bars, problem):
    if problem == "naive":
        bars.index = bars.index.tz_localize(None)
    elif problem == "duplicate":
        bars.index = pd.DatetimeIndex([OPEN] * len(bars))
    elif problem == "unsorted":
        bars = bars.iloc[::-1]
    elif problem == "nat":
        bars.index = pd.DatetimeIndex([pd.NaT, *bars.index[1:]])
    elif problem == "seconds":
        bars.index = bars.index + pd.Timedelta(seconds=1)
    else:
        bars.index = pd.RangeIndex(len(bars))
    with pytest.raises(ValueError):
        run(bars)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "bad"])
def test_nonfinite_or_nonnumeric_values_are_rejected(run, bars, value):
    bars["close"] = bars["close"].astype(object)
    bars.iloc[0, bars.columns.get_loc("close")] = value
    with pytest.raises(ValueError):
        run(bars)


@pytest.mark.parametrize("column", ["volume", "trade_count"])
def test_negative_activity_is_rejected(run, bars, column):
    bars.loc[bars.index[0], column] = -1
    with pytest.raises(ValueError):
        run(bars)


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_invalid_bar_columns_are_rejected(run, bars, problem):
    invalid = bars.drop(columns="vwap") if problem == "missing" else pd.concat([bars, bars[["close"]]], axis=1)
    with pytest.raises(ValueError):
        run(invalid)


@pytest.mark.parametrize("options", [
    {"output_interval": "0min"}, {"output_interval": "90s"},
    {"session_open": OPEN.tz_localize(None)}, {"session_close": OPEN},
])
def test_invalid_grid_arguments_are_rejected(run, bars, options):
    with pytest.raises(ValueError):
        run(bars, **options)
