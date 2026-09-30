import pandas as pd
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass
from pandas.api.types import is_bool_dtype, is_integer_dtype
from data.calendars import validate_schedule
from data.archive import BAR_COLUMNS

# Defines time intervals that are expected to exists in a trading session. 
def make_session_bins(*, session_open, session_close, output_interval):
    try:
        opening = pd.Timestamp(session_open)
        closing = pd.Timestamp(session_close)
        interval = pd.Timedelta(output_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid session boundary or output interval") from exc

    if (pd.isna(opening) or pd.isna(closing) or opening.tzinfo is None or closing.tzinfo is None):
        raise ValueError("Session boundaries must be timezone-aware")

    opening = opening.tz_convert("UTC")
    closing = closing.tz_convert("UTC")
    minute = pd.Timedelta(minutes=1)

    if opening >= closing:
        raise ValueError("Session open must precede session close")

    if opening.floor("min") != opening or closing.floor("min") != closing:
        raise ValueError("Session boundaries must align to whole minutes")

    if (pd.isna(interval) or interval <= pd.Timedelta(0) or interval % minute != pd.Timedelta(0)):
        raise ValueError("Output interval must be positive whole minutes")

    starts = pd.date_range(opening, closing, freq=interval, inclusive="left")
    ends = pd.DatetimeIndex([min(start + interval, closing) for start in starts], name="bar_end")
    durations = ends - starts

    return pd.DataFrame(
        {"bar_start": starts, "expected_minutes": (durations // minute).astype("int64"), "is_partial": durations < interval}, 
        index=ends
    )
    
# Aggregates one-minute bars into session-aligned OHLCV bins and reports observation coverage.
def resample_session(bars, *, session_open, session_close, output_interval):
    # Construct the expected intervals independently of observed prices.
    grid = make_session_bins(
        session_open=session_open,
        session_close=session_close,
        output_interval=output_interval,
    )
    opening = grid["bar_start"].iloc[0]
    closing = grid.index[-1]
    interval = pd.Timedelta(output_interval)

    # Validate timestamps before assigning observations to intervals.
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Bars need a timezone-aware DatetimeIndex")

    if (bars.index.hasnans or not bars.index.is_unique or not bars.index.is_monotonic_increasing):
        raise ValueError("Bar timestamps must be present, unique, and sorted")

    timestamps = bars.index.tz_convert("UTC")
    if not timestamps.equals(timestamps.floor("min")):
        raise ValueError("Bars must be aligned to whole minutes")

    if not bars.columns.is_unique or not set(BAR_COLUMNS).issubset(bars.columns):
        raise ValueError("Missing or duplicate bar columns")

    # Copy and validate values so the original observations remain intact.
    frame = bars.loc[:, list(BAR_COLUMNS)].astype(float).copy()
    frame.index = timestamps

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Bar values must be finite")
    if (frame[["volume", "trade_count"]] < 0).any().any():
        raise ValueError("Volume and trade count cannot be negative")

    # Keep this session and assign each minute to its output bin.
    frame = frame.loc[(frame.index >= opening) & (frame.index < closing)].copy()

    bin_numbers = ((frame.index - opening) // interval).astype("int64")
    frame.index = grid.index.take(bin_numbers)
    frame["weighted_vwap"] = frame["vwap"] * frame["volume"]

    # Aggregate the observations belonging to each bin.
    grouped = frame.groupby(level=0).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        weighted_vwap=("weighted_vwap", "sum"),
        observed_minutes=("close", "size")
    )
    grouped["vwap"] = (grouped.pop("weighted_vwap") / grouped["volume"].where(grouped["volume"] > 0))

    # Preserve empty bins and describe how much data was observed.
    result = grid.join(grouped, how="left")
    result["observed_minutes"] = (result["observed_minutes"].fillna(0).astype("int64"))
    result["coverage_fraction"] = (result["observed_minutes"] / result["expected_minutes"])
    result["is_complete"] = (result["observed_minutes"] == result["expected_minutes"])
    return result

@dataclass(frozen=True)
class BarEligibilityPolicy:
    """Rules for deciding whether a resampled bar can feed a feature."""

    minimum_coverage: float = 1.0
    partial_bar_policy: str = "keep_and_flag"

    def __post_init__(self):
        if isinstance(self.minimum_coverage, bool):
            raise ValueError("minimum_coverage must be a number between 0 and 1")

        try:
            minimum_coverage = float(self.minimum_coverage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "minimum_coverage must be a number between 0 and 1"
            ) from exc

        if not np.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between 0 and 1")

        if self.partial_bar_policy not in {"keep_and_flag", "exclude"}:
            raise ValueError(
                "partial_bar_policy must be 'keep_and_flag' or 'exclude'"
            )

        object.__setattr__(self, "minimum_coverage", minimum_coverage)

def resample_sessions(bars, schedule, output_interval):
    """Resample minute bars across several non-overlapping market sessions.

    The returned index remains the UTC bar-end timestamp. ``session_id`` records
    the calendar session associated with each bar. No bin can cross a session
    boundary.
    """

    validate_schedule(schedule)

    try:
        interval = pd.Timedelta(output_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid output interval") from exc

    minute = pd.Timedelta(minutes=1)
    if (
        pd.isna(interval)
        or interval <= pd.Timedelta(0)
        or interval % minute != pd.Timedelta(0)
    ):
        raise ValueError("Output interval must be positive whole minutes")

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")

    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Bars need a timezone-aware DatetimeIndex")

    if (
        bars.index.hasnans
        or not bars.index.is_unique
        or not bars.index.is_monotonic_increasing
    ):
        raise ValueError("Bar timestamps must be present, unique, and sorted")

    timestamps = bars.index.tz_convert("UTC")
    if not timestamps.equals(timestamps.floor("min")):
        raise ValueError("Bars must be aligned to whole minutes")

    if not bars.columns.is_unique or not set(BAR_COLUMNS).issubset(bars.columns):
        raise ValueError("Missing or duplicate bar columns")

    try:
        validated_bars = bars.loc[:, list(BAR_COLUMNS)].astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Bar fields must be numeric") from exc

    validated_bars.index = timestamps

    if not np.isfinite(validated_bars.to_numpy()).all():
        raise ValueError("Bar values must be finite")

    if (validated_bars[["volume", "trade_count"]] < 0).any().any():
        raise ValueError("Volume and trade count cannot be negative")

    session_frames = []

    for session_id, session in schedule.iterrows():
        session_open = session["session_open"]
        session_close = session["session_close"]

        # Slice before calling resample_session so each invocation processes
        # only one session and cannot aggregate across an overnight boundary.
        in_session = (
            (validated_bars.index >= session_open)
            & (validated_bars.index < session_close)
        )
        session_bars = validated_bars.loc[in_session]

        sampled = resample_session(
            session_bars,
            session_open=session_open,
            session_close=session_close,
            output_interval=interval,
        )
        sampled.insert(0, "session_id", session_id)
        session_frames.append(sampled)

    if session_frames:
        result = pd.concat(session_frames)
    else:
        index = pd.DatetimeIndex([], tz="UTC", name="bar_end")
        result = pd.DataFrame(
            {
                "session_id": pd.Series(index=index, dtype="datetime64[ns]"),
                "bar_start": pd.Series(
                    index=index, dtype="datetime64[ns, UTC]"
                ),
                "expected_minutes": pd.Series(index=index, dtype="int64"),
                "is_partial": pd.Series(index=index, dtype="bool"),
                "open": pd.Series(index=index, dtype="float64"),
                "high": pd.Series(index=index, dtype="float64"),
                "low": pd.Series(index=index, dtype="float64"),
                "close": pd.Series(index=index, dtype="float64"),
                "volume": pd.Series(index=index, dtype="float64"),
                "trade_count": pd.Series(index=index, dtype="float64"),
                "observed_minutes": pd.Series(index=index, dtype="int64"),
                "vwap": pd.Series(index=index, dtype="float64"),
                "coverage_fraction": pd.Series(index=index, dtype="float64"),
                "is_complete": pd.Series(index=index, dtype="bool"),
            },
            index=index,
        )

    # Retain provenance for tomorrow's dataset adapter.
    result.attrs.update(bars.attrs)
    result.attrs.update(schedule.attrs)
    result.attrs["output_interval"] = str(interval)

    return result

def apply_bar_eligibility(frame, policy):
    """Add an eligibility mask without changing aggregated bar values.

    A shortened final session bin can still be eligible when it contains all
    expected observations. Missing field values remain visible; consumers should
    combine ``is_eligible`` with the selected field's ``notna()`` mask.
    """

    if isinstance(policy, Mapping):
        try:
            policy = BarEligibilityPolicy(**dict(policy))
        except TypeError as exc:
            raise ValueError("Invalid bar eligibility policy") from exc

    if not isinstance(policy, BarEligibilityPolicy):
        raise TypeError(
            "policy must be a BarEligibilityPolicy or compatible mapping"
        )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    required = {
        "expected_minutes",
        "observed_minutes",
        "coverage_fraction",
        "is_complete",
        "is_partial",
    }

    if not frame.columns.is_unique:
        raise ValueError("Frame columns must be unique")

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Frame is missing coverage columns: {', '.join(sorted(missing))}"
        )

    if not is_integer_dtype(frame["expected_minutes"]):
        raise ValueError("expected_minutes must be integer-valued")

    if not is_integer_dtype(frame["observed_minutes"]):
        raise ValueError("observed_minutes must be integer-valued")

    if not is_bool_dtype(frame["is_complete"]):
        raise ValueError("is_complete must be Boolean")

    if not is_bool_dtype(frame["is_partial"]):
        raise ValueError("is_partial must be Boolean")

    expected = frame["expected_minutes"]
    observed = frame["observed_minutes"]

    try:
        coverage = pd.to_numeric(
            frame["coverage_fraction"], errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage_fraction must be numeric") from exc

    if (expected <= 0).any():
        raise ValueError("expected_minutes must be positive")

    if ((observed < 0) | (observed > expected)).any():
        raise ValueError(
            "observed_minutes must be between zero and expected_minutes"
        )

    if (
        not np.isfinite(coverage.to_numpy()).all()
        or ((coverage < 0) | (coverage > 1)).any()
    ):
        raise ValueError("coverage_fraction must be finite and between 0 and 1")

    calculated_coverage = observed / expected
    if not np.allclose(
        coverage.to_numpy(),
        calculated_coverage.to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "coverage_fraction does not match observed/expected minutes"
        )

    calculated_complete = observed == expected
    if not frame["is_complete"].equals(calculated_complete):
        raise ValueError("is_complete does not match the coverage fields")

    eligible = (
        (observed > 0)
        & (coverage >= policy.minimum_coverage)
    )

    if policy.partial_bar_policy == "exclude":
        eligible &= ~frame["is_partial"]

    result = frame.copy()
    result["is_eligible"] = eligible.astype(bool)
    return result

