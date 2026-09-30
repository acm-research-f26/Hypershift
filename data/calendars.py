"""Exchange-calendar adapter used by the market-data pipeline.

The public functions in this module expose a project-owned schedule schema.  That
keeps the rest of the pipeline independent of the column names and date-range
conventions used by the calendar provider.
"""

from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import exchange_calendars
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


CALENDAR_PROVIDER = "exchange_calendars"
SUPPORTED_CALENDARS = frozenset({"XNYS"})
SCHEDULE_COLUMNS = (
    "session_open",
    "break_start",
    "break_end",
    "session_close",
)


@dataclass(frozen=True)
class MarketCalendar:
    """A versioned handle to a provider calendar.

    ``provider_calendar`` is intentionally private to this module.  Callers use
    :func:`session_schedule` so provider-specific details do not spread through
    the data pipeline.
    """

    name: str
    provider: str
    provider_version: str
    _provider_calendar: Any = field(repr=False, compare=False)


def load_calendar(name: str) -> MarketCalendar:
    """Load a supported exchange calendar and record its provider version.

    XNYS regular sessions are the first supported contract.  Other exchanges
    must be enabled deliberately after their breaks and session conventions are
    supported by the resampling pipeline.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Calendar name must be a non-empty string")

    normalized_name = name.strip().upper()
    if normalized_name not in SUPPORTED_CALENDARS:
        supported = ", ".join(sorted(SUPPORTED_CALENDARS))
        raise ValueError(
            f"Unsupported calendar {normalized_name!r}; supported calendars: {supported}"
        )

    provider_calendar = exchange_calendars.get_calendar(normalized_name)
    return MarketCalendar(
        name=normalized_name,
        provider=CALENDAR_PROVIDER,
        provider_version=version("exchange-calendars"),
        _provider_calendar=provider_calendar,
    )


def session_schedule(calendar: MarketCalendar, start, end) -> pd.DataFrame:
    """Return sessions whose labels are in the half-open range ``[start, end)``.

    ``start`` and ``end`` are calendar dates rather than market instants.  They
    must be midnight values without a timezone.  The returned index contains
    timezone-naive session labels, while all actual boundaries are UTC.
    Holidays and weekends therefore produce no rows and early closes retain
    their real close time.
    """

    if not isinstance(calendar, MarketCalendar):
        raise TypeError("calendar must be returned by load_calendar")

    start_date = _parse_session_date(start, "start")
    end_date = _parse_session_date(end, "end")
    if start_date >= end_date:
        raise ValueError("start must precede end")

    provider_schedule = calendar._provider_calendar.schedule
    available_start = provider_schedule.index[0]
    available_end = provider_schedule.index[-1] + pd.Timedelta(days=1)
    if start_date < available_start or end_date > available_end:
        raise ValueError(
            "Requested range is outside the loaded calendar bounds "
            f"[{available_start.date()}, {available_end.date()})"
        )

    selected = provider_schedule.loc[
        (provider_schedule.index >= start_date)
        & (provider_schedule.index < end_date),
        ["open", "break_start", "break_end", "close"],
    ].copy()
    selected.index = selected.index.rename("session_id")
    selected.columns = list(SCHEDULE_COLUMNS)
    selected.attrs.update(
        calendar_name=calendar.name,
        calendar_provider=calendar.provider,
        calendar_provider_version=calendar.provider_version,
        range_semantics="[start, end)",
    )

    validate_schedule(selected)
    return selected


def validate_schedule(schedule: pd.DataFrame) -> None:
    """Validate the schedule shape and the supported contiguous-session model."""

    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("schedule must be a pandas DataFrame")
    if not schedule.columns.is_unique:
        raise ValueError("Schedule columns must be unique")

    missing_columns = set(SCHEDULE_COLUMNS).difference(schedule.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Schedule is missing required columns: {missing}")

    if not isinstance(schedule.index, pd.DatetimeIndex):
        raise ValueError("Schedule needs a DatetimeIndex of session IDs")
    if schedule.index.tz is not None:
        raise ValueError("Session IDs must be timezone-naive calendar dates")
    if (
        schedule.index.hasnans
        or not schedule.index.is_unique
        or not schedule.index.is_monotonic_increasing
    ):
        raise ValueError("Session IDs must be present, unique, and sorted")
    if not schedule.index.equals(schedule.index.normalize()):
        raise ValueError("Session IDs must be normalized calendar dates")

    frame = schedule.loc[:, list(SCHEDULE_COLUMNS)]
    for column in SCHEDULE_COLUMNS:
        values = frame[column]
        if not is_datetime64_any_dtype(values.dtype) or values.dt.tz is None:
            raise ValueError(f"{column} must contain timezone-aware timestamps")
        if str(values.dt.tz) != "UTC":
            raise ValueError(f"{column} must use UTC")

    if frame[["session_open", "session_close"]].isna().any().any():
        raise ValueError("Session open and close must be present")
    if (frame["session_open"] >= frame["session_close"]).any():
        raise ValueError("Every session open must precede its close")

    for column in SCHEDULE_COLUMNS:
        present = frame[column].dropna()
        if not present.equals(present.dt.floor("min")):
            raise ValueError("Session boundaries must align to whole minutes")

    break_start_present = frame["break_start"].notna()
    break_end_present = frame["break_end"].notna()
    if not break_start_present.equals(break_end_present):
        raise ValueError("Break start and end must either both be present or both be absent")
    if break_start_present.any():
        raise NotImplementedError(
            "Intraday-break calendars are not supported by the session resampler yet"
        )

    if len(frame) > 1:
        later_opens = frame["session_open"].iloc[1:].reset_index(drop=True)
        earlier_closes = frame["session_close"].iloc[:-1].reset_index(drop=True)
        if (later_opens < earlier_closes).any():
            raise ValueError("Sessions must not overlap")


def _parse_session_date(value, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid calendar date") from exc

    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid calendar date")
    if timestamp.tzinfo is not None:
        raise ValueError(f"{label} must be timezone-naive because it is a session label")
    if timestamp != timestamp.normalize():
        raise ValueError(f"{label} must be a calendar date at midnight")
    return timestamp
