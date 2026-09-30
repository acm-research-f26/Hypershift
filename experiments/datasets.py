from pathlib import Path

from .config import ComponentConfig, DatasetGroup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = PROJECT_ROOT / "local-data"
ALPACA_15MIN_ARCHIVE = "raw/alpaca/us_equities/15min/2021-09-26_2026-09-26"
ALPACA_1MIN_ARCHIVE = "raw/alpaca/us_equities/1min/2021-09-26_2026-09-26" # I have yet to get this data btw
UNIVERSE_FILE = (
    LOCAL_DATA / "metadata" / "universes" / "us_equities_250"
    / "2026-09-26" / "universe.csv"
)

# region Alpaca

def alpaca_group(*, name: str, archive_directory: str, native_interval: str, output_interval: str) -> DatasetGroup:
    """Factory to create a DatasetGroup for Alpaca data."""

    return DatasetGroup(
        name=name,
        domain="finance",
        source=ComponentConfig(
            name="alpaca_archive",
            params={
                "path": str(LOCAL_DATA / archive_directory),
                "native_interval": native_interval,
                "feed": "sip",
                "adjustment": "raw",
                "start": "2021-09-26",
                "end_exclusive": "2026-09-26",
                "timestamp_role": "interval_start"
            },
        ),
        nodes=ComponentConfig(
            name="universe_csv",
            params={
                "path": str(UNIVERSE_FILE),
                "expected_count": 250
            },
        ),
        sampling=ComponentConfig(
            name="market_ohlcv",
            params={
                "output_interval": output_interval,
                "calendar": "XNYS",
                "timezone": "America/New_York",
                "session": "regular",
                "intraday_anchor": "session_open",
                "weekly_anchor": "calendar_week",
                "output_timestamp_role": "interval_end",
                "partial_bar_policy": "keep_and_flag",
                "missing_policy": "mask",
                "aggregation": {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "trade_count": "sum",
                    "vwap": "volume_weighted_mean"
                },
            },
        ),
    )


ALPACA_DATA_15_MIN_INTERVALS_250_GROUP = alpaca_group(
    name="alpaca_250_15min",
    archive_directory=ALPACA_15MIN_ARCHIVE,
    native_interval="15min",
    output_interval="15min"
)

ALPACA_DATA_HOURLY_250_GROUP = alpaca_group(
    name="alpaca_250_hourly",
    archive_directory=ALPACA_15MIN_ARCHIVE,
    native_interval="15min",
    output_interval="1h"
)

ALPACA_DATA_WEEKLY_250_GROUP = alpaca_group(
    name="alpaca_250_weekly",
    archive_directory=ALPACA_15MIN_ARCHIVE,
    native_interval="15min",
    output_interval="1week"
)

# This declares a future data source; it does not download it.
ALPACA_DATA_1_MIN_INTERVALS_250_GROUP = alpaca_group(
    name="alpaca_250_1min",
    archive_directory=ALPACA_1MIN_ARCHIVE,
    native_interval="1min",
    output_interval="1min"
)

# Look up table for dataset groups by name. This is used to resolve the dataset group in the experiment config.
DATASET_GROUPS = {
    group.name: group
    for group in (
        ALPACA_DATA_15_MIN_INTERVALS_250_GROUP,
        ALPACA_DATA_HOURLY_250_GROUP,
        ALPACA_DATA_WEEKLY_250_GROUP,
        ALPACA_DATA_1_MIN_INTERVALS_250_GROUP
    )
}

# endregion