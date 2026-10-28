"""Interval and timestamp helpers.

All timestamps in this project are timezone-aware UTC with millisecond resolution.
A bar is identified by its *open* time, matching Binance's convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440,
}


def interval_minutes(interval: str) -> int:
    try:
        return INTERVAL_MINUTES[interval]
    except KeyError:
        raise ValueError(
            f"Unsupported interval {interval!r}; choose one of {sorted(INTERVAL_MINUTES)}"
        ) from None


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=interval_minutes(interval))


def interval_to_ms(interval: str) -> int:
    return interval_minutes(interval) * 60_000


def bars_per_day(interval: str) -> float:
    return 1440 / interval_minutes(interval)


def bars_per_year(interval: str) -> float:
    # Crypto trades 24/7, so a year is 365 full days.
    return 365 * bars_per_day(interval)


def to_utc_index(epoch_values) -> pd.DatetimeIndex:
    """Convert epoch integers to a UTC DatetimeIndex.

    Binance switched some public data files from milliseconds to microseconds,
    so the unit is detected from magnitude: ms epochs are ~1.7e12 today, us ~1.7e15.
    """
    arr = np.asarray(epoch_values, dtype="int64")
    unit = "us" if arr.size and arr.max() > 10**14 else "ms"
    idx = pd.to_datetime(arr, unit=unit, utc=True)
    return pd.DatetimeIndex(idx).as_unit("ms")


def to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").as_unit("ms")
