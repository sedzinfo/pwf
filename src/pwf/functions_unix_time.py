# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_UNIX_TIME.R (convert_excel_unix_timestamp,
decompose_datetime). Outputs match rwf, checked against R reference output
in tests/test_functions_unix_time.py.

rwf's decompose_datetime writes the POSIXct as text and splits it on
" ", ":", "-", "/" and ".". The port reads the components with pandas'
.dt accessor and reproduces what that text gives: R prints a time only when
it is not midnight, so date-only input (every value at midnight) has no
HOUR/MINUTE/SECOND/FULL_TIME columns, and fractional seconds add a
MILLISECOND column (the digits after the decimal point, "5" for .5 s).

Differences from R that cannot be avoided:
  - Numeric input is seconds since `origin`. In R 4.6 rwf's call
    as.POSIXct(x, format = "", ...) counts numbers from the current time
    instead (rwf output depends on the day it is run).
  - Text input with format="" (the default) is parsed by pandas, which infers
    the format. R's strptime with format "" returns today's date for every
    value.
  - When some values are at midnight and others are not, or only some have
    fractional seconds, or values are missing, R's text pieces do not line
    up and rwf returns shifted columns (or fails). The port gives each value
    its own components (missing values give missing cells).
  - "%e" (day without zero padding) in `format` is read as "%d".
  - rwf's `...` (passed on to as.POSIXct) has no Python equivalent.
  - Month and weekday names are English, as in R under an English locale.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import datetime

import numpy as np
import pandas as pd
##########################################################################################
# CONVERT EXCEL TIMESTAMP TO UNIX TIMESTAMP
##########################################################################################
def convert_excel_unix_timestamp(timestamp):
    """
    Convert between Excel and UNIX timestamp conventions.

    Parameters:
    timestamp (float or array-like): A UNIX or Excel serial timestamp.

    Returns:
    dict: {'unix_timestamp': ..., 'excel_timestamp': ...} (R's named list):
    `timestamp` read as an Excel serial date converted to UNIX seconds,
    (timestamp - 25569) * 86400, and `timestamp` read as UNIX seconds
    converted with rwf's formula (timestamp / 86400) + 24107. Both are
    always computed. Array input gives arrays.

    Examples:
    >>> convert_excel_unix_timestamp(1)
    >>> convert_excel_unix_timestamp([1, 45000])
    """
    if np.ndim(timestamp) > 0:
        timestamp = np.asarray(timestamp, dtype=float)
    unix_timestamp = (timestamp - 25569) * 86400
    excel_timestamp = (timestamp / 86400) + 24107
    return {'unix_timestamp': unix_timestamp, 'excel_timestamp': excel_timestamp}
##########################################################################################
# DECOMPOSE DATE TIME
##########################################################################################
def _to_datetime(x, format, origin, tz):
    """as.POSIXct(x, format, origin, tz): tz-aware timestamps, wall-clock input read in `tz`."""
    is_scalar = isinstance(x, (str, bytes)) or not hasattr(x, "__iter__")
    series = pd.Series([x] if is_scalar else list(x), dtype=object)
    non_missing = series.dropna()
    if len(non_missing) and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                                for v in non_missing):
        seconds = pd.to_numeric(series)
        return pd.to_datetime(seconds, unit="s", origin=pd.Timestamp(origin)).dt.tz_localize("UTC").dt.tz_convert(tz)
    if len(non_missing) and all(isinstance(v, str) for v in non_missing):
        fmt = format.replace("%e", "%d") if format else None
        dt = pd.to_datetime(series, format=fmt)
    else:
        values = [pd.NaT if pd.isna(v) else pd.Timestamp(v) for v in series]
        aware = [v for v in values if v is not pd.NaT and v.tzinfo is not None]
        if aware:
            values = [v if v is pd.NaT or v.tzinfo is not None else v.tz_localize(tz) for v in values]
            return pd.Series(pd.to_datetime(values, utc=True)).dt.tz_convert(tz)
        dt = pd.Series(pd.to_datetime(values))
    if dt.dt.tz is None:
        return dt.dt.tz_localize(tz)
    return dt.dt.tz_convert(tz)


def decompose_datetime(x, format="", origin="1970-01-01", tz="GMT", extended=False,
                       breaks=(-1, 5, 13, 16, 20, 23)):
    """
    Decompose date-time value(s) into data frame columns: YEAR,
    MONTH_NUMERIC, DAY_NUMERIC, HOUR, MINUTE, SECOND (MILLISECOND when there
    are fractional seconds), FULL_DATE and FULL_TIME, all as zero-padded text.
    Date-only input (every value at midnight) has no time columns, as in R.

    Parameters:
    x: A date-time scalar or sequence: numbers (seconds since `origin`),
        text, or date/datetime/pandas Timestamp objects. Dates and naive
        datetimes are read as wall-clock times in `tz` (R's as.POSIXct);
        timezone-aware values are converted to `tz`.
    format (str, optional): strptime format for text input. "" (default)
        lets pandas infer it (see the module notes for how R differs).
    origin (str, optional): Origin of numeric input. Defaults to
        "1970-01-01" (the UNIX epoch).
    tz (str, optional): Time zone. Defaults to "GMT".
    extended (bool, optional): If True, the columns QUARTER ("Q1".."Q4"),
        MONTH (month name), JULIAN (days since 1970-01-01 00:00 GMT,
        fractional, R's julian()), WEEKDAY (weekday name) and DAY_PERIOD
        are put in front. Defaults to False.
    breaks (sequence, optional): Hour-of-day breaks (intervals closed on the
        right, as R's cut()) labelling DAY_PERIOD "Night", "Morning",
        "Noon", "Afternoon", "Evening". Defaults to (-1, 5, 13, 16, 20, 23).
        As in rwf, DAY_PERIOD is missing for date-only input.

    Returns:
    pandas.DataFrame: One row per value of x, with the columns above.

    Examples:
    >>> import datetime
    >>> d1 = datetime.date(2026, 1, 15)
    >>> d2 = pd.Timestamp("2026-01-15 14:30:00", tz="GMT")
    >>> decompose_datetime(x=d1)
    >>> decompose_datetime(x=d2)
    >>> decompose_datetime(x=d1, extended=True)
    >>> decompose_datetime(x=d2, extended=True)
    >>> decompose_datetime(x="01/15/1900", format="%m/%e/%Y")
    >>> decompose_datetime(x="01/15/1900", format="%m/%e/%Y", extended=True)
    >>> decompose_datetime(x=[0, 86400 * 365.25])
    """
    dt = _to_datetime(x, format, origin, tz)
    valid = dt.notna()
    has_time = bool(((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)
                     | (dt.dt.microsecond != 0))[valid].any())
    has_fraction = bool((dt.dt.microsecond != 0)[valid].any())

    def pad2(component):
        return component.map(lambda v: f"{int(v):02d}" if pd.notna(v) else np.nan)

    def text(component):
        return component.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)

    result = pd.DataFrame({
        "YEAR": text(dt.dt.year),
        "MONTH_NUMERIC": pad2(dt.dt.month),
        "DAY_NUMERIC": pad2(dt.dt.day),
    })
    if has_time:
        result["HOUR"] = pad2(dt.dt.hour)
        result["MINUTE"] = pad2(dt.dt.minute)
        result["SECOND"] = pad2(dt.dt.second)
        if has_fraction:
            result["MILLISECOND"] = dt.dt.microsecond.map(
                lambda v: (f"{int(v):06d}".rstrip("0") or "0") if pd.notna(v) else np.nan)
    result["FULL_DATE"] = (result["YEAR"] + "-" + result["MONTH_NUMERIC"] + "-" + result["DAY_NUMERIC"]).where(valid)
    if has_time:
        result["FULL_TIME"] = (result["HOUR"] + ":" + result["MINUTE"]).where(valid)

    if extended:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        julian = (dt - epoch).dt.total_seconds() / 86400
        if has_time:
            day_period = pd.cut(dt.dt.hour, bins=list(breaks),
                                labels=["Night", "Morning", "Noon", "Afternoon", "Evening"])
            day_period = day_period.astype(object).where(day_period.notna(), np.nan)
        else:
            day_period = pd.Series(np.nan, index=dt.index, dtype=object)
        front = pd.DataFrame({
            "QUARTER": dt.dt.quarter.map(lambda q: f"Q{int(q)}" if pd.notna(q) else "QNA"),
            "MONTH": dt.dt.month_name().where(valid, np.nan),
            "JULIAN": julian,
            "WEEKDAY": dt.dt.day_name().where(valid, np.nan),
            "DAY_PERIOD": day_period,
        })
        result = pd.concat([front, result], axis=1)
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nconvert_excel_unix_timestamp\n", "=" * 80, sep="")
    print(convert_excel_unix_timestamp(1))

    print("\n" + "=" * 80, "\ndecompose_datetime\n", "=" * 80, sep="")
    d1 = datetime.date(2026, 1, 15)
    d2 = pd.Timestamp("2026-01-15 14:30:00", tz="GMT")
    print(decompose_datetime(d1))
    print(decompose_datetime(d1, extended=True))
    print(decompose_datetime(d2))
    print(decompose_datetime(d2, extended=True))
    print(decompose_datetime("01/15/1900", format="%m/%e/%Y", extended=True))
