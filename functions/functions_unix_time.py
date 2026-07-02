# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_UNIX_TIME.R.

Deviations from the R original, by design:
  - decompose_datetime uses pandas' native .dt accessor to extract date
    components, rather than replicating R's approach of coercing the
    datetime to a separator-delimited string (via str_mgsub) and
    splitting it back apart. Both approaches produce the same output
    columns; pandas' native extraction is far more robust (R's string
    round-trip is sensitive to locale/format quirks in how R renders a
    POSIXct as text).
  - R distinguishes a date-only `Date` object from a full `POSIXct`
    datetime as separate classes, and only includes HOUR/MINUTE/SECOND
    columns for the latter. Python has no such built-in distinction for
    a plain numeric/string timestamp, so this uses a heuristic instead:
    if every parsed value has hour=minute=second=0, it's treated as
    date-only. A POSIXct where every value coincidentally falls at
    midnight would be mis-detected as date-only under this heuristic —
    an unavoidable tradeoff without R's separate Date/POSIXct classes.
  - R's julian(dt) (default origin) returns days since 1970-01-01, not
    day-of-year — reimplemented as that same days-since-epoch count,
    not pandas' dayofyear (which is a different, easy-to-confuse
    quantity that would silently produce wrong values here).
  - R's FULL_DATE/FULL_TIME construction relies on partial `$` column
    name matching (`df$MONTH` resolving to the actual column
    "MONTH_NUMERIC", since it's an unambiguous prefix) — Python has no
    such mechanism, so the real column names are referenced explicitly.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd

try:
    from .functions_strings import str_pad
except ImportError:
    from functions_strings import str_pad
##########################################################################################
# CONVERT EXCEL TIMESTAMP TO UNIX TIMESTAMP
##########################################################################################
def convert_excel_unix_timestamp(timestamp):
    """
    Convert between Excel and UNIX timestamp conventions.

    Parameters:
    timestamp (float or array-like): A UNIX or Excel serial timestamp.

    Returns:
    dict: {'unix_timestamp': ..., 'excel_timestamp': ...} — treats
    `timestamp` as an Excel serial date to get unix_timestamp, and as a
    UNIX timestamp (seconds) to get excel_timestamp, matching R's
    (order-independent) dual computation.

    Examples:
    >>> convert_excel_unix_timestamp(1)
    """
    unix_timestamp = (timestamp - 25569) * 86400
    excel_timestamp = (timestamp / 86400) + 24107
    return {'unix_timestamp': unix_timestamp, 'excel_timestamp': excel_timestamp}
##########################################################################################
# DECOMPOSE DATE TIME
##########################################################################################
def decompose_datetime(x, format=None, origin="1970-01-01", tz="GMT", extended=False,
                        breaks=(-1, 5, 13, 16, 20, 23)):
    """
    Decompose datetime value(s) into separate YEAR/MONTH_NUMERIC/
    DAY_NUMERIC/HOUR/MINUTE/SECOND/MILLISECOND columns, plus FULL_DATE
    (and FULL_TIME, if time components are present).

    Parameters:
    x: A datetime-like scalar or sequence — a numeric timestamp
        (seconds since `origin`), a date/datetime string, or a
        pandas/numpy/datetime object.
    format (str, optional): strptime-style format for parsing string
        input. None (default) lets pandas infer the format.
    origin (str, optional): Epoch reference for numeric input. Defaults
        to "1970-01-01" (the UNIX epoch).
    tz (str, optional): Timezone to convert to. Defaults to "GMT".
    extended (bool, optional): If True, also prepend QUARTER, MONTH
        (full name), JULIAN (days since 1970-01-01), WEEKDAY (full
        name), and DAY_PERIOD (hour-of-day bucket). Defaults to False.
    breaks (sequence, optional): Hour-of-day bin edges for DAY_PERIOD,
        labeled "Night"/"Morning"/"Noon"/"Afternoon"/"Evening". Defaults
        to (-1, 5, 13, 16, 20, 23).

    Returns:
    pandas.DataFrame: One row per input value, with the columns
    described above (as zero-padded strings, matching R's output).

    Examples:
    >>> import pandas as pd
    >>> decompose_datetime(pd.Timestamp("2024-01-15"))
    >>> decompose_datetime(pd.Timestamp("2024-01-15 10:30:45"), extended=True)
    >>> decompose_datetime("01/15/1900", format="%m/%d/%Y")
    """
    is_scalar = np.isscalar(x) or isinstance(x, str) or not hasattr(x, "__iter__")
    series = pd.Series([x]) if is_scalar else pd.Series(list(x))

    if pd.api.types.is_numeric_dtype(series):
        dt = pd.to_datetime(series, unit="s", origin=origin, utc=True)
    else:
        dt = pd.to_datetime(series, format=format, utc=True)
    dt = dt.dt.tz_convert(tz)

    has_time = not ((dt.dt.hour == 0) & (dt.dt.minute == 0) & (dt.dt.second == 0)).all()

    def pad2(component):
        return component.apply(lambda v: str_pad(str(int(v)), 2, side="left", pad="0") if pd.notna(v) else pd.NA)

    result = pd.DataFrame({
        "YEAR": dt.dt.year.astype("Int64").astype("string"),
        "MONTH_NUMERIC": pad2(dt.dt.month),
        "DAY_NUMERIC": pad2(dt.dt.day),
    })
    if has_time:
        result["HOUR"] = pad2(dt.dt.hour)
        result["MINUTE"] = pad2(dt.dt.minute)
        result["SECOND"] = pad2(dt.dt.second)
        result["MILLISECOND"] = (dt.dt.microsecond // 1000).astype("string")

    result["FULL_DATE"] = result["YEAR"] + "-" + result["MONTH_NUMERIC"] + "-" + result["DAY_NUMERIC"]
    if has_time:
        result["FULL_TIME"] = result["HOUR"] + ":" + result["MINUTE"]

    if extended:
        julian_days = (dt.dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.days
        day_period = pd.cut(dt.dt.hour, bins=list(breaks),
                             labels=["Night", "Morning", "Noon", "Afternoon", "Evening"])
        result.insert(0, "DAY_PERIOD", day_period.astype(str))
        result.insert(0, "WEEKDAY", dt.dt.day_name())
        result.insert(0, "JULIAN", julian_days)
        result.insert(0, "MONTH", dt.dt.month_name())
        result.insert(0, "QUARTER", "Q" + dt.dt.quarter.astype(str))

    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nconvert_excel_unix_timestamp\n", "=" * 80, sep="")
    print(convert_excel_unix_timestamp(1))

    print("\n" + "=" * 80, "\ndecompose_datetime\n", "=" * 80, sep="")
    d1 = pd.Timestamp("2024-06-15")
    d2 = pd.Timestamp("2024-06-15 10:30:45")

    print("bare date:")
    print(decompose_datetime(d1))
    print()
    print("bare date, extended=True:")
    print(decompose_datetime(d1, extended=True))
    print()
    print("full datetime:")
    print(decompose_datetime(d2))
    print()
    print("full datetime, extended=True:")
    print(decompose_datetime(d2, extended=True))
    print()
    print("string with explicit format:")
    print(decompose_datetime("01/15/1900", format="%m/%d/%Y"))
    print()
    print("multiple values:")
    print(decompose_datetime([d1, d2]))
