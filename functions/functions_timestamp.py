# -*- coding: utf-8 -*-
"""
Small helpers for decomposing UNIX timestamps into calendar fields.

Note: this file previously defined timestamp_info_df() twice -- once for
a single scalar timestamp (returning a one-row DataFrame) and once for a
list of timestamps (returning a multi-row DataFrame). The second
definition silently shadowed the first (dead code, never reachable). The
single-timestamp version is kept here as timestamp_info_row instead, so
both behaviors remain available under distinct names.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import math
from datetime import datetime
import pandas as pd
##########################################################################################
# TIMESTAMP INFO (single timestamp, tuple of fields)
##########################################################################################
def timestamp_info(timestamp):
    """
    Decompose a single UNIX timestamp into calendar fields.

    Parameters:
    timestamp (float): Seconds since the UNIX epoch.

    Returns:
    tuple: (date, time, day_of_week, week_of_year, quarter, month).

    Examples:
    >>> timestamp_info(1718452245)
    """
    dt = datetime.fromtimestamp(timestamp)
    date = dt.strftime('%Y-%m-%d')
    time = dt.strftime('%H:%M:%S')
    day_of_week = dt.strftime('%A')
    week_of_year = dt.strftime('%U')
    month = dt.strftime('%B')
    quarter = (dt.month - 1) // 3 + 1
    return date, time, day_of_week, week_of_year, quarter, month
##########################################################################################
# TIMESTAMP INFO (single timestamp, one-row DataFrame)
##########################################################################################
def timestamp_info_row(timestamp):
    """
    Decompose a single UNIX timestamp into calendar fields, as a one-row DataFrame.

    Parameters:
    timestamp (float): Seconds since the UNIX epoch.

    Returns:
    pandas.DataFrame: One row, columns Date, Time, Year, Week_Year,
    Month_No, Day_No, Hour, Minute, Second, Day, Quarter, Month.

    Examples:
    >>> timestamp_info_row(1718452245)
    """
    dt = datetime.fromtimestamp(timestamp)
    data = {
        'Date': [dt.strftime('%Y-%m-%d')],
        'Time': [dt.strftime('%H:%M:%S')],
        'Year': [dt.strftime('%Y')],
        'Week_Year': [dt.strftime('%U')],
        'Month_No': [dt.strftime('%m')],
        'Day_No': [dt.strftime('%d')],
        'Hour': [dt.strftime('%H')],
        'Minute': [dt.strftime('%M')],
        'Second': [dt.strftime('%S')],
        'Day': [dt.strftime('%A')],
        'Quarter': [(dt.month - 1) // 3 + 1],
        'Month': [dt.strftime('%B')],
    }
    return pd.DataFrame(data)
##########################################################################################
# TIMESTAMP INFO (list of timestamps, multi-row DataFrame)
##########################################################################################
def timestamp_info_df(timestamps):
    """
    Decompose a list of UNIX timestamps into calendar fields, one row per timestamp.

    Parameters:
    timestamps (list of float): Seconds since the UNIX epoch.

    Returns:
    pandas.DataFrame: Columns year, month_n, day, date, time, hour,
    minute, second, day_of_week, week_of_year, month, quarter.

    Examples:
    >>> timestamp_info_df([1718452245, 1704067200])
    """
    dt = [datetime.fromtimestamp(t) for t in timestamps]
    year = [d.year for d in dt]
    month_n = [d.month for d in dt]
    day = [d.day for d in dt]
    date = [d.strftime('%Y-%m-%d') for d in dt]
    time = [d.strftime('%H:%M:%S') for d in dt]
    hour = [d.strftime('%H') for d in dt]
    minute = [d.strftime('%M') for d in dt]
    second = [d.strftime('%S') for d in dt]
    day_of_week = [d.strftime('%A') for d in dt]
    week_of_year = [d.strftime('%U') for d in dt]
    month = [d.strftime('%B') for d in dt]
    quarter = [math.ceil(d.month / 3) for d in dt]
    return pd.DataFrame(
        list(zip(year, month_n, day, date, time, hour, minute, second, day_of_week, week_of_year, month, quarter)),
        columns=["year", "month_n", "day", "date", "time", "hour", "minute", "second",
                 "day_of_week", "week_of_year", "month", "quarter"])
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    ts = 1718452245  # 2024-06-15 10:30:45 UTC-ish (local time zone dependent)

    print("=" * 80, "\ntimestamp_info\n", "=" * 80, sep="")
    print(timestamp_info(ts))

    print("\n" + "=" * 80, "\ntimestamp_info_row\n", "=" * 80, sep="")
    print(timestamp_info_row(ts))

    print("\n" + "=" * 80, "\ntimestamp_info_df\n", "=" * 80, sep="")
    timestamps = [1718452245, 1704067200, 1609459200]
    print(timestamp_info_df(timestamps))
