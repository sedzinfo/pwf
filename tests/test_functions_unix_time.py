import datetime

import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_unix_time as fu

D1 = datetime.date(2026, 1, 15)
D2 = pd.Timestamp("2026-01-15 14:30:00", tz="GMT")
TIMES = ["2026-03-01 23:00:00", "2026-03-01 05:05:09", "2026-07-04 13:00:01", "2026-12-31 00:30:00"]
TEXT_COLUMNS = ["YEAR", "MONTH_NUMERIC", "DAY_NUMERIC", "HOUR", "MINUTE", "SECOND", "MILLISECOND"]


def _reference(reference, name):
    """Components are zero-padded text: read them as text, JULIAN as a number."""
    expected = reference(name, dtype=str)
    if "JULIAN" in expected:
        expected["JULIAN"] = expected["JULIAN"].astype(float)
    return expected


@pytest.mark.parametrize("name, kwargs", [
    ("date", dict(x=D1)),
    ("date_extended", dict(x=D1, extended=True)),
    ("datetime", dict(x=D2)),
    ("datetime_extended", dict(x=D2, extended=True)),
    ("text_date", dict(x="01/15/1900", format="%m/%e/%Y", extended=True)),
    ("text_datetime", dict(x=TIMES, format="%Y-%m-%d %H:%M:%S", extended=True)),
    ("text_breaks", dict(x=TIMES, format="%Y-%m-%d %H:%M:%S", extended=True, breaks=(-1, 6, 12, 18, 22, 24))),
    ("text_tz", dict(x=TIMES, format="%Y-%m-%d %H:%M:%S", tz="America/New_York", extended=True)),
    ("fraction", dict(x=[D2 + pd.Timedelta(seconds=0.5), D2 + pd.Timedelta(seconds=3600.25)], extended=True)),
])
def test_decompose_datetime_matches_rwf(reference, name, kwargs):
    assert_matches_reference(fu.decompose_datetime(**kwargs), _reference(reference, f"unix_time_{name}"), rtol=1e-12)


def test_decompose_datetime_numeric_counts_seconds_from_origin():
    # rwf (R 4.6) counts numbers from the current time; the port uses origin, as documented
    result = fu.decompose_datetime([0, 86400 * 31 + 3600], extended=True)
    assert result["FULL_DATE"].tolist() == ["1970-01-01", "1970-02-01"]
    assert result["FULL_TIME"].tolist() == ["00:00", "01:00"]
    np.testing.assert_allclose(result["JULIAN"], [0, 31 + 1 / 24])


def test_decompose_datetime_missing_values():
    result = fu.decompose_datetime(["2026-01-15 14:30:00", None], extended=True)
    assert result["FULL_DATE"].iloc[0] == "2026-01-15"
    assert pd.isna(result["FULL_DATE"].iloc[1]) and result["QUARTER"].iloc[1] == "QNA"


def test_convert_excel_unix_timestamp_matches_rwf(reference):
    result = fu.convert_excel_unix_timestamp([1, 25569, 45000.5])
    assert_matches_reference(pd.DataFrame(result), reference("unix_time_excel"))
    assert fu.convert_excel_unix_timestamp(1) == {"unix_timestamp": -2209075200, "excel_timestamp": 1 / 86400 + 24107}
