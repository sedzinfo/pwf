import datetime

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR, assert_matches_reference
from pwf import functions_cdf


@pytest.fixture
def df_types():
    """Same frame as df_types in tests/reference/make_reference.R."""
    titanic = pd.read_csv(DATA_DIR / "titanic.csv", keep_default_na=False, na_values=["NA"])
    df = titanic[["survived", "pclass", "name", "sex", "age", "fare", "cabin", "body"]].copy()
    df["body"] = df["body"].astype("Int64")
    df["sex"] = pd.Categorical(df["sex"])
    df["class"] = pd.Categorical(df["pclass"].map({1: "first", 2: "second", 3: "third"}),
                                 categories=["first", "second", "third"], ordered=True)
    alive = (df["survived"] == 1).astype(object)
    alive[[4, 8]] = np.nan
    df["alive"] = alive.astype("boolean")
    df["date"] = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(len(df))]
    df.loc[[0, 1], "fare"] = [np.inf, -np.inf]
    return df


def _reference_check(reference, name):
    """Text columns read as text; R's empty strings stay empty and NA becomes NaN."""
    return reference(name, keep_default_na=False, na_values=["NA"],
                     dtype={c: str for c in ["NAMES", "MIN", "MAX", "MODE", "TYPE", "CLASS", "UNIQUES", "LEVELS"]})


@pytest.mark.parametrize("name, kwargs, rows", [
    ("cdff_types", {}, None),
    ("cdff_types_nuniques", {"nuniques": 3}, None),
    ("cdff_head_nuniques", {"nuniques": 25, "name_length": 5}, 20),
])
def test_cdff(df_types, reference, name, kwargs, rows):
    data = df_types if rows is None else df_types.head(rows)
    result = functions_cdf.cdff(data, **kwargs)
    assert_matches_reference(result["check"], _reference_check(reference, f"{name}_check"))
    assert_matches_reference(result["summary"], reference(f"{name}_summary"))


def test_cdff_writes_excel(df_types, tmp_path):
    functions_cdf.cdff(df_types, file=str(tmp_path / "check"))
    sheets = pd.read_excel(tmp_path / "check.xlsx", sheet_name=None)
    assert list(sheets) == ["variables", "summary"]


def test_cdff_parralel_matches_serial(df_types):
    serial = functions_cdf.cdff(df_types, nuniques=3)
    parallel = functions_cdf.cdff(df_types, nuniques=3, parralel=True)
    pd.testing.assert_frame_equal(serial["check"], parallel["check"])


def test_r_number_formats_like_r():
    cases = {1e6: "1e+06", 1e5: "1e+05", 123456.0: "123456", 0.1 + 0.2: "0.3", 1 / 3: "0.333333333333333",
             2.5: "2.5", 1e-5: "1e-05", 0.0001: "1e-04", 1234567.1: "1234567.1", 100.0: "100", 1e15: "1e+15",
             -0.5: "-0.5", 7: "7"}
    assert {k: functions_cdf._r_number(k) for k in cases} == cases


@pytest.mark.parametrize("name, kwargs, rows", [
    ("cdf_types", {}, None),
    ("cdf_types_nuniques", {"nuniques": 3}, None),
    ("cdf_head_nuniques", {"nuniques": 25, "name_length": 5}, 20),
])
def test_cdf(df_types, reference, name, kwargs, rows):
    data = df_types.assign(empty=np.nan)
    data = data if rows is None else data.head(rows)
    result = functions_cdf.cdf(data, **kwargs)
    assert_matches_reference(result["check"], _reference_check(reference, f"{name}_check"))
    assert_matches_reference(result["summary"], reference(f"{name}_summary"))
