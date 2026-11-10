import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_recode as fr


@pytest.mark.parametrize("name, vector", [
    ("text", ["Agree,Hi", "All", "Hi,All,Agree", "Hi"]),
    ("numeric", [1, 3, 2, 4, 1]),
    ("case", ["b", "B", "a", "A"]),
    ("spaces", ["A, B", "B", ""]),
    ("na", ["A,B", None, "C"]),
    ("sort", ["2,10", "1", "10,3"]),
])
def test_dummy_arrange_matches_rwf(reference, name, vector):
    expected = reference(f"recode_dummy_arrange_{name}", dtype=str, keep_default_na=False)
    result = fr.dummy_arrange(vector)
    assert_matches_reference(result, expected)
    assert all(result[c].map(type).eq(str).all() for c in result.columns)  # rwf's cells are "1"/"0" text


def test_dummy_arrange_single_value_is_a_vector(reference):
    expected = reference("recode_dummy_arrange_single", dtype=str)
    result = fr.dummy_arrange(["x", "x"])
    assert isinstance(result, np.ndarray)
    assert result.tolist() == expected["value"].tolist()


def test_dummy_arrange_numeric_float_input():
    assert list(fr.dummy_arrange(np.array([1.0, 2.0, 1.0])).columns) == ["1", "2"]


@pytest.mark.parametrize("name, mydata", [
    ("named", {"a": {"x": [1, 2], "y": ["p", "q"]}, "b": {"x": [3, 4]}}),
    ("unnamed", [{"x": [1, 2]}, {"x": [3, 4], "z": 5}]),
    ("vectors", {"a": [1, 2], "b": [3, 4]}),
    ("frames", {"a": {"m": 1, "n": "z"}, "b": pd.DataFrame({"m": [2]})}),
])
def test_flatten_list_matches_rwf(reference, name, mydata):
    assert_matches_reference(fr.flatten_list(mydata), reference(f"recode_flatten_list_{name}"))


@pytest.mark.parametrize("name", ["numeric", "na", "text", "decimal"])
def test_swap_matches_rwf(reference, name):
    expected = reference(f"recode_swap_{name}", keep_default_na=name != "text")
    result = fr.swap(expected["input"].tolist() if name != "na" else [1, 3, 3, np.nan, 2])
    if name == "text":
        assert list(result) == expected["output"].tolist()
    else:
        np.testing.assert_array_equal(np.asarray(result, dtype=float), expected["output"].to_numpy(dtype=float))


def test_swap_keeps_integer_dtype():
    assert fr.swap(np.array([1, 2, 5])).dtype.kind == "i"
    assert fr.swap(np.array([1, 2, 5])).tolist() == [5, 2, 1]


def _drop_levels_frame(df):
    """drop_levels_df() of make_reference_utilities.R: categories as text plus their levels joined by |."""
    out = {}
    for c in df.columns:
        series = df[c]
        out[c] = series.astype(object).where(series.notna(), np.nan)
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[f"{c}_levels"] = ["|".join(map(str, series.cat.categories))] * len(df)
    return pd.DataFrame(out)


def _df_levels():
    return pd.DataFrame({
        "numeric1": range(1, 21),
        "factor1": pd.Categorical(["A"] * 10 + ["B"] * 10, categories=["A", "B", "C", "D"]),
        "factor2": pd.Categorical(["A"] * 3 + ["B"] * 17),
    })


@pytest.mark.parametrize("name, kwargs", [
    ("9", dict(minimum_frequency=9)),
    ("10", dict(minimum_frequency=10)),
    ("index", dict(factor_index=3, minimum_frequency=3)),
])
def test_drop_levels_matches_rwf(reference, name, kwargs):
    expected = reference(f"recode_drop_levels_{name}", keep_default_na=False)
    assert_matches_reference(_drop_levels_frame(fr.drop_levels(_df_levels(), **kwargs)), expected)


def test_drop_levels_existing_other_and_character_column(reference):
    df = pd.DataFrame({"g": pd.Categorical(["Other", "x", "x", "y"]), "h": ["a", "a", "b", "c"]})
    assert_matches_reference(_drop_levels_frame(fr.drop_levels(df, minimum_frequency=1)),
                             reference("recode_drop_levels_other", keep_default_na=False))
    result = _drop_levels_frame(fr.drop_levels(df, factor_index=2, minimum_frequency=1))
    expected = reference("recode_drop_levels_character")
    assert result["h"].isna().all() and expected["h"].isna().all()
    assert result["h_levels"].tolist() == [""] * 4 and expected["h_levels"].isna().all()


def test_drop_levels_accepts_column_names():
    by_name = fr.drop_levels(_df_levels(), factor_index=["factor2"], minimum_frequency=3)
    by_index = fr.drop_levels(_df_levels(), factor_index=3, minimum_frequency=3)
    pd.testing.assert_frame_equal(by_name, by_index)
