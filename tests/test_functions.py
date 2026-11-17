import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions

TEXT = {"dtype": str, "keep_default_na": False, "na_values": ["NA"]}


def _types_data():
    return pd.DataFrame({"f": [1, 0.1 + 0.2, np.nan, 1e-20, 123456789.123],
                         "i": pd.array([1, None, 3, 4, 5], dtype="Int64"),
                         "s": ["\t a \n", "b", None, " c", "1.5"],
                         "l": pd.array([True, False, None, True, True], dtype="boolean")})


@pytest.mark.parametrize("type", ["round", "ceiling", "floor", "tenth"])
def test_round_dataframe_matches_rwf(reference, type):
    # R rounds 0.15 to 0.1, 2.675 to 2.67 and -0.05 to 0 (numpy: 0.2, 2.68 and -0.1)
    df = pd.DataFrame({"x": [0.15, 2.675, 1.005, -0.5, 2.5, np.nan, 0.285, -0.05], "y": range(1, 9),
                       "label": list("abcdefgh")})
    expected = reference(f"round_dataframe_{type}")
    assert_matches_reference(functions.round_dataframe(df, digits=1, type=type), expected, rtol=0)


def test_change_data_type_character_matches_rwf(reference):
    # R's as.character(): 1 -> "1", 0.1 + 0.2 -> "0.3", TRUE -> "TRUE"; only tabs and newlines are trimmed
    result = functions.change_data_type(_types_data(), "character")
    assert_matches_reference(result, reference("change_data_type_character", **TEXT))
    assert result["f"].isna().tolist() == [False, False, True, False, False]


def test_change_data_type_numeric_matches_rwf(reference):
    result = functions.change_data_type(_types_data(), "numeric")
    assert_matches_reference(result, reference("change_data_type_numeric"), rtol=0)


def test_change_data_type_factor_levels_match_rwf(reference):
    data = _types_data()[["f", "i", "l"]].assign(s=["b", "B", "a", None, "A10"])
    result = functions.change_data_type(data, "factor")
    levels = pd.DataFrame({"column": result.columns,
                           "levels": ["|".join(result[c].cat.categories) for c in result.columns]})
    assert_matches_reference(levels, reference("change_data_type_factor_levels", **TEXT))


def test_change_data_type_character_factor_converts_text_columns():
    # pandas 3 stores text as "str", not object
    result = functions.change_data_type(pd.DataFrame({"s": ["b", "a"], "n": [1.5, 2.5]}), "character_factor")
    assert isinstance(result["s"].dtype, pd.CategoricalDtype) and list(result["s"].cat.categories) == ["a", "b"]
    assert result["n"].dtype == float
    back = functions.change_data_type(result, "factor_character")
    assert back["s"].tolist() == ["b", "a"] and not isinstance(back["s"].dtype, pd.CategoricalDtype)


def test_comparison_combinations_matches_rwf(reference):
    # R's order() sorts text by the locale (a < b < C), and the rows keep their positions as names
    result = functions.comparison_combinations(pd.DataFrame(columns=["b", "a", "C", "d e"]))
    expected = reference("comparison_combinations", **TEXT)
    assert_matches_reference(result.reset_index(drop=True), expected[["X1", "X2"]])
    assert [str(i + 1) for i in result.index] == expected["row"].tolist()


def test_rbind_all_matches_rwf(reference):
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}, index=["1", "2"])
    df2 = pd.DataFrame({"B": ["z", "w"], "C": [7.5, 8.5]}, index=["r3", "r4"])
    result = functions.rbind_all(df1, df2)
    expected = reference("rbind_all", keep_default_na=False, na_values=["NA"])
    assert_matches_reference(result.reset_index(drop=True), expected.drop(columns="row"))
    assert [str(i) for i in result.index] == expected["row"].tolist()


def _nc_data():
    return pd.DataFrame({"a": [1, np.nan, np.inf, 4], "b": [np.nan] * 4, "c": [5.0] * 4, "d": [-np.inf, np.nan, 2, 3]})


def test_remove_nc_matches_rwf(reference):
    assert_matches_reference(functions.remove_nc(_nc_data(), value=0), reference("remove_nc_value"))
    assert_matches_reference(functions.remove_nc(_nc_data(), remove_cols=True, remove_zero_variance=True),
                             reference("remove_nc_cols"))
    for aggressive in [False, True]:
        result = functions.remove_nc(_nc_data()[["a", "d"]], remove_rows=True, aggressive=aggressive)
        expected = reference(f"remove_nc_rows_{str(aggressive).lower()}")
        assert_matches_reference(result.reset_index(drop=True), expected.drop(columns="row"))
        assert [i + 1 for i in result.index] == expected["row"].tolist()


def test_replace_na_with_previous_matches_rwf(reference):
    result = functions.replace_na_with_previous([np.nan, np.nan, 3, np.nan, 5, np.nan])
    np.testing.assert_array_equal(np.asarray(result, dtype=float), reference("replace_na_with_previous")["value"])


def test_padNA_and_c_bind_match_rwf(reference):
    padded = functions.padNA(pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}), 4, first=False)
    assert_matches_reference(padded.reset_index(drop=True), reference("padNA", keep_default_na=False, na_values=["NA"]))
    bound = functions.c_bind(v1=[1, 2, 3], v2=[10.5, 20.5], m=pd.DataFrame({"p": [1, 2, 3, 4], "q": list("abcd")}))
    assert_matches_reference(bound.reset_index(drop=True), reference("c_bind", keep_default_na=False, na_values=["NA"]))


def test_min_max_index_matches_rwf(reference):
    expected = reference("min_max_index")
    result = functions.min_max_index([3, 1, 3, 2, 1])
    for which in ["max", "min"]:
        assert list(result[f"{which}_index"]) == expected.loc[expected["which"] == which, "index"].tolist()
