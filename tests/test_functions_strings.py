import re

import pandas as pd
import pytest

from conftest import _load_data, assert_matches_reference
from pwf import functions_strings as fs

WORDS = ["_hello world_", "TES.T", "a  b\tc", "mixed-Case_text", "x", "Long sentence that needs wrapping at twelve"]


def test_vectorised_string_helpers_match_rwf(reference):
    result = pd.DataFrame({
        "input": WORDS, "proper": fs.str_proper(WORDS), "aes": fs.str_aes(WORDS),
        "aes_plain": fs.str_aes(WORDS, proper=False), "right": fs.str_sub(WORDS, 3, "right"),
        "left": fs.str_sub(WORDS, 3, "left"), "squish": fs.str_squish(WORDS), "wrap": fs.str_wrap(WORDS, 12),
        "pad_left": fs.str_pad(WORDS, 8, "left", "*"), "pad_both": fs.str_pad(WORDS, 9, "both", "-"),
        "count_e": fs.str_count(WORDS, "e"), "count_fixed": fs.str_count(WORDS, fs.fixed(".")),
        "replace": fs.str_replace(WORDS, "[aeiou]", "#"), "replace_all": fs.str_replace_all(WORDS, "[aeiou]", "#"),
        "replace_fixed": fs.str_replace_all(WORDS, fs.fixed("."), "!"),
        "mgsub": fs.str_mgsub(WORDS, ["[.]", "_", "-"], "+"),
    })
    assert_matches_reference(result, reference("strings_vectorised", keep_default_na=False))


def test_str_split_matches_rwf(reference):
    assert_matches_reference(fs.str_split(["a/b/c", "d/e/f"], include_original=True), reference("str_split"))
    split = pd.DataFrame(fs.str_split_fixed(["a-b-c-d", "e-f", "g"], "-", 3), columns=["X1", "X2", "X3"])
    assert_matches_reference(split, reference("str_split_fixed", keep_default_na=False))


def test_str_split_edge_cases_match_rwf(reference):
    """R's strsplit() drops a trailing empty piece; matrix() refills the rows of ragged splits, with a warning;
    data.frame() names a one-column matrix after rwf's expression."""
    assert_matches_reference(fs.str_split(["a/b/", "c/d/"]), reference("str_split_trailing"))
    warning = reference("string_conditions").set_index("call")["message"]["str_split(ragged)"]
    with pytest.warns(UserWarning, match=re.escape(warning)):
        ragged = fs.str_split(["a/b", "c/d/e"])
    assert_matches_reference(ragged, reference("str_split_ragged"))
    assert_matches_reference(fs.str_split(["a", "b"], include_original=True), reference("str_split_one_piece"))


def test_str_split_df_matches_rwf(reference):
    df = pd.DataFrame({"X1": [10, 20], "b": ["u", "v"]}, index=["1/aa", "2/bb"])
    result = fs.str_split_df(df)  # data.frame() makes the names unique: the data's X1 becomes X1.1
    expected = reference("str_split_df_row")
    assert list(result.index) == expected["row"].tolist()
    assert_matches_reference(result, expected.drop(columns="row"))
    ids = pd.DataFrame({"id": ["x-1-p", "y-2-q"], "v": [1, 2]})
    assert_matches_reference(fs.str_split_df(ids, split="-", type="collumn", index=1, include_original=True),
                             reference("str_split_df_collumn"))
    assert_matches_reference(fs.str_split_df(ids, split="-", type="collumn", index="id", include_original=True),
                             reference("str_split_df_collumn"))
    # pandas' default index 0, 1, 2 is R's automatic row names 1, 2, 3
    assert_matches_reference(fs.str_split_df(pd.DataFrame({"v": [1, 2, 3]})), reference("str_split_df_automatic"))
    message = reference("string_conditions").set_index("call")["message"]["str_split_df(type = column)"]
    with pytest.raises(ValueError, match=re.escape(message)):
        fs.str_split_df(ids, type="column", index=1)


def test_str_trim_df_matches_rwf(reference):
    """strwrap() collapses whitespace but keeps two spaces after a sentence end."""
    df = pd.DataFrame({"s": ["  a   b  ", " c", "x.  y", "p\tq", "end?  Yes"], "t": ["u", " v ", "w  ", "z", ""]})
    assert_matches_reference(fs.str_trim_df(df), reference("str_trim_df", keep_default_na=False))
    # not copied: rwf turns numbers into text, missing values into "NA", and drops an all-numeric frame's columns
    kept = fs.str_trim_df(pd.DataFrame({"s": [" a ", None], "n": [1.5, 10.0]}))
    assert kept["s"].iloc[0] == "a" and pd.isna(kept["s"].iloc[1]) and kept["n"].tolist() == [1.5, 10.0]


def test_call_to_string_matches_rwf(reference):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    string_insurance = _load_data("insurance")  # the call names the data by its variable, as R's
    string_infert = _load_data("infert")
    calls = {
        "lm": smf.ols("charges ~ age + bmi", data=string_insurance).fit(),
        "glm_long": smf.glm("case ~ education + age + parity + induced + spontaneous + age:parity + age:induced",
                            data=string_infert, family=sm.families.Binomial()).fit(),
        "poisson": smf.glm("spontaneous ~ age + parity", data=string_infert, family=sm.families.Poisson()).fit(),
        "Call": {"Call": 'f(x = 1, y = "a b")'},
        "none": {"a": 1},
    }
    expected = reference("call_to_string", keep_default_na=False).set_index("case")["call"]
    for case, model in calls.items():
        assert fs.call_to_string(model) == expected[case], case


def test_output_separator_matches_rwf(reference, capsys):
    """The same lines as rwf, which print()s each text as [1] "..."."""
    fs.output_separator("TITLE", output="OUT", instruction="Read me", length=25)
    fs.output_separator("TITLE2", length=22.5)  # rep() cuts 22.5 and 11.25 to whole characters
    expected = [re.fullmatch(r'\[1\] "(.*)"', line).group(1) for line in reference("output_separator")["line"]]
    assert capsys.readouterr().out.splitlines() == expected
