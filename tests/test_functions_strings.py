import pandas as pd

from conftest import assert_matches_reference
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
