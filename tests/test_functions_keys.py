import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_keys as fk


def _keys_frame(key_list):
    rows = [(i, q) for i in sorted(key_list) for q in key_list[i]]
    return pd.DataFrame(rows, columns=["dimension", "question"])


@pytest.mark.parametrize("name, key", [
    ("", [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
    ("_uneven", [3, 1, 1, 2, 3, 3]),
])
def test_questions_by_keys_matches_rwf(reference, name, key):
    assert_matches_reference(_keys_frame(fk.questions_by_keys(key)), reference(f"keys_questions_by_keys{name}"))


def test_questions_by_keys_without_dimension_one_is_empty(reference):
    # rwf's which(match(key, i) %in% key) finds nothing when 1 is not in key
    key_list = fk.questions_by_keys([2, 2, 3])
    lengths = pd.DataFrame({"lengths": [len(key_list[i]) for i in sorted(key_list)]})
    assert_matches_reference(lengths, reference("keys_questions_by_keys_no_one"))


def test_questions_dimensions_dataframe_matches_rwf(reference):
    result = fk.questions_dimensions_dataframe(
        [3, 1, 1, 2, 3, 3], [f"Dimension{i}" for i in range(1, 4)],
        [f"Elaborated_Dimension{i}" for i in range(1, 4)], [f"Question{i}" for i in range(1, 7)])
    assert_matches_reference(result, reference("keys_questions_dimensions_dataframe"))
