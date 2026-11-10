"""
rwf's generators are random, so only what does not depend on the draws is compared
with R: shapes, column names, types, factor levels, the pools values are drawn
from, and the fully deterministic outputs (balanced factors, the default target
correlation matrix).
"""
import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_generate_data as fg

R_CLASSES = {"f": "numeric", "i": "integer", "b": "logical", "O": "character"}


def _structure(call, x):
    """structure_row() of make_reference_utilities.R."""
    x = pd.DataFrame(x)
    classes, levels = [], []
    for c in x.columns:
        col = x[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            classes.append("factor")
            levels.append(",".join(fg._r_as_character(v) for v in col.cat.categories))
        elif col.isna().all() and col.dtype.kind == "f" and call.endswith("other"):
            classes.append("logical")  # R's all-NA columns are logical
            levels.append("")
        else:
            kind = "O" if col.dtype.kind in "OUT" or pd.api.types.is_string_dtype(col) else col.dtype.kind
            if kind == "f" and call == "generate_missing_frame" and c == "a":
                kind = "i"  # R keeps an integer column integer with NA; pandas needs floats
            classes.append(R_CLASSES[kind])
            levels.append("")
    return {"call": call, "nrow": len(x), "ncol": x.shape[1], "names": "|".join(map(str, x.columns)),
            "classes": "|".join(classes), "levels": "|".join(levels), "na": int(x.isna().sum().sum())}


def test_structures_match_rwf(reference):
    np.random.seed(12345)
    rows = [
        _structure("generate_data_normal", fg.generate_data(nrows=10, ncols=5)),
        _structure("generate_data_uniform", fg.generate_data(nrows=10, ncols=5, min=1, max=5, type="uniform")),
        _structure("generate_data_decimal", fg.generate_data(nrows=10, ncols=2, min=1.5, max=4, type="uniform")),
        _structure("generate_data_other", fg.generate_data(nrows=3, ncols=2, type="other")),
        _structure("generate_factor_random", fg.generate_factor(vector=list("ABCDE"), ncols=5, nrows=10, type="random")),
        _structure("generate_factor_balanced", fg.generate_factor(vector=list("ABCDE"), ncols=5, nrows=10, type="balanced")),
        _structure("generate_factor_numeric", fg.generate_factor(vector=[3, 1, 2], ncols=2, nrows=4)),
        _structure("generate_correlation_matrix_default", fg.generate_correlation_matrix(nrows=4)),
        _structure("generate_correlation_matrix_named",
                   fg.generate_correlation_matrix(pd.DataFrame([[1, .5], [.5, 1]], index=["p", "q"], columns=["p", "q"]),
                                                  nrows=3)),
        _structure("simulate_correlation_from_sample",
                   fg.simulate_correlation_from_sample(pd.DataFrame({"u": [1, 2, 3, 5], "v": [2, 1, 4, 4]}), nrows=3)),
        _structure("generate_missing_frame", fg.generate_missing(pd.DataFrame({"a": [1, 2, 3, 4], "b": list("abcd")}),
                                                                 missing=2)),
    ]
    expected = reference("generate_data_structure", keep_default_na=False)
    assert_matches_reference(pd.DataFrame(rows), expected)


def test_balanced_factors_match_rwf(reference):
    result = fg.generate_factor(vector=list("ABCDE"), ncols=3, nrows=10, type="balanced")
    assert_matches_reference(result.astype(str), reference("generate_data_balanced"))
    single = fg.generate_factor(vector=list("ABCDE"), ncols=1, nrows=10, type="balanced")
    assert isinstance(single, pd.Series) and single.name is None
    assert single.astype(str).tolist() == reference("generate_data_balanced_single")["x"].tolist()


@pytest.mark.parametrize("row", range(4))
def test_uniform_pools_match_rwf(reference, row):
    np.random.seed(row)
    expected = reference("generate_data_uniform_pool").iloc[row]
    data = fg.generate_data(nrows=200, ncols=5, min=expected["min"], max=expected["max"], type="uniform")
    values = sorted(np.unique(data.to_numpy()))
    assert [float(v) for v in values] == [float(v) for v in str(expected["values"]).split("|")]


def test_default_target_correlation_matrix_matches_rwf(reference):
    target = fg._default_correlation_matrix(4)
    assert_matches_reference(pd.DataFrame(target, columns=[f"V{i}" for i in range(1, 5)]),
                             reference("generate_data_default_target"))
    np.random.seed(1)
    assert fg.generate_correlation_matrix(nrows=10).shape == (10, 10)  # positive definite, as in R


def test_correlation_matrix_reproduces_target():
    np.random.seed(2)
    target = np.array([[1, .7], [.7, 1]])
    data = fg.generate_correlation_matrix(target, nrows=20000)
    np.testing.assert_allclose(data.corr().to_numpy(), target, atol=0.02)
    # the earlier, correctly spelled argument name still works
    np.random.seed(2)
    pd.testing.assert_frame_equal(fg.generate_correlation_matrix(correlation_matrix=target, nrows=20000), data)


def test_multiple_response_vector_matches_rwf_pools(reference):
    np.random.seed(3)
    mr = fg.generate_multiple_response_vector(responses=["a", "b", "c"], responded=2, length=200)
    pieces = [s.split(", ") for s in mr]
    assert sorted({p for row in pieces for p in row}) == reference("generate_data_multiple_response")["value"].tolist()
    assert sorted({len(row) for row in pieces}) == reference("generate_data_multiple_response_counts")["counts"].tolist()
    assert all(len(set(row)) == len(row) for row in pieces)  # drawn without replacement


def test_generate_missing_keeps_types():
    np.random.seed(4)
    result = fg.generate_missing(pd.DataFrame({"a": [1, 2, 3, 4], "b": list("abcd")}), missing=2)
    assert result.isna().sum().tolist() == [2, 2]
    assert set(result["b"].dropna()) <= set("abcd")
    text = fg.generate_missing(list("abcd"), missing=1)
    assert sum(pd.isna(v) for v in text) == 1
    numbers = fg.generate_missing(np.arange(10), missing=3)
    assert np.isnan(numbers).sum() == 3


def test_generate_string_shape():
    np.random.seed(5)
    strings = fg.generate_string(nchar=10, vector_length=4)
    assert len(strings) == 4 and all(len(s) == 10 and s.isalnum() for s in strings)
