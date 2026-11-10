import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_matrix as fx

M1 = np.arange(1, 10).reshape(3, 3, order="F")
M2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 0.5]).reshape(3, 3, order="F")
DF = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=["r1", "r2", "r3"])


def _frame(m):
    """matrix_df() of make_reference_utilities.R: row labels in a "row" column, cells as V1..Vn or by name."""
    if isinstance(m, pd.DataFrame):
        out = m.reset_index().rename(columns={"index": "row"})
        return out
    return pd.DataFrame(m, columns=[f"V{i + 1}" for i in range(m.shape[1])])


@pytest.mark.parametrize("name, kwargs", [
    ("lower", dict(m=M1)),
    ("lower_na", dict(m=M1, diagonal=np.nan, type="lower")),
    ("upper_na", dict(m=M1, diagonal=np.nan, type="upper")),
    ("upper", dict(m=M1, diagonal=None, type="upper")),
    ("frame", dict(m=DF, off_diagonal=0, diagonal=[-1, -2, -3], type="upper")),
    ("wide", dict(m=np.arange(1, 7).reshape(2, 3, order="F"))),
    ("tall", dict(m=np.arange(1, 7).reshape(3, 2, order="F"), type="upper")),
])
def test_matrix_triangle_matches_rwf(reference, name, kwargs):
    assert_matches_reference(_frame(fx.matrix_triangle(**kwargs)), reference(f"matrix_triangle_{name}"))


def test_matrix_triangle_rejects_unknown_type():
    with pytest.raises(ValueError):
        fx.matrix_triangle(M1, type="diagonal")


@pytest.mark.parametrize("name, diagonal", [
    ("upper", "upper"), ("lower", "lower"), ("na", np.nan), ("value", 1), ("values", [1, 2, 3]),
])
def test_display_upper_lower_triangle_matches_rwf(reference, name, diagonal):
    result = fx.display_upper_lower_triangle(m_upper=M1, m_lower=M2, diagonal=diagonal)
    assert_matches_reference(_frame(result), reference(f"matrix_display_{name}"))


def test_display_upper_lower_triangle_text_diagonal_makes_text(reference):
    result = fx.display_upper_lower_triangle(m_upper=M1, m_lower=M2, diagonal=["X1", "X2", "X3"])
    assert_matches_reference(_frame(result), reference("matrix_display_text", dtype=str))


def test_symmetric_matrix_matches_rwf(reference):
    m_lower = fx.matrix_triangle(M1, type="lower", diagonal=np.nan)
    m_upper = fx.matrix_triangle(np.arange(11, 20).reshape(3, 3, order="F"), type="upper", diagonal=np.nan)
    assert_matches_reference(_frame(fx.symmetric_matrix(matrix=m_lower, duplicate="lower", diagonal=np.nan)),
                             reference("matrix_symmetric_lower"))
    assert_matches_reference(_frame(fx.symmetric_matrix(matrix=m_upper, duplicate="upper", diagonal=np.nan)),
                             reference("matrix_symmetric_upper"))
    # a labelled matrix: R sets the row names to the column names
    assert_matches_reference(_frame(fx.symmetric_matrix(DF, duplicate="upper")), reference("matrix_symmetric_frame"))


def test_off_diagonal_index_matches_rwf(reference):
    assert_matches_reference(fx.off_diagonal_index(length=6), reference("matrix_off_diagonal_index"))
