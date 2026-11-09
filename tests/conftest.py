import pathlib

import numpy as np
import pandas as pd
import pytest


def assert_matches_reference(result, expected, rtol=1e-8, atol=0.0, column_tol=None):
    """
    Assert a pwf result equals an rwf reference: same column names in the same
    order, same row count, equal text columns, and numeric columns within
    tolerance. column_tol maps column name -> (rtol, atol) overrides.

    R writes logicals as TRUE/FALSE, which pandas reads back as bools; they are
    compared as the strings "TRUE"/"FALSE", R's as.character() spelling.
    """
    column_tol = column_tol or {}
    assert list(result.columns) == list(expected.columns)
    assert len(result) == len(expected)
    for col in expected.columns:
        exp, res = expected[col], result[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(exp):
            exp = exp.map({True: "TRUE", False: "FALSE"})
        if pd.api.types.is_bool_dtype(res):
            res = res.map({True: "TRUE", False: "FALSE"})
        if pd.api.types.is_numeric_dtype(exp):
            col_rtol, col_atol = column_tol.get(col, (rtol, atol))
            np.testing.assert_allclose(res.to_numpy(dtype=float), exp.to_numpy(dtype=float),
                                       rtol=col_rtol, atol=col_atol, err_msg=col)
        else:
            assert res.astype(str).tolist() == exp.astype(str).tolist(), col

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REFERENCE_DIR = pathlib.Path(__file__).resolve().parent / "reference"


def _load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def _load_reference(name, **read_csv_kwargs):
    path = REFERENCE_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"no R reference {path.name}; run tests/reference/make_reference.R")
    return pd.read_csv(path, **read_csv_kwargs)


@pytest.fixture
def load_data():
    """Load a bundled dataset by name, e.g. load_data("co2")."""
    return _load_data


@pytest.fixture
def reference():
    """Load an rwf reference output by name (see tests/reference/make_reference.R)."""
    return _load_reference


@pytest.fixture
def df_blood_pressure():
    return _load_data("blood_pressure")
