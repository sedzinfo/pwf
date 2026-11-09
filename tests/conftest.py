import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REFERENCE_DIR = pathlib.Path(__file__).resolve().parent / "reference"


def _load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def _load_reference(name):
    path = REFERENCE_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"no R reference {path.name}; run tests/reference/make_reference.R")
    return pd.read_csv(path)


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
