import pathlib

import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def df_blood_pressure():
    return pd.read_csv(DATA_DIR / "blood_pressure.csv")
