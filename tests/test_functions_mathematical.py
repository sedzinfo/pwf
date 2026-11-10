import math

import numpy as np
import pandas as pd

from conftest import assert_matches_reference
from pwf import functions_mathematical as fm


def test_angle_conversions_match_rwf(reference):
    expected = reference("mathematical_angles")
    x = expected["x"].to_numpy()
    result = pd.DataFrame({"x": x, "rad2deg": fm.rad2deg(x), "deg2rad": fm.deg2rad(list(x))})
    assert_matches_reference(result, expected, rtol=1e-14)


def test_angle_conversions_of_numbers():
    assert fm.rad2deg(math.pi) == 180.0
    assert fm.deg2rad(180) == math.pi
    assert isinstance(fm.rad2deg(np.float64(1.0)), float)
