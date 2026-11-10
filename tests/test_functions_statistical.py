import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import functions_statistical as stats

VECTOR = np.array([3, 7, 1, 7, np.nan, 12, 5, 9, 2, 7, 15, 4])
TYPES = ["z", "uz", "sten", "t", "stanine", "center", "center_reversed", "percent", "percentile", "scale_zero_one",
         "normal_density", "cumulative_density"]


def test_compute_standard_matches_rwf(reference):
    result = pd.DataFrame({t: stats.compute_standard(VECTOR, mean=2, sd=3, type=t) for t in TYPES})
    assert_matches_reference(result, reference("compute_standard_types"), rtol=1e-12)
    assert_matches_reference(stats.compute_standard(VECTOR, type="all"), reference("compute_standard_all"),
                             rtol=1e-12)


def test_statistical_helpers_match_rwf(reference):
    noise = reference("dissatenuation_data")
    expected = reference("statistical_helpers").iloc[0]
    adjustment = stats.compute_adjustment(0.05, 7)
    got = {"dissatenuation": stats.compute_dissatenuation(noise["v1"], noise["e1"], noise["v2"], noise["e2"]),
           "skewness": stats.compute_skewness(VECTOR), "kurtosis": stats.compute_kurtosis(VECTOR),
           "standard_error": stats.compute_standard_error(VECTOR),
           "confidence_interval": stats.compute_confidence_inteval(VECTOR),
           "sidak": adjustment["sidak"], "bonferroni": adjustment["bonferroni"]}
    for name, value in got.items():
        assert value == pytest.approx(expected[name], rel=1e-12), name
