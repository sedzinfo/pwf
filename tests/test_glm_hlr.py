import numpy as np
import pandas as pd

from conftest import assert_matches_reference
from pwf import glm_hlr


def test_report_hlr_matches_rwf(reference, tmp_path):
    sleepstudy = reference("sleepstudy")
    result = glm_hlr.report_hlr(df=sleepstudy, corlist=[1, 4], factorlist=3, predictor="Days",
                                random_effect="Subject", file=str(tmp_path / "hlr"))
    expected = reference("report_hlr_sleepstudy", keep_default_na=False, na_values=["NA"])
    # statsmodels' optimizer and nlme's stop at slightly different points (log-likelihoods agree to ~1e-8)
    assert_matches_reference(result, expected, rtol=1e-7, atol=1e-6)
    assert list(pd.read_excel(tmp_path / "hlr.xlsx", sheet_name=None)) == ["report"]
    assert np.isnan(result["L.Ratio"].iloc[0])
