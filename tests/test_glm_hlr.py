import numpy as np
import pandas as pd

from conftest import assert_matches_reference
from pwf import glm_hlr


def test_report_hlr_matches_rwf(reference, tmp_path):
    sleepstudy = reference("sleepstudy")
    result = glm_hlr.report_hlr(df=sleepstudy, corlist=[1, 4], factorlist=3, predictor="Days",
                                random_effect="Subject", file=str(tmp_path / "hlr"))
    expected = reference("report_hlr_sleepstudy", keep_default_na=False, na_values=["NA"])
    # Fatigue's random slope is perfectly correlated with its intercept: nlme fails and rwf leaves the model out
    assert_matches_reference(result, expected)
    assert list(pd.read_excel(tmp_path / "hlr.xlsx", sheet_name=None)) == ["report"]
    assert np.isnan(result["L.Ratio"].iloc[0])


def test_report_hlr_zero_variances_match_rwf(reference):
    # Grouping by the predictor, as rwf's example does, puts the random-effect variances at 0 at the maximum;
    # statsmodels' mixedlm returned an infinite log-likelihood there. nlme stops just short of 0, so where pwf's
    # likelihood ratio is 0, rwf's is up to ~1.6e-7 of optimizer noise, and a 1-df p-value moves with its root.
    infert = reference("infert")
    result = glm_hlr.report_hlr(df=infert, corlist=[2, 3, 4, 6], factorlist=1, predictor="case",
                                random_effect="case")
    expected = reference("report_hlr_infert", keep_default_na=False, na_values=["NA"])
    assert_matches_reference(result.drop(columns="p.value"), expected.drop(columns="p.value"), rtol=1e-7,
                             atol=1e-6)
    noise = (expected["L.Ratio"] < 1e-6).to_numpy()
    np.testing.assert_allclose(result["p.value"][~noise], expected["p.value"][~noise], rtol=1e-7)
    assert (result["p.value"][noise] > 0.999).all() and (expected["p.value"][noise] > 0.999).all()
