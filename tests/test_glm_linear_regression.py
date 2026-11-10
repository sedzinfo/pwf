import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from conftest import _load_data  # noqa: E402
from pwf import glm_linear_regression  # noqa: E402

CASES = {"simple": "charges ~ age", "mixed": "charges ~ age * bmi + smoker + region",
         "additive": "charges ~ age + bmi + children"}


def assert_frame_matches(got, expected, label, rtol=1e-8, atol=1e-10):
    assert list(got.columns) == list(expected.columns), label
    assert len(got) == len(expected), label
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
            np.testing.assert_allclose(pd.to_numeric(got[column]).to_numpy(float), expected[column].to_numpy(float),
                                       rtol=rtol, atol=atol, err_msg=f"{label} {column}")
        else:
            assert got[column].astype(str).tolist() == expected[column].astype(str).tolist(), (label, column)


@pytest.fixture(scope="module")
def results():
    df_insurance = _load_data("insurance")
    np.random.seed(1)
    return {case: glm_linear_regression.report_regression(smf.ols(formula, data=df_insurance).fit(),
                                                          plot_diagnostics=False)
            for case, formula in CASES.items()}


@pytest.mark.parametrize("case", list(CASES))
def test_report_regression_matches_rwf(results, reference, case):
    result = results[case]
    for part in ["r", "coeficients", "deviance", "call"]:
        assert_frame_matches(result[part], reference(f"report_regression_{case}_{part}"), part)
    anova = result["anova"].rename_axis("term").reset_index()
    assert_frame_matches(anova, reference(f"report_regression_{case}_anova"), "anova")
    assert_frame_matches(result["variance_covariance"].reset_index(drop=True),
                         reference(f"report_regression_{case}_variance_covariance"), "vcov")
    outliers = result["outlier_test"].rename_axis("observation").reset_index()
    outliers["observation"] = outliers["observation"] + 1  # R numbers rows from 1
    assert_frame_matches(outliers, reference(f"report_regression_{case}_outlier_test"), "outliers")
    assert_frame_matches(result["diagnostics"].reset_index(drop=True),
                         reference(f"report_regression_{case}_diagnostics"), "diagnostics", rtol=1e-7, atol=1e-8)
    dw = reference(f"report_regression_{case}_durbin_watson")
    assert list(result["durbin_watson"].columns) == list(dw.columns)
    np.testing.assert_allclose(result["durbin_watson"][["dw.r", "dw.dw"]].to_numpy(float),
                               dw[["dw.r", "dw.dw"]].to_numpy(float), rtol=1e-10)
    assert result["durbin_watson"]["dw.alternative"].tolist() == ["two.sided"]
    # the bootstrap p value differs between R's and numpy's random draws
    assert abs(result["durbin_watson"]["dw.p"][0] - dw["dw.p"][0]) < 0.1


@pytest.mark.parametrize("case", list(CASES))
def test_report_regression_vif_matches_rwf(results, reference, case):
    vif = results[case]["vif"]
    if case == "additive":
        assert_frame_matches(vif.reset_index(drop=True), reference(f"report_regression_{case}_vif"), "vif")
    else:
        assert vif.empty


def test_report_regression_writes_files(tmp_path):
    file = str(tmp_path / "regression")
    model = smf.ols(CASES["additive"], data=_load_data("insurance")).fit()
    glm_linear_regression.report_regression(model, file=file, plot_diagnostics=False)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["r", "Coefficients", "ANOVA", "Variance Covariance", "Outliers", "Durbin Watson", "VIF",
                      "Diagnostics", "Call"]
