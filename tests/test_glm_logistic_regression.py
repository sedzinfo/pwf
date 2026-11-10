import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from conftest import _load_reference  # noqa: E402
from pwf import glm_logistic_regression  # noqa: E402

# a cut search over few distinct predictions draws one-point lines; R's ggplot warns the same way
pytestmark = pytest.mark.filterwarnings("ignore:geom_path. Each group consist of only one observation")

FAMILIES = {"binomial": sm.families.Binomial(), "poisson": sm.families.Poisson(),
            "gaussian": sm.families.Gaussian()}
# Same cases as logistic_cases in tests/reference/make_reference.R
CASES = {
    "education": ("case ~ education", "binomial", False),
    "numeric": ("case ~ age + parity + spontaneous", "binomial", False),
    "mixed": ("case ~ education + age", "binomial", False),
    "validation": ("case ~ age + parity + spontaneous", "binomial", True),
    "poisson": ("spontaneous ~ age + parity", "poisson", False),
    "gaussian": ("age ~ parity + induced", "gaussian", False),
}


def assert_frame_matches(got, expected, label, rtol=1e-7, atol=1e-9):
    assert list(got.columns) == list(expected.columns), label
    assert len(got) == len(expected), label
    for column in expected.columns:
        values = expected[column]
        if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
            assert got[column].astype(str).tolist() == values.astype(str).tolist(), (label, column)
        else:
            np.testing.assert_allclose(pd.to_numeric(got[column]).to_numpy(float), values.to_numpy(float),
                                       rtol=rtol, atol=atol, err_msg=f"{label} {column}")


@pytest.fixture(scope="module")
def results():
    infert_data = _load_reference("infert")
    out = {}
    for case, (formula, family, validation) in CASES.items():
        model = smf.glm(formula, data=infert_data, family=FAMILIES[family]).fit()
        out[case] = glm_logistic_regression.report_logistic(model, validation_data=infert_data if validation else None)
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_report_logistic_matches_rwf(results, reference, case):
    result = results[case]
    for part in ["result_R2_logistic", "result_X2_logistic", "logistic_output", "model_call"]:
        assert_frame_matches(result[part], reference(f"report_logistic_{case}_{part}", keep_default_na=False), part)
    coefficients = result["coefficients"].rename_axis("term").reset_index()
    assert_frame_matches(coefficients, reference(f"report_logistic_{case}_coefficients"), "coefficients")
    assert_frame_matches(result["scores"].reset_index(drop=True), reference(f"report_logistic_{case}_scores"),
                         "scores", rtol=1e-6, atol=1e-8)
    descriptives = reference(f"report_logistic_{case}_score_descriptives")
    assert_frame_matches(result["score_descriptives"], descriptives, "score_descriptives", rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("case", list(CASES))
def test_report_logistic_vif_matches_rwf(results, reference, case):
    expected = reference(f"report_logistic_{case}_VIF")
    vif = results[case]["VIF"]
    if expected.shape[1] <= 1:
        assert vif.empty
    else:
        assert_frame_matches(vif.rename_axis("term").reset_index(), expected, "VIF")


@pytest.mark.parametrize("case", list(CASES))
def test_report_logistic_confusion_matches_rwf(results, reference, case):
    result = results[case]
    cmatrix = reference(f"report_logistic_{case}_cmatrix", dtype=str, keep_default_na=False)
    assert result["cmatrix"].columns.tolist() == cmatrix.columns.tolist()
    assert result["cmatrix"].to_numpy().tolist() == cmatrix.to_numpy().tolist()
    assert_frame_matches(result["confusion_performance"]["cut_performance"],
                         reference(f"report_logistic_{case}_cut_performance"), "cut_performance", rtol=1e-9)


def test_report_logistic_writes_files(tmp_path):
    infert_data = _load_reference("infert")
    model = smf.glm("case ~ age + parity", data=infert_data, family=sm.families.Binomial()).fit()
    file = str(tmp_path / "logistic")
    glm_logistic_regression.report_logistic(model, file=file)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["summary", "coefficients", "Confusion Matrix", "confusion matrix performance", "X^2",
                      "pseudo R^2", "scores residuals", "score residual descriptives", "VIF", "Call"]
