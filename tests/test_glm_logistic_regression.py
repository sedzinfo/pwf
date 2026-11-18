import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from conftest import _load_reference, assert_autoplot_matches, assert_matches_reference  # noqa: E402
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


def _report_case(case):
    formula, family, validation = CASES[case]
    infert_data = _load_reference("infert")
    model = smf.glm(formula, data=infert_data, family=FAMILIES[family]).fit()
    return glm_logistic_regression.report_logistic(model, validation_data=infert_data if validation else None)


@pytest.fixture(scope="module")
def results():
    return {case: _report_case(case) for case in CASES}


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


# In "case ~ education" the 0-5yrs and 6-11yrs groups both have a fitted probability of exactly 1/3 (4/12 and
# 40/120). The cut search starts at the smallest prediction, so which of the two groups is classed 0 depends on
# the last bit of their fitted values, which differs between platforms: rwf (R 4.6.2) classes the 6-11yrs group
# 0 on Windows (the reference, 80/40/120) and the 0-5yrs group 0 on Linux (8/4/12), as pwf does there. rwf's
# fitted values, saved with 15 digits, keep its order (...379 < ...380), so this case's confusion results are
# computed from them.
TIED_FITTED = {"education"}


@pytest.mark.parametrize("case", list(CASES))
def test_report_logistic_confusion_matches_rwf(results, reference, case, monkeypatch):
    result = results[case]
    if case in TIED_FITTED:
        r_fitted = reference(f"report_logistic_{case}_scores")["fitted"].to_numpy(float)
        r_glm = glm_logistic_regression._r_glm
        monkeypatch.setattr(glm_logistic_regression, "_r_glm",
                            lambda model, **kwargs: {**r_glm(model, **kwargs), "fitted": r_fitted})
        result = _report_case(case)
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


def test_output_compare_model_logistic_matches_rwf(reference):
    infert = reference("infert")
    null = smf.glm("case ~ 1", data=infert, family=sm.families.Binomial()).fit()
    full = smf.glm("case ~ age + parity + spontaneous", data=infert, family=sm.families.Binomial()).fit()
    result = glm_logistic_regression.output_compare_model_logistic(null, full)
    # p is 1 - pchisq(), as in rwf, so it carries the cancellation error of a value near 1
    assert_matches_reference(result, reference("output_compare_model_logistic"), rtol=1e-6)


def test_compute_y_logistic_matches_rwf(reference):
    expected = reference("compute_y_logistic")
    result = glm_logistic_regression.compute_y_logistic(intercept=0.5, coefficient=-1.2, x=expected["x"])
    np.testing.assert_allclose(result, expected["y"], rtol=1e-14)


def test_plot_logistic_model_matches_rwf(reference):
    """rwf's title argument, melt()'s predictor order and the binomial smooth of a predictor without separation."""
    data = pd.DataFrame({"outcome": [1] * 10 + [0] * 10, "z2": [1] * 11 + [0] * 9, "b1": [1] * 9 + [0] * 11,
                         "a3": [5.2, 4.1, 6.3, 3.9, 5.5, 7.1, 4.8, 6.0, 5.9, 3.5, 8.2, 9.1, 7.7, 10.4, 8.8, 9.9, 7.2,
                                11.0, 8.5, 6.6]})
    p = glm_logistic_regression.plot_logistic_model(data, title="T")
    labels = reference("plot_logistic_model_labels")
    assert p.labels.get("title", None) == labels["title"].iloc[0]
    assert p.labels.get("caption", None) == labels["caption"].iloc[0]
    assert ",".join(p.data["Predictor"].cat.categories) == labels["predictors"].iloc[0]
    p.draw()
    smooth = p.layers[0].data
    smooth = smooth[smooth["group"] == 3].sort_values("x")
    expected = reference("plot_logistic_model_smooth_a3")
    np.testing.assert_allclose(smooth["x"], expected["x"], rtol=1e-12)
    np.testing.assert_allclose(smooth["y"], expected["y"], rtol=1e-6)  # statsmodels' IRLS stops near R's glm


def test_report_logistic_diagnostics_match_rwf(reference, monkeypatch, capsys):
    """autoplot() of the glm: panels 1, 2, 3 and 5 from the linear predictor, deviance and standardized Pearson
    residuals (R 4.6's rstandard) and the hat values of R's last IRLS iteration; labs(title = title) on each."""
    drawn = []
    monkeypatch.setattr(glm_logistic_regression, "report_pdf", lambda *plots, **kwargs: drawn.extend(plots))
    model = smf.glm("case ~ age + parity + spontaneous", data=_load_reference("infert"),
                    family=sm.families.Binomial()).fit()
    glm_logistic_regression.report_logistic(model, title="T")
    assert_autoplot_matches(drawn[0], reference("report_logistic_diagnostics_layers", dtype={"label": str}),
                            reference("report_logistic_diagnostics_labels"), rtol=1e-7, atol=1e-9)
