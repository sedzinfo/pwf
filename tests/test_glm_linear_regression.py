import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from conftest import _load_data, _load_reference, assert_autoplot_matches  # noqa: E402
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


MORE_CASES = {"poly": "charges ~ age + I(age**2) + sex", "log": "np.log(charges) ~ age + smoker * sex",
              "noint": "charges ~ 0 + age + bmi", "na": "charges ~ bmi + children + smoker"}


@pytest.mark.parametrize("case", list(MORE_CASES))
def test_report_regression_more_models_match_rwf(reference, case):
    # R's spelling of patsy terms (I(age^2), log(charges)); car::vif fails without an intercept or when a
    # predictor spans terms (age in age and I(age^2)), so rwf has no VIF table there
    data = _load_data("insurance")
    if case == "na":
        data.loc[[4, 49, 499], "bmi"] = np.nan
        data.loc[[6, 69], "children"] = np.nan
    result = glm_linear_regression.report_regression(smf.ols(MORE_CASES[case], data=data).fit(),
                                                     plot_diagnostics=False)
    for part in ["r", "coeficients", "deviance"]:
        assert_frame_matches(result[part], reference(f"report_regression_{case}_{part}"), part)
    anova = result["anova"].rename_axis("term").reset_index()
    assert_frame_matches(anova, reference(f"report_regression_{case}_anova"), "anova")
    vif = reference(f"report_regression_{case}_vif")
    if vif.shape[1] == 0 or (vif.shape == (0, 1)):
        assert result["vif"].empty
    else:
        assert_frame_matches(result["vif"].reset_index(drop=True), vif, "vif")


def test_report_regression_diagnostics_match_rwf(reference, monkeypatch, capsys):
    """autoplot(model, which = 1:6, ncol = 2) + labs(title = title, ...): rwf's default title "" blanks the six titles."""
    drawn = []
    monkeypatch.setattr(glm_linear_regression, "report_pdf", lambda *plots, **kwargs: drawn.extend(plots))
    model = smf.ols("age ~ parity + induced", data=_load_reference("infert")).fit()
    glm_linear_regression.report_regression(model)
    assert_autoplot_matches(drawn[0], reference("report_regression_diagnostics_layers", dtype={"label": str}),
                            reference("report_regression_diagnostics_labels", keep_default_na=False, na_values=[]))


def test_plot_scatterplot_matches_rwf(reference):
    """rwf's plot names, its captions (pairwise n, r, explained variance, line and angle for y ~ x; n alone for
    another formula) and no plot for a pair with fewer than 2 complete rows."""
    data = _load_data("insurance")
    data.loc[[1, 4], "bmi"] = np.nan  # make_reference.R's plots_insurance
    data.loc[6, "charges"] = np.nan
    plots = {"default": glm_linear_regression.plot_scatterplot(data[["age", "bmi", "children"]], title="T"),
             "poly": glm_linear_regression.plot_scatterplot(data[["age", "bmi"]], formula="y ~ poly(x, 2)",
                                                            str_aes=False),
             "too_few": glm_linear_regression.plot_scatterplot(pd.DataFrame({"a": [1, np.nan, 3],
                                                                             "b": [np.nan, 2, np.nan]}))}
    expected = reference("plot_scatterplot_captions")
    for case, result in plots.items():
        rows = expected[expected["case"] == case]
        assert list(result) == rows["name"].tolist(), case
        for (_, row), (name, g) in zip(rows.iterrows(), result.items()):
            assert (g is not None) == bool(row["drawn"]), name
            if g is not None:
                assert [t.get_text() for t in g.ax_joint.texts] == [row["caption"]], name
    labels = {name: (g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()) for name, g in plots["default"].items()}
    assert labels["age_bmi"] == ("Age", "Bmi") and plots["poly"]["age_bmi"].ax_joint.get_xlabel() == "age"
