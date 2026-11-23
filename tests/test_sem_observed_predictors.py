"""
SEM with observed predictors against lavaan 0.7.3 (make_reference.R, "LAVAAN OBSERVED PREDICTORS"): fixed.x,
lavaan's dummy latent variables in its matrices, auto.cov.y, the conditional likelihood (logl, information,
modification indices), lavPredict's factor scores, and rwf::report_cfa and plot_cfa_gg of such a model.
"""
import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from _lavaan_checks import (check_fit_measures, check_modification_indices, check_parameters,  # noqa: E402
                            check_rsquare_inspect_scores, reference_case)
from conftest import _load_reference  # noqa: E402
from pwf import glm_sem  # noqa: E402
from test_glm_sem import _assert_cfa_layers_match  # noqa: E402

CASES = {
    "mimic": dict(model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender"),
    "mimic_direct": dict(model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; E5 ~ age"),
    "two_outcomes": dict(model="E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; E ~ age; N ~ age + gender"),
    "mediation": dict(model="E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; N ~ E + age; E ~ gender"),
    "path": dict(model="E1 ~ age + gender; E2 ~ E1 + age"),
    "ov_outcome": dict(model="E =~ E2 + E3 + E4; E1 ~ E + age"),
    "ov_mediator": dict(model="E =~ E1 + E2 + E3 + E4 + E5; N1 ~ age; E ~ N1"),
    "random_x": dict(model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender", fixed_x=False),
    "user_x": dict(model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; age ~~ gender"),
    "fiml": dict(model="E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3; E ~ age + gender; N ~ E", missing="ml"),
    "stdlv": dict(model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender", std_lv=True),
    "labels": dict(model="E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; N ~ b*E + age; E ~ a*gender; ab := a*b"),
}
# with FIML, lavaan's robust ("FIML corrected") measures and the classic GFI family are not ported
# (notes/TODO.md); pwf gives them as missing
NOT_PORTED_FIML = {"cfi.robust", "tli.robust", "nnfi.robust", "rni.robust", "rmsea.robust", "rmsea.ci.lower.robust",
                   "rmsea.ci.upper.robust", "rmsea.pvalue.robust", "rmsea.notclose.pvalue.robust", "gfi.robust",
                   "gfi.ci.lower.robust", "gfi.ci.upper.robust", "gfi_lisrel", "agfi_lisrel", "pgfi"}


@pytest.fixture(scope="module")
def fits():
    data = {"complete": _load_reference("lavaan_x_data"), "missing": _load_reference("lavaan_x_missing_data")}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case, arguments in CASES.items():
            arguments = dict(arguments)
            model = arguments.pop("model")
            out[case] = glm_sem.sem(model, data["missing" if arguments.get("missing") == "ml" else "complete"],
                                    **arguments)
    return out


def _case(name, case):
    return reference_case("lavaan_x", name, case)


@pytest.mark.parametrize("case", CASES)
def test_parameters_match_lavaan(fits, case):
    check_parameters(fits[case], _case("parameters", case))


@pytest.mark.parametrize("case", CASES)
def test_fit_measures_match_lavaan(fits, case):
    check_fit_measures(fits[case], _case("fit", case), NOT_PORTED_FIML if case == "fiml" else ())


@pytest.mark.parametrize("case", CASES)
def test_rsquare_inspect_and_scores_match_lavaan(fits, case):
    check_rsquare_inspect_scores(fits[case], _case("rsquare", case), _case("inspect", case), _case("predict", case))


@pytest.mark.parametrize("case", CASES)
def test_modification_indices_match_lavaan(fits, case):
    check_modification_indices(fits[case], _case("mi", case))


def test_lavaans_warnings():
    data = _load_reference("lavaan_x_data")
    with pytest.warns(UserWarning, match=r"exogenous variable\(s\): \[age gender\]; these variables will now be "
                                         "treated as random"):
        glm_sem.sem(CASES["user_x"]["model"], data)
    with pytest.warns(UserWarning, match="20 cases were deleted due to missing values in exogenous variable"):
        glm_sem.sem(CASES["fiml"]["model"], _load_reference("lavaan_x_missing_data"), missing="ml")


def test_report_cfa_of_a_model_with_predictors_matches_rwf(fits, reference, capsys):
    result = glm_sem.report_cfa(fits["mimic_direct"])
    expected = reference("lavaan_x_report_parameters", keep_default_na=False, na_values=["NA"])
    got = result["parameters"]
    assert list(got.columns) == list(expected.columns[1:])  # rwf's columns: exo after rhs, std.nox last
    assert got.index.tolist() == expected["row"].tolist()
    assert got["exo"].tolist() == expected["exo"].tolist()
    for column in ["est", "se", "std.lv", "std.all", "std.nox"]:
        np.testing.assert_allclose(got[column].to_numpy(float), expected[column].to_numpy(float), rtol=2e-4,
                                   atol=2e-5, err_msg=column)
    r2 = reference("lavaan_x_report_r_squared")
    assert result["r_squared"].index.tolist() == r2["name"].tolist()
    np.testing.assert_allclose(result["r_squared"]["r_squared"], r2["r_squared"], rtol=2e-4)


@pytest.mark.parametrize("layout,what", [("tree", "std"), ("tree", "est"), ("circle", "std"), ("circle", "est")])
def test_plot_cfa_gg_of_a_model_with_predictors_matches_rwf(fits, reference, layout, what):
    expected = reference(f"lavaan_x_plot_cfa_gg_{layout}_{what}", keep_default_na=False, na_values=["NA"])
    p = glm_sem.plot_cfa_gg(fits["mimic_direct"], what=what, layout=layout)
    assert p.labels.get("title", None) == expected["title"].iloc[0]
    _assert_cfa_layers_match(p, expected)


def test_plot_cfa_keys_of_a_model_with_predictors_match_rwf(fits, reference):
    with contextlib.redirect_stdout(io.StringIO()):
        keys = list(glm_sem.plot_cfa(fits["mimic_direct"]))
    assert keys == reference("lavaan_x_plot_cfa_keys")["key"].tolist()
