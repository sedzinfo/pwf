"""
SEM with a mean structure against lavaan 0.7.3 (make_reference.R, "LAVAAN MEAN STRUCTURES"): meanstructure = TRUE
with complete data (the mean term of the ML discrepancy and of the expected information), intercepts in the
syntax (y ~ 1, y ~ 0*1, a free latent mean F ~ 1, labelled intercepts), which imply a mean structure, the fit
measures of a non-saturated mean structure and of a saturated model (df = 0), and rwf::report_cfa of such a model.
"""
import warnings

import numpy as np
import pytest

from _lavaan_checks import (check_fit_measures, check_modification_indices, check_parameters,
                            check_rsquare_inspect_scores, reference_case)
from conftest import _load_reference
from pwf import glm_sem

DATA = {"ld": "lavaan_data", "lm": "lavaan_fiml_data", "xd": "lavaan_x_data"}
CASES = {
    "cfa": dict(data="ld", model="E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", meanstructure=True),
    "sem": dict(data="ld", model="E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; "
                                 "A ~ E + N; E1 ~~ E2", meanstructure=True),
    "mimic": dict(data="xd", model="E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; E5 ~ age", meanstructure=True),
    "path": dict(data="xd", model="E1 ~ age + gender; E2 ~ E1 + age", meanstructure=True),
    "stdlv": dict(data="ld", model="E =~ E1 + E2 + E3 + E4 + E5", meanstructure=True, std_lv=True),
    "latent_mean": dict(data="ld", model="E =~ E1 + E2 + E3 + E4 + E5; E1 ~ 0*1; E ~ 1"),
    "equal_intercepts": dict(data="ld", model="E =~ E1 + E2 + E3 + E4 + E5; E2 ~ a*1; E4 ~ a*1; E5 ~ 3*1",
                             meanstructure=True),
    "intercept_only": dict(data="ld", model="E =~ E1 + E2 + E3; E1 ~ 1"),
    "fiml_latent_mean": dict(data="lm", model="E =~ E1 + E2 + E3 + E4 + E5; E1 ~ 0*1; E ~ 1", missing="ml"),
}
# with FIML, lavaan's robust ("FIML corrected") measures and the classic GFI family are not ported
# (notes/TODO.md); pwf gives them as missing
NOT_PORTED_FIML = {"cfi.robust", "tli.robust", "nnfi.robust", "rni.robust", "rmsea.robust", "rmsea.ci.lower.robust",
                   "rmsea.ci.upper.robust", "rmsea.pvalue.robust", "rmsea.notclose.pvalue.robust", "gfi.robust",
                   "gfi.ci.lower.robust", "gfi.ci.upper.robust", "gfi_lisrel", "agfi_lisrel", "pgfi"}


@pytest.fixture(scope="module")
def fits():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case, arguments in CASES.items():
            arguments = dict(arguments)
            model, data = arguments.pop("model"), _load_reference(DATA[arguments.pop("data")])
            out[case] = glm_sem.sem(model, data, **arguments)
    return out


def _case(name, case):
    return reference_case("lavaan_ms", name, case)


@pytest.mark.parametrize("case", CASES)
def test_parameters_match_lavaan(fits, case):
    check_parameters(fits[case], _case("parameters", case))


@pytest.mark.parametrize("case", CASES)
def test_fit_measures_match_lavaan(fits, case):
    check_fit_measures(fits[case], _case("fit", case), NOT_PORTED_FIML if case.startswith("fiml") else ())


@pytest.mark.parametrize("case", CASES)
def test_rsquare_inspect_and_scores_match_lavaan(fits, case):
    check_rsquare_inspect_scores(fits[case], _case("rsquare", case), _case("inspect", case), _case("predict", case))


@pytest.mark.parametrize("case", CASES)
def test_modification_indices_match_lavaan(fits, case):
    check_modification_indices(fits[case], _case("mi", case))


def test_intercepts_in_the_syntax_imply_a_mean_structure(fits):
    assert fits["latent_mean"].meanstructure and fits["intercept_only"].meanstructure
    # lavaan's parameter table: the user's rows in their order (intercepts included), then the automatic ones
    table = fits["latent_mean"].parameter_estimates()
    assert list(zip(table["lhs"], table["op"]))[5:7] == [("E1", "~1"), ("E", "~1")]
    assert table.loc[table["op"] == "~1", "lhs"].tolist() == ["E1", "E", "E2", "E3", "E4", "E5"]


def test_report_cfa_of_a_model_with_a_latent_mean_matches_rwf(fits, reference, capsys):
    result = glm_sem.report_cfa(fits["latent_mean"])
    expected = reference("lavaan_ms_report_parameters", keep_default_na=False, na_values=["NA"])
    got = result["parameters"]
    assert list(got.columns) == list(expected.columns[1:])
    assert got.index.tolist() == expected["row"].tolist()
    for column in ["est", "se", "std.lv", "std.all"]:
        np.testing.assert_allclose(got[column].to_numpy(float), expected[column].to_numpy(float), rtol=2e-4,
                                   atol=2e-5, err_msg=column)
    assert set(result["unstandardized_estimates"]) == {"lambda", "theta", "psi", "nu", "alpha"}
