import numpy as np
import pandas as pd
import pytest

from conftest import assert_matches_reference
from pwf import _sem

# Same models as lavaan_cases in tests/reference/make_reference.R
LAVAAN_CASES = {
    "cfa2": ("E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", False),
    "cfa2_stdlv": ("E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", True),
    "sem": ("E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2", False),
}
# lavaan stops its optimizer at a relative tolerance of about 1e-6 to 1e-5; pwf converges further
ESTIMATE_TOL = 5e-5


@pytest.fixture(scope="module")
def lavaan_data():
    from conftest import _load_reference
    return _load_reference("lavaan_data")


@pytest.mark.parametrize("case", LAVAAN_CASES)
def test_parameter_estimates_match_lavaan(lavaan_data, reference, case):
    model, std_lv = LAVAAN_CASES[case]
    result = _sem.sem(model, lavaan_data, std_lv=std_lv).parameter_estimates()
    expected = reference(f"lavaan_{case}_parameters")
    tol = {c: (ESTIMATE_TOL, 1e-5) for c in ["est", "se", "z", "ci.lower", "ci.upper", "std.lv", "std.all"]}
    tol["pvalue"] = (1e-3, 1e-12)
    assert_matches_reference(result, expected, column_tol=tol)


@pytest.mark.parametrize("case", LAVAAN_CASES)
def test_fit_measures_match_lavaan(lavaan_data, reference, case):
    model, std_lv = LAVAAN_CASES[case]
    result = _sem.sem(model, lavaan_data, std_lv=std_lv).fit_measures()
    expected = reference(f"lavaan_{case}_fit")
    assert list(result) == expected["measure"].tolist()
    got = np.array([result[m] for m in expected["measure"]], dtype=float)
    # chi-square p values this small move with the last digits of the optimum
    tolerance = np.where(expected["measure"] == "pvalue", 1e-3, ESTIMATE_TOL)
    for measure, g, e, t in zip(expected["measure"], got, expected["value"], tolerance):
        assert g == pytest.approx(e, rel=t, abs=1e-10), measure


def test_parse_rejects_unsupported(lavaan_data):
    with pytest.raises(NotImplementedError):
        _sem.sem("E =~ a*E1 + E2 + E3", lavaan_data)
    with pytest.raises(NotImplementedError):
        _sem.sem("E =~ E1 + E2 + E3; E ~ N1", lavaan_data)


def test_listwise_deletion(lavaan_data):
    data = lavaan_data.copy()
    data.loc[[0, 5, 9], "E2"] = np.nan
    fit = _sem.cfa("E =~ E1 + E2 + E3 + E4 + E5", data)
    assert fit.n == len(data) - 3


@pytest.mark.parametrize("case", ["cfa2", "sem"])
def test_modification_indices_match_lavaan(lavaan_data, reference, case):
    model, std_lv = LAVAAN_CASES[case]
    result = _sem.sem(model, lavaan_data, std_lv=std_lv).modification_indices()
    expected = reference(f"lavaan_{case}_modindices", keep_default_na=False, na_values=["NA"])
    assert len(result) == len(expected)
    # rows of free parameters have indices of about 0, whose order is numerical noise; compare the rest
    informative = expected["mi"] > 1e-4
    assert result["mi"].is_monotonic_decreasing
    # rows whose indices agree to 4 digits can swap places, so match rows by parameter
    keys = ["lhs", "op", "rhs"]
    want = expected[informative].sort_values(keys).reset_index(drop=True)
    got = result[result["mi"] > 1e-4].sort_values(keys).reset_index(drop=True)
    # indices are differences of the score at the optimum, so lavaan's looser convergence shows more here
    tol = {c: (1e-3, 1e-6) for c in ["mi", "epc", "sepc.all", "ncp", "power"]}
    assert_matches_reference(got, want, column_tol=tol)


@pytest.mark.parametrize("case", ["cfa2", "sem"])
def test_predict_matches_lavaan(lavaan_data, reference, case):
    model, std_lv = LAVAAN_CASES[case]
    scores = _sem.sem(model, lavaan_data, std_lv=std_lv).predict()
    np.testing.assert_allclose(scores.to_numpy(), reference(f"lavaan_{case}_predict").to_numpy(), atol=5e-5)


FIML_MODEL = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5"
# not ported for missing="ml" (pwf returns them as missing): lavaan's FIML-corrected robust indices,
# and the GFI family and CRMR, which lavaan computes from differently weighted moments with missing data
FIML_NOT_PORTED = {"crmr", "gfi", "gfi.ci.lower", "gfi.ci.upper", "gfi_lisrel", "agfi_lisrel", "pgfi"}


@pytest.fixture(scope="module")
def fiml_fit():
    from conftest import _load_reference
    return _sem.sem(FIML_MODEL, _load_reference("lavaan_fiml_data"), missing="ml")


def test_fiml_parameter_estimates(fiml_fit, reference):
    expected = reference("lavaan_fiml_parameters", keep_default_na=False, na_values=["NA"])
    assert fiml_fit.n == 299  # the respondent without any answer is dropped, as in lavaan
    tol = {c: (ESTIMATE_TOL, 1e-5) for c in ["est", "se", "z", "ci.lower", "ci.upper", "std.lv", "std.all"]}
    tol["pvalue"] = (1e-3, 1e-12)
    assert_matches_reference(fiml_fit.parameter_estimates(), expected, column_tol=tol)


def test_fiml_fit_measures(fiml_fit, reference):
    result = fiml_fit.fit_measures()
    expected = reference("lavaan_fiml_fit")
    assert list(result) == expected["measure"].tolist()
    for measure, value in zip(expected["measure"], expected["value"]):
        if "robust" in measure or measure in FIML_NOT_PORTED:
            assert np.isnan(result[measure]), measure
            continue
        assert result[measure] == pytest.approx(value, rel=1e-4, abs=1e-10), measure


def test_fiml_modification_indices(fiml_fit, reference):
    result = fiml_fit.modification_indices()
    expected = reference("lavaan_fiml_modindices", keep_default_na=False, na_values=["NA"])
    assert len(result) == len(expected)
    keys = ["lhs", "op", "rhs"]
    want = expected[expected["mi"] > 1e-4].sort_values(keys).reset_index(drop=True)
    got = result[result["mi"] > 1e-4].sort_values(keys).reset_index(drop=True)
    assert_matches_reference(got, want, column_tol={c: (1e-3, 1e-6) for c in ["mi", "epc", "sepc.all", "ncp", "power"]})


def test_fiml_predict(fiml_fit, reference):
    expected = reference("lavaan_fiml_predict").drop(index=6)  # row 7 in R has no answers
    np.testing.assert_allclose(fiml_fit.predict().to_numpy(), expected.to_numpy(), atol=5e-5)
