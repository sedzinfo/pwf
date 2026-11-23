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
    with pytest.raises(NotImplementedError, match="higher-order"):  # observed predictors work since 23/11
        _sem.sem("E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; G =~ E + N", lavaan_data)
    with pytest.raises(NotImplementedError, match="inequality"):
        _sem.sem("E =~ E1 + a*E2 + E3; a > 0", lavaan_data)
    with pytest.raises(NotImplementedError, match="not linear"):
        _sem.sem("E =~ E1 + a*E2 + b*E3; a == b*b", lavaan_data)
    with pytest.raises(ValueError, match="not the label of a free parameter"):
        _sem.sem("E =~ E1 + a*E2 + E3; z == 1", lavaan_data)
    with pytest.raises(ValueError, match="already a label"):
        _sem.sem("E =~ E1 + a*E2 + E3; a := 2", lavaan_data)


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
# not ported for missing="ml" (pwf returns them as missing): lavaan's FIML-corrected robust indices, and the
# classic GFI family, whose weights lavaan computes differently with missing data (crmr and gfi match since 23/11)
FIML_NOT_PORTED = {"gfi_lisrel", "agfi_lisrel", "pgfi"}


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


# Same models as labels_cases in tests/reference/make_reference.R: shared labels, == constraints, := definitions
LABEL_CASES = {
    "equal": ("E =~ E1 + a*E2 + a*E3 + E4 + E5; N =~ N1 + b*N2 + b*N3 + N4 + N5", {}),
    "constraint": ("E =~ E1 + l2*E2 + l3*E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5; l2 == l3 + 0.2", {}),
    "variances": ("E =~ E1 + E2 + E3; E1 ~~ v*E1; E2 ~~ v*E2; E3 ~~ v*E3", {}),
    "combined": ("E =~ E1 + a*E2 + a*E3 + l4*E4; N =~ N1 + N2 + l3*N3 + N4; A =~ A1 + A2 + A3; def1 := l4*2; "
                 "N ~ r*E; l4 == -l3 - 0.4; A ~ s*N; ind := r*s; N2 ~~ v*N2; N4 ~~ v*N4", {}),
    "stdlv": ("E =~ a*E1 + a*E2 + a*E3 + E4 + E5", {"std_lv": True}),
    "fixed_label": ("E =~ a*E1 + a*E3 + E2 + E4 + E5", {}),  # a label shared with the fixed marker fixes E3 too
    "fiml": ("E =~ E1 + a*E2 + a*E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", {"missing": "ml"}),
}
# lavaan's optimizer stops a little earlier on constrained models (its chi-square is a hair above pwf's)
LABEL_TOL = 1e-4


@pytest.fixture(scope="module")
def label_fits(lavaan_data):
    from conftest import _load_reference
    fiml_data = _load_reference("lavaan_fiml_data")
    return {case: _sem.sem(model, fiml_data if kwargs.get("missing") == "ml" else lavaan_data, **kwargs)
            for case, (model, kwargs) in LABEL_CASES.items()}


@pytest.mark.parametrize("case", LABEL_CASES)
def test_labelled_parameter_estimates_match_lavaan(label_fits, reference, case):
    """Equal values and standard errors under shared labels and constraints, := rows with delta-method standard
    errors and standardized values, the label column and lavaan's row numbers (its hidden == rows skipped)."""
    result = label_fits[case].parameter_estimates()
    expected = reference(f"labels_{case}_parameters", keep_default_na=False, na_values=["NA"])
    assert [str(i) for i in result.index] == expected["row"].astype(str).tolist()
    assert_matches_reference(result, expected.drop(columns="row"), rtol=LABEL_TOL, atol=LABEL_TOL / 10)


@pytest.mark.parametrize("case", LABEL_CASES)
def test_labelled_fit_measures_match_lavaan(label_fits, reference, case):
    """npar counts the free parameters less the constraints, so df gains one for each."""
    result = label_fits[case].fit_measures()
    expected = reference(f"labels_{case}_fit").set_index("measure")["value"]
    for name in ["npar", "chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic", "logl"]:
        assert result[name] == pytest.approx(expected[name], rel=LABEL_TOL, abs=1e-8), name


@pytest.mark.parametrize("case", LABEL_CASES)
def test_labelled_modification_indices_match_lavaan(label_fits, reference, case):
    """Under the constraints, as lavaan's, for the same rows and with lavaan's row numbers, which count its
    rows of definitions and constraints."""
    result = label_fits[case].modification_indices()
    expected = reference(f"labels_{case}_modindices", keep_default_na=False, na_values=["NA"])
    assert sorted(str(i) for i in result.index) == sorted(expected["row"].astype(str))
    got = result.assign(row=[str(i) for i in result.index]).set_index("row")
    want = expected.assign(row=expected["row"].astype(str)).set_index("row").loc[got.index]
    assert (got[["lhs", "op", "rhs"]] == want[["lhs", "op", "rhs"]]).all().all()
    # indices are differences of the score at the optimum, so lavaan's earlier stop shows more here
    np.testing.assert_allclose(got["mi"], want["mi"], rtol=3e-3, atol=1e-3)


def test_definitions_and_constraints_hold(label_fits):
    """The fitted values keep the constraints exactly, and a := row is its expression of the estimates."""
    fit = label_fits["combined"]
    values = {p.label: p.value for p in fit.table if p.label}
    assert values["l4"] == pytest.approx(-values["l3"] - 0.4, abs=1e-12)
    loadings = {p.rhs: p.value for p in fit.table if p.op == "=~"}
    assert loadings["E2"] == loadings["E3"]
    estimates = fit.parameter_estimates().set_index("lhs")
    assert estimates.loc["ind", "est"] == pytest.approx(values["r"] * values["s"], rel=1e-12)
    assert fit.npar == len(fit.free) - 3  # two shared labels and one constraint
