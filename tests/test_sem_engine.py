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
    tol = {c: (ESTIMATE_TOL, 1e-8) for c in ["est", "se", "z", "ci.lower", "ci.upper", "std.lv", "std.all"]}
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
