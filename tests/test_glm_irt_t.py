import numpy as np
import pandas as pd
import pytest

from conftest import REFERENCE_DIR, _load_reference
from pwf import glm_irt_t

# lavaan's WLSMV optimum of this weakly determined model moves by ~1e-4 with the ~1e-8
# differences of its polychoric correlations; the ULSMV fit agrees to ~1e-5
TOLERANCE = {"ULSMV": 2e-5, "WLSMV": 3e-3}


@pytest.fixture(scope="module")
def fits():
    model = (REFERENCE_DIR / "tirt_lavaan_model.txt").read_text(encoding="utf-8")
    data = _load_reference("tirt_triplets")
    return {estimator: glm_irt_t.fit_tirt(model, data, estimator=estimator) for estimator in TOLERANCE}


@pytest.mark.parametrize("estimator", list(TOLERANCE))
def test_fit_tirt_matches_lavaan(fits, reference, estimator):
    fit = fits[estimator]
    assert fit.converged
    mine = fit.parameter_estimates(standardized=False)
    expected = reference(f"tirt_{estimator.lower()}_parameters", keep_default_na=False, na_values=["NA"])
    keys = ["lhs", "op", "rhs", "label"]
    assert mine[keys].astype(str).values.tolist() == expected[keys].astype(str).values.tolist()
    tol = TOLERANCE[estimator]
    for column in ["est", "se"]:
        np.testing.assert_allclose(mine[column].to_numpy(float), expected[column].to_numpy(float),
                                   atol=tol, rtol=tol, err_msg=column)
    measures = fit.fit_measures()
    reference_fit = reference(f"tirt_{estimator.lower()}_fit").set_index("measure")["value"]
    for name in ["npar", "chisq", "df", "chisq.scaled", "pvalue.scaled", "baseline.chisq", "cfi", "tli",
                 "cfi.scaled", "rmsea", "rmsea.scaled", "srmr", "wrmr"]:
        assert measures[name] == pytest.approx(reference_fit[name], rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize("estimator", list(TOLERANCE))
def test_extract_tirt_params_matches_rwf(fits, reference, estimator):
    pars = glm_irt_t.extract_tirt_params(fits[estimator])
    tol = TOLERANCE[estimator]
    lam = reference(f"tirt_{estimator.lower()}_lambda").set_index("indicator")
    assert list(pars["lambda"].index) == list(lam.index)
    np.testing.assert_allclose(pars["lambda"].to_numpy(), lam.to_numpy(), atol=tol, rtol=tol)
    vectors = reference(f"tirt_{estimator.lower()}_vectors").set_index("indicator")
    for name in ["theta_diag", "tau", "nu"]:
        np.testing.assert_allclose(pars[name].to_numpy(float), vectors[name].to_numpy(float), atol=tol, rtol=tol)
    np.testing.assert_allclose(pars["Psi"].to_numpy(), reference(f"tirt_{estimator.lower()}_psi").to_numpy(),
                               atol=tol, rtol=tol)


@pytest.mark.parametrize("estimator", list(TOLERANCE))
def test_check_heywood_matches_rwf(fits, reference, estimator):
    result = glm_irt_t.check_heywood(fits[estimator], verbose=False)
    expected = reference(f"tirt_{estimator.lower()}_heywood").set_index("item")["value"]
    assert result["has_issues"] == bool(expected["has_issues"])
    assert result["converged"] == bool(expected["converged"])
    assert result["issues"] == {}


def test_check_heywood_flags_problems(fits):
    fit = fits["ULSMV"]
    row = next(r for r in fit.table if r["op"] == "~~" and r["lhs"] == "trait1" and r["rhs"] == "trait2")
    original = row["value"]
    try:
        fit.converged = False
        result = glm_irt_t.check_heywood(fit, verbose=False)
    finally:
        fit.converged = True
        row["value"] = original
    assert result["has_issues"] and "convergence" in result["issues"]
