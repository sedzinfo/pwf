import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import _load_reference, assert_matches_reference  # noqa: E402
from pwf import glm_sem  # noqa: E402

SEM_MODEL = "E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2"


@pytest.fixture(scope="module")
def fit():
    return glm_sem.sem(SEM_MODEL, _load_reference("lavaan_data"))


@pytest.mark.parametrize("what", ["est", "std"])
def test_inspect_matches_lavaan(fit, reference, what):
    matrices = fit.inspect(what)
    for name in ["lambda", "theta", "psi", "beta"]:
        expected = reference(f"lavaan_sem_inspect_{what}_{name}")
        np.testing.assert_allclose(matrices[name].to_numpy(), expected.to_numpy(), atol=5e-5, err_msg=name)
        assert list(matrices[name].columns) == list(expected.columns)


def test_report_cfa(fit, reference, tmp_path):
    file = str(tmp_path / "cfa")
    result = glm_sem.report_cfa(fit, file=file)
    assert set(result) == {"r_squared", "fit_indices", "parameters", "modification_indices", "sample_covariance",
                           "unstandardized_estimates", "standardized_estimates", "group", "predict", "call"}
    fit_expected = reference("lavaan_sem_fit")
    # lavaan writes chi-square p values below about 1e-20 as 0
    np.testing.assert_allclose(result["fit_indices"].loc[fit_expected["measure"], "fit"], fit_expected["value"],
                               rtol=1e-3, atol=1e-20)
    parameters = reference("lavaan_sem_parameters")
    r2 = parameters[parameters["op"] == "r2"].set_index("lhs")["est"]
    np.testing.assert_allclose(result["r_squared"].loc[r2.index, "r_squared"], r2, rtol=1e-4)
    assert list(result["predict"].columns[-3:]) == ["E", "N", "A"]
    assert result["predict"].shape[0] == fit.n
    for suffix in ["_diagram.pdf", ".log", ".xlsx"]:
        assert (tmp_path / f"cfa{suffix}").exists()
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["R_Squared", "Fit_Indices", "Parameters", "Modification_Indices", "Groups", "Sample_Covariance",
                      "Scores", "Call"]


def test_plot_cfa(fit):
    plots = glm_sem.plot_cfa(fit)
    assert list(plots) == [f"{lay}_{what}" for lay in ["circle", "tree", "spring"]
                           for what in ["estimates", "standard_estimates", "parameters_wih_equality_constraints"]]


@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened")  # one plot per fit index, as in rwf
def test_simulate_cfa_fit():
    sim_results, plots = glm_sem.simulate_cfa_fit(model_sim="F =~ 1*X1 + 0.8*X2 + 0.6*X3 + 0.9*X4",
                                                  model="F =~ X1 + X2 + X3 + X4", minnobs=100, maxnobs=300,
                                                  stepping=100)
    assert sim_results["observations"].tolist() == [100, 200, 300]
    assert {"chisq", "cfi", "rmsea", "srmr"} <= set(sim_results.columns)
    assert len(plots) > 0


def test_simulate_data_recovers_model():
    data = glm_sem.simulate_data("F =~ 1*X1 + 0.8*X2 + 0.6*X3 + 0.9*X4", sample_nobs=20000, seed=1)
    loadings = glm_sem.cfa("F =~ X1 + X2 + X3 + X4", data).parameter_estimates()
    loadings = loadings[loadings["op"] == "=~"].set_index("rhs")["est"]
    np.testing.assert_allclose(loadings, [1, 0.8, 0.6, 0.9], atol=0.05)


def test_report_cfa_rejects_other_models():
    with pytest.raises(TypeError):
        glm_sem.report_cfa(object())
