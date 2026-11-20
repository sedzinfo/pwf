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


def _ulsmv_parameters():
    lam = _load_reference("tirt_ulsmv_lambda").set_index("indicator")
    vectors = _load_reference("tirt_ulsmv_vectors")
    return lam, vectors["theta_diag"].to_numpy(), vectors["tau"].to_numpy(), _load_reference("tirt_ulsmv_psi").to_numpy(), \
        vectors["nu"].to_numpy()


def test_score_tirt_matches_rwf(reference):
    # MAP scores from R's optim(method = "BFGS") itself (_rcompat.r_optim_bfgs): scipy's BFGS stopped ~7e-6 away
    lam, theta_diag, tau, psi, nu = _ulsmv_parameters()
    patterns = reference("tirt_triplets").iloc[:25][list(lam.index)].astype(float)
    patterns.iloc[2, 1] = np.nan
    patterns.iloc[4, [0, 6]] = np.nan
    result = glm_irt_t.score_tirt(patterns, lam, theta_diag, tau, psi, nu)
    expected = reference("score_tirt")
    assert list(result.columns) == list(expected.columns)
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), atol=1e-7)


def test_r_optim_bfgs_follows_r():
    # ?optim's Rosenbrock example: R returns this par, value and counts (function 110, gradient 43)
    from pwf._rcompat import r_optim_bfgs

    def fr(x):
        return 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2

    def grr(x):
        return np.array([-400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]), 200 * (x[1] - x[0] * x[0])])

    result = r_optim_bfgs(fr, grr, [-1.2, 1])
    np.testing.assert_allclose(result["par"], [0.99999999690491403, 0.999999993797419], rtol=1e-12)
    assert result["counts"] == (110, 43) and result["convergence"] == 0


def test_tirt_data_helpers_match_rwf(reference):
    ranks = pd.DataFrame({"a": [1, 3, 2, 2, 1, 3], "b": [2, 1, 3, 1, 3, 2], "c": [3, 2, 1, 3, 2, 1],
                          "d": [2, 3, 1, 1, 2, 3], "e": [1, 2, 3, 3, 1, 2], "f": [3, 1, 2, 2, 3, 1]})
    binary = glm_irt_t.rank_to_binary(ranks.iloc[:, :3], 3)
    np.testing.assert_array_equal(np.asarray(binary), reference("rank_to_binary").to_numpy())
    both = glm_irt_t.rank_df_to_binary(ranks, 3)  # R's data.frame() names the second block i12.1, ...
    expected = reference("rank_df_to_binary")
    assert list(both.columns) == list(expected.columns)
    np.testing.assert_array_equal(both.to_numpy(), expected.to_numpy())
    triplets = glm_irt_t.rank3_to_triplets(pd.DataFrame(np.asarray(binary)))
    np.testing.assert_array_equal(np.asarray(triplets, dtype=float), reference("rank3_to_triplets").to_numpy(float))
    np.testing.assert_array_equal(glm_irt_t.generate_matrix_A(2, 4), reference("generate_matrix_A").to_numpy())
    np.testing.assert_array_equal(glm_irt_t.generate_matrix_lambda_hat(2, 3),
                                  reference("generate_matrix_lambda_hat").to_numpy())
    assert list(glm_irt_t.name_triplet_pairs(7, strict=False)) == reference("name_triplet_pairs")["pair"].tolist()


def test_compute_ability_and_solve_match_rwf(reference):
    eta = np.round(np.arange(-3, 3.0001, 0.25), 10)
    ability = glm_irt_t.compute_ability([1, 0, 1], eta, gamma=[0.2, -0.3, 0.1], lambda_=[1, 1.5, 0.8],
                                        psi=[0.5, 0.7, 0.6])
    expected = reference("compute_ability_icc")
    np.testing.assert_allclose(ability["icc"][list(expected.columns)].to_numpy(float), expected.to_numpy(float),
                               rtol=1e-12)
    estimates = reference("compute_ability")
    assert float(np.atleast_1d(ability["ability_ml"])[0]) == pytest.approx(estimates["ability_ml"][0])
    assert float(np.atleast_1d(ability["ability_map"])[0]) == pytest.approx(estimates["ability_map"][0])
    solved = glm_irt_t.compute_solve(np.array([[4, 1, 2], [1, 3, 0.5], [2, 0.5, 5]]),
                                     np.array([[1, 4], [2, 5], [3, 6]]))
    np.testing.assert_allclose(solved, reference("compute_solve").to_numpy(), rtol=1e-12)


def test_plot_icc_thurstonian_keeps_item_order(reference):
    """reshape2::melt keeps the columns' order (item1, item2, ..., item10), not text order."""
    icc = glm_irt_t.compute_icc_thurstonian(eta=np.arange(-3, 4), gamma=[0.5] * 12, lambda_=[1] * 12, psi=[1] * 12)["icc"]
    p = glm_irt_t.plot_icc_thurstonian(icc)
    assert list(p.data["variable"].cat.categories) == reference("plot_icc_thurstonian_levels")["level"].tolist()


def test_comparison_indices_match_rwf(reference):
    """R's row names, its 2:1 loop for one block (three rows) and its 1:0 loop for one item (an error)."""
    counts = reference("compute_dummy_comparisons")
    np.testing.assert_array_equal(glm_irt_t.compute_dummy_comparisons(counts["items"].tolist()), counts["comparisons"])
    for items, rows in reference("generate_unique_comparisons_index").groupby("items", sort=False):
        result = glm_irt_t.generate_unique_comparisons_index(items)
        assert list(result.columns) == ["i1", "i2"] and list(result.index) == rows["row"].tolist()
        np.testing.assert_array_equal(result.to_numpy(), rows[["i1", "i2"]].to_numpy())
    assert glm_irt_t.generate_unique_comparisons_index(1).shape == (0, 2)
    np.testing.assert_array_equal(glm_irt_t.generate_comparisons_matrix(4),
                                  reference("generate_comparisons_matrix").to_numpy())
    for (blocks, items), cells in reference("increase_index").groupby(["blocks", "items"], sort=False):
        result = glm_irt_t.increase_index(blocks, items)
        assert result.shape == (cells["row"].max(), cells["col"].max()), (blocks, items)
        np.testing.assert_array_equal(result[cells["row"] - 1, cells["col"] - 1], cells["value"])
    messages = reference("tirt_helper_conditions").set_index("call")["message"]
    with pytest.raises(IndexError, match=messages["generate_comparisons_matrix(1)"]):
        glm_irt_t.generate_comparisons_matrix(1)
    with pytest.raises(IndexError, match=messages["rank_to_binary(one item)"]):
        glm_irt_t.rank_to_binary(pd.DataFrame({"a": [1, 2, 3]}), 1)
    with pytest.raises(ValueError, match=messages["increase_index(2, 0)"]):
        glm_irt_t.increase_index(2, 0)


def test_response_dimension_and_cfa_icc_index_match_rwf(reference):
    """rwf shifts the responses by a block per pick (NA past the end); one factor gives twice the indices."""
    cases = [(range(1, 19), 3, [1, 2]), (range(1, 13), 3, [2, 3]), (range(1, 9), 3, [1, 3]), (range(1, 3), 3, [1, 2]),
             (range(1, 10), 3, [1, 4]), ([0.5, 2, 3.5, 4, 5, 6], 2, 2)]
    expected = reference("response_dimension")
    for case, args in enumerate(cases, start=1):
        np.testing.assert_array_equal(np.asarray(glm_irt_t.response_dimension(*args), dtype=float),
                                      expected.loc[expected["case"] == case, "value"].to_numpy(float), err_msg=str(case))
    for (nitems, nfactors), cells in reference("cfa_icc_index").groupby(["nitems", "nfactors"], sort=False):
        result = glm_irt_t.cfa_icc_index(nitems, nfactors)
        matrix = result["index_matrix"]
        assert matrix.shape == (cells["row"].max(), cells["col"].max()), (nitems, nfactors)
        np.testing.assert_allclose(matrix[cells["row"] - 1, cells["col"] - 1], cells["matrix"], rtol=1e-12)
        np.testing.assert_allclose(result["index_vector"], cells["vector"], rtol=1e-12)


def test_icc_cfa_and_compute_map_match_rwf(reference):
    icc = reference("icc_cfa")
    results = [glm_irt_t.icc_cfa(np.arange(-2.0, 3.0), 1, 1, 1),
               glm_irt_t.icc_cfa([-1, 0, 1], [0.5, -0.2, 0.1], [1, 2, -1], [1, 2, 0.5]),
               glm_irt_t.icc_cfa([-1, 0, 1, 2], [0.5, -0.2], 1, 1)]  # gamma recycled, as R's arithmetic
    for case, result in enumerate(results, start=1):
        np.testing.assert_allclose(result, icc.loc[icc["case"] == case, "value"], rtol=1e-12, err_msg=str(case))
    assert glm_irt_t.icc_cfa(0.0, 0.0, 1.0, 1.0) == 0.5
    priors = reference("compute_map")
    np.testing.assert_allclose(glm_irt_t.compute_map(priors["eta"]), priors["standard"], rtol=1e-12)
    np.testing.assert_allclose(glm_irt_t.compute_map(priors["eta"], 0.5, 2), priors["shifted"], rtol=1e-12)


def _tirt_example():
    """rwf's example of compute_ability/compute_scores: the first two traits' items, eta = seq(-6, 6, 0.1)."""
    gamma = np.array([0.556, -1.253, -1.729, 0.618, 0.937, 0.295, -0.672, -1.127, -0.446, 0.632, 1.147, 0.498])
    psi = np.array([2.172, 1.883, 2.055, 1.869, 2.231, 2.100, 1.762, 1.803, 1.565, 1.892, 1.794, 1.686])
    items = glm_irt_t.response_dimension(range(1, 13), 3, [1, 2]) - 1
    eta = -6 + np.arange(121) * 0.1  # R's seq(-6, 6, by = 0.1)
    return eta, gamma[items], [1.082, 1.082, -1.297, -1.297, 0.802, 0.802, 1.083, 1.083], psi[items]


def test_compute_scores_matches_rwf(reference):
    """rwf's example passes eta, gamma, lambda and psi through ... by position."""
    eta, gamma, lam, psi = _tirt_example()
    responses = pd.DataFrame([[0] * 8, [1] * 8, [1, 0] * 4, [0, 1] * 4, [1, 1, 0, 0] * 2])
    scores = glm_irt_t.compute_scores(responses, eta, gamma, lam, psi, map=glm_irt_t.compute_map(eta), plot=False)
    np.testing.assert_allclose(scores, reference("compute_scores")["ability"], atol=1e-12)


def test_compute_ability_plot_matches_rwf(reference, monkeypatch):
    """With plot=True rwf draws the curves and returns the likelihood and posterior scaled up by tens."""
    titles = []
    draw = glm_irt_t.plot_icc_thurstonian
    monkeypatch.setattr(glm_irt_t, "plot_icc_thurstonian", lambda data, title: titles.append(title) or draw(data, title))
    eta, gamma, lam, psi = _tirt_example()
    result = glm_irt_t.compute_ability([1, 1, 0, 0, 1, 1, 0, 0], eta, gamma, lam, psi, plot=True)
    expected = reference("compute_ability_plot")
    assert list(result["icc"].columns) == list(expected.columns)
    np.testing.assert_allclose(result["icc"].to_numpy(float), expected.to_numpy(float), rtol=1e-12)
    np.testing.assert_allclose(result["product"], expected["product"], rtol=1e-12)
    assert titles == reference("compute_ability_plot_title")["title"].tolist()


def test_score_tirt_pattern_options_match_rwf(reference):
    """optim's controls (maxit, reltol, parscale, fnscale), a starting point and missing responses, as rwf."""
    lam, theta_diag, tau, psi, nu = _ulsmv_parameters()
    pattern = reference("tirt_triplets").loc[0, list(lam.index)].to_numpy(float)
    missing = pattern.copy()
    missing[[1, 4, 8]] = np.nan
    cases = {"default": {}, "maxit": {"control": {"maxit": 3}}, "init": {"init": [0.5, -0.5, 0.2]},
             "reltol": {"control": {"reltol": 1e-4}}, "parscale": {"control": {"parscale": [2, 1, 0.5], "maxit": 5}},
             "fnscale": {"control": {"fnscale": 10, "maxit": 5}}, "missing": {"pattern": missing},
             "all_missing": {"pattern": np.full(len(pattern), np.nan)}}
    expected = reference("score_tirt_pattern")
    for case, options in cases.items():
        options = dict(options)
        estimate = glm_irt_t.score_tirt_pattern(options.pop("pattern", pattern), lam, theta_diag, tau, psi, nu, **options)
        rows = expected[expected["case"] == case]
        assert list(estimate.index) == rows["trait"].tolist(), case
        np.testing.assert_allclose(estimate.to_numpy(), rows["value"].to_numpy(), atol=1e-8, err_msg=case)
    message = reference("tirt_helper_conditions").set_index("call")["message"]["score_tirt_pattern(control = list(foo = 1))"]
    with pytest.warns(UserWarning, match=message):
        glm_irt_t.score_tirt_pattern(pattern, lam, theta_diag, tau, psi, control={"foo": 1})
