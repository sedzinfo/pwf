import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import assert_matches_reference  # noqa: E402
from pwf import glm_one_way_anova as anova  # noqa: E402

# R's ptukey and scipy's studentized_range integrate numerically to about 1e-10 (absolute)
PTUKEY_TOL = {"p": (1e-6, 1e-9)}


@pytest.fixture
def df_blood_pressure_na(df_blood_pressure):
    """Same NAs as df_blood_pressure_na in tests/reference/make_reference.R (1-based rows there)."""
    df = df_blood_pressure.copy()
    df.loc[[2, 49, 98], "bp_before"] = np.nan
    df.loc[[9, 69], "agegrp"] = np.nan
    return df


def test_compute_kruskal_wallis_test(df_blood_pressure, reference):
    result = anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure)
    assert_matches_reference(result, reference("compute_kruskal_wallis_test"), rtol=1e-10)


def test_compute_kruskal_wallis_test_drops_missing(df_blood_pressure_na, reference):
    result = anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure_na)
    assert_matches_reference(result, reference("compute_kruskal_wallis_test_na"), rtol=1e-10)


def test_compute_kruskal_wallis_test_ci(df_blood_pressure, reference):
    """Point estimates match R exactly; bootstrap bounds only within Monte Carlo error."""
    np.random.seed(1)
    result = anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure,
                                               ci=True, nboot=5000)
    bounds = ["etasq_lower", "etasq_upper", "epsilonsq_lower", "epsilonsq_upper"]
    assert_matches_reference(result, reference("compute_kruskal_wallis_test_ci"), rtol=1e-10,
                             column_tol={b: (0, 0.02) for b in bounds})
    row = result.iloc[0]
    assert row["etasq_lower"] < row["etasq"] < row["etasq_upper"]
    assert row["epsilonsq_lower"] < row["epsilonsq"] < row["epsilonsq_upper"]


@pytest.mark.parametrize("var_equal", [True, False])
def test_compute_one_way_test(df_blood_pressure, df_blood_pressure_na, load_data, reference, var_equal):
    cases = {"": ("bp_before ~ agegrp", df_blood_pressure), "na_": ("bp_before ~ agegrp", df_blood_pressure_na),
             "region_": ("charges ~ region", load_data("insurance"))}
    for prefix, (formula, data) in cases.items():
        result = anova.compute_one_way_test(formula, data, var_equal=var_equal)
        assert_matches_reference(result, reference(f"compute_one_way_test_{prefix}{str(var_equal).upper()}"))


@pytest.mark.parametrize("case", ["bp", "region"])
def test_compute_posthoc(df_blood_pressure, load_data, reference, case):
    if case == "bp":
        y, x = df_blood_pressure["bp_before"], df_blood_pressure["agegrp"]
    else:
        insurance = load_data("insurance")
        y, x = insurance["charges"], insurance["region"]
    output = anova.compute_posthoc(y, x)["output"]
    for table, name in [("tukey", "tukey"), ("games.howell", "games_howell")]:
        result = output[table].rename_axis("pair").reset_index()
        assert_matches_reference(result, reference(f"compute_posthoc_{case}_{name}"), column_tol=PTUKEY_TOL)


def test_compute_friedman_test(load_data, reference):
    co2 = load_data("co2")
    assert_matches_reference(anova.compute_friedman_test("uptake ~ conc | Plant", co2),
                             reference("compute_friedman_test"))
    incomplete = co2.drop(index=[2, 39])  # rows 3 and 40 in R
    assert_matches_reference(anova.compute_friedman_test("uptake ~ conc | Plant", incomplete),
                             reference("compute_friedman_test_incomplete"))


def test_compute_friedman_test_ci(load_data, reference):
    np.random.seed(1)
    result = anova.compute_friedman_test("uptake ~ conc | Plant", load_data("co2"), ci=True, nboot=5000)
    assert_matches_reference(result, reference("compute_friedman_test_ci"),
                             column_tol={"kendall_w_lower": (0, 0.02), "kendall_w_upper": (0, 0.02)})


def test_compute_friedman_test_rejects_replicates():
    df = pd.DataFrame({"y": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "b": [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="at most one observation"):
        anova.compute_friedman_test("y ~ g | b", df)


@pytest.mark.parametrize("table", ["fisher", "welch", "kruskal_wallis", "games_howell", "tukey", "homogeneity"])
def test_report_oneway(df_blood_pressure_na, reference, table):
    result = anova.report_oneway(df=df_blood_pressure_na, dv=[4, 5], iv=[2, 3])
    assert_matches_reference(result[table], reference(f"report_oneway_{table}"), column_tol=PTUKEY_TOL)


def test_report_oneway_writes_files(df_blood_pressure, tmp_path):
    file = str(tmp_path / "anova")
    anova.report_oneway(df=df_blood_pressure, dv=["bp_before"], iv=["agegrp"], file=file,
                        plot_means=True, plot_diagnostics=True)
    sheets = pd.read_excel(f"{file}.xlsx", sheet_name=None)
    assert list(sheets) == ["Fisher", "Welch", "Kruskal", "Homogeneity", "Tukey", "Games-Howell", "Descriptives"]
    assert (tmp_path / "anova_means.pdf").exists()
    assert (tmp_path / "anova_diagnostics.pdf").exists()


FACTORIAL_CASES = {
    "between": dict(between=["B1", "B2"]),
    "between_unadjusted": dict(between=["B1", "B2"], white_adjust=False),
    "within": dict(within=["W1", "W2"], within_full=["W1", "W2"]),
    "within_unadjusted": dict(within=["W1", "W2"], white_adjust=False),
    "mixed": dict(within=["W1"], between=["B1"]),
    "mixed_2x2": dict(within=["W1", "W2"], between=["B1", "B2"]),
    "mixed_observed": dict(within=["W1"], between=["B1"], white_adjust=False, observed=["B1"]),
}


@pytest.fixture(scope="module")
def df_factorial():
    from conftest import _load_reference
    return _load_reference("factorial_data")


@pytest.mark.parametrize("case", FACTORIAL_CASES)
def test_report_factorial_anova(df_factorial, reference, case):
    result = anova.report_factorial_anova(df=df_factorial, wid="id", dv=["DV1", "DV2"], **FACTORIAL_CASES[case])
    assert_matches_reference(result["omnibus"], reference(f"report_factorial_anova_{case}_omnibus"))
    # sjstats rounds to 3 decimals, so allow one unit in the last place
    assert_matches_reference(result["omnibus_effect_size"], reference(f"report_factorial_anova_{case}_effect_size"),
                             rtol=0, atol=1.001e-3)
    assert_matches_reference(result["post_hoc"], reference(f"report_factorial_anova_{case}_post_hoc"),
                             column_tol={"p.value": (1e-6, 1e-9)})


def test_report_factorial_anova_unbalanced(df_factorial, reference):
    unbalanced = df_factorial[~df_factorial["id"].isin([1, 2, 14])]
    result = anova.report_factorial_anova(df=unbalanced, wid="id", dv="DV1", within=["W1"], between=["B1"])
    assert_matches_reference(result["omnibus"], reference("report_factorial_anova_unbalanced_omnibus"))
    assert_matches_reference(result["omnibus_effect_size"], reference("report_factorial_anova_unbalanced_effect_size"),
                             rtol=0, atol=1.001e-3)
    assert_matches_reference(result["post_hoc"], reference("report_factorial_anova_unbalanced_post_hoc"),
                             column_tol={"p.value": (1e-6, 1e-9)})


def test_report_factorial_anova_writes_excel(df_factorial, tmp_path):
    file = str(tmp_path / "factorial")
    anova.report_factorial_anova(df=df_factorial, wid="id", dv="DV1", within=["W1"], between=["B1"], file=file)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["ANOVA within", "effect size", "post hoc", "descriptives", "call"]


def test_report_factorial_anova_rejects_unported_options(df_factorial):
    with pytest.raises(NotImplementedError):
        anova.report_factorial_anova(df=df_factorial, wid="id", dv="DV1", within=["W1"], within_covariates=["DV2"])
