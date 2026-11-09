import numpy as np
import pytest

from conftest import assert_matches_reference
from pwf import glm_means

# wilcox.test finds its normal-approximation CI with uniroot(tol = 1e-4); the
# Python port uses brentq with the same tolerance, so bounds agree to ~1e-4.
WTEST_TOL = {"CI_l": (0, 1e-3), "CI_u": (0, 1e-3)}


@pytest.fixture
def df_blood_pressure_na(df_blood_pressure):
    """Same NAs as df_blood_pressure_na in tests/reference/make_reference.R (1-based rows there)."""
    df = df_blood_pressure.copy()
    df.loc[[2, 49, 98], "bp_before"] = np.nan
    df.loc[[9, 69], "agegrp"] = np.nan
    return df


def test_compute_cohens_d(load_data, reference):
    expected = reference("compute_cohens_d")
    data = {"bp_before ~ sex": load_data("blood_pressure"), "bp_after ~ sex": load_data("blood_pressure"),
            "charges ~ smoker": load_data("insurance")}
    result = [glm_means.compute_cohens_d(f, data[f]) for f in expected["formula"]]
    np.testing.assert_allclose(result, expected["d"], rtol=1e-10)


def test_compute_cohens_d_needs_two_levels(df_blood_pressure):
    with pytest.raises(ValueError, match="exactly 2 levels"):
        glm_means.compute_cohens_d("bp_before ~ agegrp", df_blood_pressure)


def test_compute_wilcoxon_effect_size(load_data, reference):
    expected = reference("compute_wilcoxon_effect_size").set_index("case")["r"]
    bp = load_data("blood_pressure")
    bp = bp[bp["agegrp"].isin(["30-45", "46-59"])]
    insurance = load_data("insurance")
    children_4_5 = insurance[insurance["children"].isin([4, 5])]
    result = {
        "bp_before_agegrp": glm_means.compute_wilcoxon_effect_size("bp_before ~ agegrp", bp),
        "bp_before_agegrp_nocorrect": glm_means.compute_wilcoxon_effect_size("bp_before ~ agegrp", bp, correct=False),
        "charges_children_4_5": glm_means.compute_wilcoxon_effect_size("charges ~ children", children_4_5),
    }
    for case, r in result.items():
        np.testing.assert_allclose(r, expected[case], rtol=1e-10, err_msg=case)


@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_report_ttests(df_blood_pressure, reference, alternative):
    result = glm_means.report_ttests(df=df_blood_pressure, dv=[4, 5], iv=[2, 3], alternative=alternative)
    assert_matches_reference(result, reference(f"report_ttests_{alternative}"))


def test_report_ttests_accepts_column_names(df_blood_pressure):
    by_position = glm_means.report_ttests(df=df_blood_pressure, dv=[4, 5], iv=[2, 3])
    by_name = glm_means.report_ttests(df=df_blood_pressure, dv=["bp_before", "bp_after"], iv=["sex", "agegrp"])
    assert by_position.equals(by_name)


def test_report_ttests_var_equal(df_blood_pressure, reference):
    result = glm_means.report_ttests(df=df_blood_pressure, dv=4, iv=[2, 3], var_equal=True)
    assert_matches_reference(result, reference("report_ttests_var_equal"))


def test_report_ttests_na(df_blood_pressure_na, reference):
    result = glm_means.report_ttests(df=df_blood_pressure_na, dv=4, iv=3)
    assert_matches_reference(result, reference("report_ttests_na"))


def test_report_ttests_insurance(load_data, reference):
    result = glm_means.report_ttests(df=load_data("insurance"), dv=7, iv=[2, 4])
    assert_matches_reference(result, reference("report_ttests_insurance"))


@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
def test_report_wtests(df_blood_pressure, reference, alternative):
    result = glm_means.report_wtests(df=df_blood_pressure, dv=[4, 5], iv=[2, 3], alternative=alternative)
    assert_matches_reference(result, reference(f"report_wtests_{alternative}"), column_tol=WTEST_TOL)


def test_report_wtests_na(df_blood_pressure_na, reference):
    result = glm_means.report_wtests(df=df_blood_pressure_na, dv=4, iv=3)
    assert_matches_reference(result, reference("report_wtests_na"), column_tol=WTEST_TOL)


def test_report_wtests_insurance(load_data, reference):
    """Includes children 4 vs 5 (both groups < 50, no ties), which uses the exact test."""
    result = glm_means.report_wtests(df=load_data("insurance"), dv=7, iv=[2, 4])
    assert_matches_reference(result, reference("report_wtests_insurance"), column_tol=WTEST_TOL)


def test_report_wtests_insurance_less(load_data, reference):
    result = glm_means.report_wtests(df=load_data("insurance"), dv=7, iv=4, alternative="less")
    assert_matches_reference(result, reference("report_wtests_insurance_less"), column_tol=WTEST_TOL)


def test_report_ttests_writes_excel(df_blood_pressure, tmp_path):
    file = tmp_path / "ttest"
    result = glm_means.report_ttests(df=df_blood_pressure, dv=4, iv=2, file=str(file))
    assert (tmp_path / "ttest.xlsx").exists()
    assert len(result) == 1
