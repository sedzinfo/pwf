import numpy as np

from conftest import assert_matches_reference
from pwf import glm_one_way_anova


def test_compute_kruskal_wallis_test(df_blood_pressure, reference):
    result = glm_one_way_anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure)
    assert_matches_reference(result, reference("compute_kruskal_wallis_test"), rtol=1e-10)


def test_compute_kruskal_wallis_test_drops_missing(df_blood_pressure, reference):
    df = df_blood_pressure.copy()
    df.loc[[2, 49, 98], "bp_before"] = np.nan  # rows 3, 50, 99 in make_reference.R
    df.loc[[9, 69], "agegrp"] = np.nan  # rows 10, 70
    result = glm_one_way_anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df)
    assert_matches_reference(result, reference("compute_kruskal_wallis_test_na"), rtol=1e-10)


def test_compute_kruskal_wallis_test_ci(df_blood_pressure, reference):
    """Point estimates match R exactly; bootstrap bounds only within Monte Carlo error."""
    np.random.seed(1)
    result = glm_one_way_anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure,
                                                           ci=True, nboot=5000)
    bounds = ["etasq_lower", "etasq_upper", "epsilonsq_lower", "epsilonsq_upper"]
    assert_matches_reference(result, reference("compute_kruskal_wallis_test_ci"), rtol=1e-10,
                             column_tol={b: (0, 0.02) for b in bounds})
    row = result.iloc[0]
    assert row["etasq_lower"] < row["etasq"] < row["etasq_upper"]
    assert row["epsilonsq_lower"] < row["epsilonsq"] < row["epsilonsq_upper"]
