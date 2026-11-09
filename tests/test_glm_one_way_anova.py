import numpy as np

from pwf import glm_one_way_anova


def test_compute_kruskal_wallis_test_matches_rwf(df_blood_pressure, reference):
    expected = reference("compute_kruskal_wallis_test")
    result = glm_one_way_anova.compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure)
    for col in ["etasq", "epsilonsq", "H", "df", "p"]:
        np.testing.assert_allclose(result[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float),
                                   rtol=1e-10, err_msg=col)
