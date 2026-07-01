from pwf import glm_means


def test_report_ttests(df_blood_pressure):
    result = glm_means.report_ttests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    assert len(result) == 4  # 1 pair for sex + 3 pairs for agegrp
    assert {"T", "p-val", "cohen-d"}.issubset(result.columns)


def test_report_wtests(df_blood_pressure):
    result = glm_means.report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    assert len(result) == 4
    assert {"U-val", "p-val", "RBC"}.issubset(result.columns)
