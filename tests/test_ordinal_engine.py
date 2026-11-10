import numpy as np
import pandas as pd
import pytest

from conftest import _load_reference
from pwf._ordinal import OrdinalCFA, OrdinalSampleStats

ITEMS = [f"E{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]
MODEL = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5"
ROBUST = ["cfi.robust", "tli.robust", "nnfi.robust", "rni.robust", "rmsea.robust", "rmsea.ci.lower.robust",
          "rmsea.ci.upper.robust", "rmsea.pvalue.robust", "rmsea.notclose.pvalue.robust", "gfi.robust",
          "gfi.ci.lower.robust", "gfi.ci.upper.robust"]


@pytest.fixture(scope="module")
def datasets():
    return {"ordinal": _load_reference("lavaan_data")[ITEMS], "binary": _load_reference("wlsmv_binary_data")}


@pytest.fixture(scope="module")
def fits(datasets):
    return {(case, par): OrdinalCFA(MODEL, datasets[case], parameterization=par)
            for case in datasets for par in ["delta", "theta"]}


@pytest.mark.parametrize("case", ["ordinal", "binary"])
def test_sample_statistics_match_lavaan(datasets, reference, case):
    stats = OrdinalSampleStats(datasets[case])
    th = reference(f"wlsmv_{case}_th")
    assert stats.th_names == th["name"].tolist()
    np.testing.assert_allclose(stats.th, th["th"], atol=1e-12)
    # lavaan's nlminb stops the polychoric correlations at about 1e-8
    np.testing.assert_allclose(stats.cor, reference(f"wlsmv_{case}_cor").to_numpy(), atol=1e-7)
    gamma = reference(f"wlsmv_{case}_gamma")
    assert stats.th_names + stats.cor_names == list(gamma.columns)
    np.testing.assert_allclose(stats.nacov, gamma.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(stats.wls_v, reference(f"wlsmv_{case}_wls_v")["w"], rtol=1e-6)


@pytest.mark.parametrize("par", ["delta", "theta"])
@pytest.mark.parametrize("case", ["ordinal", "binary"])
def test_parameters_match_lavaan(fits, reference, case, par):
    mine = fits[(case, par)].parameter_estimates()
    expected = reference(f"wlsmv_{case}_{par}_parameters", keep_default_na=False, na_values=["NA"])
    assert mine[["lhs", "op", "rhs"]].astype(str).values.tolist() == expected[["lhs", "op", "rhs"]].astype(str).values.tolist()
    # lavaan's optimizer stops earlier in the theta parameterization (its objective is above the delta optimum)
    atol = 1e-6 if par == "delta" else 1e-4
    for column in ["est", "se", "z", "std.lv", "std.all"]:
        np.testing.assert_allclose(mine[column].to_numpy(float), expected[column].to_numpy(float),
                                   atol=atol, rtol=atol, err_msg=column)


@pytest.mark.parametrize("par", ["delta", "theta"])
@pytest.mark.parametrize("case", ["ordinal", "binary"])
def test_fit_measures_match_lavaan(fits, reference, case, par):
    mine = fits[(case, par)].fit_measures()
    expected = reference(f"wlsmv_{case}_{par}_fit").set_index("measure")["value"]
    for name, value in expected.items():
        if np.isnan(value):
            assert np.isnan(mine[name]), name
        else:
            assert mine[name] == pytest.approx(value, rel=1e-5, abs=1e-7), name
