import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import _load_reference  # noqa: E402
from pwf import glm_efa  # noqa: E402

ITEMS = [f"E{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]
# The oblimin rotation stops at a tolerance of 1e-5 (psych/GPArotation), so values agree to about 1e-4;
# psych labels its factors (MR1, MR2, ...) by the order of a rotation from random starts, so factor
# columns are compared by position.
ATOL = 3e-4


@pytest.fixture(scope="module")
def datasets():
    return {"complete": _load_reference("lavaan_data")[ITEMS], "missing": _load_reference("lavaan_fiml_data")}


def by_position(frame):
    return frame.to_numpy(dtype=float)


@pytest.mark.parametrize("nfactors", [1, 2, 3])
@pytest.mark.parametrize("fm", ["minres", "ml"])
@pytest.mark.parametrize("case", ["complete", "missing"])
def test_fa_matches_psych(datasets, reference, case, fm, nfactors):
    model = glm_efa.fa(datasets[case], nfactors=nfactors, fm=fm)
    prefix = f"psych_fa_full_{case}_{fm}_{nfactors}"
    np.testing.assert_allclose(by_position(model.loadings), by_position(reference(f"{prefix}_loadings")), atol=ATOL)
    items = reference(f"{prefix}_items")
    mine = pd.DataFrame({"communality": model.communality.to_numpy(), "communalities": model.communalities.to_numpy(),
                         "uniquenesses": model.uniquenesses.to_numpy(), "complexity": model.complexity.to_numpy(),
                         "values": model.values, "e.values": model.e_values})
    np.testing.assert_allclose(mine.to_numpy(), items.to_numpy(), atol=2e-3 if nfactors == 3 else ATOL)
    np.testing.assert_allclose(by_position(model.weights), by_position(reference(f"{prefix}_weights")), atol=1e-3)
    np.testing.assert_allclose(by_position(model.scores), by_position(reference(f"{prefix}_scores")), atol=2e-3)
    np.testing.assert_allclose(by_position(model.Vaccounted), by_position(reference(f"{prefix}_vaccounted")),
                               atol=ATOL)
    stats = reference(f"{prefix}_stats").set_index("stat")["value"]
    got = {"crms": model.crms, "BIC": model.BIC, "SABIC": model.SABIC, "EBIC": model.EBIC, "ESABIC": model.ESABIC,
           "fit": model.fit, "fit.off": model.fit_off, "rms": model.rms, "TLI": model.TLI, "CFI": model.CFI,
           "RMSEA": model.RMSEA["RMSEA"], "STATISTIC": model.STATISTIC}
    for name, value in got.items():
        if np.isnan(stats[name]):
            assert np.isnan(value), name
        else:
            assert value == pytest.approx(stats[name], rel=1e-3, abs=1e-6), name


@pytest.mark.parametrize("case", ["complete", "missing"])
def test_report_efa_matches_rwf(datasets, reference, case):
    report = glm_efa.report_efa(glm_efa.fa(datasets[case], nfactors=2, fm="minres"), datasets[case])
    for part in ["residual_stats", "determinant_test", "bartlett_test", "kmo_test", "correlations"]:
        expected = reference(f"report_efa_{case}_{part}", keep_default_na=False, na_values=["NA"])
        got = report[part].reset_index(drop=True)
        assert list(got.columns) == list(expected.columns), part
        for column in expected.columns:
            if pd.api.types.is_bool_dtype(expected[column]):
                assert got[column].tolist() == expected[column].tolist(), (part, column)
            elif pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(pd.to_numeric(got[column]).to_numpy(float),
                                           expected[column].to_numpy(float), atol=1e-4, rtol=1e-6,
                                           err_msg=f"{part} {column}")
            else:
                assert got[column].astype(str).tolist() == expected[column].astype(str).tolist(), (part, column)
    loadings = report["loadings"]
    expected = reference(f"report_efa_{case}_loadings", keep_default_na=False, na_values=["NA"])
    assert loadings.shape == expected.shape
    assert loadings.columns[[0, 1, -2, -1]].tolist() == expected.columns[[0, 1, -2, -1]].tolist()
    for column in [0, 1, -2, -1]:
        assert loadings.iloc[:, column].astype(str).tolist() == expected.iloc[:, column].astype(str).tolist()
    np.testing.assert_allclose(loadings.iloc[:, 2:4].to_numpy(float), expected.iloc[:, 2:4].to_numpy(float),
                               atol=1e-4)


def test_report_efa_writes_files(datasets, tmp_path):
    file = str(tmp_path / "efa")
    glm_efa.report_efa(glm_efa.fa(datasets["complete"], nfactors=2), datasets["complete"], file=file, scores=True)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets[:4] == ["r", "n", "residual stats", "fit_index"] and "scores" in sheets
    assert (tmp_path / "efa.pdf").exists()
