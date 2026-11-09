import numpy as np
import pandas as pd
import pytest

from conftest import _load_reference
from pwf import _efa

ITEMS = [f"E{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]
STATS = {"RMSEA": lambda m: m.RMSEA["RMSEA"], "RMSEA.lower": lambda m: m.RMSEA["lower"],
         "RMSEA.upper": lambda m: m.RMSEA["upper"], "RMSEA.conf": lambda m: m.RMSEA["confidence"],
         "rms": lambda m: m.rms, "CFI": lambda m: m.CFI, "TLI": lambda m: m.TLI, "STATISTIC": lambda m: m.STATISTIC,
         "dof": lambda m: m.dof, "objective": lambda m: m.objective, "null.chisq": lambda m: m.null_chisq,
         "null.dof": lambda m: m.null_dof, "n.obs": lambda m: m.n_obs}


@pytest.fixture(scope="module")
def datasets():
    return {"complete": _load_reference("lavaan_data")[ITEMS], "missing": _load_reference("lavaan_fiml_data")}


@pytest.mark.parametrize("nfactors", [1, 2, 3])
@pytest.mark.parametrize("case", ["complete", "missing"])
def test_fa_matches_psych(datasets, reference, case, nfactors):
    model = _efa.fa(datasets[case], nfactors=nfactors)
    prefix = f"psych_fa_{case}_{nfactors}"
    # the oblimin rotation stops at psych's tolerance (1e-5), so loadings agree to about 1e-4
    for name, frame in [("loadings", model.loadings), ("structure", model.Structure)]:
        expected = reference(f"{prefix}_{name}").set_index("variable")
        assert list(frame.index) == list(expected.index)
        np.testing.assert_allclose(frame.to_numpy(), expected.to_numpy(), atol=3e-4, err_msg=name)
    np.testing.assert_allclose(model.residual.to_numpy(), reference(f"{prefix}_residual").to_numpy(), atol=1e-4)
    if nfactors > 1:
        np.testing.assert_allclose(model.Phi.to_numpy(), reference(f"{prefix}_phi").to_numpy(), atol=3e-4)
    for _, row in reference(f"{prefix}_stats").iterrows():
        value = STATS[row["stat"]](model)
        if np.isnan(row["value"]):
            assert np.isnan(value), row["stat"]
        else:
            assert value == pytest.approx(row["value"], rel=1e-4, abs=1e-8), row["stat"]


@pytest.mark.parametrize("nfactors", [1, 2, 3])
@pytest.mark.parametrize("case", ["complete", "missing"])
def test_model_loadings_matches_rwf(datasets, reference, case, nfactors):
    result = _efa.model_loadings(_efa.fa(datasets[case], nfactors=nfactors), cut=0.001)
    expected = reference(f"psych_fa_{case}_{nfactors}_model_loadings", keep_default_na=False, dtype=str)
    assert result.iloc[:, :2].astype(str).values.tolist() == expected.iloc[:, :2].values.tolist()
    assert result.iloc[:, 2:].astype(str).values.tolist() == expected.iloc[:, 2:].values.tolist()


def test_gpf_oblique_returns_unit_factor_variances(datasets):
    loadings, phi = _efa.gpf_oblique(_efa.fa(datasets["complete"], nfactors=2, rotate="none").loadings.to_numpy())
    np.testing.assert_allclose(np.diag(phi), 1)
