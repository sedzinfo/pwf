"""
psych::fa's extraction methods and rotations against psych itself (make_reference.R, "EFA METHODS AND
ROTATIONS"), and rwf::report_efa on models of other methods.

psych rotates from the identity and 19 random starts; pwf from the identity only. The references of
GPArotation's rotations are therefore psych's with n.rotations = 1; varimax, promax and equamax have no
random starts and use psych's defaults.
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import _load_reference
from pwf import _efa
from pwf.glm_efa import report_efa

ITEMS = [f"E{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]
STATS = ["rms", "crms", "RMSEA", "RMSEA.lower", "RMSEA.upper", "TLI", "CFI", "STATISTIC", "dof", "objective", "BIC",
         "SABIC", "EBIC", "ESABIC", "fit", "fit.off", "null.chisq"]


def _stats(model):
    return [model.rms, model.crms, model.RMSEA["RMSEA"], model.RMSEA["lower"], model.RMSEA["upper"], model.TLI,
            model.CFI, model.STATISTIC, model.dof, model.objective, model.BIC, model.SABIC, model.EBIC, model.ESABIC,
            model.fit, model.fit_off, model.null_chisq]


def _data(data_case="complete"):
    # "rotation" models are fitted on the complete data
    if data_case == "missing":
        return _load_reference("lavaan_fiml_data")
    return _load_reference("lavaan_data")[ITEMS]


def _matrix(block):
    shape = (block["row"].max(), block["column"].max())
    result = np.full(shape, np.nan)
    result[block["row"] - 1, block["column"] - 1] = block["value"].to_numpy(dtype=float)
    return result


try:
    METHODS = _load_reference("psych_fa_methods")
    COLUMNS = _load_reference("psych_fa_methods_columns").set_index("model")["columns"]
    ERRORS = _load_reference("psych_fa_methods_errors").set_index("model")["error"]
    KEYS = list(dict.fromkeys(METHODS["model"]))
except pytest.skip.Exception:  # no references: the tests below skip themselves
    METHODS, COLUMNS, ERRORS, KEYS = None, None, None, []


@pytest.mark.parametrize("key", KEYS)
def test_fa_matches_psych(key):
    data_case, fm, nfactors, rotate = key.split("|")
    # the extraction methods with rwf's oblique.scores = TRUE, the rotations with psych's default FALSE
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        model = _efa.fa(_data(data_case), nfactors=int(nfactors), fm=fm, rotate=rotate,
                        oblique_scores=data_case != "rotation")
    parts = dict(list(METHODS[METHODS["model"] == key].groupby("part")))
    # psych leaves the columns of "ols" unnamed (empty in the reference); pwf numbers them as rwf's data frames
    expected_columns = COLUMNS[key] if isinstance(COLUMNS[key], str) else ",".join(
        f"X{i + 1}" for i in range(int(nfactors)))
    assert ",".join(model.loadings.columns) == expected_columns
    for part, got in [("loadings", model.loadings), ("weights", model.weights), ("Vaccounted", model.Vaccounted),
                      ("Phi", model.Phi)]:
        if part not in parts:
            assert got is None, part
            continue
        np.testing.assert_allclose(np.asarray(got, dtype=float), _matrix(parts[part]), atol=1e-6, err_msg=part)
    for part, got in [("communality", model.communality), ("communalities", model.communalities),
                      ("uniquenesses", model.uniquenesses), ("complexity", model.complexity),
                      ("values", model.values), ("e.values", model.e_values)]:
        np.testing.assert_allclose(np.atleast_1d(np.asarray(got, dtype=float)), _matrix(parts[part])[:, 0],
                                   atol=1e-6, err_msg=part)
    expected = _matrix(parts["stats"])[:, 0]
    for name, got, want in zip(STATS, _stats(model), expected):
        if np.isnan(want):
            assert np.isnan(got), name
        else:
            assert got == pytest.approx(want, rel=1e-6, abs=1e-8), name


def test_unknown_method_is_minres_with_psychs_message():
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        model = _efa.fa(_data(), nfactors=2, fm="foo")
    assert "factor method not specified correctly" in stderr.getvalue()
    assert model.fm == "minres"
    assert ",".join(model.loadings.columns) == COLUMNS["complete|foo|2|oblimin"]


@pytest.mark.parametrize("key", ["complete|alpha|1|oblimin", "missing|alpha|1|oblimin", "rotation|minres|2|foo"])
def test_psychs_errors(key):
    data_case, fm, nfactors, rotate = key.split("|")
    # psych's alpha factoring of one factor takes diag() of one number (an identity matrix of that size)
    assert ERRORS[key] == ("non-conformable arguments" if fm == "alpha" else "object 'rot.mat' not found")
    message = "non-conformable" if fm == "alpha" else "rotation not found"
    with pytest.raises(ValueError, match=message), contextlib.redirect_stderr(io.StringIO()):
        _efa.fa(_data(data_case), nfactors=int(nfactors), fm=fm, rotate=rotate)


def test_methods_and_rotations_not_ported():
    with pytest.raises(NotImplementedError, match="minrank"):
        _efa.fa(_data(), nfactors=2, fm="minrank")
    with pytest.raises(NotImplementedError, match="bifactor"):
        _efa.fa(_data(), nfactors=2, rotate="bifactor")
    # one factor is never rotated, so psych accepts any rotation name there
    assert _efa.fa(_data(), nfactors=1, rotate="bifactor").Phi is None


@pytest.mark.parametrize("fm", ["pa", "ols", "wls"])
def test_report_efa_with_other_methods(reference, fm):
    data = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _efa.fa(data, nfactors=2, fm=fm, rotate="oblimin", oblique_scores=True)
        report = report_efa(model, data, scores=True)
    expected = reference(f"report_efa_methods_{fm}_loadings", keep_default_na=False, dtype=str)
    got = report["loadings"].astype(object).where(report["loadings"].notna(), "NA")
    assert list(got.columns) == list(expected.columns)
    assert got.iloc[:, :2].astype(str).values.tolist() == expected.iloc[:, :2].values.tolist()
    numbers = expected.columns[2:-2]
    np.testing.assert_allclose(pd.to_numeric(got[numbers].stack(), errors="coerce").to_numpy(dtype=float),
                               pd.to_numeric(expected[numbers].stack(), errors="coerce").to_numpy(dtype=float),
                               atol=1e-6)
    np.testing.assert_allclose(report["weights"].to_numpy(dtype=float),
                               reference(f"report_efa_methods_{fm}_weights").to_numpy(dtype=float), atol=1e-6)
    np.testing.assert_allclose(report["scores"].to_numpy(dtype=float),
                               reference(f"report_efa_methods_{fm}_scores").to_numpy(dtype=float), atol=1e-6)
    correlations = reference(f"report_efa_methods_{fm}_correlations")
    np.testing.assert_allclose(report["correlations"][ITEMS].to_numpy(dtype=float),
                               correlations[ITEMS].to_numpy(dtype=float), atol=1e-6)
    stats = reference(f"report_efa_methods_{fm}_residual_stats")
    np.testing.assert_allclose(report["residual_stats"]["value"].to_numpy(dtype=float),
                               stats["value"].to_numpy(dtype=float), atol=1e-8)
