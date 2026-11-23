"""
Checks of a pwf SEM fit against lavaan's outputs as make_reference.R writes them in long format (one CSV per kind
of output, with a case column): parameterEstimates with rwf's arguments, fitMeasures, R squared, modification
indices, the inspect() matrices and lavPredict's factor scores. Used by the tests of the "LAVAAN OBSERVED
PREDICTORS" and "LAVAAN MEAN STRUCTURES" sections.
"""
import warnings

import numpy as np
import pytest

from conftest import _load_reference


def reference_case(prefix, name, case):
    """The rows of one case of a long-format reference (lavaan_x_parameters, lavaan_ms_fit, ...)."""
    frame = _load_reference(f"{prefix}_{name}", keep_default_na=False, na_values=["NA"])
    return frame[frame["case"] == case].drop(columns="case").reset_index(drop=True)


def check_parameters(fit, expected):
    """parameterEstimates: row numbers, rows, lavaan's optional columns (std.nox, label) and the values."""
    expected = expected.dropna(axis=1, how="all")
    got = fit.parameter_estimates(remove_eq=False)
    assert got.index.tolist() == expected["row"].astype(int).tolist()
    assert (got["lhs"] + got["op"] + got["rhs"]).tolist() == (expected["lhs"] + expected["op"]
                                                               + expected["rhs"].fillna("")).tolist()
    # lavaan has std.nox only with fixed predictors, and a label column only with labels
    assert ("std.nox" in got) == ("std.nox" in expected) and ("label" in got) == ("label" in expected)
    for column in ["est", "se", "z", "pvalue", "ci.lower", "ci.upper", "std.lv", "std.all", "std.nox"]:
        if column in expected:
            np.testing.assert_allclose(got[column].to_numpy(float), expected[column].to_numpy(float), rtol=2e-4,
                                       atol=2e-5, err_msg=column)


# root mean square residuals: of a saturated model (df = 0) they are rounding noise (about 1e-9) in both
RESIDUAL_MEASURES = {"rmr", "rmr_nomean", "srmr", "srmr_bentler", "srmr_bentler_nomean", "crmr", "crmr_nomean",
                     "srmr_mplus", "srmr_mplus_nomean"}


def check_fit_measures(fit, expected, not_ported=()):
    """fitMeasures in lavaan's order; measures not ported are missing in pwf."""
    got = fit.fit_measures()
    expected = expected.set_index("measure")["value"]
    assert list(got) == list(expected.index)
    for measure, value in expected.items():
        if measure in not_ported or np.isnan(value):
            assert np.isnan(got[measure]), measure
        else:  # lavaan writes chi-square p values below about 1e-20 as 0
            noise = 1e-7 if measure in RESIDUAL_MEASURES else 1e-12
            assert got[measure] == pytest.approx(value, rel=2e-4, abs=noise), measure


def check_rsquare_inspect_scores(fit, rsquare, inspect, scores):
    """R squared, lavaan's matrices (its dummy latent variables included) and the factor scores."""
    assert list(fit.rsquare()) == rsquare["name"].tolist()
    np.testing.assert_allclose(list(fit.rsquare().values()), rsquare["value"], rtol=2e-4, atol=1e-6)
    for what in ["est", "std"]:
        matrices = fit.inspect(what)
        rows = inspect[inspect["what"] == what]
        assert list(matrices) == list(dict.fromkeys(rows["matrix"]))
        for name, matrix in matrices.items():
            block = rows[rows["matrix"] == name]
            expected = block.pivot(index="row", columns="column", values="value")
            expected = expected.loc[list(dict.fromkeys(block["row"])), list(dict.fromkeys(block["column"]))]
            assert list(matrix.index) == list(expected.index) and list(matrix.columns) == list(expected.columns)
            np.testing.assert_allclose(matrix.to_numpy(float), expected.to_numpy(float), rtol=2e-4, atol=2e-5,
                                       err_msg=f"{what} {name}")
    predicted = fit.predict()
    assert list(predicted.columns) == list(dict.fromkeys(scores["column"]))  # the model's latent variables only
    if len(scores):
        assert fit.n == scores["nobs"].iloc[0]
        expected = scores.pivot(index="row", columns="column", values="value")[list(predicted.columns)]
        expected = expected[expected.notna().any(axis=1)]  # lavaan's NA row of a case without answers (FIML)
        np.testing.assert_allclose(predicted.to_numpy(float), expected.to_numpy(float), rtol=0, atol=2e-5)


def check_modification_indices(fit, expected):
    """
    modificationIndices with rwf's arguments. Equal indices (e.g. a covariance and the equivalent regression, or
    the model's own parameters, whose indices are rounding noise) are sorted by their last bits, so rows are
    compared by parameter, not position; lavaan's optimizer stops a little before pwf's (as in test_sem_engine),
    which moves the indices by up to about 1e-3 relative, and the expected changes of near-zero indices are
    ratios of two noisy numbers.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = fit.modification_indices()
    key = ["lhs", "op", "rhs"]
    got = got.assign(row=got.index).reset_index(drop=True)
    merged = got.merge(expected.assign(rhs=expected["rhs"].fillna("")), on=key, how="outer", suffixes=("", "_r"),
                       indicator=True)
    assert (merged["_merge"] == "both").all(), merged.loc[merged["_merge"] != "both", key]
    assert merged["row"].tolist() == merged["row_r"].tolist()  # lavaan's row numbers
    assert (np.diff(got["mi"].to_numpy()) <= 0).all()  # sorted as lavaan's sort. = TRUE
    real, informative = merged["mi_r"] > 1e-6, merged["mi_r"] > 0.1
    np.testing.assert_allclose(merged.loc[real, "mi"], merged.loc[real, "mi_r"], rtol=1e-3, atol=1e-3)
    for column in ["epc", "sepc.all"]:
        np.testing.assert_allclose(merged.loc[informative, column], merged.loc[informative, f"{column}_r"], rtol=1e-3,
                                   atol=1e-6, err_msg=column)
    assert merged.loc[informative, "decision"].tolist() == merged.loc[informative, "decision_r"].tolist()
