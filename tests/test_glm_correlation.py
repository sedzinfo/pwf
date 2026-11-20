import matplotlib

matplotlib.use("Agg")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.stats  # noqa: E402

from conftest import _load_reference, assert_matches_reference  # noqa: E402
from pwf import glm_correlation  # noqa: E402

ITEMS = [f"E{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]
# psych finds rho with R's optimize(), which pwf runs itself (_rcompat.r_optimize): rho agrees to ~2e-11
ATOL = 1e-9


@pytest.fixture(scope="module")
def datasets():
    complete = _load_reference("lavaan_data")[ITEMS]
    unequal = complete.copy()
    unequal["N5"] = unequal["N5"].clip(upper=3)
    return {"complete": complete, "missing": _load_reference("lavaan_fiml_data"), "small": complete.iloc[:40],
            "unequal": unequal}


def dichotomize(items):
    """1 above 3, 0 otherwise, keeping missing values missing (as R's (items > 3) * 1)."""
    return items.gt(3).astype(float).where(items.notna())


def run_quietly(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def test_bivariate_normal_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(50):
        h, k = rng.normal(size=2) * 2
        r = rng.uniform(-0.999, 0.999)
        expected = scipy.stats.multivariate_normal([0, 0], [[1, r], [r, 1]], abseps=1e-13, releps=1e-13).cdf([-h, -k])
        assert glm_correlation._bvn_upper(h, k, r) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("case", ["complete", "missing", "small", "unequal"])
def test_tetrachoric_matches_psych(datasets, reference, case):
    result = run_quietly(glm_correlation.report_choric_serial, dichotomize(datasets[case]), type="tetrachoric")
    expected = reference(f"tetrachoric_{case}_rho")
    assert list(result["rho"].columns) == list(expected.columns)
    np.testing.assert_allclose(result["rho"].to_numpy(), expected.to_numpy(), atol=ATOL)
    tau = reference(f"tetrachoric_{case}_tau")
    assert result["tau"].index.tolist() == tau["item"].tolist()
    np.testing.assert_allclose(result["tau"].to_numpy(), tau["tau"].to_numpy(), atol=1e-12)
    assert result["n_obs"] == len(datasets[case])


def test_tetrachoric_deletes_constant_items(datasets):
    with pytest.warns(UserWarning, match="N5 had no variance"):
        result = glm_correlation.tetrachoric(dichotomize(datasets["unequal"]))
    assert "N5" not in result["rho"].columns


@pytest.mark.parametrize("case", ["complete", "missing", "small", "unequal"])
def test_polychoric_matches_psych(datasets, reference, case):
    result = run_quietly(glm_correlation.report_choric_serial, datasets[case], type="polychoric")
    expected = reference(f"polychoric_{case}_rho")
    assert list(result["rho"].columns) == list(expected.columns)
    np.testing.assert_allclose(result["rho"].to_numpy(), expected.to_numpy(), atol=ATOL)
    tau = reference(f"polychoric_{case}_tau")
    np.testing.assert_allclose(result["tau"].to_numpy(), tau.to_numpy(), atol=1e-12)


def test_polychoric_unequal_categories_sets_global_false(datasets):
    with pytest.warns(UserWarning, match="global set to FALSE"):
        glm_correlation.polychoric(datasets["unequal"])


@pytest.mark.parametrize("type", ["polyserial", "biserial"])
def test_serial_matches_psych(reference, type):
    data = reference("polyserial_data")
    y = data[["N1", "N2", "N3"]]
    if type == "biserial":
        y = y.gt(3).astype(int)
    result = run_quietly(glm_correlation.report_choric_serial, data[["E", "A"]], y, type=type)
    expected = reference(type)
    assert result.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_report_choric_serial_writes_files(datasets, tmp_path):
    file = str(tmp_path / "polychoric")
    result = run_quietly(glm_correlation.report_choric_serial, datasets["complete"][ITEMS[:4]], file=file,
                         type="polychoric")
    sheets = pd.read_excel(f"{file}.xlsx", sheet_name=None)
    assert list(sheets) == ["R", "R Squared", "Tau", "Observations", "Call"]
    assert sheets["Observations"]["Observations"].tolist() == [result["n_obs"]]
    assert (tmp_path / "polychoric_polychoric.pdf").exists()


CORRELATION_CASES = {
    "complete": dict(columns=["age", "bmi", "children", "charges"], y=None, use="pairwise", method="pearson",
                     adjust="holm"),
    "missing": dict(columns=ITEMS[:5], y=None, use="pairwise", method="pearson", adjust="holm"),
    "spearman": dict(columns=ITEMS[:5], y=None, use="pairwise", method="spearman", adjust="BH"),
    "kendall": dict(columns=ITEMS[:5], y=None, use="complete", method="kendall", adjust="none"),
    "xy": dict(columns=ITEMS[:3], y=ITEMS[5:8], use="pairwise", method="pearson", adjust="hommel"),
}


@pytest.mark.parametrize("case", list(CORRELATION_CASES))
def test_report_correlation_matches_rwf(load_data, reference, case):
    k = CORRELATION_CASES[case]
    data = load_data("insurance") if case == "complete" else _load_reference("lavaan_fiml_data")
    y = data[k["y"]] if k["y"] else None
    result = glm_correlation.report_correlation(data[k["columns"]], y, use=k["use"], method=k["method"],
                                                adjust=k["adjust"], scatterplot=False)
    for part in ["r_lower", "r_squared_lower", "p_lower", "p_lower_adjusted", "t_lower", "n_lower", "se_lower"]:
        expected = reference(f"report_correlation_{case}_{part}")
        assert list(result[part].columns) == list(expected.columns), part
        np.testing.assert_allclose(result[part].to_numpy(float), expected.to_numpy(float), rtol=1e-9, atol=1e-12,
                                   err_msg=part)
    expected = reference(f"report_correlation_{case}_ci")
    ci = result["ci"]
    assert ci.index.tolist() == expected["pair"].tolist()
    assert list(ci.columns) == list(expected.columns[1:])
    np.testing.assert_allclose(ci.to_numpy(float), expected.iloc[:, 1:].to_numpy(float), rtol=1e-9, atol=1e-12)
    call = reference(f"report_correlation_{case}_call")
    assert result["call"].astype(str).values.tolist() == call.astype(str).values.tolist()


def test_report_correlation_writes_files(load_data, tmp_path):
    file = str(tmp_path / "correlation")
    glm_correlation.report_correlation(load_data("insurance")[["age", "bmi", "charges"]], file=file, scatterplot=False)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["r", "r_squared", "p", "p_adjusted", "t", "N", "SE", "CI", "Call"]
    assert (tmp_path / "correlation_corrplot.pdf").exists()


def test_r_p_adjust_and_abbreviate():
    from pwf._rcompat import r_abbreviate, r_p_adjust
    words = ["children", "charges", "Extraversion", "Sepal.Length", "Petal Length", "petal width", "Agreeableness1",
             "Agreeableness2", "response_time_ms", "age2"]
    assert r_abbreviate(words, 5) == ["chldr", "chrgs", "Extrv", "Spl.L", "PtlLn", "ptlwd", "Agrb1", "Agrb2", "rsp__",
                                      "age2"]
    p = [0.01, 0.04, 0.03, 0.2, 0.04, np.nan, 0.5, 0.001]
    expected = {"holm": [0.06, 0.16, 0.15, 0.4, 0.16, np.nan, 0.5, 0.007],
                "hommel": [0.06, 0.12, 0.09, 0.4, 0.12, np.nan, 0.5, 0.007],
                "BY": [0.09075, 0.1452, 0.1452, 0.605, 0.1452, np.nan, 1, 0.01815]}
    for method, values in expected.items():
        np.testing.assert_allclose(r_p_adjust(p, method), values, rtol=1e-12)


POWER_CASES = {"two_sided": dict(n=30, r=0.3, sig_level=0.05, alternative="two.sided"),
               "less": dict(n=20, r=-0.25, sig_level=0.01, alternative="less"),
               "greater": dict(n=25, r=0.2, sig_level=0.05, alternative="greater")}


@pytest.mark.parametrize("case", list(POWER_CASES))
def test_compute_power_r_matches_rwf(reference, case):
    # pwr::pwr.r.test: the t test's critical r and a bias-corrected Fisher z; r is reported as |r| or -r
    result = glm_correlation.compute_power_r(**POWER_CASES[case])["power_table"]
    assert_matches_reference(result, reference(f"compute_power_r_{case}"), rtol=1e-12)


def test_compute_power_r_matrix_matches_rwf(reference):
    m = np.array([[1, 0.3, -0.5], [0.3, 1, 0.1], [-0.5, 0.1, 1]])
    result = glm_correlation.compute_power_r_matrix(m, n=15)["power_table"]
    assert_matches_reference(result, reference("compute_power_r_matrix"), rtol=1e-12)


def test_compute_power_r_accepts_scipy_spelling():
    two_sided = glm_correlation.compute_power_r(n=12, r=0.4, alternative="two-sided")["power_table"]
    assert two_sided["alternative"].unique().tolist() == ["two.sided"]
    with pytest.raises(ValueError):
        glm_correlation.compute_power_r(n=12, r=0.4, alternative="both")


def test_plot_corrplot_labels_match_rwf(load_data):
    """ggplot2 writes the rounded correlations with as.character() (1, 0.3), at rwf's base_size / 4 mm."""
    data = load_data("insurance")[["age", "bmi", "children", "charges"]]
    p = glm_correlation.plot_corrplot(data.corr(), title="T")
    p.draw()
    # rwf: ggplot_build(plot_corrplot(cor(...)))$data[[2]]$label
    assert p.layers[1].data["label"].tolist() == ["1", "0.11", "1", "0.04", "0.01", "1", "0.3", "0.2", "0.07", "1"]
    assert p.layers[1].geom.aes_params["size"] == pytest.approx(10 / 4 * 72.27 / 25.4)


def test_plot_corrplot_fills_match_rwf(load_data, reference):
    """ggplot2's scale_fill_gradient2 colours, interpolated in Lab (plotnine's would be in RGB)."""
    data = load_data("insurance")[["age", "bmi", "children", "charges"]]
    p = glm_correlation.plot_corrplot(data.corr(), title="T")
    p.draw()
    tiles = p.layers[0].data
    expected = reference("plot_corrplot_fills")
    assert sorted(zip(tiles["x"].astype(int), tiles["y"].astype(int), tiles["fill"].str.upper())) == \
        sorted(zip(expected["x"], expected["y"], expected["fill"]))
