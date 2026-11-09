import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from conftest import assert_matches_reference  # noqa: E402
from pwf import explore_assumptions as ea  # noqa: E402


@pytest.fixture
def df_blood_pressure_na(df_blood_pressure):
    """Same NAs as df_blood_pressure_na in tests/reference/make_reference.R (1-based rows there)."""
    df = df_blood_pressure.copy()
    df.loc[[2, 49, 98], "bp_before"] = np.nan
    df.loc[[9, 69], "agegrp"] = np.nan
    return df


@pytest.mark.parametrize("method", ["mean", "median", "boxplot"])
def test_plot_outlier(df_blood_pressure_na, reference, method):
    plots = ea.plot_outlier(df=df_blood_pressure_na[["bp_before", "bp_after"]], method=method, title="bp")
    expected = reference(f"plot_outlier_{method}", dtype={"id": str})
    assert_matches_reference(plots["bp_before"].data, expected)
    labels = reference(f"plot_outlier_{method}_labels", keep_default_na=False)
    assert list(plots) == labels["name"].tolist()
    for (name, p), (_, row) in zip(plots.items(), labels.iterrows()):
        assert p.labels.x == row["x"]
        assert p.labels.y == row["y"]
        assert p.labels.caption == row["caption"]
        hlines = [float(layer.geom.data["yintercept"].iloc[0]) for layer in p.layers
                  if type(layer.geom).__name__ == "geom_hline"]
        np.testing.assert_allclose([float(v) for v in hlines], [float(v) for v in row["hlines"].split(";")],
                                   rtol=1e-12)


def test_plot_outlier_draws(df_blood_pressure_na, tmp_path):
    p = ea.plot_outlier(df=df_blood_pressure_na["bp_before"], method="mean")["bp_before"]
    p.save(tmp_path / "outlier.png", width=5, height=8, verbose=False)


@pytest.fixture
def df_normality(load_data):
    """Same frame as df_normality in tests/reference/make_reference.R."""
    insurance, bp = load_data("insurance"), load_data("blood_pressure")
    n = len(insurance)
    return pd.DataFrame({
        "age": insurance["age"], "bmi": insurance["bmi"], "charges": insurance["charges"],
        "bp": list(bp["bp_before"]) + [np.nan] * (n - len(bp)),
        "few": list(range(1, 6)) + [np.nan] * (n - 5),
        "constant": 1,
    })


def test_report_normality_tests(df_normality, reference, capsys):
    result = ea.report_normality_tests(df=df_normality)
    expected = reference("report_normality_tests", keep_default_na=False, na_values=["NA"],
                         dtype={"alternative": str})
    assert_matches_reference(result, expected)
    printed = capsys.readouterr().out
    assert "OUT OF BOUNDS SAMPLE SIZE FOR few" in printed
    assert "OUT OF BOUNDS SAMPLE SIZE FOR constant" in printed


def test_report_normality_tests_writes_files(df_normality, tmp_path):
    ea.report_normality_tests(df=df_normality[["bmi"]], file=str(tmp_path / "normality"))
    assert (tmp_path / "normality.log").exists()
    assert pd.read_excel(tmp_path / "normality.xlsx").shape[0] == 8


def test_kolmogorov_limit_matches_r_series():
    """Values of R's stats:::pkolmogorov_two_asymp(x, 1, lower.tail = FALSE)."""
    r_values = {0.5: 0.963945243664875, 0.8: 0.54414250308288703, 1.0: 0.26999967167737998,
                1.35: 0.052241888564693299, 2.0: 0.00067092525577969501, 5.0: 3.8574996959278399e-22}
    for x, expected in r_values.items():
        assert ea._kolmogorov_limit_upper(x) == pytest.approx(expected, rel=1e-13)


def test_kolmogorov_exact_small_sample():
    x = np.array([0.61, -1.2, 0.33, 2.1, -0.4, 0.05, 1.7, -0.9, 0.12, -2.3])
    ks = ea._kolmogorov_smirnov(x)
    assert ks["method"] == "Exact one-sample Kolmogorov-Smirnov test"
    expected = stats.kstest(x, lambda v: stats.norm.cdf(v, loc=x.mean(), scale=x.std(ddof=1)), method="exact")
    assert ks["statistic"] == pytest.approx(expected.statistic)
    assert ks["p"] == pytest.approx(expected.pvalue)
