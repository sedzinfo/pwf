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


def test_outlier_summary_matches_rwf(reference):
    # R's sd() (n - 1) with na.rm = TRUE, and paste(round(...), "%"): "4.29 %", "5 %"
    vector = reference("outlier_vector")["v"].to_numpy()
    result = pd.concat([ea.outlier_summary(vector),
                        ea.outlier_summary(np.r_[np.arange(1, 20), 100])], ignore_index=True)
    assert_matches_reference(result, reference("outlier_summary", dtype=str))


def test_remove_outliers_matches_rwf(reference):
    vector = reference("outlier_vector")["v"].to_numpy()
    expected = reference("remove_outliers")
    np.testing.assert_array_equal(ea.remove_outliers(vector), expected["default"].to_numpy())
    np.testing.assert_array_equal(ea.remove_outliers(vector, probs=(0.1, 0.9), type=6),
                                  expected["type6"].to_numpy())
    with pytest.raises(ValueError):
        ea.remove_outliers(vector, na_rm=False)


def test_explore_plot_captions_match_rwf(load_data, reference):
    # one plot per numeric column, captioned as rwf's paste0() prints R's round(): "SD=6.1", "Median=39"
    data = load_data("insurance")[["age", "sex", "bmi", "charges"]].copy()
    data.loc[[2, 8], "bmi"] = np.nan
    captions = {f"histogram_{n}": p.labels.get("caption", None) for n, p in ea.plot_histogram(data).items()}
    captions.update({f"qq_{n}": p.labels.get("caption", None) for n, p in ea.plot_qq(data).items()})
    captions["boxplot"] = ea.plot_boxplot(data).labels.get("caption", None)
    expected = reference("explore_plot_captions", dtype=str)
    assert list(captions) == expected["plot"].tolist()
    assert list(captions.values()) == expected["caption"].tolist()


def test_plot_qq_points_match_rwf(load_data, reference):
    """ggplot2's stat_qq puts the sorted values at qnorm(ppoints(n)) (plotnine's default positions differ)."""
    p = ea.plot_qq(load_data("insurance")[["bmi"]])["bmi"]
    p.draw()
    points = p.layers[0].data.sort_values("x")
    expected = reference("plot_qq_bmi_points")
    np.testing.assert_allclose(points["x"], expected["x"], rtol=1e-13)
    np.testing.assert_allclose(points["y"], expected["y"], rtol=1e-13)


@pytest.mark.parametrize("breaks", ["Sturges", 5])
def test_plot_normality_diagnostics_matches_rwf(reference, load_data, breaks):
    """rwf's four panels are R's hist(), density(), boxplot() and qqnorm() with qqline(): the same bars,
    curve, box, points and line, and rwf's caption."""
    data = load_data("insurance")
    data.loc[[1, 4], "bmi"] = np.nan  # make_reference.R's plots_insurance
    figure = ea.plot_normality_diagnostics(data[["bmi"]], breaks=None if breaks == "Sturges" else breaks)["bmi"]
    histogram, density, box, qq = figure.axes
    bars = reference(f"plot_normality_hist_{breaks}")
    np.testing.assert_allclose([patch.get_x() for patch in histogram.patches], bars["left"], rtol=1e-12)
    np.testing.assert_allclose([patch.get_x() + patch.get_width() for patch in histogram.patches], bars["right"],
                               rtol=1e-12)
    np.testing.assert_array_equal([patch.get_height() for patch in histogram.patches], bars["count"])
    curve = reference("plot_normality_density")
    np.testing.assert_allclose(density.lines[0].get_xdata(), curve["x"], rtol=1e-12)
    np.testing.assert_allclose(density.lines[0].get_ydata(), curve["y"], rtol=1e-10, atol=1e-15)
    assert density.get_xlabel() == f"N = 1336   Bandwidth = {curve['bw'].iloc[0]:.4g}"
    parts = reference("plot_normality_box")
    fliers = [line for line in box.lines if line.get_marker() == "o"]
    drawn = {float(v) for line in box.lines if line not in fliers for v in line.get_ydata()}
    np.testing.assert_allclose(sorted(drawn), sorted(set(parts.loc[parts["part"] == "stats", "value"])), rtol=1e-12)
    np.testing.assert_allclose(sorted(fliers[0].get_ydata()), sorted(parts.loc[parts["part"] == "out", "value"]),
                               rtol=1e-12)
    points = reference("plot_normality_qq")
    got = np.array(sorted(map(tuple, qq.collections[0].get_offsets())))
    np.testing.assert_allclose(got, np.array(sorted(zip(points["x"], points["y"]))), rtol=1e-12)
    line = qq.lines[0]
    np.testing.assert_allclose(line.get_slope(), points["slope"].iloc[0], rtol=1e-12)
    np.testing.assert_allclose(line.get_xy1()[1] - line.get_slope() * line.get_xy1()[0], points["intercept"].iloc[0],
                               rtol=1e-12)
    caption = reference("plot_normality_caption")["caption"].iloc[0]
    assert caption.lstrip("\n") in [text.get_text() for text in figure.texts]
