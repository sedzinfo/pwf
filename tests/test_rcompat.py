import numpy as np
import pytest

from pwf._rcompat import (r_boxplot_stats, r_curve, r_density, r_hist, r_layout_as_tree, r_layout_in_circle,
                          r_lowess, r_merge_rows, r_ppoints, r_pretty, r_qqnorm, r_quantile7)


def test_density_matches_r(reference, load_data):
    """R's density(): bw.nrd0, linear binning and the FFT convolution, within rounding of the FFT."""
    insurance = load_data("insurance")
    for name, column, kwargs in (("r_density_bmi", "bmi", {}),
                                 ("r_density_age_from_to", "age", {"n": 100, "from_": 10, "to": 70})):
        expected = reference(name)
        x, y, bw = r_density(insurance[column].to_numpy(dtype=float), **kwargs)
        np.testing.assert_allclose(bw, expected["bw"].iloc[0], rtol=1e-14)
        np.testing.assert_allclose(x, expected["x"], rtol=1e-14)
        np.testing.assert_allclose(y, expected["y"], rtol=1e-12, atol=1e-15)


def test_pretty_matches_r(reference):
    """R's pretty(), including tiny and zero ranges and ranges ending at 0 that need min.n divisions."""
    expected = reference("r_pretty")
    for case, rows in expected.groupby("case", sort=False):
        first = rows.iloc[0]
        result = r_pretty([first["lo"], first["hi"]], n=int(first["n"]), min_n=int(first["min_n"]))
        np.testing.assert_allclose(result, rows["value"], rtol=1e-14, atol=0, err_msg=case)


@pytest.mark.parametrize("variable", ["age", "bmi", "charges", "children"])
def test_hist_matches_r(reference, load_data, variable):
    """R's hist(): pretty() breaks on Sturges', Scott's, Freedman-Diaconis' or a given number of classes, right-closed."""
    expected = reference("r_hist")
    values = load_data("insurance")[variable].to_numpy(dtype=float)
    for breaks, rows in expected[expected["variable"] == variable].groupby("breaks", sort=False):
        edges, counts = r_hist(values, int(breaks) if breaks.isdigit() else breaks)
        np.testing.assert_allclose(edges[:-1], rows["left"], rtol=1e-14, err_msg=breaks)
        np.testing.assert_allclose(edges[1:], rows["right"], rtol=1e-14, err_msg=breaks)
        assert counts.tolist() == rows["count"].tolist(), breaks


def test_boxplot_stats_match_r(reference, load_data):
    """R's boxplot.stats(): Tukey's hinges (fivenum), whiskers within 1.5 IQR and the notch."""
    expected = reference("r_boxplot_stats")
    insurance = load_data("insurance")
    for variable, rows in expected.groupby("variable", sort=False):
        box = r_boxplot_stats(insurance[variable])
        result = np.r_[box["stats"], box["conf"], len(box["out"])]
        np.testing.assert_allclose(result, rows["value"], rtol=1e-14, err_msg=variable)


def test_ppoints_follow_r():
    np.testing.assert_allclose(r_ppoints(5), (np.arange(1, 6) - 3 / 8) / (5 + 1 - 3 / 4))
    np.testing.assert_allclose(r_ppoints(11), (np.arange(1, 12) - 0.5) / 11)


def test_layout_as_tree_matches_igraph(reference):
    """igraph's Reingold-Tilford layout_as_tree(root = roots, flip.y = TRUE): several roots, cycles, unreachable vertices."""
    expected = reference("r_layout_as_tree")
    for case, rows in expected.groupby("case", sort=False):
        first = rows.iloc[0]
        # a graph without edges is written as ">" (paste0 of empty vectors)
        edges = [tuple(map(int, e.split(">"))) for e in str(first["edges"]).split(";") if e != ">"]
        roots = [int(r) for r in str(first["roots"]).split(";")]
        layout = r_layout_as_tree(int(first["n"]), edges, roots)
        np.testing.assert_allclose(layout[:, 0], rows["x"], rtol=0, atol=1e-12, err_msg=str(case))
        np.testing.assert_allclose(layout[:, 1], rows["y"], rtol=0, atol=1e-12, err_msg=str(case))


def test_layout_in_circle_matches_igraph(reference):
    expected = reference("r_layout_in_circle")
    for n, rows in expected.groupby("n"):
        np.testing.assert_allclose(r_layout_in_circle(int(n)), rows[["x", "y"]].to_numpy(), rtol=0, atol=1e-15)


def test_curve_matches_grid(reference):
    """grid's curveGrob (geom_curve): calcControlPoints and the open X-spline, point for point (inches, 1200 ppi steps)."""
    expected = reference("r_curve_points")
    for case, rows in expected.groupby("case", sort=False):
        first = rows.iloc[0]
        x, y = r_curve(first["x1"], first["y1"], first["x2"], first["y2"], curvature=first["curvature"], angle=90, ncp=5)
        assert len(x) == len(rows), case
        np.testing.assert_allclose(x, rows["x"], rtol=0, atol=1e-12, err_msg=str(case))
        np.testing.assert_allclose(y, rows["y"], rtol=0, atol=1e-12, err_msg=str(case))


def test_lowess_matches_r(reference):
    """R's lowess (clowess): sorted x, ties fitted once, delta skipping and robustness iterations."""
    expected = reference("r_lowess")
    for case, rows in expected.groupby("case", sort=False):
        x, y = r_lowess(rows["x"].to_numpy(), rows["y"].to_numpy(), f=rows["f"].iloc[0], iter=int(rows["iter"].iloc[0]))
        np.testing.assert_allclose(x, rows["lowess_x"], rtol=1e-15, err_msg=str(case))
        np.testing.assert_allclose(y, rows["lowess_y"], rtol=1e-13, atol=1e-14, err_msg=str(case))


def test_qqnorm_and_quantile7_follow_r():
    """qqnorm(): ranks with ties in order, NA kept in place; quantile(type = 7): (1 - h) * x[lo] + h * x[hi]."""
    x, y = r_qqnorm([3.0, 1.0, np.nan, 1.0, 2.0])
    # R: qqnorm(c(3, 1, NA, 1, 2), plot.it = FALSE)$x
    np.testing.assert_allclose(x[[1, 3, 4, 0]], [-1.0491313979639711, -0.29930691046566704, 0.29930691046566704,
                                                 1.0491313979639711], rtol=1e-14)
    assert np.isnan(x[2]) and np.isnan(y[2])
    np.testing.assert_allclose(r_quantile7([4.0, 1.0, 3.0, 2.0, np.nan], [0.25, 0.75]), [1.75, 3.25], rtol=0)


def test_merge_rows_follow_r():
    """merge(x, y, by.x = "key", by.y = "name"): R's Shell sort leaves the tied "a" rows as 7, 2 (not 2, 7)."""
    x = ["b", "E3", "a", "N1", "b", "E3", "C", "a", "N1", "e3", "B"]
    y = ["N1", "a", "E3", "b", "C", "e3"]
    rows, names = r_merge_rows(x, y)
    assert rows == [7, 2, 0, 4, 6, 9, 1, 5, 3, 8]  # R: m$row
    assert names == [1, 1, 3, 3, 4, 5, 2, 2, 0, 0]  # R: m$yrow


def _hex(colour):
    """An R or plotnine colour as "#RRGGBB" (R gives missing values the name grey50)."""
    from pwf._rcompat import _r_rgb
    return "#{:02X}{:02X}{:02X}".format(*_r_rgb(colour))


def test_r_scale_fill_gradient2_matches_ggplot2(reference):
    """ggplot2's colours, interpolated in Lab, values outside the limits censored before the mid-point
    rescaling (grey50, as missing values), and R's colour names (green, gray and purple are X11's)."""
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from plotnine import aes, geom_tile, ggplot

    from pwf._rcompat import r_scale_fill_gradient2
    cases = {"limits": dict(midpoint=0, limits=(-1, 1)), "asymmetric": dict(midpoint=0, limits=(-0.5, 1)),
             "colours": dict(low="#5E912C", mid="white", high="#5F2C91", midpoint=0),
             "names": dict(low="green", mid="gray", high="purple", midpoint=1)}
    expected = reference("scale_fill_gradient2_fills")
    for case, arguments in cases.items():
        rows = expected[expected["case"] == case]
        data = pd.DataFrame({"x": np.arange(1, len(rows) + 1), "value": rows["value"].to_numpy()})
        p = ggplot(data, aes("x", 1, fill="value")) + geom_tile() + r_scale_fill_gradient2(**arguments)
        p.draw()
        assert [_hex(f) for f in p.layers[0].data["fill"]] == [_hex(f) for f in rows["fill"]], case
    assert _hex(r_scale_fill_gradient2(mid="grey50").palette([0.5])[0]) == "#7F7F7F"  # R's name, not matplotlib's
    with pytest.raises(ValueError, match="unknown colour"):
        r_scale_fill_gradient2(low="steelblue3")
