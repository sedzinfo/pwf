import numpy as np
import pytest

from pwf._rcompat import r_boxplot_stats, r_density, r_hist, r_ppoints, r_pretty


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
