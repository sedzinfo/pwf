import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from plotnine import aes, geom_point, ggplot  # noqa: E402

from pwf.functions_plot import plot_multiplot  # noqa: E402


@pytest.fixture(scope="module")
def plots():
    data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    return [ggplot(data, aes("x", "y")) + geom_point() for _ in range(5)]


def _bounds(page):
    return [np.round(ax.get_position().bounds, 6).tolist() for ax in page.axes]


def test_plot_multiplot_pages_follow_rwf(plots):
    """rwf: ceiling(nplots / max(layout)) pages, a single plot returned as it is."""
    assert plot_multiplot(plots[0]) is plots[0]
    assert [len(page.axes) for page in plot_multiplot(*plots, cols=2)] == [5]
    assert [len(page.axes) for page in plot_multiplot(plotlist=plots, layout=[[1, 2]])] == [2, 2, 1]


def test_plot_multiplot_layout_cells_follow_rwf(plots):
    """A plot whose index fills several cells spans them (R's viewport over the range of rows and columns);
    cells with 0 or NaN stay empty; the grid fills the page, as grid.layout."""
    page = plot_multiplot(plotlist=plots[:3], layout=[[1, 1], [2, 3]])[0]
    assert _bounds(page) == [[0.0, 0.5, 1.0, 0.5], [0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 0.5, 0.5]]
    page = plot_multiplot(plotlist=plots[:2], layout=[[1, np.nan], [0, 2]])[0]
    assert _bounds(page) == [[0.0, 0.5, 0.5, 0.5], [0.5, 0.0, 0.5, 0.5]]
