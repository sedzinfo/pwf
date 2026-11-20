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


def test_plot_duplicate_y_axis_copies_p2s_axis():
    """rwf pastes p2's left y axis (ticks, labels, title) to the right of p1's panel, in a column of its own
    before the legend: the copy keeps the heights p2's ticks have in p2's panel."""
    import io

    import matplotlib.pyplot as plt

    from pwf.functions_plot import plot_duplicate_y_axis
    data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9], "z": [100, 300, 500], "g": ["a", "b", "a"]})
    p1 = ggplot(data, aes("x", "y")) + geom_point()
    figure = plot_duplicate_y_axis(p1, p1)
    panel = figure.axes[0]
    mirror = panel.child_axes[0]
    labels = [t.get_text() for t in panel.get_yticklabels()]
    assert all(labels) and [t.get_text() for t in mirror.get_yticklabels()] == labels
    assert mirror.get_ylabel() == "y"
    np.testing.assert_allclose(mirror.get_yticks(), panel.get_yticks())
    assert "figure" not in vars(p1)  # drawn as a copy: p1 is left as it was
    p2 = ggplot(data, aes("x", "z")) + geom_point()
    figure = plot_duplicate_y_axis(p1, p2)
    panel, mirror = figure.axes[0], figure.axes[0].child_axes
    assert len(mirror) == 1 and mirror[0].get_ylabel() == "z"
    figure.savefig(io.BytesIO(), format="png")  # plotnine's figures have the base canvas: laid out when saved
    source = p2.draw().axes[0]
    low, high = source.get_ylim()
    np.testing.assert_allclose(mirror[0].get_ylim(), (low, high))  # p2's range over p1's panel height
    np.testing.assert_allclose(mirror[0].get_yticks(), [t for t in source.get_yticks() if low <= t <= high])
    # with a legend on the right, the axis comes between the panel and the legend
    with_legend = ggplot(data, aes("x", "y", colour="g")) + geom_point()
    figure = plot_duplicate_y_axis(with_legend)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    legends = [artist for artist in figure.artists if type(artist).__name__.endswith("AnchoredOffsetbox")]
    assert legends  # plotnine's legend box
    axis_right = figure.axes[0].child_axes[0].get_tightbbox(renderer).x1
    assert all(axis_right <= legend.get_window_extent(renderer).x0 for legend in legends)
    fig, ax = plt.subplots()  # the earlier pwf form: an Axes
    assert plot_duplicate_y_axis(ax) is ax and len(ax.child_axes) == 1
