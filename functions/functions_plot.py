# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_PLOT.R.

Deviations from the R originals, by design:
  - plot_multiplot composites each plot as a rasterized image into a
    matplotlib subplot grid, rather than R's grid::viewport() approach
    of drawing each plot object into a page region. Moving Axes between
    matplotlib Figures isn't reliably supported, so rasterizing (still
    "place a fully-rendered object into a page region", just via a
    pixel buffer instead of a grid viewport) is the robust option.
    Works with both plotnine ggplot objects and matplotlib Figures.
  - plot_hinvert_title_grob is dropped entirely — it's pure grid/gtable
    grob-manipulation plumbing (mirroring a title grob's internal width
    layout) with no meaning outside R's grid graphics system.
  - plot_duplicate_y_axis no longer does grob surgery to graft one
    plot's y-axis onto another's right side. It takes a single
    matplotlib Axes and calls ax.secondary_yaxis("right") — the
    built-in, idiomatic matplotlib mechanism for the same visible
    result (a duplicated y-axis on the right), rather than R's
    (pre-ggplot2-native-sec.axis) hand-rolled workaround.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
##########################################################################################
# MULTIPLOT
##########################################################################################
def _plot_to_array(p, dpi=150):
    """Rasterize a plotnine ggplot or matplotlib Figure to an RGB(A) array."""
    buf = io.BytesIO()
    if hasattr(p, "save"):  # plotnine ggplot
        p.save(buf, format="png", dpi=dpi, verbose=False)
    elif hasattr(p, "savefig"):  # matplotlib Figure
        p.savefig(buf, format="png", dpi=dpi)
    else:
        raise TypeError(f"Unsupported plot type: {type(p)!r}")
    buf.seek(0)
    return mpimg.imread(buf)


def plot_multiplot(*plots, plotlist=None, cols=2, layout=None):
    """
    Arrange multiple plots (plotnine ggplot objects and/or matplotlib
    Figures) in a grid layout, paginated if there are more plots than
    fit in one layout.

    Parameters:
    *plots: Plot objects passed directly.
    plotlist (list, optional): Additional plot objects; combined with
        any passed via *plots.
    cols (int, optional): Number of grid columns. Ignored if `layout`
        is given. Defaults to 2.
    layout (array-like, optional): A 2D array where each cell holds the
        1-based index of the plot to display at that grid position. If
        None (default), a layout is generated from `cols`.

    Returns:
    The single plot directly if only one was given; otherwise a list of
    matplotlib.figure.Figure, one per page.

    Examples:
    >>> from plotnine import ggplot, aes, geom_point
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
    >>> p1 = ggplot(df, aes('x', 'y')) + geom_point()
    >>> p2 = ggplot(df, aes('x', 'y')) + geom_point()
    >>> pages = plot_multiplot(p1, p2, cols=2)
    """
    all_plots = list(plots) + (list(plotlist) if plotlist else [])
    nplots = len(all_plots)
    if nplots == 1:
        return all_plots[0]

    if layout is None:
        nrows = math.ceil(nplots / cols)
        layout = np.arange(1, nrows * cols + 1).reshape(nrows, cols)
    else:
        layout = np.asarray(layout)

    plots_per_page = int(layout.max())
    pages_count = math.ceil(nplots / plots_per_page)
    nrows_layout, ncols_layout = layout.shape

    pages = []
    counter = 0
    for _ in range(pages_count):
        fig, axes = plt.subplots(nrows_layout, ncols_layout,
                                  figsize=(ncols_layout * 5, nrows_layout * 4))
        axes = np.atleast_2d(axes)
        for i in range(1, plots_per_page + 1):
            positions = np.argwhere(layout == i)
            ax = None
            if len(positions) > 0:
                row, col = positions[0]
                ax = axes[row, col]
                ax.axis("off")
            if counter < nplots:
                if ax is not None:
                    ax.imshow(_plot_to_array(all_plots[counter]))
                counter += 1
        pages.append(fig)
    return pages
##########################################################################################
# DUPLICATE Y AXIS
##########################################################################################
def plot_duplicate_y_axis(ax):
    """
    Mirror an Axes' y-axis onto the right side of the plot.

    Parameters:
    ax (matplotlib.axes.Axes): Axes to add a mirrored right-hand y-axis to.

    Returns:
    matplotlib.axes.Axes: The same `ax`.

    Examples:
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> plot_duplicate_y_axis(ax)
    """
    ax.secondary_yaxis("right")
    return ax
##########################################################################################
# REPORT PDF
##########################################################################################
def _to_figure(p, w, h):
    """Get a sized matplotlib Figure from a plotnine ggplot or matplotlib Figure."""
    fig = p.draw() if hasattr(p, "draw") and not hasattr(p, "savefig") else p
    fig.set_size_inches(w, h)
    return fig


def report_pdf(*plots, plotlist=None, file=None, title=None, w=10, h=10, print_plot=True):
    """
    Save one or more plots (plotnine ggplot objects and/or matplotlib
    Figures) to a multi-page PDF, optionally also displaying them.

    Parameters:
    *plots: Plot objects passed directly.
    plotlist (list, optional): Additional plot objects; combined with
        any passed via *plots.
    file (str, optional): Output filename without extension. If None
        (default), no PDF is written.
    title (str, optional): Suffix appended to `file` (with an
        underscore) to form the final filename. Defaults to None.
    w (float, optional): Page width in inches. Defaults to 10.
    h (float, optional): Page height in inches. Defaults to 10.
    print_plot (bool, optional): If True (default), also render the
        plots to the active display.

    Returns:
    None.

    Examples:
    >>> from plotnine import ggplot, aes, geom_point
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
    >>> p1 = ggplot(df, aes('x', 'y')) + geom_point()
    >>> report_pdf(p1, file="report", print_plot=False)
    """
    all_plots = list(plots) + (list(plotlist) if plotlist else [])
    suffix = f"_{title}" if title is not None else ""

    if file is not None:
        with PdfPages(f"{file}{suffix}.pdf") as pdf:
            for p in all_plots:
                fig = _to_figure(p, w, h)
                pdf.savefig(fig)
                if not print_plot:
                    plt.close(fig)

    if print_plot:
        for p in all_plots:
            _to_figure(p, w, h)
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from plotnine import ggplot, aes, geom_point, geom_line, labs
    import pandas as pd

    df = pd.DataFrame({'x': range(10), 'y': [v**2 for v in range(10)]})
    p1 = ggplot(df, aes('x', 'y')) + geom_point() + labs(title="p1")
    p2 = ggplot(df, aes('x', 'y')) + geom_line() + labs(title="p2")
    p3 = ggplot(df, aes('x', 'y')) + geom_point() + geom_line() + labs(title="p3")
    p4 = ggplot(df, aes('y', 'x')) + geom_point() + labs(title="p4")

    print("=" * 80, "\nplot_multiplot\n", "=" * 80, sep="")
    single = plot_multiplot(p1)
    print("single plot passthrough:", type(single))

    pages = plot_multiplot(p1, p2, p3, p4, cols=2)
    print("4 plots, cols=2 -> pages:", len(pages))
    pages[0].savefig("multiplot_cols2.png")
    print("saved multiplot_cols2.png")

    pages2 = plot_multiplot(p1, p2, p3, cols=1)
    print("3 plots, cols=1 -> pages:", len(pages2))

    print("\n" + "=" * 80, "\nplot_duplicate_y_axis\n", "=" * 80, sep="")
    fig, ax = plt.subplots()
    ax.plot(df['x'], df['y'])
    plot_duplicate_y_axis(ax)
    print("secondary axes on figure:", len(fig.axes))
    fig.savefig("duplicate_y_axis.png")
    print("saved duplicate_y_axis.png")

    print("\n" + "=" * 80, "\nreport_pdf\n", "=" * 80, sep="")
    report_pdf(p1, p2, file="report_example", print_plot=False)
    import os
    print("report_example.pdf written:", os.path.exists("report_example.pdf"),
          "size:", os.path.getsize("report_example.pdf"), "bytes")
