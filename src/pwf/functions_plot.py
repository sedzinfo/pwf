# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_PLOT.R.

Deviations from the R originals, by design:
  - plot_multiplot composites each plot as a rasterized image into a
    matplotlib grid (a plot spans the rows and columns its index fills,
    as R's viewports), rather than R's grid::viewport() approach
    of drawing each plot object into a page region. Moving Axes between
    matplotlib Figures isn't reliably supported, so rasterizing (still
    "place a fully-rendered object into a page region", just via a
    pixel buffer instead of a grid viewport) is the robust option.
    Works with both plotnine ggplot objects and matplotlib Figures.
  - plot_hinvert_title_grob is dropped entirely — it's pure grid/gtable
    grob-manipulation plumbing (mirroring a title grob's internal width
    layout) with no meaning outside R's grid graphics system.
  - plot_duplicate_y_axis(p1, p2) takes rwf's two plots, but instead of
    grafting p2's y-axis grobs onto p1's right side it draws p1 and adds
    a matplotlib secondary y axis carrying p2's ticks, labels and title,
    and returns the figure. An Axes is still taken as p1, as the earlier
    pwf did.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import copy
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
##########################################################################################
# MULTIPLOT
##########################################################################################
def _plot_to_array(p, dpi=150, size=None):
    """
    Rasterize a plotnine ggplot or composition, or a matplotlib Figure, to an RGB(A) array, drawn at size
    (width, height) in inches when given (R draws each plot into its viewport's size).
    """
    from plotnine import ggplot
    buf = io.BytesIO()
    if isinstance(p, ggplot):
        extra = {"width": size[0], "height": size[1]} if size is not None else {}
        p.save(buf, format="png", dpi=dpi, verbose=False, **extra)
    elif hasattr(p, "draw") and hasattr(p, "save"):  # plotnine composition
        figure = p.draw()
        if size is not None:
            figure.set_size_inches(*size)
        figure.savefig(buf, format="png", dpi=dpi)
    elif hasattr(p, "savefig"):  # matplotlib Figure
        old_size = p.get_size_inches()
        if size is not None:
            p.set_size_inches(*size)
        p.savefig(buf, format="png", dpi=dpi)
        p.set_size_inches(*old_size)
    else:
        raise TypeError(f"Unsupported plot type: {type(p)!r}")
    buf.seek(0)
    return mpimg.imread(buf)


def plot_multiplot(*plots, plotlist=None, cols=2, layout=None):
    """
    Arrange multiple plots (plotnine ggplot objects and/or matplotlib
    Figures) in a grid layout, paginated if there are more plots than
    fit in one layout (port of rwf::plot_multiplot).

    Parameters:
    *plots: Plot objects passed directly.
    plotlist (list, optional): Additional plot objects; combined with
        any passed via *plots.
    cols (int, optional): Number of grid columns. Ignored if `layout`
        is given. Defaults to 2.
    layout (array-like, optional): A 2D array where each cell holds the
        1-based index of the plot to display at that grid position. As in
        rwf, a plot whose index fills several cells spans their rows and
        columns, and cells holding 0 or NaN stay empty. If None (default),
        a layout is generated from `cols`, filled by rows.

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
    >>> pages = plot_multiplot(p1, p2, p1, layout=[[1, 1], [2, 3]])  # the first plot spans the top row
    """
    all_plots = list(plots) + (list(plotlist) if plotlist else [])
    nplots = len(all_plots)
    if layout is None:
        nrows = math.ceil(nplots / cols)
        layout = np.arange(1, nrows * cols + 1).reshape(nrows, cols)
    else:
        layout = np.asarray(layout, dtype=float)
    if nplots == 1:
        return all_plots[0]

    plots_per_page = int(np.nanmax(layout))
    pages_count = math.ceil(nplots / plots_per_page)
    nrows_layout, ncols_layout = layout.shape

    pages = []
    counter = 0
    for _ in range(pages_count):
        cell_width, cell_height = 5, 4
        fig = plt.figure(figsize=(ncols_layout * cell_width, nrows_layout * cell_height))
        # grid.layout(nrow(layout), ncol(layout)): equal cells filling the page
        grid = fig.add_gridspec(nrows_layout, ncols_layout, left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        for i in range(1, plots_per_page + 1):
            positions = np.argwhere(layout == i)
            if counter < nplots and len(positions) > 0:
                # viewport(layout.pos.row = rows, layout.pos.col = cols) covers the range of the rows and columns
                rows, columns = positions[:, 0], positions[:, 1]
                height = (rows.max() - rows.min() + 1) * cell_height
                width = (columns.max() - columns.min() + 1) * cell_width
                ax = fig.add_subplot(grid[rows.min():rows.max() + 1, columns.min():columns.max() + 1])
                ax.axis("off")
                ax.imshow(_plot_to_array(all_plots[counter], size=(width, height)), aspect="auto")
            counter += 1
        pages.append(fig)
    return pages
##########################################################################################
# DUPLICATE Y AXIS
##########################################################################################
def plot_duplicate_y_axis(p1, p2=None):
    """
    Draw p1 with p2's left y axis (its ticks, tick labels and title) copied
    to its right, as rwf::plot_duplicate_y_axis, which pastes p2's axis
    grobs next to p1's panel: p2's ticks sit at the heights they have in
    p2's panel.

    Parameters:
    p1 (plotnine.ggplot): The plot, with one panel (as rwf's). A
        matplotlib Axes is also taken, as the earlier pwf: its own y axis
        is mirrored on its right and the Axes returned.
    p2 (plotnine.ggplot, optional): The plot whose y axis is copied;
        typically p1 itself, the default.

    Returns:
    matplotlib.figure.Figure: p1 drawn anew with the second axis, which
    gets the room p2's axis takes, between the panel and a legend on the
    right, as rwf's added gtable columns (rwf draws it on the current
    device and returns NULL). p1 and p2 are left as they are.

    Examples:
    >>> from plotnine import ggplot, aes, geom_line
    >>> p1 = ggplot(df, aes("x", "y")) + geom_line()
    >>> plot_duplicate_y_axis(p1, p1)
    """
    if isinstance(p1, plt.Axes):
        p1.secondary_yaxis("right")
        return p1
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from plotnine import theme

    p2 = p1 if p2 is None else p2
    # p2's axis: its ticks, labels and title, and the width they take beside its panel
    drawn_p2 = _fresh_plot(p2)
    reference = drawn_p2.draw()
    source = _one_panel(reference, "p2")
    canvas = FigureCanvasAgg(reference)  # plotnine's figures have matplotlib's base canvas, which can't measure
    canvas.draw()
    renderer = canvas.get_renderer()
    title = drawn_p2.labels.get("y", None) or ""  # plotnine fills the labels when it draws
    title_text = next((text for text in reference.texts if text.get_text() == title), None)
    pieces = [t for t in source.get_yticklabels() if t.get_text()] + ([title_text] if title_text else [])
    width = (source.get_window_extent(renderer).x0 - min(p.get_window_extent(renderer).x0 for p in pieces)
             if pieces else 0.0) / reference.bbox.width  # a fraction of the figure's width
    # p1 drawn anew with that room on its right, and between its panel and a legend there
    measured = _fresh_plot(p1)
    measured.draw()
    margin, spacing = measured.theme.getp("plot_margin_right"), measured.theme.getp("legend_box_spacing")
    plt.close(measured.figure)
    final = _fresh_plot(p1) + theme(plot_margin_right=margin + width, legend_box_spacing=spacing + width)
    figure = final.draw()
    panel = _one_panel(figure, "p1")
    (low1, high1), (low2, high2) = panel.get_ylim(), source.get_ylim()

    def forward(y):
        return low2 + (np.asarray(y, dtype=float) - low1) * (high2 - low2) / (high1 - low1)

    def inverse(y):
        return low1 + (np.asarray(y, dtype=float) - low2) * (high1 - low1) / (high2 - low2)

    second = panel.secondary_yaxis("right", functions=(forward, inverse))
    ticks = [tick for tick in source.get_yticks() if low2 <= tick <= high2]
    labels = {tick: label.get_text() for tick, label in zip(source.get_yticks(), source.get_yticklabels())}
    second.set_yticks(ticks, [labels[tick] for tick in ticks])
    # p2's styling of its axis: the spine (none in theme_gray), the ticks and their labels
    left, right = source.spines["left"], second.spines["right"]
    right.set_visible(left.get_visible())
    right.set_edgecolor(left.get_edgecolor())
    right.set_linewidth(left.get_linewidth())
    major = source.yaxis.get_major_ticks()
    if major:
        line = major[0].tick1line
        second.tick_params(axis="y", color=line.get_color(), length=line.get_markersize(),
                           width=line.get_markeredgewidth(), pad=major[0].get_pad())
    style = source.get_yticklabels()[0] if source.get_yticklabels() else None
    if style is not None:
        second.tick_params(axis="y", labelsize=style.get_fontsize(), labelcolor=style.get_color())
    # plotnine draws the axis titles as figure texts: p2's y title, with the size and colour it has there
    second.set_ylabel(title, **({} if title_text is None else {"fontsize": title_text.get_fontsize(),
                                                               "color": title_text.get_color()}))
    plt.close(reference)
    return figure


def _fresh_plot(plot):
    """A copy of a plotnine plot that draws on a figure of its own (plotnine draws a plot into the figure it
    kept from an earlier draw)."""
    copied = copy.deepcopy(plot)
    for key in ("figure", "_gridspec", "axs"):
        copied.__dict__.pop(key, None)
    return copied


def _one_panel(figure, name):
    """The panel of a drawn one-panel plotnine figure (rwf's plot_duplicate_y_axis takes a layout's 'panel')."""
    panels = [axes for axes in figure.axes if axes.get_label() != "<colorbar>"]
    if len(panels) != 1:
        raise ValueError(f"{name} must have one panel, as rwf's (its layout has one 'panel')")
    return panels[0]
##########################################################################################
# REPORT PDF
##########################################################################################
def _to_figure(p, w, h):
    """Get a sized matplotlib Figure from a plotnine ggplot, a seaborn grid (e.g. JointGrid) or a Figure."""
    if hasattr(p, "savefig") and hasattr(p, "set_size_inches"):
        fig = p
    elif hasattr(p, "figure") and hasattr(p.figure, "set_size_inches"):
        fig = p.figure  # seaborn JointGrid / FacetGrid / PairGrid
    else:
        fig = p.draw()
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
    fig = plot_duplicate_y_axis(p1, p1)
    print("secondary y-axis attached:", len(fig.axes[0].child_axes) == 1)
    fig.savefig("duplicate_y_axis.png")
    print("saved duplicate_y_axis.png")

    print("\n" + "=" * 80, "\nreport_pdf\n", "=" * 80, sep="")
    report_pdf(p1, p2, file="report_example", print_plot=False)
    import os
    print("report_example.pdf written:", os.path.exists("report_example.pdf"),
          "size:", os.path.getsize("report_example.pdf"), "bytes")
