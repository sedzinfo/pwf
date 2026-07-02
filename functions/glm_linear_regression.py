# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LINEAR_REGRESSION.R — plot_scatterplot only, for
now. R's version lives in this same file alongside a larger set of
regression-modeling functions not yet ported; this file is scoped to just
the one function needed as a dependency of glm_correlation.py's
report_correlation(), and is meant to be extended later.

Deviations from the R original, by design:
  - ggExtra::ggMarginal (scatter + marginal histograms) has no plotnine
    equivalent — plotnine doesn't support marginal plots natively. Uses
    seaborn's jointplot(kind="reg") instead, which natively combines a
    scatter plot, OLS regression line with CI band, and marginal
    histograms in one call. Returns a seaborn JointGrid per pair
    (with a real matplotlib Figure at .figure), not a plotnine ggplot
    like the rest of this project's plotting functions.
  - Dropped the generic `method`/`formula` parameters (R forwards these
    to geom_smooth, supporting arbitrary smoothers/formulas). This
    always fits a simple OLS y~x line and always shows the full
    r/slope/angle caption — which is exactly what R itself does when
    the default y~x formula is used anyway.
  - The future/future.apply parallel-plotting branch is not replicated,
    for the same reason as in glm_anova_plot.py and functions_plot.py:
    plot objects don't reliably serialize across process boundaries in
    Python. Runs sequentially, matching R's own sequential fallback path.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .functions import comparison_combinations
    from .functions_strings import str_aes
    from .functions_mathematical import rad2deg
except ImportError:
    from functions import comparison_combinations
    from functions_strings import str_aes
    from functions_mathematical import rad2deg
##########################################################################################
# SCATTERPLOT
##########################################################################################
def plot_scatterplot(df, base_size=10, coord_equal=False, all_orders=False, title="",
                      combinations=None, str_aes_labels=True):
    """
    Scatter plot with an OLS regression line, CI band, and marginal
    histograms for variable pairs in a data frame. Each plot's caption
    reports pairwise n, Pearson r, explained variance, the regression
    equation, and the regression angle.

    Parameters:
    df (pandas.DataFrame): Numeric variables. When df has exactly 2
        columns and combinations=None, only that one (x, y) pair is used.
    base_size (int, optional): Base font size / figure height driver. Defaults to 10.
    coord_equal (bool, optional): If True, both axes share the same
        scale and limits. Defaults to False.
    all_orders (bool, optional): If True, both (X, Y) and (Y, X)
        orderings are plotted for each pair. Ignored if `combinations`
        is given. Defaults to False.
    title (str, optional): Plot title applied to every panel. Defaults to "".
    combinations (pandas.DataFrame, optional): Two-column frame of
        (x, y) column-name pairs to plot. If None (default), all pairs
        are derived automatically via comparison_combinations.
    str_aes_labels (bool, optional): If True (default), axis labels are
        cleaned via str_aes().

    Returns:
    dict: {"x_y": seaborn.axisgrid.JointGrid or None}, one per pair.
    None when a pair has fewer than 2 complete-case observations.

    Examples:
    >>> plot_scatterplot(df=df_insurance[["age", "bmi", "charges"]], coord_equal=True)
    """
    if combinations is None:
        combinations = comparison_combinations(df, all_orders=all_orders)
        combinations.columns = ['x', 'y']
    combinations = combinations[combinations.iloc[:, 0] != combinations.iloc[:, 1]]

    plots = {}
    for _, row in combinations.iterrows():
        xcol, ycol = row.iloc[0], row.iloc[1]
        key = f"{xcol}_{ycol}"
        tempdata = df[[xcol, ycol]].dropna()
        if len(tempdata) < 2:
            plots[key] = None
            continue

        x = tempdata[xcol].to_numpy(dtype=float)
        y = tempdata[ycol].to_numpy(dtype=float)
        pearsonr = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        degrees = rad2deg(np.arctan(slope))
        if degrees > 180:
            degrees = 360 - degrees

        note = (f"Pairwise n = {len(tempdata)}\n"
                f"Pearson r = {round(pearsonr, 4)}\n"
                f"Explained Variance = {round(pearsonr ** 2, 4) * 100}%\n"
                f"y = {round(slope, 4)}x + {round(intercept, 4)}\n"
                f"Angle = {round(degrees, 2)}")

        xlabel = str_aes(xcol) if str_aes_labels else xcol
        ylabel = str_aes(ycol) if str_aes_labels else ycol

        g = sns.jointplot(data=tempdata, x=xcol, y=ycol, kind="reg", height=max(base_size / 1.2, 4),
                           joint_kws={'scatter_kws': {'alpha': 0.3}})
        g.ax_joint.set_xlabel(xlabel)
        g.ax_joint.set_ylabel(ylabel)
        g.figure.suptitle(title)
        g.ax_joint.annotate(note, xy=(0.02, -0.3), xycoords='axes fraction',
                             fontsize=max(base_size * 0.6, 6), va='top')

        if x.min() <= 0 <= x.max():
            g.ax_joint.axvline(0, alpha=.5, color='gray')
        if y.min() <= 0 <= y.max():
            g.ax_joint.axhline(0, alpha=.5, color='gray')

        if coord_equal:
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            g.ax_joint.set_xlim(lo, hi)
            g.ax_joint.set_ylim(lo, hi)
            g.ax_joint.set_aspect('equal')

        plots[key] = g

    return plots
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os

    df_insurance = pd.read_csv("data/insurance.csv") if os.path.exists("data/insurance.csv") \
        else pd.read_csv("../data/insurance.csv")

    print("=" * 80, "\nplot_scatterplot\n", "=" * 80, sep="")
    plots = plot_scatterplot(df_insurance[["age", "bmi", "charges"]], coord_equal=False)
    print("keys:", list(plots.keys()))
    plots["age_charges"].figure.savefig("plot_scatterplot_age_charges.png")
    print("saved plot_scatterplot_age_charges.png")
