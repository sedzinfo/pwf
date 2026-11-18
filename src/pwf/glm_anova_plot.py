# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_ANOVA_PLOT.R.

Deviations from the R originals, by design:
  - Rmisc::summarySE has no Python package equivalent; reimplemented
    directly (N, mean, sd, se, and a t-distribution based 95% CI,
    matching Rmisc::summarySE's exact formula: ci = se * qt(0.975, N-1)).
  - The future/future.apply parallel-plotting branch is not replicated.
    Plot objects (plotnine ggplot / matplotlib Figure) don't reliably
    serialize across process boundaries the way R's future framework
    handles this, so everything here always runs the equivalent of R's
    sequential fallback path — which is exactly what R itself falls
    back to whenever the combination count doesn't exceed 4x the CPU
    core count anyway.
  - ggpubr::as_ggplot(gridExtra::arrangeGrob(plot)) is a no-op for a
    single plot (it exists to combine *multiple* grobs) — dropped
    entirely; the plotnine ggplot object is returned directly.
  - ggfortify::autoplot(model, which = 1:6, ncol = 2) is ported as
    _autoplot_lm: the same six plotnine panels (points, R's lowess
    smoothers, qqnorm and its quartile line, Cook's distance contours,
    the three most extreme row names, the constant-leverage panel of
    balanced designs), composed in ncol columns with plotnine's | and /
    (ggfortify's ggmultiplot). It also draws the diagnostics of
    report_regression and report_logistic, as rwf's autoplot() calls.
    Where tied leverages (equal group sizes) make R's lowess depend on the
    last-bit rounding of its leverages, the smoother over leverage can
    differ; stringr::str_wrap's line breaks of long factor levels are
    textwrap's greedy ones.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import re
import textwrap
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_errorbar, geom_text, labs,
    theme_bw, scale_x_discrete, coord_flip, position_dodge,
    geom_hline, geom_abline, geom_linerange, expand_limits, theme, element_text,
)
from plotnine.composition import plot_spacer

try:
    from ._rcompat import (gg_linewidth, gg_text_size, r_levels, r_lowess, r_pretty, r_qqnorm, r_quantile7,
                           resolve_columns)
    from .functions_strings import str_aes
except ImportError:
    from _rcompat import (gg_linewidth, gg_text_size, r_levels, r_lowess, r_pretty, r_qqnorm, r_quantile7,
                          resolve_columns)
    from functions_strings import str_aes

_ERROR_BAR_LABELS = {'se': 'standard errors', 'ci': 'confidence intervals', 'sd': 'standard deviations'}
##########################################################################################
# SUMMARY SE (Rmisc::summarySE equivalent)
##########################################################################################
def _summary_se(df, measurevar, groupvars):
    """Group summary: N, mean, sd, se, and a t-based 95% CI, matching Rmisc::summarySE."""
    if isinstance(groupvars, str):
        groupvars = [groupvars]
    grouped = df.groupby(groupvars, observed=True)[measurevar]
    n = grouped.count().rename('N')
    mean = grouped.mean().rename(measurevar)
    sd = grouped.std().rename('sd')
    result = pd.concat([n, mean, sd], axis=1).reset_index()
    result['se'] = result['sd'] / np.sqrt(result['N'])
    result['ci'] = result['se'] * scipy.stats.t.ppf(0.975, result['N'] - 1)
    return result


def _resolve_columns(df, cols):
    """Column names from names or 1-based positions, as rwf's names(df)[index]."""
    return resolve_columns(df, cols)


def _as_r_factor(series):
    """R's factor(): categories in R's level order (numbers by value, text in R's collation)."""
    return pd.Categorical(series, categories=r_levels(series))


def _combine(plot_data, columns):
    """rwf's plot_data_df: plyr::rbind.fill of the summaries, columns in rwf's order."""
    valid = [v for v in plot_data.values() if v is not None]
    if not valid:
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True)
    for frame_column in combined.columns:
        if isinstance(combined[frame_column].dtype, pd.CategoricalDtype):
            combined[frame_column] = combined[frame_column].astype(object)
    return combined[[c for c in columns if c in combined.columns]]
##########################################################################################
# PLOT ONE WAY ANOVA
##########################################################################################
def _plot_oneway_panel(summary, iv_col, dv_col, base_size, type, order_factor, title, note, width):
    data = summary.copy()
    if order_factor:
        order = data.sort_values(dv_col, ascending=False)[iv_col].tolist()
        data[iv_col] = pd.Categorical(data[iv_col], categories=order, ordered=True)

    p = (ggplot(data, aes(x=iv_col, y=dv_col))
         + geom_point()
         + labs(y=str_aes(dv_col), x=textwrap.fill(str_aes(iv_col), width), title=title, caption=note)
         + theme_bw(base_size=base_size)
         + scale_x_discrete(labels=lambda labels: [textwrap.fill(str(l), width) for l in labels])
         + coord_flip())

    if type in _ERROR_BAR_LABELS:
        data['ymin_err'] = data[dv_col] - data[type]
        data['ymax_err'] = data[dv_col] + data[type]
        p = (p + geom_errorbar(aes(ymin='ymin_err', ymax='ymax_err'), width=.1)
               + labs(caption=f"Bars are {_ERROR_BAR_LABELS[type]} {note}"))

    min_y = (data['ymin_err'] if type in _ERROR_BAR_LABELS else data[dv_col]).min()
    ann = pd.DataFrame({iv_col: data[iv_col], 'y': min_y, 'label': "N:" + data['N'].astype(str)})
    p = p + geom_text(ann, aes(x=iv_col, y='y', label='label'), alpha=.5, size=gg_text_size(base_size / 10 * 2),
                       ha='left', va='top', inherit_aes=False)
    return p


def plot_oneway(df, dv, iv, base_size=20, type="se", order_factor=True, title="", note="", width=60):
    """
    For every IV-DV combination, plot group means with optional error
    bars (standard error, 95% CI, or standard deviation) and per-group
    sample size annotations.

    Parameters:
    df (pandas.DataFrame): Data containing the independent and
        dependent variables.
    dv (list of int or str): Column 1-based positions/names of continuous
        dependent variables.
    iv (list of int or str): Column 1-based positions/names of categorical
        independent variables. Coerced to category dtype automatically.
    base_size (int, optional): Base font size for theme_bw(). Defaults to 20.
    type (str, optional): Error bar type: "se", "ci", "sd", or "" (none).
        Defaults to "se".
    order_factor (bool, optional): If True (default), sort the x-axis
        by group mean of the DV (descending).
    title (str, optional): Plot title applied to every panel. Defaults to "".
    note (str, optional): Caption appended to every panel. Defaults to "".
    width (int, optional): Character width for wrapping long axis
        labels. Defaults to 60.

    Returns:
    dict: {
        'plot_data': {"iv_dv": summary DataFrame or None}, one per pair,
        'plot_data_df': all summary DataFrames combined row-wise,
        'plots': {"iv_dv": plotnine.ggplot or None}, one per pair,
    }
    Pairs where the IV has fewer than 2 observed levels (after dropping
    incomplete rows) get None in both plot_data and plots.

    Examples:
    >>> plot_oneway(df=df_insurance, dv=[3, 7], iv=[2, 5, 6])
    >>> plot_oneway(df=df_insurance, dv=[7], iv=[5], type="ci")
    """
    df = df.copy()
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)
    for c in iv_cols:
        df[c] = _as_r_factor(df[c])

    plot_data, plots = {}, {}
    for dv_col in dv_cols:
        for iv_col in iv_cols:
            key = f"{iv_col}_{dv_col}"
            subset = df[[iv_col, dv_col]].dropna()
            if subset[iv_col].nunique() <= 1:
                plot_data[key], plots[key] = None, None
                continue
            summary = _summary_se(subset, dv_col, iv_col)
            plot_data[key] = summary
            plots[key] = _plot_oneway_panel(summary, iv_col, dv_col, base_size, type, order_factor, title, note, width)

    plot_data_df = _combine(plot_data, iv_cols + dv_cols + ["N", "sd", "se", "ci"])
    return {'plot_data': plot_data, 'plot_data_df': plot_data_df, 'plots': plots}
##########################################################################################
# PLOT TWO WAY INTERACTION
##########################################################################################
def _plot_interaction_panel(summary, iv1, iv2, dv_col, base_size, type, order_factor, title, note):
    data = summary.copy()
    if order_factor:
        order = data.groupby(iv1, observed=True)[dv_col].mean().sort_values(ascending=False).index.tolist()
        data[iv1] = pd.Categorical(data[iv1], categories=order, ordered=True)

    if type in _ERROR_BAR_LABELS:
        data['ymin_err'] = data[dv_col] - data[type]
        data['ymax_err'] = data[dv_col] + data[type]

    p = (ggplot(data, aes(x=iv1, y=dv_col, color=iv2, group=iv2))
         + geom_line()
         + geom_point(size=5)
         + theme_bw(base_size=base_size)
         + labs(y=str_aes(dv_col), x=str_aes(iv1), title=title, caption=note, color=str_aes(iv2))
         + coord_flip())

    if type in _ERROR_BAR_LABELS:
        p = (p + geom_errorbar(aes(ymin='ymin_err', ymax='ymax_err'), width=.1, position=position_dodge(0.1))
               + labs(caption=f"Bars are {_ERROR_BAR_LABELS[type]} {note}"))

    n_per_iv1 = data.groupby(iv1, observed=True)['N'].sum().reset_index()
    n_per_iv1['y'] = (data['ymin_err'] if type in _ERROR_BAR_LABELS else data[dv_col]).min()
    n_per_iv1['label'] = "N:" + n_per_iv1['N'].astype(str)
    p = p + geom_text(n_per_iv1, aes(x=iv1, y='y', label='label'), alpha=.5, size=gg_text_size(base_size / 10 * 2),
                       ha='left', va='top', inherit_aes=False)
    return p


def plot_interaction(df, dv, iv, base_size=20, type="se", order_factor=True, title="", note=""):
    """
    For every ordered pair of independent variables (IV1 x IV2, IV1 !=
    IV2) and every dependent variable, plot a line-and-point two-way
    interaction graph: IV1 on the (flipped) x-axis, IV2 as color/group,
    with optional error bars and sample size annotations.

    Both (A, B) and (B, A) orderings are produced for each unordered
    pair — showing A grouped by B is a different, complementary view to
    B grouped by A — matching R's behavior (only exact duplicate rows,
    not unordered-pair duplicates, are dropped there).

    Parameters:
    df (pandas.DataFrame)
    dv (list of int or str): Dependent variable column 1-based positions/names.
    iv (list of int or str): Independent variable column 1-based positions/names.
        Coerced to category dtype automatically.
    base_size (int, optional): Base font size. Defaults to 20.
    type (str, optional): Error bar type: "se", "ci", "sd", or "" (none).
        Defaults to "se".
    order_factor (bool, optional): If True (default), order IV1's
        levels by their overall mean DV (descending).
    title (str, optional): Plot title. Defaults to "".
    note (str, optional): Caption. Defaults to "".

    Returns:
    dict: {'plot_data': {...}, 'plot_data_df': ..., 'plots': {...}},
    keyed "iv1_iv2_dv".

    Examples:
    >>> plot_interaction(df=df_insurance, dv=[7], iv=[2, 5])
    """
    df = df.copy()
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)
    for c in iv_cols:
        df[c] = _as_r_factor(df[c])

    # R's expand.grid(iv, iv, dv): the first factor varies fastest
    combos = [(iv1, iv2, dv_col) for dv_col in dv_cols for iv2 in iv_cols for iv1 in iv_cols if iv1 != iv2]

    plot_data, plots = {}, {}
    for iv1, iv2, dv_col in combos:
        key = f"{iv1}_{iv2}_{dv_col}"
        subset = df[[iv1, iv2, dv_col]].dropna()
        if len(subset) <= 1:
            plot_data[key], plots[key] = None, None
            continue
        summary = _summary_se(subset, dv_col, [iv1, iv2])
        plot_data[key] = summary
        plots[key] = _plot_interaction_panel(summary, iv1, iv2, dv_col, base_size, type, order_factor, title, note)

    plot_data_df = _combine(plot_data, iv_cols + dv_cols + ["N", "sd", "se", "ci"])
    return {'plot_data': plot_data, 'plot_data_df': plot_data_df, 'plots': plots}
##########################################################################################
# PLOT ANOVA DIAGNOSTICS
##########################################################################################
def r_row_names(labels):
    """
    R's row names of the rows a model used: a data frame read by R is numbered from 1, so pandas' integer
    labels (positions from 0) become label + 1; other labels are kept as text.
    """
    labels = list(labels)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in labels):
        return [str(int(v) + 1) for v in labels]
    return [str(v) for v in labels]


def model_factors(results):
    """
    The text/categorical variables of a statsmodels formula model's frame (R's model frame columns that are
    character or factor), for _autoplot_lm's constant-leverage panel; None when there are none.
    """
    try:
        design_info = results.model.data.design_info
        frame = results.model.data.frame
    except AttributeError:
        return None
    rows = results.model.data.row_labels
    columns = []
    for factor, info in design_info.factor_infos.items():
        if info.type != "categorical":
            continue
        code = factor.name()
        match = re.fullmatch(r"C\(\s*(?:Q\(['\"])?([^,'\")]+)(?:['\"]\))?.*\)", code)
        name = match.group(1).strip() if match else code
        if name in frame.columns and name not in columns:
            columns.append(name)
    return frame.loc[rows, columns] if columns else None


def lm_diagnostics(fit):
    """
    ggfortify's lm_diagnostics() of a statsmodels OLS fit (unweighted): fitted values, residuals, standardized
    residuals (rstandard; also the .wstdresid of the Q-Q and scale-location panels), leverages, Cook's distances
    and the rank, as R's lm.influence, rstandard and cooks.distance give them.
    """
    hat = np.asarray(fit.get_influence().hat_matrix_diag, dtype=float)
    resid = np.asarray(fit.resid, dtype=float)
    s = np.sqrt(fit.ssr / fit.df_resid)
    rank = int(np.linalg.matrix_rank(np.asarray(fit.model.exog, dtype=float)))
    stdresid = resid / (s * np.sqrt(1 - hat))
    return {"is_glm": False, "fitted": np.asarray(fit.fittedvalues, dtype=float), "resid": resid,
            "stdresid": stdresid, "wstdresid": stdresid, "hat": hat,
            "cooksd": (resid / (s * (1 - hat))) ** 2 * hat / rank, "rank": rank}


def _autoplot_lm(diagnostics, labels, factors=None, which=(1, 2, 3, 4, 5, 6), ncol=2, label_size=None,
                 base_size=10, caption=None, title=None, axis_angle=45):
    """
    Port of ggfortify::autoplot() of an lm or glm (ggfortify 0.4.25's autoplot.lm, label.n = 3) followed by
    rwf's + labs(caption = caption) [+ labs(title = title)] + theme_bw(base_size) +
    theme(axis.text.x = element_text(angle = axis_angle, hjust = 1)), applied to every panel.

    Parameters:
    diagnostics (dict): lm_diagnostics(fit) for an lm; for a glm (is_glm True) its linear predictor (fitted),
        deviance residuals (resid), standardized Pearson residuals (stdresid, R 4.6's rstandard), deviance
        residuals / sqrt(dispersion * (1 - hat)) (wstdresid), hat, Cook's distances and rank.
    labels (list of str): R's row names of the rows the model used (labels of the 3 most extreme points).
    factors (pandas.DataFrame, optional): The model frame's text/factor columns; with a constant leverage
        panel 5 plots residuals against their level combinations, as ggfortify's.
    which (sequence of int): Panels 1 to 6 (Residuals vs Fitted, Normal Q-Q, Scale-Location, Cook's distance,
        Residuals vs Leverage, Cook's dist vs Leverage).
    ncol (int): Columns of the composition (ggmultiplot's ncol; rows follow).
    label_size (float, optional): ggplot2 text size (mm) of the point labels; None is ggplot2's 3.88.
    title (str, optional): When not None it replaces every panel's title (rwf's labs(title = title), so ""
        leaves them blank).

    Returns:
    plotnine composition of the panels (a single ggplot for one panel).
    """
    is_glm = diagnostics["is_glm"]
    labels = [str(v) for v in labels]
    fitted, resid = diagnostics["fitted"], diagnostics["resid"]
    stdresid, wstdresid = diagnostics["stdresid"], diagnostics["wstdresid"]
    hat, cooksd, rank = diagnostics["hat"], diagnostics["cooksd"], diagnostics["rank"]
    n = len(resid)
    data = pd.DataFrame({".fitted": fitted, ".resid": resid, ".stdresid": stdresid, ".wstdresid": wstdresid,
                         ".hat": hat, ".cooksd": cooksd, ".index": np.arange(1, n + 1), ".label": labels})
    data[".sqrtresid"] = np.sqrt(np.abs(wstdresid))
    data[".qqx"], data[".qqy"] = r_qqnorm(wstdresid)
    hat_range = (np.nanmin(hat), np.nanmax(hat))
    constant_leverage = (hat_range[0] == 0 and hat_range[1] == 0) or \
        hat_range[1] - hat_range[0] < 1e-10 * np.nanmean(hat)
    if constant_leverage and factors is not None and factors.shape[1] > 0:
        combination = factors.astype(str).agg(":".join, axis=1).to_numpy()
        wrapped = ["\n".join(textwrap.wrap(v, width=10, break_long_words=False, break_on_hyphens=False)) or v
                   for v in combination]
        data[".nf"] = pd.Categorical(wrapped, categories=r_levels(wrapped))
    else:
        constant_leverage = False

    def top(column):  # dplyr::arrange(desc(abs(x))) keeps ties in order; head(3)
        return data.iloc[np.argsort(-np.abs(data[column].to_numpy()), kind="stable")[:3]]

    r_data, cd_data = top(".resid"), top(".cooksd")
    fitted_label = "Predicted values" if is_glm else "Fitted values"
    y23 = "Std. deviance resid." if is_glm else "Standardized residuals"
    text_size = gg_text_size(3.88 if label_size is None else label_size)
    ad = {"linetype": "dashed", "size": gg_linewidth(0.2), "colour": "#888888"}
    point = geom_point(colour="#444444")

    def smoother(x, y):
        sx, sy = r_lowess(data[x].to_numpy(), data[y].to_numpy())
        return geom_line(pd.DataFrame({"x": sx, "y": sy}), aes(x="x", y="y"), colour="#0000FF",
                         linetype="solid", size=gg_linewidth(0.5), inherit_aes=False)

    def text(frame):
        return geom_text(frame, aes(label=".label"), colour="#000000", size=text_size)

    panels = {}
    panels[1] = (ggplot(data, aes(x=".fitted", y=".resid")) + point + smoother(".fitted", ".resid")
                 + geom_hline(yintercept=0, **ad) + text(r_data)
                 + labs(x=fitted_label, y="Residuals", title="Residuals vs Fitted"))
    qy = r_quantile7(wstdresid, [0.25, 0.75])
    qx = scipy.stats.norm.ppf([0.25, 0.75])
    slope = (qy[1] - qy[0]) / (qx[1] - qx[0])
    panels[2] = (ggplot(data, aes(x=".qqx", y=".qqy")) + point
                 + geom_abline(intercept=qy[0] - slope * qx[0], slope=slope, **ad) + text(r_data)
                 + labs(x="Theoretical Quantiles", y=y23, title="Normal Q-Q"))
    y3 = r"$\sqrt{|\mathrm{Std.\ deviance\ resid.}|}$" if is_glm else r"$\sqrt{|\mathrm{Standardized\ residuals}|}$"
    panels[3] = (ggplot(data, aes(x=".fitted", y=".sqrtresid")) + point + smoother(".fitted", ".sqrtresid")
                 + text(r_data) + labs(x=fitted_label, y=y3, title="Scale-Location"))
    panels[4] = (ggplot(data, aes(x=".index", y=".cooksd", ymin=0, ymax=".cooksd"))
                 + geom_linerange(colour="#444444", size=gg_linewidth(0.5)) + text(cd_data)
                 + labs(x="Obs. Number", y="Cook's distance", title="Cook's distance"))
    y5 = "Std. Pearson resid." if is_glm else "Standardized Residuals"
    if constant_leverage:
        panels[5] = (ggplot(data, aes(x=".nf", y=".stdresid")) + point + geom_hline(yintercept=0, **ad)
                     + expand_limits(x=0) + text(cd_data)
                     + labs(x="Factor Level Combination", y=y5,
                            title="Constant Leverage:\nResiduals vs Factor Levels"))
    else:
        panels[5] = (ggplot(data, aes(x=".hat", y=".stdresid")) + point + smoother(".hat", ".stdresid")
                     + geom_hline(yintercept=0, **ad) + expand_limits(x=0) + text(cd_data)
                     + labs(x="Leverage", y=y5, title="Residuals vs Leverage"))
    panel6 = (ggplot(data, aes(x=".hat", y=".cooksd")) + point + smoother(".hat", ".cooksd")
              + expand_limits(x=0, y=0) + text(cd_data)
              + labs(x="Leverage", y="Cook's distance", title="Cook's dist vs Leverage"))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(hat >= 1, np.nan, hat / (1 - hat))
        contours = r_pretty(np.sqrt(rank * cooksd / g), 5)
    for b in contours:
        panel6 = panel6 + geom_abline(intercept=0, slope=b ** 2, **ad)
    panels[6] = panel6

    chosen = []
    for k in which:
        panel = panels[k] + labs(caption=caption) if caption is not None else panels[k]
        if title is not None:
            panel = panel + labs(title=title)
        chosen.append(panel + theme_bw(base_size=base_size)
                      + theme(axis_text_x=element_text(rotation=axis_angle, ha="right"),
                              plot_title=element_text(ha="left")))  # ggplot2's titles are left-aligned
    if len(chosen) == 1:
        return chosen[0]
    rows = []
    for start in range(0, len(chosen), ncol):  # ggmultiplot fills its grid by rows
        row = chosen[start:start + ncol]
        row = row + [plot_spacer()] * (ncol - len(row))
        composed = row[0]
        for panel in row[1:]:
            composed = composed | panel
        rows.append(composed)
    composed = rows[0]
    for row in rows[1:]:
        composed = composed / row
    return composed


def plot_oneway_diagnostics(df, dv, iv, base_size=10):
    """
    For every IV-DV combination, fit dv ~ iv (iv as a factor) and draw
    ggfortify's six lm diagnostic panels (port of rwf::plot_oneway_diagnostics):
    Residuals vs Fitted, Normal Q-Q, Scale-Location, Cook's distance,
    Residuals vs Leverage (Residuals vs Factor Levels when every group has
    the same size) and Cook's dist vs Leverage, in two columns. Smoothers
    are R's lowess; the three most extreme points carry their row numbers.

    Parameters:
    df (pandas.DataFrame)
    dv (list of int or str): Continuous dependent variable column indices/names.
    iv (list of int or str): Categorical independent variable column indices/names.
    base_size (int, optional): Base font size of theme_bw. Defaults to 10.

    Returns:
    dict: {"iv_dv": plotnine composition of the six panels, or None when
    the IV has fewer than 2 levels with more than one observation each}.

    Examples:
    >>> plot_oneway_diagnostics(df=df_insurance, dv=[7], iv=[5, 6])
    """
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)
    frame = df.reset_index(drop=True)  # rows numbered by position, as R's row names

    plots = {}
    for dv_col in dv_cols:
        for iv_col in iv_cols:
            key = f"{iv_col}_{dv_col}"
            subset = frame[[dv_col, iv_col]].dropna()
            counts = subset[iv_col].astype(str).value_counts()
            subset = subset[subset[iv_col].astype(str).isin(counts[counts > 1].index)].copy()
            if subset[iv_col].nunique() <= 1:
                plots[key] = None
                continue
            subset[iv_col] = pd.Categorical(subset[iv_col], categories=r_levels(subset[iv_col]))
            model = smf.ols(f"Q('{dv_col}') ~ C(Q('{iv_col}'))", data=subset).fit()
            labels = r_row_names(df.index[subset.index])  # R keeps the data frame's row names
            caption = f"{dv_col} ~ {iv_col}\nobservations={int(model.nobs)}"
            plots[key] = _autoplot_lm(lm_diagnostics(model), labels, factors=subset[[iv_col]],
                                      which=(1, 2, 3, 4, 5, 6), ncol=2, label_size=3, base_size=base_size,
                                      caption=caption, axis_angle=45)
    return plots
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    import matplotlib
    matplotlib.use("Agg")

    df_insurance = load_dataset("insurance")

    print("=" * 80, "\nplot_oneway\n", "=" * 80, sep="")
    result = plot_oneway(df=df_insurance, dv=["charges"], iv=["sex", "smoker", "region"], type="se")
    print("keys:", list(result['plots'].keys()))
    print(result['plot_data_df'])
    result['plots']['sex_charges'].save("plot_oneway_sex_charges.png", verbose=False)
    print("saved plot_oneway_sex_charges.png")

    print("\n" + "=" * 80, "\nplot_interaction\n", "=" * 80, sep="")
    result2 = plot_interaction(df=df_insurance, dv=["charges"], iv=["sex", "smoker"], type="se")
    print("keys:", list(result2['plots'].keys()))
    result2['plots']['sex_smoker_charges'].save("plot_interaction_sex_smoker_charges.png", verbose=False)
    print("saved plot_interaction_sex_smoker_charges.png")

    print("\n" + "=" * 80, "\nplot_oneway_diagnostics\n", "=" * 80, sep="")
    result3 = plot_oneway_diagnostics(df=df_insurance, dv=["charges"], iv=["smoker", "region"])
    print("keys:", list(result3.keys()))
    result3['smoker_charges'].save("plot_diagnostics_smoker_charges.png", width=10, height=12, verbose=False)
    print("saved plot_diagnostics_smoker_charges.png")
