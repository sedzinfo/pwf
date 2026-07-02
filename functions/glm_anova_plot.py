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
  - ggfortify::autoplot(model, which=1:6)'s 6-panel lm diagnostic plot
    is reimplemented from scratch as a matplotlib 3x2 grid, using
    statsmodels' OLSInfluence for leverage/Cook's distance and
    scipy.stats.probplot for the Q-Q panel — there's no one-line
    equivalent in the Python plotting ecosystem.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import textwrap
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_errorbar, geom_text, labs,
    theme_bw, scale_x_discrete, coord_flip, position_dodge,
)

try:
    from .functions_strings import str_aes
except ImportError:
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
    return [df.columns[c] for c in cols] if all(isinstance(c, (int, np.integer)) for c in cols) else list(cols)
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
    p = p + geom_text(ann, aes(x=iv_col, y='y', label='label'), alpha=.5, size=base_size / 10 * 2,
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
    dv (list of int or str): Column indices/names of continuous
        dependent variables.
    iv (list of int or str): Column indices/names of categorical
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
    >>> plot_oneway(df=df_insurance, dv=[2, 6], iv=[1, 4, 5])
    >>> plot_oneway(df=df_insurance, dv=[6], iv=[4], type="ci")
    """
    df = df.copy()
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)
    for c in iv_cols:
        df[c] = df[c].astype('category')

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

    valid = [v for v in plot_data.values() if v is not None]
    plot_data_df = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
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
    p = p + geom_text(n_per_iv1, aes(x=iv1, y='y', label='label'), alpha=.5, size=base_size / 10 * 2,
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
    dv (list of int or str): Dependent variable column indices/names.
    iv (list of int or str): Independent variable column indices/names.
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
    >>> plot_interaction(df=df_insurance, dv=[6], iv=[1, 4])
    """
    df = df.copy()
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)
    for c in iv_cols:
        df[c] = df[c].astype('category')

    combos = [(iv1, iv2, dv_col) for dv_col in dv_cols for iv1 in iv_cols for iv2 in iv_cols if iv1 != iv2]

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

    valid = [v for v in plot_data.values() if v is not None]
    plot_data_df = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
    return {'plot_data': plot_data, 'plot_data_df': plot_data_df, 'plots': plots}
##########################################################################################
# PLOT ANOVA DIAGNOSTICS
##########################################################################################
def _lm_diagnostics_figure(model, base_size, caption):
    """6-panel lm diagnostic plot, the Python equivalent of ggfortify::autoplot(model, which=1:6)."""
    fitted = model.fittedvalues
    influence = model.get_influence()
    standardized_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    sqrt_abs_std_resid = np.sqrt(np.abs(standardized_resid))

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    axes[0, 0].scatter(fitted, model.resid, alpha=0.6)
    axes[0, 0].axhline(0, linestyle='--', color='gray')
    axes[0, 0].set(xlabel='Fitted values', ylabel='Residuals', title='Residuals vs Fitted')

    (osm, osr), (slope, intercept, _) = scipy.stats.probplot(standardized_resid, dist='norm')
    axes[0, 1].scatter(osm, osr, alpha=0.6)
    axes[0, 1].plot(osm, intercept + slope * osm, 'r--')
    axes[0, 1].set(xlabel='Theoretical Quantiles', ylabel='Standardized residuals', title='Normal Q-Q')

    axes[1, 0].scatter(fitted, sqrt_abs_std_resid, alpha=0.6)
    axes[1, 0].set(xlabel='Fitted values', ylabel='sqrt(|Standardized residuals|)', title='Scale-Location')

    axes[1, 1].vlines(range(len(cooks_d)), 0, cooks_d)
    axes[1, 1].set(xlabel='Obs. number', ylabel="Cook's distance", title="Cook's Distance")

    axes[2, 0].scatter(leverage, standardized_resid, alpha=0.6)
    axes[2, 0].axhline(0, linestyle='--', color='gray')
    axes[2, 0].set(xlabel='Leverage', ylabel='Standardized residuals', title='Residuals vs Leverage')

    axes[2, 1].scatter(leverage, cooks_d, alpha=0.6)
    axes[2, 1].set(xlabel='Leverage', ylabel="Cook's distance", title="Cook's dist vs Leverage")

    fig.suptitle(caption, fontsize=base_size)
    fig.tight_layout()
    return fig


def plot_oneway_diagnostics(df, dv, iv, base_size=10):
    """
    For every IV-DV combination, fit dv ~ C(iv) and produce a 6-panel
    lm diagnostic plot: Residuals vs Fitted, Normal Q-Q, Scale-Location,
    Cook's Distance, Residuals vs Leverage, Cook's Distance vs Leverage.

    Parameters:
    df (pandas.DataFrame)
    dv (list of int or str): Continuous dependent variable column indices/names.
    iv (list of int or str): Categorical independent variable column indices/names.
    base_size (int, optional): Base font size for the figure suptitle. Defaults to 10.

    Returns:
    dict: {"iv_dv": matplotlib.figure.Figure or None}. None when the IV
    has fewer than 2 levels with more than one observation each.

    Examples:
    >>> plot_oneway_diagnostics(df=df_insurance, dv=[6], iv=[1, 4])
    """
    df = df.copy()
    iv_cols = _resolve_columns(df, iv)
    dv_cols = _resolve_columns(df, dv)

    plots = {}
    for dv_col in dv_cols:
        for iv_col in iv_cols:
            key = f"{iv_col}_{dv_col}"
            subset = df[[dv_col, iv_col]].dropna().copy()
            counts = subset[iv_col].value_counts()
            subset = subset[subset[iv_col].isin(counts[counts > 1].index)]
            subset[iv_col] = subset[iv_col].astype('category')
            if subset[iv_col].nunique() <= 1:
                plots[key] = None
                continue
            model = smf.ols(f"{dv_col} ~ C({iv_col})", data=subset).fit()
            caption = f"{dv_col} ~ {iv_col}\nobservations={int(model.nobs)}"
            plots[key] = _lm_diagnostics_figure(model, base_size=base_size, caption=caption)
    return plots
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    df_insurance = pd.read_csv("data/insurance.csv") if __import__("os").path.exists("data/insurance.csv") \
        else pd.read_csv("../data/insurance.csv")

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
    result3['smoker_charges'].savefig("plot_diagnostics_smoker_charges.png")
    print("saved plot_diagnostics_smoker_charges.png")
