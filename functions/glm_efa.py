# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_EFA.R: scree plots, loading tables and plots,
residual statistics and the EFA report (report_efa).

Models come from fa() (also as _fit_efa), pwf's port of psych::fa with
fm = "minres" (psych's default) or "ml" and oblimin rotation, which
reproduces psych's loadings, communalities, eigenvalues, fit statistics
and regression factor scores. Bartlett's test and KMO follow
psych::cortest.bartlett and psych::KMO. Attribute names with dots in R
(model$e.values, model$fit.off) use underscores (e_values, fit_off).
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
from plotnine import (
    ggplot, aes, geom_hline, geom_line, geom_point, geom_tile, geom_text, geom_bar,
    theme_bw, theme, scale_x_continuous, scale_fill_gradient2, labs, annotate,
    element_blank, element_text, coord_flip, facet_wrap, lims,
)

try:
    from ._efa import fa, model_loadings as _model_loadings
    from .functions_plot import plot_multiplot, report_pdf
    from .functions_strings import str_proper, call_to_string
    from .explore_assumptions import plot_histogram
except ImportError:
    from _efa import fa, model_loadings as _model_loadings
    from functions_plot import plot_multiplot, report_pdf
    from functions_strings import str_proper, call_to_string
    from explore_assumptions import plot_histogram
##########################################################################################
# FIT EFA (internal helper, R psych::fa-like model wrapper)
##########################################################################################
def _fit_efa(df, n_factors, rotation="oblimin", method="minres", **kwargs):
    """
    Fit an exploratory factor analysis, as psych::fa(df, nfactors, rotate,
    fm) (rwf's functions take a psych::fa model). A thin wrapper around
    pwf's psych-compatible engine, fa(); kept for code that calls it.

    Parameters:
    df (pandas.DataFrame): Raw data; correlations are pairwise.
    n_factors (int): Number of factors.
    rotation (str, optional): "oblimin" (default) or "none".
    method (str, optional): "minres" (default, psych's default) or "ml".

    Returns:
    FAResult: The fitted model (see fa()).

    Examples:
    >>> model = _fit_efa(df_ocean[["E1", "E2", "E3", "E4", "E5"]], n_factors=1)
    """
    return fa(df, nfactors=n_factors, rotate=rotation, fm=method)
##########################################################################################
# PLOT SCREE
##########################################################################################
def plot_scree(df, base_size=15, title="", color=("#5F2C91", "#5E912C")):
    """
    Scree plot showing eigenvalues of the correlation matrix, with the
    Kaiser (eigenvalue > 1) and Jolliffe (eigenvalue > 0.7) extraction
    criteria annotated.

    Parameters:
    df (pandas.DataFrame): Data to compute the correlation matrix from.
    base_size (int, optional): Base font size. Defaults to 15.
    title (str, optional): Appended to the plot title. Defaults to "".
    color (tuple, optional): (Kaiser line color, Jolliffe line color).

    Returns:
    plotnine.ggplot

    Examples:
    >>> plot_scree(df=df_insurance.select_dtypes("number"), title="")
    """
    corr_matrix = df.corr(method='pearson', numeric_only=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix.to_numpy()))[::-1]
    eigenvalues_df = pd.DataFrame({'x': np.arange(1, len(eigenvalues) + 1), 'eigenvalues': eigenvalues})

    kaiser = int(np.sum(eigenvalues > 1))
    jolliffe = int(np.sum(eigenvalues > 0.7))

    return (ggplot(eigenvalues_df, aes(x='x', y='eigenvalues'))
            + geom_hline(yintercept=1, color=color[0])
            + geom_hline(yintercept=0.7, color=color[1])
            + geom_line(color=color[0])
            + geom_point(size=base_size / 4, color=color[0])
            + scale_x_continuous(breaks=eigenvalues_df['x'].tolist())
            + theme_bw(base_size=base_size)
            + labs(x="Index", y="Eigenvalue", title=f"Scree plot {title}")
            + annotate("text", x=eigenvalues_df['x'].max(), y=eigenvalues_df['eigenvalues'].max(),
                       label=f"Top line: Kaiser criterion:{kaiser}\nBottom line: Jolliffe criterion:{jolliffe}",
                       ha='right', va='top', size=base_size / 4)
            + theme(legend_title=element_blank(), legend_position='bottom', axis_title_x=element_blank()))
##########################################################################################
# FACTOR PATTERN STRUCTURE MATRIX
##########################################################################################
_LOADING_CRITICAL = pd.DataFrame({
    'sample': [50, 60, 70, 85, 100, 120, 150, 200, 250, 350],
    'critical_loading': [.75, .70, .65, .60, .55, .50, .45, .40, .35, .30],
})


def model_loadings(model, cut=None, matrix_type="pattern", sort=True):
    """
    Pattern and/or structure loadings rounded to 2 decimals, sorted like
    psych::fa.sort, with loadings below cut blanked (port of
    rwf::model_loadings).

    Parameters:
    model (FAResult): A model from fa() / _fit_efa().
    cut (float, optional): Smallest absolute loading shown. By default the
        critical loading for the sample size (Stevens' table), as rwf.
    matrix_type (str, optional): "pattern" (default), "structure" or "all".
    sort (bool, optional): Sort the items. Defaults to True.

    Returns:
    pandas.DataFrame: Columns Matrix, variable and one per factor. As in rwf,
    the loadings are text when some are blanked and numbers otherwise.

    Examples:
    >>> model_loadings(model=model, cut=None, matrix_type="pattern")
    """
    return _model_loadings(model, cut=cut, matrix_type=matrix_type, sort=sort)
##########################################################################################
# LOADINGS PLOT STACKED WITH CORRELATION
##########################################################################################
def plot_loadings(model, matrix_type=None, title="", base_size=10,
                   color=("#5E912C", "white", "#5F2C91"), sort=True):
    """
    Plot factor loadings as a stacked bar chart alongside the variable
    correlation matrix heatmap, plus a per-factor faceted loading bar plot.

    Parameters:
    model: An EFA model from _fit_efa.
    matrix_type (str): "pattern" or "structure" (required).
    title (str, optional): Appended to the faceted plot's title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.
    color (tuple, optional): (low, mid, high) fill colors. Defaults to
        ("#5E912C", "white", "#5F2C91").
    sort (bool, optional): Sort loadings by dominant factor. Defaults to True.

    Returns:
    dict: {'correlation_loadings': list of matplotlib Figures (via
    plot_multiplot: correlation heatmap + stacked loadings side by
    side), 'plot_barplot': plotnine.ggplot (per-factor faceted bars)}.

    Examples:
    >>> model = _fit_efa(df_insurance.select_dtypes("number"), n_factors=2)
    >>> plot_loadings(model=model, matrix_type="pattern")
    """
    if matrix_type is None:
        raise ValueError("specify matrix_type either pattern or structure")

    load = model_loadings(model, cut=0, matrix_type=matrix_type, sort=sort)
    factor_names = [c for c in load.columns if c not in ("Matrix", "variable")]
    variable_order = list(load['variable'])

    correlation_matrix = model.r.copy()
    correlation_matrix.insert(0, 'variable', correlation_matrix.index)

    loadings_long = load.melt(id_vars='variable', value_vars=factor_names, var_name='Factor', value_name='Loading')
    loadings_long['Loading'] = pd.to_numeric(loadings_long['Loading'], errors='coerce')
    loadings_long['AbsLoading'] = loadings_long['Loading'].abs()
    loadings_long['variable'] = pd.Categorical(loadings_long['variable'], categories=variable_order, ordered=True)

    correlations_long = correlation_matrix.melt(id_vars='variable', var_name='Relation', value_name='Correlation')
    correlations_long['CorrelationRounded'] = correlations_long['Correlation'].round(2)
    correlations_long['variable'] = pd.Categorical(correlations_long['variable'], categories=variable_order, ordered=True)

    correlation_plot = (ggplot(correlations_long, aes('Relation', 'variable', fill='Correlation'))
                         + geom_tile()
                         + scale_fill_gradient2(low=color[0], mid=color[1], high=color[2], midpoint=0)
                         + geom_text(aes(label='CorrelationRounded'), size=base_size / 4)
                         + labs(y="Loading", title="Correlation Matrix")
                         + theme_bw(base_size=base_size)
                         + theme(axis_title_x=element_blank(), axis_title_y=element_blank(),
                                 axis_text_x=element_text(color="white"), legend_position="none"))

    stacked_loadings = (ggplot(loadings_long, aes('variable', 'AbsLoading', fill='Factor'))
                         + geom_bar(stat="identity")
                         + coord_flip()
                         + labs(y="Loading", title=f"{str_proper(matrix_type)} Matrix Loadings")
                         + theme_bw(base_size=base_size)
                         + theme(axis_title_y=element_blank(), axis_title_x=element_blank(),
                                 axis_text_y=element_blank()))

    plot_barplot = (ggplot(loadings_long, aes('variable', 'AbsLoading', fill='Loading'))
                    + facet_wrap('~Factor', nrow=1)
                    + geom_bar(stat="identity")
                    + scale_fill_gradient2(name="Loading", high=color[0], mid=color[1], low=color[2], midpoint=0)
                    + coord_flip()
                    + labs(y="Loading", x="", title=f"{str_proper(matrix_type)} Matrix {title}")
                    + theme_bw(base_size=base_size)
                    + lims(y=(0, 1)))

    correlation_loadings = plot_multiplot(correlation_plot, stacked_loadings, cols=2)
    return {'correlation_loadings': correlation_loadings, 'plot_barplot': plot_barplot}
##########################################################################################
# EFA OBSERVED AND EXPECTED CORRELATION MATRIX RESIDUALS
##########################################################################################
def compute_residual_stats(model, data=None):
    """
    Root Mean Squared Residual, count and proportion of large
    (>0.05) absolute residuals. Accepts an EFA model, or two
    correlation/covariance matrices to compare directly.

    Parameters:
    model: An EFA model from _fit_efa (uses model.residual), or a
        correlation/covariance matrix if `data` is given.
    data (array-like, optional): A second matrix to subtract from
        `model` directly (residuals = model - data), instead of using
        a fitted EFA model's own residual matrix.

    Returns:
    pandas.DataFrame: 3 rows (RMSR, count, proportion), with 'value',
    'critical', and 'formula' columns.

    Examples:
    >>> model = _fit_efa(df_insurance.select_dtypes("number"), n_factors=2)
    >>> compute_residual_stats(model)
    """
    if data is not None:
        residual_matrix = np.asarray(model, dtype=float) - np.asarray(data, dtype=float)
    else:
        residual_matrix = model.residual.to_numpy()
    residuals = residual_matrix[np.triu_indices_from(residual_matrix, k=1)]

    large_residuals = np.abs(residuals) > 0.05
    n_large = int(large_residuals.sum())
    prop_large = n_large / len(residuals)
    rmsr = float(np.sqrt(np.mean(residuals ** 2)))

    return pd.DataFrame({
        'residual_statistics': ["Root Mean Squared Residual", "Number of absolute residuals > 0.05",
                                 "Proportion of absolute residuals > 0.05"],
        'value': [rmsr, n_large, prop_large],
        'critical': [np.nan, np.nan, 0.5],
        'formula': ["sqrt(mean(residuals^2))", "abs(residuals)>0.05", "numberLargeResiduals/nrow(residuals)"],
    })
##########################################################################################
# REPORT EFA
##########################################################################################
def _bartlett_sphericity(r, n):
    """psych::cortest.bartlett(R, n)."""
    p = r.shape[0]
    statistic = -np.log(np.linalg.det(r)) * (n - 1 - (2 * p + 5) / 6)
    df = p * (p - 1) / 2
    return statistic, df, stats.chi2.sf(statistic, df)


def _kmo(r):
    """psych::KMO(r): overall and per item measures of sampling adequacy."""
    q = np.linalg.inv(r)
    d = 1 / np.sqrt(np.diag(q))
    partial = q * np.outer(d, d)
    np.fill_diagonal(partial, 0)
    off = r.copy()
    np.fill_diagonal(off, 0)
    msa = np.sum(off ** 2) / (np.sum(off ** 2) + np.sum(partial ** 2))
    msa_items = (off ** 2).sum(axis=0) / ((off ** 2).sum(axis=0) + (partial ** 2).sum(axis=0))
    return msa, msa_items


def report_efa(model, df, file=None, w=10, h=5, cut=0, base_size=10, scores=False):
    """
    Exploratory factor analysis report (port of rwf::report_efa): observed,
    reproduced and residual correlations, residual statistics, determinant,
    Bartlett's test of sphericity, KMO, fit indices, loadings with variance
    accounted, score weights and scree, loading and residual plots,
    optionally written to <file>.xlsx and <file>.pdf.

    Parameters:
    model (FAResult): A model from fa() / _fit_efa() (rwf: psych::fa).
    df (pandas.DataFrame): The data the model was fitted on.
    file (str, optional): Output filename without extension.
    w, h (float, optional): PDF page size in inches. Defaults to 10, 5.
    cut (float, optional): Loading cut passed to model_loadings. Defaults to 0.
    base_size (int, optional): Base font size of the plots. Defaults to 10.
    scores (bool, optional): Add the factor scores. Defaults to False.

    Returns:
    dict: correlations, npobs, residual_stats, determinant_test,
    bartlett_test, kmo_test, loadings, instruction_loading_critical_values,
    weights and (scores=True) scores, as in rwf.

    Examples:
    >>> model = fa(df_ocean[["E1", "E2", "E3", "E4", "E5"]], nfactors=1)
    >>> report_efa(model=model, df=df_ocean[["E1", "E2", "E3", "E4", "E5"]])
    """
    correlationmatrix = df.corr().to_numpy()
    statistic, bartlett_df, p_value = _bartlett_sphericity(correlationmatrix, len(df))
    bartlett_test = pd.DataFrame({"x_squared[bartlett]": [statistic], "df[bartlett]": [bartlett_df],
                                  "p[bartlett]": [p_value]})
    determinant = float(np.linalg.det(correlationmatrix))
    determinant_test = pd.DataFrame({"determinant": [determinant], "above_critical": [determinant > 0.00001]})
    msa, msa_items = _kmo(correlationmatrix)
    kmo_test = pd.DataFrame({"Overall_MSA": msa, "MSA": msa_items, "Kaiser_1974": np.nan}, index=df.columns)

    fit_index = pd.DataFrame({
        "RMSEA": [model.RMSEA["RMSEA"]], "lower": [model.RMSEA["lower"]], "upper": [model.RMSEA["upper"]],
        "confidence": [model.RMSEA["confidence"]], "RMS": [model.rms], "CRMS": [model.crms], "TLI": [model.TLI],
        "CFI": [model.CFI], "BIC": [model.BIC], "SABIC": [model.SABIC], "EBIC": [model.EBIC],
        "ESABIC": [model.ESABIC], "fit": [model.fit], "fit.off": [model.fit_off],
    })

    plots = {
        "plot_scree": plot_scree(df, base_size=base_size, title=""),
        "plot_loadings_structure": plot_loadings(model, matrix_type="structure", base_size=base_size),
        "plot_loadings_pattern": plot_loadings(model, matrix_type="pattern", base_size=base_size),
    }
    residual_offdiag = model.residual.to_numpy()[np.triu_indices_from(model.residual.to_numpy(), k=1)]
    plots["plot_residual_histogram"] = plot_histogram(
        pd.DataFrame({"data": residual_offdiag}), bins=30, title="Histogram of residuals", base_size=base_size)
    if file is not None:
        flat_plots = [plots["plot_scree"], plots["plot_loadings_structure"]["plot_barplot"],
                      plots["plot_loadings_pattern"]["plot_barplot"], plots["plot_residual_histogram"]]
        report_pdf(plotlist=flat_plots, file=file, w=w, h=h, print_plot=False)

    residual_stats = compute_residual_stats(model)
    model_communality = pd.DataFrame({
        "complexity": model.complexity, "communality": model.communality, "communalities": model.communalities,
        "uniquenesses": model.uniquenesses, "values": model.values, "e.values": model.e_values})
    correlations = pd.concat([
        pd.concat([pd.DataFrame({"type": "reproduced correlations"}, index=model.model.index), model.model], axis=1),
        pd.concat([pd.DataFrame({"type": "observed correlations"}, index=model.r.index), model.r], axis=1),
        pd.concat([pd.DataFrame({"type": "residual correlations"}, index=model.residual.index), model.residual],
                  axis=1),
    ])
    vaccounted = model.Vaccounted.copy()
    vaccounted.insert(0, "row.names.model.Vaccounted.", vaccounted.index)
    vaccounted.insert(0, "type", "variance accounted")
    loadings = pd.concat([model_loadings(model, cut, matrix_type="all"), vaccounted.reset_index(drop=True)],
                         ignore_index=True)
    loadings = loadings[[c for c in loadings.columns if c not in ("type", "row.names.model.Vaccounted.")]
                        + ["type", "row.names.model.Vaccounted."]]
    model_call = pd.DataFrame({"call": [f"fa(nfactors={model.factors},fm='{model.fm}')"]})
    result = {"correlations": correlations, "npobs": model.np_obs, "residual_stats": residual_stats,
              "determinant_test": determinant_test, "bartlett_test": bartlett_test, "kmo_test": kmo_test,
              "loadings": loadings, "instruction_loading_critical_values": _LOADING_CRITICAL.copy(),
              "weights": model.weights}
    if scores:
        result["scores"] = model.scores

    if file is not None:
        try:
            from .functions_excel import excel_matrix, excel_critical_value
        except ImportError:
            from functions_excel import excel_matrix, excel_critical_value
        with pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter") as writer:
            excel_matrix(correlations.drop(columns="type"), writer, sheet="r")
            excel_matrix(model.np_obs, writer, sheet="n")
            excel_critical_value(residual_stats, writer, sheet="residual stats")
            excel_critical_value(fit_index, writer, sheet="fit_index")
            excel_critical_value(loadings, writer, sheet="loadings")
            excel_critical_value(_LOADING_CRITICAL, writer, sheet="loading critical values")
            excel_matrix(model.weights, writer, sheet="weights")
            excel_critical_value(determinant_test, writer, sheet="determinant")
            excel_critical_value(bartlett_test, writer, sheet="bartlett test")
            excel_critical_value(kmo_test.reset_index(names="variable"), writer, sheet="msa")
            excel_matrix(model_communality, writer, sheet="communalities")
            if scores:
                excel_matrix(model.scores, writer, sheet="scores")
            excel_critical_value(model_call, writer, sheet="call")
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os

    df_insurance = pd.read_csv("data/insurance.csv") if os.path.exists("data/insurance.csv") \
        else pd.read_csv("../data/insurance.csv")

    np.random.seed(0)
    cm = np.array([[1, .8, .8, .1, .1, .1], [.8, 1, .8, .1, .1, .1], [.8, .8, 1, .1, .1, .1],
                   [.1, .1, .1, 1, .8, .8], [.1, .1, .1, .8, 1, .8], [.1, .1, .1, .8, .8, 1]])
    data = np.random.multivariate_normal(mean=np.zeros(6), cov=cm, size=1000)
    df_sim = pd.DataFrame(data, columns=[f"V{i + 1}" for i in range(6)])

    print("=" * 80, "\n_fit_efa\n", "=" * 80, sep="")
    model = _fit_efa(df_sim, n_factors=2, rotation="oblimin", method="minres")
    print("loadings:\n", model.loadings.round(2))
    print("Phi (factor correlations):\n", model.Phi.round(2))
    print("RMSEA:", model.RMSEA['RMSEA'], "| rms:", model.rms, "| crms:", model.crms)

    print("\n" + "=" * 80, "\nplot_scree\n", "=" * 80, sep="")
    p = plot_scree(df_sim, base_size=15, title="")
    p.save("plot_scree_example.png", verbose=False)
    print("saved plot_scree_example.png")

    print("\n" + "=" * 80, "\nmodel_loadings\n", "=" * 80, sep="")
    print(model_loadings(model=model, cut=None, matrix_type="pattern"))

    print("\n" + "=" * 80, "\nplot_loadings\n", "=" * 80, sep="")
    result = plot_loadings(model=model, matrix_type="pattern", base_size=30)
    result['correlation_loadings'][0].savefig("plot_loadings_correlation.png")
    result['plot_barplot'].save("plot_loadings_barplot.png", verbose=False)
    print("saved plot_loadings_correlation.png, plot_loadings_barplot.png")

    print("\n" + "=" * 80, "\ncompute_residual_stats\n", "=" * 80, sep="")
    print(compute_residual_stats(model))

    print("\n" + "=" * 80, "\nreport_efa\n", "=" * 80, sep="")
    result2 = report_efa(model=model, df=df_sim, print_plot=False if False else None) if False else \
        report_efa(model=model, df=df_sim)
    print("keys:", list(result2.keys()))
    print("bartlett_test:\n", result2['bartlett_test'])
    print("kmo_test:\n", result2['kmo_test'])
