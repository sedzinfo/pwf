# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_EFA.R.

Scope, per explicit choices made during this port (factor_analyzer
doesn't expose most of psych::fa()'s rich model object):
  - A lightweight EFA model wrapper (_fit_efa) is built around
    factor_analyzer.FactorAnalyzer, exposing the R-psych::fa-style
    attributes needed by model_loadings/plot_loadings/report_efa:
    loadings (pattern), Structure, Phi, r, model (reproduced
    correlations), residual, communality/communalities, uniquenesses,
    complexity (Hoffman's index), eigenvalues, RMSEA (point estimate
    only), rms/crms.
  - TLI, CFI, BIC/SABIC/EBIC/ESABIC, and RMSEA confidence bounds are
    NOT computed — these need a null/baseline-model chi-square
    comparison (TLI/CFI/*IC family) or inverting a noncentral
    chi-square CDF (RMSEA CI), both easy to get subtly wrong without a
    reference implementation to verify against. They're present as
    fields (set to NaN) for structural parity but not populated.
  - RMSEA/chi-square use the classical residual-based approximation
    chi_sq ~= (n - 1 - (2p+5)/6 - 2*factors/3) * sum(squared
    off-diagonal residuals), the standard formula usable for any
    extraction method (Harman 1976), not just ML.
  - Attribute names with dots in R (e.g. model$e.values) become
    underscored Python attributes (e_values) since Python identifiers
    can't contain dots.

Environment note: factor_analyzer 0.5.1 (latest on PyPI) calls
sklearn's check_array(force_all_finite=...), a parameter renamed to
ensure_all_finite in scikit-learn >=1.6 and removed by 1.9 (installed
here) — factor_analyzer.fit() is completely broken without a shim.
This module monkey-patches check_array inside factor_analyzer's own
module at import time to translate the old kwarg, so FactorAnalyzer
actually works in this environment.

This file replaces the previous glm_efa.py (a simpler, self-contained
report_efa that fit its own model from raw data with a different
signature). Confirmed nothing else in this codebase called that
version before replacing it.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
from types import SimpleNamespace
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_hline, geom_line, geom_point, geom_tile, geom_text, geom_bar,
    theme_bw, theme, scale_x_continuous, scale_fill_gradient2, labs, annotate,
    element_blank, element_text, coord_flip, facet_wrap, lims,
)

try:
    from .functions_plot import plot_multiplot, report_pdf
    from .functions_strings import str_proper, call_to_string
    from .explore_assumptions import plot_histogram
except ImportError:
    from functions_plot import plot_multiplot, report_pdf
    from functions_strings import str_proper, call_to_string
    from explore_assumptions import plot_histogram

import sklearn.utils as _sku
_orig_check_array = _sku.check_array


def _patched_check_array(*args, **kwargs):
    """Compat shim: factor_analyzer 0.5.1 still passes sklearn's renamed force_all_finite kwarg."""
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)


import factor_analyzer.factor_analyzer as _fa_mod
_fa_mod.check_array = _patched_check_array
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
##########################################################################################
# FIT EFA (internal helper, R psych::fa-like model wrapper)
##########################################################################################
def _fit_efa(df, n_factors, rotation="oblimin", method="minres", **kwargs):
    """
    Fit a FactorAnalyzer model and wrap it into a SimpleNamespace with
    R psych::fa-style attributes, for use by model_loadings/
    plot_loadings/compute_residual_stats/report_efa.
    """
    df_complete = df.dropna()
    n = len(df_complete)
    p = df_complete.shape[1]

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, **kwargs)
    fa.fit(df_complete)

    loadings = pd.DataFrame(fa.loadings_, index=df_complete.columns,
                             columns=[f"PA{i + 1}" for i in range(n_factors)])
    phi = (pd.DataFrame(fa.phi_, index=loadings.columns, columns=loadings.columns)
           if getattr(fa, 'phi_', None) is not None
           else pd.DataFrame(np.eye(n_factors), index=loadings.columns, columns=loadings.columns))
    structure = pd.DataFrame(loadings.to_numpy() @ phi.to_numpy(), index=loadings.index, columns=loadings.columns)

    r = df_complete.corr()
    reproduced = pd.DataFrame(loadings.to_numpy() @ phi.to_numpy() @ loadings.to_numpy().T,
                               index=r.index, columns=r.columns)
    residual = r - reproduced

    communalities = pd.Series(np.sum(loadings.to_numpy() * structure.to_numpy(), axis=1), index=loadings.index)
    uniquenesses = 1 - communalities
    sq = loadings.to_numpy() ** 2
    complexity = pd.Series(np.sum(sq, axis=1) ** 2 / np.sum(sq ** 2, axis=1), index=loadings.index)

    dof = int(((p - n_factors) ** 2 - (p + n_factors)) / 2)
    offdiag_resid = residual.to_numpy()[np.triu_indices(p, k=1)]
    chi = max((n - 1 - (2 * p + 5) / 6 - (2 * n_factors) / 3) * np.sum(offdiag_resid ** 2), 0.0)
    rmsea = float(np.sqrt(max((chi / dof - 1) / (n - 1), 0))) if dof > 0 else np.nan
    rms = float(np.sqrt(np.mean(offdiag_resid ** 2)))
    crms = float(np.sqrt(np.sum(offdiag_resid ** 2) / dof)) if dof > 0 else np.nan

    reduced_r_arr = r.to_numpy().copy()
    np.fill_diagonal(reduced_r_arr, communalities.to_numpy())
    values = np.sort(np.linalg.eigvalsh(reduced_r_arr))[::-1]
    e_values = np.sort(np.linalg.eigvalsh(r.to_numpy()))[::-1]

    weights = None
    try:
        weights = pd.DataFrame(fa.get_factor_scores_weights() if hasattr(fa, 'get_factor_scores_weights')
                                else fa.get_uniquenesses(), index=df_complete.columns)
    except Exception:
        weights = None

    return SimpleNamespace(
        loadings=loadings, Structure=structure, Phi=phi, r=r, model=reproduced, residual=residual,
        communality=communalities, communalities=communalities, uniquenesses=uniquenesses,
        complexity=complexity, values=values, e_values=e_values,
        n_obs=n, np_obs=pd.DataFrame(n, index=r.index, columns=r.columns),
        factors=n_factors, dof=dof, chi=chi, STATISTIC=chi,
        RMSEA={'RMSEA': rmsea, 'lower': np.nan, 'upper': np.nan, 'confidence': np.nan},
        rms=rms, crms=crms,
        TLI=np.nan, CFI=np.nan, BIC=np.nan, SABIC=np.nan, EBIC=np.nan, ESABIC=np.nan,
        fit=np.nan, fit_off=np.nan,
        weights=weights, scores=None, rotation=rotation, method=method, fm=method,
        Call=f"_fit_efa(n_factors={n_factors}, rotation={rotation!r}, method={method!r})",
    )
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
    Build a pattern and/or structure loadings table, with small
    loadings blanked out below a cutoff and rows sorted by their
    dominant factor.

    Parameters:
    model: An EFA model from _fit_efa (or with matching attributes:
        .loadings, .Structure, .n_obs).
    cut (float, optional): Loadings with absolute value below this are
        blanked (""). If None (default), picked from a standard
        sample-size-to-critical-loading table (Stevens, 2002-style).
    matrix_type (str, optional): "pattern", "structure", or "all"
        (both, stacked). Defaults to "pattern".
    sort (bool, optional): If True (default), sort rows by dominant
        factor, then by loading magnitude within factor.

    Returns:
    pandas.DataFrame: Columns "Matrix", "variable", one column per factor.

    Examples:
    >>> model = _fit_efa(df_insurance.select_dtypes("number"), n_factors=2)
    >>> model_loadings(model=model, cut=None, matrix_type="pattern")
    """
    n = model.n_obs
    if cut is None:
        idx = (_LOADING_CRITICAL['sample'] - n).abs().idxmin()
        cut = _LOADING_CRITICAL.loc[idx, 'critical_loading']

    def _prep(mat, label):
        mat = mat.copy()
        if sort:
            dominant = mat.abs().idxmax(axis=1)
            magnitude = mat.abs().max(axis=1)
            order = pd.DataFrame({'dominant': dominant, 'magnitude': -magnitude}).sort_values(['dominant', 'magnitude'])
            mat = mat.loc[order.index]
        display = mat.mask(mat.abs() < cut, "")
        display.insert(0, 'variable', display.index)
        display.insert(0, 'Matrix', label)
        return display.reset_index(drop=True)

    pattern = _prep(model.loadings, "Pattern")
    structure = _prep(model.Structure, "Structure")

    if matrix_type == "pattern":
        return pattern
    elif matrix_type == "structure":
        return structure
    elif matrix_type == "all":
        return pd.concat([pattern, structure], ignore_index=True)
    raise ValueError(f"Unknown matrix_type: {matrix_type!r}")
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
def report_efa(model, df, file=None, w=10, h=5, cut=0, base_size=10, scores=False):
    """
    Full EFA report: correlation/reproduced/residual matrices,
    determinant test, Bartlett's test, KMO, fit indices, loadings,
    scree/loadings/residual-histogram plots, optionally written to a
    multi-sheet Excel workbook and a PDF of plots.

    Parameters:
    model: An EFA model from _fit_efa.
    df (pandas.DataFrame): The original data the model was fit on
        (used for Bartlett's test, KMO, and the determinant test).
    file (str, optional): Output filename (without extension). Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10, 5.
    cut (float, optional): Loading cutoff forwarded to model_loadings. Defaults to 0.
    base_size (int, optional): Base font size for plots. Defaults to 10.
    scores (bool, optional): If True, include factor scores in the
        result/Excel export. Defaults to False.

    Returns:
    dict: correlations, npobs, residual_stats, determinant_test,
    bartlett_test, kmo_test, loadings, weights, and (if scores=True) scores.

    Examples:
    >>> model = _fit_efa(df_insurance.select_dtypes("number"), n_factors=2)
    >>> report_efa(model=model, df=df_insurance.select_dtypes("number"))
    """
    df_complete = df.dropna()
    correlationmatrix = df_complete.corr(numeric_only=True)

    chi_square_value, p_value = calculate_bartlett_sphericity(df_complete)
    bartlett_test = pd.DataFrame({'x_squared[bartlett]': [chi_square_value], 'df[bartlett]': [np.nan],
                                   'p[bartlett]': [p_value]})

    kmo_all, kmo_model = calculate_kmo(df_complete)
    kmo_test = pd.DataFrame({'Overall_MSA': kmo_model, 'MSA': kmo_all}, index=df_complete.columns)

    determinant = float(np.linalg.det(correlationmatrix.to_numpy()))
    determinant_test = pd.DataFrame({'determinant': [determinant], 'above_critical': [determinant > 0.00001]})

    fit_index = pd.DataFrame({
        'RMSEA': [model.RMSEA['RMSEA']], 'lower': [model.RMSEA['lower']], 'upper': [model.RMSEA['upper']],
        'confidence': [model.RMSEA['confidence']], 'RMS': [model.rms], 'CRMS': [model.crms],
        'TLI': [model.TLI], 'CFI': [model.CFI], 'BIC': [model.BIC], 'SABIC': [model.SABIC],
        'EBIC': [model.EBIC], 'ESABIC': [model.ESABIC], 'fit': [model.fit], 'fit_off': [model.fit_off],
    })

    plots = {
        'plot_scree': plot_scree(df_complete, base_size=base_size, title=""),
        'plot_loadings_structure': plot_loadings(model, matrix_type="structure", base_size=base_size),
        'plot_loadings_pattern': plot_loadings(model, matrix_type="pattern", base_size=base_size),
    }
    residual_offdiag = model.residual.to_numpy()[np.triu_indices_from(model.residual.to_numpy(), k=1)]
    plots['plot_residual_histogram'] = plot_histogram(
        pd.DataFrame({'data': residual_offdiag}), bins=30, title="Histogram of residuals", base_size=base_size)

    flat_plots = [plots['plot_scree'], plots['plot_loadings_structure']['plot_barplot'],
                  plots['plot_loadings_pattern']['plot_barplot'], plots['plot_residual_histogram']]
    report_pdf(plotlist=flat_plots, file=file, w=w, h=h, print_plot=(file is None))

    residual_stats = compute_residual_stats(model)

    model_communality = pd.DataFrame({
        'complexity': model.complexity, 'communality': model.communality, 'communalities': model.communalities,
        'uniquenesses': model.uniquenesses, 'values': pd.Series(model.values), 'e_values': pd.Series(model.e_values),
    })

    correlations = pd.concat([
        model.model.assign(type="reproduced correlations"),
        model.r.assign(type="observed correlations"),
        model.residual.assign(type="residual correlations"),
    ])

    loadings = model_loadings(model, cut, matrix_type="all")
    model_call = pd.DataFrame({'call': [call_to_string(model)]})

    result = {
        'correlations': correlations, 'npobs': model.np_obs, 'residual_stats': residual_stats,
        'determinant_test': determinant_test, 'bartlett_test': bartlett_test, 'kmo_test': kmo_test,
        'loadings': loadings, 'weights': model.weights,
    }
    if scores:
        result['scores'] = model.scores

    if file is not None:
        try:
            from .functions_excel import excel_matrix, excel_critical_value, excel_generic_format
        except ImportError:
            from functions_excel import excel_matrix, excel_critical_value, excel_generic_format
        writer = pd.ExcelWriter(f"{file}.xlsx", engine='xlsxwriter')
        excel_matrix(correlations.select_dtypes('number'), writer, sheetname="r")
        excel_matrix(model.np_obs, writer, sheetname="n")
        excel_generic_format(residual_stats, writer, sheetname="residual stats")
        excel_critical_value(fit_index, writer, sheetname="fit_index")
        excel_generic_format(loadings, writer, sheetname="loadings")
        excel_critical_value(determinant_test, writer, sheetname="determinant")
        excel_critical_value(bartlett_test, writer, sheetname="bartlett test")
        excel_generic_format(kmo_test.reset_index(), writer, sheetname="msa")
        excel_matrix(model_communality, writer, sheetname="communalities")
        if scores and model.scores is not None:
            excel_matrix(model.scores, writer, sheetname="scores")
        excel_generic_format(model_call, writer, sheetname="call")
        writer._save()
        writer.close()

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
