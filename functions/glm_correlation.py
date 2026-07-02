# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_CORRELATION.R.

Scope, per explicit choices made during this port:
  - report_choric_serial implements polyserial/biserial correlation
    only (two-step / method-of-moments estimators, matching
    psych::polyserial's/psych::biserial's non-ML default). tetrachoric
    and polychoric correlation need real iterative maximum-likelihood
    estimation of a latent bivariate-normal correlation from
    categorical thresholds — a substantial standalone numerical
    project, not attempted here; calling with those types raises
    NotImplementedError.

Other deviations, by design:
  - report_correlation uses pingouin.pairwise_corr (already a project
    dependency) instead of hand-rolling an equivalent to
    psych::corr.test. t-statistics and standard errors (which
    pairwise_corr doesn't report directly) are derived from r and n via
    the standard formulas; confidence intervals use a Fisher
    z-transform (matching the common approach, though not necessarily
    byte-identical to psych::corr.test's exact CI table structure).
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats
import pingouin as pg
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient2, theme_bw, theme,
    element_text, element_blank, labs, geom_point, geom_line, lims,
)

try:
    from .functions import change_data_type
    from .functions_matrix import matrix_triangle
    from .functions_plot import plot_multiplot, report_pdf
    from .functions_excel import excel_matrix, excel_generic_format
    from .glm_linear_regression import plot_scatterplot
except ImportError:
    from functions import change_data_type
    from functions_matrix import matrix_triangle
    from functions_plot import plot_multiplot, report_pdf
    from functions_excel import excel_matrix, excel_generic_format
    from glm_linear_regression import plot_scatterplot
##########################################################################################
# CORRELATION MATRIX PLOT
##########################################################################################
def plot_corrplot(mydata, title="", base_size=10, fill_limits=(-1, 0, 1)):
    """
    Correlation matrix heatmap: upper triangle only, values annotated,
    diverging fill scale.

    Parameters:
    mydata (array-like or pandas.DataFrame): Correlation matrix.
    title (str, optional): Plot title. Defaults to "".
    base_size (int, optional): Base font size for theme_bw(). Defaults to 10.
    fill_limits (tuple, optional): (low, mid, high) values for the fill
        scale. Defaults to (-1, 0, 1).

    Returns:
    plotnine.ggplot

    Examples:
    >>> plot_corrplot(df_insurance.select_dtypes("number").corr(), title="Correlation")
    """
    mydata = pd.DataFrame(mydata).round(2)
    upper_tri = matrix_triangle(mydata.to_numpy(), off_diagonal=np.nan, diagonal=None, type="upper")
    upper_df = pd.DataFrame(upper_tri, index=mydata.index, columns=mydata.columns)

    melted = (upper_df.reset_index()
              .melt(id_vars='index', var_name='Var2', value_name='value')
              .rename(columns={'index': 'Var1'})
              .dropna(subset=['value']))
    melted['Var1'] = pd.Categorical(melted['Var1'], categories=list(mydata.index), ordered=True)
    melted['Var2'] = pd.Categorical(melted['Var2'], categories=list(mydata.columns), ordered=True)

    p = (ggplot(melted, aes(x='Var2', y='Var1', fill='value'))
         + geom_tile(color='white')
         + scale_fill_gradient2(midpoint=fill_limits[1], limits=(fill_limits[0], fill_limits[2]))
         + geom_text(aes(x='Var2', y='Var1', label='value'), color='black', size=base_size / 4)
         + theme_bw(base_size=base_size)
         + theme(axis_text_x=element_text(rotation=45, hjust=1),
                 axis_title_x=element_blank(),
                 axis_title_y=element_blank(),
                 panel_grid_major=element_blank(),
                 panel_grid_minor=element_blank(),
                 panel_border=element_blank(),
                 panel_background=element_blank(),
                 legend_position='none')
         + labs(title=title))
    return p
##########################################################################################
# POWER
##########################################################################################
def _pwr_r_test(n, r, sig_level=0.05, alternative="two-sided"):
    """Fisher z-approximation power for a correlation test, matching R's pwr::pwr.r.test."""
    zr = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    if alternative == "two-sided":
        z_crit = scipy.stats.norm.ppf(1 - sig_level / 2)
        return scipy.stats.norm.cdf(zr / se - z_crit) + scipy.stats.norm.cdf(-zr / se - z_crit)
    elif alternative == "greater":
        z_crit = scipy.stats.norm.ppf(1 - sig_level)
        return scipy.stats.norm.cdf(zr / se - z_crit)
    elif alternative == "less":
        z_crit = scipy.stats.norm.ppf(1 - sig_level)
        return scipy.stats.norm.cdf(-zr / se - z_crit)
    raise ValueError(f"Unknown alternative: {alternative!r}")


def compute_power_r(n=100, r=None, sig_level=0.05, alternative="two-sided", title="", base_size=10):
    """
    Compute and plot a statistical power curve for a correlation test,
    across sample sizes 10..n.

    Parameters:
    n (int, optional): Maximum sample size. Defaults to 100.
    r (float): Correlation coefficient to power against.
    sig_level (float, optional): Alpha (Type I error rate). Defaults to 0.05.
    alternative (str, optional): "two-sided" (default), "greater", or "less".
    title (str, optional): Plot title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.

    Returns:
    dict: {'plot': plotnine.ggplot, 'power_table': pandas.DataFrame}.

    Examples:
    >>> compute_power_r(n=100, r=.5, sig_level=.05, alternative="two-sided")
    """
    rows = [{'n': i, 'r': r, 'p': sig_level, 'power': _pwr_r_test(i, r, sig_level, alternative),
             'alternative': alternative, 'method': 'approximate correlation power calculation (arctanh transformation)'}
            for i in range(10, n + 1)]
    df_power = pd.DataFrame(rows)

    plot = (ggplot(df_power, aes(x='n', y='power'))
            + geom_point(alpha=1)
            + geom_line()
            + labs(x="Observations", y="Power", title=title,
                   caption=f"r power curve: r = {round(r, 2)}, alpha = {sig_level}")
            + lims(y=(0, 1))
            + theme_bw(base_size=base_size))
    return {'plot': plot, 'power_table': df_power}


def compute_power_r_matrix(m, **kwargs):
    """
    Compute power curves for the min/max/mean/median absolute
    correlation in a correlation matrix.

    Parameters:
    m (array-like): Correlation matrix (diagonal is ignored).
    **kwargs: Forwarded to compute_power_r (e.g. n, sig_level, alternative).

    Returns:
    dict: {'plot': list of 4 matplotlib Figures (via plot_multiplot),
    'power_table': combined pandas.DataFrame}.

    Examples:
    >>> compute_power_r_matrix(m=df_insurance.select_dtypes("number").corr().to_numpy(), n=100)
    """
    m = np.array(m, dtype=float)
    np.fill_diagonal(m, np.nan)
    abs_m = np.abs(m)

    minimum = compute_power_r(r=np.nanmin(abs_m), title="Min absolute r in Correlation Matrix", **kwargs)
    maximum = compute_power_r(r=np.nanmax(abs_m), title="Max absolute r in Correlation Matrix", **kwargs)
    mean_ = compute_power_r(r=np.nanmean(abs_m), title="Mean absolute r in Correlation Matrix", **kwargs)
    median_ = compute_power_r(r=np.nanmedian(abs_m), title="Median absolute r in Correlation Matrix", **kwargs)

    df_power = pd.concat([minimum['power_table'], maximum['power_table'],
                           mean_['power_table'], median_['power_table']], ignore_index=True)
    plot = plot_multiplot(minimum['plot'], maximum['plot'], mean_['plot'], median_['plot'])
    return {'plot': plot, 'power_table': df_power}
##########################################################################################
# CORRELATION MATRIX
##########################################################################################
def report_correlation(x, y=None, use="pairwise", method="pearson", adjust="holm", alpha=.05, ci=True,
                        file=None, w=10, h=10, base_size=20, scatterplot=True):
    """
    Full pairwise correlation report: r, r-squared, p (raw and
    adjusted), t, n, and standard error matrices (lower triangle), plus
    a confidence-interval table, optionally written to a multi-sheet
    Excel workbook and a correlation-plot/scatterplot PDF.

    Parameters:
    x (pandas.DataFrame): Data to correlate. Non-numeric columns are
        coerced via change_data_type(type="numeric").
    y: Unused placeholder for R signature parity (pingouin.pairwise_corr
        only supports correlating columns of one data frame).
    use (str, optional): "pairwise" (default) or "complete" (listwise
        deletion) missing-data handling.
    method (str, optional): "pearson" (default), "spearman", or "kendall".
    adjust (str, optional): Multiple-comparison p-value adjustment
        method (pingouin padjust values, e.g. "holm", "bonferroni",
        "fdr_bh", "none"). Defaults to "holm".
    alpha (float, optional): Confidence interval alpha. Defaults to 0.05.
    ci (bool, optional): Present for R signature parity; CI is always computed.
    file (str, optional): Output filename (without extension) for the
        .xlsx report and .pdf plots. Defaults to None (no files written).
    w, h (float, optional): PDF page width/height in inches. Default 10.
    base_size (int, optional): Base font size for plots. Defaults to 20.
    scatterplot (bool, optional): If True (default), also generate
        pairwise scatterplots (see plot_scatterplot).

    Returns:
    dict: {'r_lower', 'r_squared_lower', 'p_lower', 'p_lower_adjusted',
    't_lower', 'n_lower', 'se_lower' (all pandas.DataFrame matrices),
    'ci' (long-format DataFrame), 'call' (arguments used)}.

    Examples:
    >>> report_correlation(x=df_insurance[["age", "bmi", "charges"]])
    """
    x = pd.DataFrame(x)
    if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x.columns):
        x = change_data_type(x, type="numeric")

    nan_policy = 'pairwise' if use == 'pairwise' else 'listwise'
    pairwise = pg.pairwise_corr(x, method=method, padjust=adjust, nan_policy=nan_policy)

    cols = list(x.columns)

    def _lower_matrix(value_col):
        mat = pd.DataFrame(np.nan, index=cols, columns=cols)
        for _, row in pairwise.iterrows():
            mat.loc[row['Y'], row['X']] = row[value_col]
        return mat

    r_mat = _lower_matrix('r')
    p_mat = _lower_matrix('p_unc')
    p_adj_mat = _lower_matrix('p_corr') if adjust != 'none' and 'p_corr' in pairwise.columns else p_mat
    n_mat = _lower_matrix('n')
    r_squared_mat = r_mat ** 2
    t_mat = r_mat * np.sqrt((n_mat - 2) / (1 - r_squared_mat))
    se_mat = np.sqrt((1 - r_squared_mat) / (n_mat - 2))

    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    ci_rows = []
    for _, row in pairwise.iterrows():
        z = np.arctanh(row['r'])
        se_z = 1 / np.sqrt(row['n'] - 3)
        ci_rows.append({'pair': f"{row['X']}-{row['Y']}", 'r': row['r'],
                         'lower': np.tanh(z - z_crit * se_z), 'upper': np.tanh(z + z_crit * se_z),
                         'p': row['p_unc']})
    ci_df = pd.DataFrame(ci_rows)

    call_df = pd.DataFrame({
        'function_arguments': ["Use", "Method", "Adjustment for Probability values", "Alpha", "Confidence Interval"],
        'function_values': [use, method, adjust, alpha, ci],
    })

    correlation_result = {
        'r_lower': r_mat, 'r_squared_lower': r_squared_mat, 'p_lower': p_mat, 'p_lower_adjusted': p_adj_mat,
        't_lower': t_mat, 'n_lower': n_mat, 'se_lower': se_mat, 'ci': ci_df, 'call': call_df,
    }

    corrplot = plot_corrplot(x.corr(method=method), base_size=base_size)
    report_pdf(corrplot, file=file, title="corrplot", w=w, h=h, print_plot=(file is None))

    if scatterplot:
        scatter_plots = plot_scatterplot(x, base_size=base_size, coord_equal=False, all_orders=False)
        figures = [g.figure for g in scatter_plots.values() if g is not None]
        report_pdf(plotlist=figures, file=file, title="scatterplot", w=w, h=h, print_plot=(file is None))

    if file is not None:
        writer = pd.ExcelWriter(f"{file}.xlsx", engine='xlsxwriter')
        excel_matrix(r_mat, writer, sheetname="r", decimals=2)
        excel_matrix(r_squared_mat, writer, sheetname="r_squared", decimals=2)
        excel_matrix(p_mat, writer, sheetname="p", decimals=2)
        excel_matrix(p_adj_mat, writer, sheetname="p_adjusted", decimals=2)
        excel_matrix(t_mat, writer, sheetname="t", decimals=2)
        excel_matrix(n_mat, writer, sheetname="N", decimals=0)
        excel_matrix(se_mat, writer, sheetname="SE", decimals=2)
        excel_generic_format(ci_df, writer, sheetname="CI")
        excel_generic_format(call_df, writer, sheetname="Call")
        writer._save()
        writer.close()

    return correlation_result
##########################################################################################
# TETRACHORIC POLYCHORIC BISERIAL POLYSERIAL
##########################################################################################
def polyserial(continuous, ordinal):
    """
    Two-step (non-ML) polyserial correlation between a continuous
    variable and an ordinal variable, assuming the ordinal variable
    reflects a discretized underlying continuum. Matches
    psych::polyserial's default (ML=FALSE) estimator.

    Parameters:
    continuous (array-like): Continuous variable.
    ordinal (array-like): Ordinal/discrete variable (few distinct levels).

    Returns:
    float: Estimated polyserial correlation.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> latent = np.random.normal(size=500)
    >>> continuous = latent + np.random.normal(scale=0.5, size=500)
    >>> ordinal = pd.cut(latent, bins=[-np.inf, -1, 0, 1, np.inf], labels=[1, 2, 3, 4])
    >>> polyserial(continuous, ordinal)
    """
    continuous = np.asarray(continuous, dtype=float)
    codes = pd.Categorical(ordinal, ordered=True).codes.astype(float)
    valid = ~np.isnan(continuous) & (codes >= 0)
    continuous, codes = continuous[valid], codes[valid]
    n = len(continuous)
    if n < 3:
        return np.nan

    r_pb = np.corrcoef(continuous, codes)[0, 1]
    _, counts = np.unique(codes, return_counts=True)
    cum_props = np.cumsum(counts)[:-1] / n
    z = scipy.stats.norm.ppf(cum_props)
    phi_sum = np.sum(scipy.stats.norm.pdf(z))
    if phi_sum == 0:
        return np.nan
    return r_pb * codes.std(ddof=1) / phi_sum


def biserial(continuous, dichotomous):
    """
    Biserial correlation between a continuous variable and a
    dichotomous variable, assuming the dichotomy reflects an underlying
    continuum (unlike point-biserial, which assumes a genuine
    dichotomy). Matches psych::biserial's default estimator.

    Parameters:
    continuous (array-like): Continuous variable.
    dichotomous (array-like): Two-level variable.

    Returns:
    float: Estimated biserial correlation.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> latent = np.random.normal(size=500)
    >>> continuous = latent + np.random.normal(scale=0.5, size=500)
    >>> dichotomous = (latent > 0).astype(int)
    >>> biserial(continuous, dichotomous)
    """
    continuous = np.asarray(continuous, dtype=float)
    codes = pd.Categorical(dichotomous).codes.astype(float)
    valid = ~np.isnan(continuous) & (codes >= 0)
    continuous, codes = continuous[valid], codes[valid]

    p = codes.mean()
    q = 1 - p
    r_pb = np.corrcoef(continuous, codes)[0, 1]
    z = scipy.stats.norm.ppf(q)
    phi_z = scipy.stats.norm.pdf(z)
    return r_pb * np.sqrt(p * q) / phi_z


def report_choric_serial(x, y=None, file=None, w=10, h=10, type="polyserial"):
    """
    Report polyserial or biserial correlation between every column of
    x and every column of y (or x with itself if y is None), as a
    matrix, optionally with a corrplot PDF and Excel report.

    Note: only type="polyserial" and type="biserial" are implemented.
    tetrachoric/polychoric correlation need real iterative maximum-
    likelihood estimation of a latent bivariate-normal correlation from
    categorical thresholds and are not ported (raises NotImplementedError).

    Parameters:
    x (pandas.DataFrame): Continuous (for polyserial) or arbitrary (for
        biserial, expects each y column to be 2-level) variables.
    y (pandas.DataFrame, optional): Ordinal (polyserial) or dichotomous
        (biserial) variables. Defaults to x itself.
    file (str, optional): Output filename (without extension). Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10.
    type (str, optional): "polyserial" (default) or "biserial".

    Returns:
    pandas.DataFrame: Correlation matrix, index = x's columns, columns = y's columns.

    Examples:
    >>> report_choric_serial(x=df_insurance[["age", "bmi"]], y=df_insurance[["children"]], type="polyserial")
    """
    if type not in ("polyserial", "biserial"):
        raise NotImplementedError(
            f"type={type!r} is not implemented — only 'polyserial' and 'biserial' are ported. "
            "tetrachoric/polychoric correlation need real iterative ML estimation of a latent "
            "bivariate-normal correlation, which is out of scope for this port."
        )
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y) if y is not None else x_df
    func = polyserial if type == "polyserial" else biserial

    result = pd.DataFrame(
        {ycol: [func(x_df[xcol], y_df[ycol]) for xcol in x_df.columns] for ycol in y_df.columns},
        index=x_df.columns,
    )

    corrplot = plot_corrplot(result, title=type)
    report_pdf(corrplot, w=w, h=h, file=file, title=type, print_plot=(file is None))

    if file is not None:
        writer = pd.ExcelWriter(f"{file}.xlsx", engine='xlsxwriter')
        excel_generic_format(result.reset_index(), writer, sheetname=type)
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
    numeric_df = df_insurance[["age", "bmi", "children", "charges"]]

    print("=" * 80, "\nplot_corrplot\n", "=" * 80, sep="")
    p = plot_corrplot(numeric_df.corr(), title="Correlation")
    p.save("plot_corrplot_example.png", verbose=False)
    print("saved plot_corrplot_example.png")

    print("\n" + "=" * 80, "\ncompute_power_r\n", "=" * 80, sep="")
    res = compute_power_r(n=100, r=.5, sig_level=.05, alternative="two-sided")
    print(res['power_table'].tail())

    print("\n" + "=" * 80, "\ncompute_power_r_matrix\n", "=" * 80, sep="")
    res2 = compute_power_r_matrix(m=numeric_df.corr().to_numpy(), n=100)
    print(res2['power_table'].groupby('n', as_index=False).first().head())

    print("\n" + "=" * 80, "\nreport_correlation\n", "=" * 80, sep="")
    result = report_correlation(x=numeric_df, scatterplot=True)
    print("r_lower:\n", result['r_lower'])
    print("ci:\n", result['ci'])

    print("\n" + "=" * 80, "\npolyserial / biserial\n", "=" * 80, sep="")
    import numpy as np
    np.random.seed(1)
    latent = np.random.normal(size=500)
    continuous = latent + np.random.normal(scale=0.5, size=500)
    ordinal = pd.cut(latent, bins=[-np.inf, -1, 0, 1, np.inf], labels=[1, 2, 3, 4])
    dichotomous = (latent > 0).astype(int)
    print("polyserial (expect high, since ordinal derives from latent):", polyserial(continuous, ordinal))
    print("biserial (expect high, since dichotomous derives from latent):", biserial(continuous, dichotomous))

    print("\n" + "=" * 80, "\nreport_choric_serial\n", "=" * 80, sep="")
    df_cont = pd.DataFrame({'x1': continuous, 'x2': continuous * 0.8 + np.random.normal(scale=0.3, size=500)})
    df_ord = pd.DataFrame({'y1': ordinal})
    print(report_choric_serial(x=df_cont, y=df_ord, type="polyserial"))
    try:
        report_choric_serial(x=df_cont, y=df_ord, type="tetrachoric")
    except NotImplementedError as e:
        print("Expected NotImplementedError:", e)
