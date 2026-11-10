# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_CORRELATION.R.

tetrachoric, polychoric, polyserial and biserial are native ports of
the psych functions of the same names (tests/test_glm_correlation.py
compares them with psych); the bivariate normal probabilities use Genz's
algorithm (as mnormt::sadmvn) and the likelihood is optimised to tighter
tolerance than R's optimize, so values agree with psych to about 1e-4.
report_choric_serial matches rwf, including its default type
"tetrachoric".

report_correlation matches rwf on a native port of psych::corr.test
(corr_test), with R's p.adjust and abbreviate (in _rcompat).
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient2, theme_bw, theme,
    element_text, element_blank, labs, geom_point, geom_line, lims,
)

try:
    from ._rcompat import r_abbreviate, r_label, r_make_names, r_p_adjust
    from .functions import change_data_type
    from .functions_matrix import matrix_triangle
    from .functions_plot import plot_multiplot, report_pdf
    from .glm_linear_regression import plot_scatterplot
except ImportError:
    from _rcompat import r_abbreviate, r_label, r_make_names, r_p_adjust
    from functions import change_data_type
    from functions_matrix import matrix_triangle
    from functions_plot import plot_multiplot, report_pdf
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
def _r_cor(x, y, use, method):
    """R's cor(x, y, use, method): pairwise or complete cases; spearman ranks each pair's complete cases."""
    x_values, y_values = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    if use == "complete":
        rows = ~(np.isnan(x_values).any(axis=1) | np.isnan(y_values).any(axis=1))
        x_values, y_values = x_values[rows], y_values[rows]
    r = np.full((x_values.shape[1], y_values.shape[1]), np.nan)
    for i in range(x_values.shape[1]):
        for j in range(y_values.shape[1]):
            a, b = x_values[:, i], y_values[:, j]
            keep = ~(np.isnan(a) | np.isnan(b))
            a, b = a[keep], b[keep]
            if len(a) < 2:
                continue
            if method == "kendall":
                r[i, j] = scipy.stats.kendalltau(a, b).statistic
                continue
            if method == "spearman":
                a, b = scipy.stats.rankdata(a), scipy.stats.rankdata(b)
            a, b = a - a.mean(), b - b.mean()
            denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
            r[i, j] = np.sum(a * b) / denominator if denominator > 0 else np.nan
    return r


def _fisherz(r):
    with np.errstate(divide="ignore"):
        return 0.5 * np.log((1 + r) / (1 - r))


def _fisherz2r(z):
    return (np.exp(2 * z) - 1) / (1 + np.exp(2 * z))


def _lower_tri(shape, diag=False):
    rows, columns = np.indices(shape)
    return rows > columns if not diag else rows >= columns


def _column_major(matrix, mask):
    """matrix[mask] in R's (column-major) order."""
    return matrix.T[mask.T]


def corr_test(x, y=None, use="pairwise", method="pearson", adjust="holm", alpha=0.05, ci=True, minlength=5):
    """
    Correlations with their t tests, as psych::corr.test(x, y, use, method,
    adjust, alpha, ci) with normal = TRUE.

    r is R's cor (pairwise or complete cases; spearman ranks and kendall's
    tau-b are computed on each pair's complete cases); n counts each pair's
    complete cases and becomes one number when they are all equal or use is
    "complete" (psych then takes the smallest pairwise count). For every r,
    t = r * sqrt(n - 2) / sqrt(1 - r^2), p is its two-sided t probability
    and se = sqrt((1 - r^2) / (n - 2)). Without y the upper triangle of p
    holds the p values adjusted by p.adjust(adjust); with y the adjusted
    values are in p_adj. The confidence intervals use Fisher's z; their
    adjusted bounds use alpha / (2 * number of correlations), or for holm
    alpha / (2 * rank of |z|).

    Parameters:
    x (pandas.DataFrame): Numeric variables.
    y (pandas.DataFrame, optional): Second set of variables. Defaults to None.
    use (str, optional): "pairwise" (default) or "complete".
    method (str, optional): "pearson" (default), "spearman" or "kendall".
    adjust (str, optional): p.adjust method. Defaults to "holm".
    alpha (float, optional): Confidence interval alpha. Defaults to 0.05.
    ci (bool, optional): Compute confidence intervals. Defaults to True.
    minlength (int, optional): Length of the abbreviated names of the CI rows. Defaults to 5.

    Returns:
    dict: r, n, t, p, p_adj, se (DataFrames, or n a number), sef, ci, ci2 and ci_adj (DataFrames), sym, adjust.

    Examples:
    >>> corr_test(df_insurance[["age", "bmi", "charges"]])["ci"]
    """
    x = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
    sym = y is None
    y = x if sym else pd.DataFrame(y).apply(pd.to_numeric, errors="coerce")
    r = _r_cor(x, y, use, method)
    n = (x.notna().to_numpy(int).T @ y.notna().to_numpy(int)).astype(float)
    if use == "complete" or n.min() == n.max():
        n = float(n.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
        p = 2 * scipy.stats.t.sf(np.abs(t), n - 2)
        se = np.sqrt((1 - r * r) / (n - 2))
    p = np.minimum(p, 1)
    pa = p.copy()
    if adjust != "none":
        if sym:
            transposed = p.T.copy()
            pa = r_p_adjust(transposed[_lower_tri(p.shape)], adjust)  # p[upper.tri(p)], column-major
            transposed[_lower_tri(p.shape)] = pa
            p = transposed.T
        else:
            pa = r_p_adjust(p.T.ravel(), adjust).reshape(p.T.shape).T
    names_x, names_y = list(x.columns), list(y.columns)

    def frame(values):
        return pd.DataFrame(values, index=names_x, columns=names_y)

    result = {"r": frame(r), "n": n if np.isscalar(n) else frame(n), "t": frame(t), "p": frame(p),
              "p_adj": pa if sym and adjust != "none" else frame(pa), "se": frame(se), "sef": None,
              "adjust": adjust, "sym": sym, "ci": None, "ci2": None, "ci_adj": None}
    if not ci:
        return result
    if np.min(n) < 4:
        warnings.warn("Number of subjects must be greater than 3 to find confidence intervals.")
    abbreviated_rows = r_abbreviate(names_x, minlength)
    abbreviated_columns = r_abbreviate(names_y, minlength)
    if sym:
        lower = _lower_tri(r.shape)
        z = _fisherz(_column_major(r, lower))
        ncors = len(names_x) * (len(names_x) - 1) / 2
        if adjust != "holm":
            dif_corrected = scipy.stats.norm.ppf(1 - alpha / (2 * ncors))
        else:
            rank = np.empty(len(z))
            rank[np.argsort(np.abs(z), kind="stable")] = np.arange(1, len(z) + 1)
            dif_corrected = scipy.stats.norm.ppf(1 - alpha / (2 * rank))
        dif = scipy.stats.norm.ppf(1 - alpha / 2)
        sef = 1 / np.sqrt((_column_major(n, lower) if not np.isscalar(n) else n) - 3)
        labels = [f"{abbreviated_columns[i]}-{abbreviated_rows[j]}"
                  for i in range(len(names_x) - 1) for j in range(i + 1, len(names_x))]
        p_lower = _column_major(p, lower)
        p_adjusted = _column_major(p.T, lower)
        r_lower = _column_major(r, lower)
        result["ci"] = pd.DataFrame({"lower": _fisherz2r(z - dif * sef), "r": r_lower,
                                     "upper": _fisherz2r(z + dif * sef), "p": p_lower}, index=labels)
        result["ci2"] = result["ci"].assign(**{"p.adj": p_adjusted})
        result["ci_adj"] = pd.DataFrame({"lower.adj": _fisherz2r(z - dif_corrected * sef),
                                         "upper.adj": _fisherz2r(z + dif_corrected * sef)}, index=labels)
    else:
        z = _fisherz(r.T.ravel())
        if adjust != "holm":
            dif_corrected = scipy.stats.norm.ppf(1 - (alpha / 2) / (len(names_x) * len(names_y)))
        else:
            rank = np.empty(len(z))
            rank[np.argsort(np.abs(z), kind="stable")] = np.arange(1, len(z) + 1)
            dif_corrected = scipy.stats.norm.ppf(1 - (alpha / 2) / rank)
        dif = scipy.stats.norm.ppf(1 - alpha / 2)
        sef = 1 / np.sqrt((n.T.ravel() if not np.isscalar(n) else n) - 3)
        labels = [f"{abbreviated_rows[j]}-{abbreviated_columns[i]}"
                  for i in range(len(names_y)) for j in range(len(names_x))]
        r_vector = r.T.ravel()
        result["ci"] = pd.DataFrame({"lower": _fisherz2r(z - dif * sef), "r": r_vector,
                                     "upper": _fisherz2r(z + dif * sef), "p": p.T.ravel()}, index=labels)
        result["ci2"] = result["ci"].assign(pa=pa.T.ravel())
        result["ci_adj"] = pd.DataFrame({"lower.adj": _fisherz2r(z - dif_corrected * sef), "r": r_vector,
                                         "upper.adj": _fisherz2r(z + dif_corrected * sef)}, index=labels)
    result["sef"] = sef
    return result


def _rwf_lower(matrix):
    """rwf's matrix_triangle(m, off_diagonal = NA, diagonal = NA, type = "lower")."""
    values = np.array(matrix, dtype=float)
    values[~_lower_tri(values.shape, diag=True)] = np.nan
    np.fill_diagonal(values, np.nan)
    return values


def report_correlation(x, y=None, use="pairwise", method="pearson", adjust="holm", alpha=.05, ci=True,
                       file=None, w=10, h=10, base_size=20, scatterplot=True):
    """
    Correlation report, as rwf::report_correlation: psych::corr.test
    (ported as corr_test) shown as lower triangles of r, r squared, p,
    adjusted p, t, n and se, the confidence intervals, a corrplot, and
    optionally the scatterplots of every pair; with file, an Excel
    workbook (sheets r, r_squared, p, p_adjusted, t, N, SE, CI, Call) and
    the PDFs file_corrplot.pdf and file_scatterplot.pdf.

    The matrices keep the variable names as their index and, as R's
    data.frame, syntactic names (make.names) as columns. When n is the
    same for every pair n_lower is a one-cell DataFrame with column "n",
    as in rwf. With y the correlations of x with y are reported; as in
    rwf, x and y must then have the same number of columns.

    Parameters:
    x (pandas.DataFrame): Data to correlate (non-numeric columns are made numeric).
    y (pandas.DataFrame, optional): Second set of variables. Defaults to None.
    use (str, optional): "pairwise" (default) or "complete".
    method (str, optional): "pearson" (default), "spearman" or "kendall".
    adjust (str, optional): p.adjust method for the adjusted p values. Defaults to "holm".
    alpha (float, optional): Confidence interval alpha. Defaults to 0.05.
    ci (bool, optional): Compute confidence intervals. Defaults to True.
    file (str, optional): Output filename without extension. Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10.
    base_size (int, optional): Base font size of the plots. Defaults to 20.
    scatterplot (bool, optional): Also plot every pair. Defaults to True.

    Returns:
    dict: r_lower, r_squared_lower, p_lower, p_lower_adjusted, t_lower,
    n_lower, se_lower (DataFrames), ci (the CI table with its adjusted
    bounds, rows named by abbreviated variable pairs) and call (the
    arguments).

    Examples:
    >>> report_correlation(x=df_insurance[["age", "bmi", "charges"]], scatterplot=False)
    """
    x = pd.DataFrame(x)
    if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x.columns):
        x = change_data_type(x, type="numeric")
    result = corr_test(x, y, use, method, adjust, alpha, ci)
    rows = list(result["r"].index)
    columns = r_make_names(result["r"].columns)

    def frame(values):
        return pd.DataFrame(values, index=rows, columns=columns)

    if np.isscalar(result["n"]):
        n_lower = pd.DataFrame({"n": [result["n"]]})
    else:
        n_lower = frame(_rwf_lower(result["n"]))
    p = result["p"].to_numpy(dtype=float)
    if p.shape[0] != p.shape[1]:
        raise ValueError("length of 'dimnames' [1] not equal to array extent (rwf needs as many y as x columns)")
    # rwf: symmetric_matrix(matrix_triangle(p, type = "upper"), duplicate = "upper") mirrors the adjusted upper
    # triangle into the lower one
    p_adjusted = p.T
    r = result["r"].to_numpy(dtype=float)
    ci_table = pd.concat([result["ci"], result["ci_adj"]], axis=1) if ci else pd.DataFrame()
    if ci:
        ci_table.columns = r_make_names(ci_table.columns)
    correlation_result = {
        "r_lower": frame(_rwf_lower(r)), "r_squared_lower": frame(_rwf_lower(r * r)), "p_lower": frame(_rwf_lower(p)),
        "p_lower_adjusted": frame(_rwf_lower(p_adjusted)), "t_lower": frame(_rwf_lower(result["t"])),
        "n_lower": n_lower, "se_lower": frame(_rwf_lower(result["se"])), "ci": ci_table,
        "call": pd.DataFrame({"function_arguments": ["Use", "Method", "Adjustment for Probability values", "Alpha",
                                                     "Confidence Interval"],
                              "function_values": [use, method, adjust, r_label(alpha), r_label(ci)]}),
    }

    corrplot = plot_corrplot(result["r"], base_size=base_size)
    report_pdf(corrplot, file=file, title="corrplot", w=w, h=h, print_plot=(file is None))
    if scatterplot:
        scatter_plots = plot_scatterplot(x, base_size=base_size, coord_equal=False, all_orders=False)
        figures = [g.figure for g in scatter_plots.values() if g is not None]
        report_pdf(plotlist=figures, file=file, title="scatterplot", w=w, h=h, print_plot=(file is None))

    if file is not None:
        try:
            from .functions_excel import excel_matrix
        except ImportError:
            from functions_excel import excel_matrix
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            for key, sheet, number_format in [("r_lower", "r", "#0.00"), ("r_squared_lower", "r_squared", "#0.00"),
                                              ("p_lower", "p", "#0.00"), ("p_lower_adjusted", "p_adjusted", "#0.00"),
                                              ("t_lower", "t", "#0.00"), ("n_lower", "N", "#0"),
                                              ("se_lower", "SE", "#0.00")]:
                excel_matrix(correlation_result[key], writer, sheet=sheet, numFmt=number_format,
                             conditional_formatting=True)
            excel_matrix(correlation_result["ci"], writer, sheet="CI", numFmt="#0.00")
            excel_matrix(correlation_result["call"], writer, sheet="Call")
    return correlation_result
##########################################################################################
# TETRACHORIC POLYCHORIC BISERIAL POLYSERIAL
##########################################################################################
_GL_WEIGHTS = {
    6: ([0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
        [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
    12: ([.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659, 0.2334925365383547,
          0.2491470458134029],
         [0.9815606342467191, 0.9041172563704750, 0.7699026741943050, 0.5873179542866171, 0.3678314989981802,
          0.1252334085114692]),
    20: ([.01761400713915212, .04060142980038694, .06267204833410906, .08327674157670475, 0.1019301198172404,
          0.1181945319615184, 0.1316886384491766, 0.1420961093183821, 0.1491729864726037, 0.1527533871307259],
         [0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188, 0.7463319064601508,
          0.6360536807265150, 0.5108670019508271, 0.3737060887154196, 0.2277858511416451, 0.07652652113349733]),
}


def _bvn_upper(h, k, r):
    """P(X > h, Y > k) of a standard bivariate normal with correlation r (Genz's BVNU, double precision)."""
    phi = scipy.special.ndtr
    if h == np.inf or k == np.inf:
        return 0.0
    if h == -np.inf:
        return 1.0 if k == -np.inf else float(phi(-k))
    if k == -np.inf:
        return float(phi(-h))
    if r == 0:
        return float(phi(-h) * phi(-k))
    size = 6 if abs(r) < 0.3 else 12 if abs(r) < 0.75 else 20
    w, x = (np.array(v) for v in _GL_WEIGHTS[size])
    w = np.concatenate([w, w])
    x = np.concatenate([1 - x, 1 + x])
    hk = h * k
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2
        asr = np.arcsin(r) / 2
        sn = np.sin(asr * x)
        bvn = np.sum(w * np.exp((sn * hk - hs) / (1 - sn ** 2)))
        bvn = bvn * asr / (2 * np.pi) + phi(-h) * phi(-k)
    else:
        if r < 0:
            k, hk = -k, -hk
        bvn = 0.0
        if abs(r) < 1:
            a_s = 1 - r ** 2
            a = np.sqrt(a_s)
            bs = (h - k) ** 2
            asr = -(bs / a_s + hk) / 2
            c = (4 - hk) / 8
            d = (12 - hk) / 80
            if asr > -100:
                bvn = a * np.exp(asr) * (1 - c * (bs - a_s) * (1 - d * bs) / 3 + c * d * a_s ** 2)
            if hk > -100:
                b = np.sqrt(bs)
                sp = np.sqrt(2 * np.pi) * phi(-b / a)
                bvn = bvn - np.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3)
            a = a / 2
            xs = (a * x) ** 2
            asr = -(bs / xs + hk) / 2
            keep = asr > -100
            xs, asr, wk = xs[keep], asr[keep], w[keep]
            sp = 1 + c * xs * (1 + 5 * d * xs)
            rs = np.sqrt(1 - xs)
            ep = np.exp(-(hk / 2) * xs / (1 + rs) ** 2) / rs
            bvn = (a * np.sum(np.exp(asr) * (sp - ep) * wk) - bvn) / (2 * np.pi)
        if r > 0:
            bvn = bvn + phi(-max(h, k))
        elif h >= k:
            bvn = -bvn
        else:
            low = phi(k) - phi(h) if h < 0 else phi(-h) - phi(-k)
            bvn = low - bvn
    return float(min(1.0, max(0.0, bvn)))


def _cell_probabilities(rho, row_cuts, column_cuts):
    """Bivariate normal probability of every cell of the table cut at row_cuts (first variable) and column_cuts."""
    rows = np.concatenate([[-np.inf], np.asarray(row_cuts, dtype=float), [np.inf]])
    columns = np.concatenate([[-np.inf], np.asarray(column_cuts, dtype=float), [np.inf]])
    cdf = np.array([[_bvn_upper(-a, -b, rho) for b in columns] for a in rows])
    return cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]


def _ml_correlation(table, row_cuts, column_cuts):
    """
    Correlation maximising the bivariate normal likelihood of a contingency
    table with fixed thresholds (psych's polyF/tetraF: cells of probability
    zero are left out of the likelihood).
    """
    def objective(rho):
        p = _cell_probabilities(rho, row_cuts, column_cuts)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.log(np.where(p > 0, p, np.nan))
        log_p[~np.isfinite(log_p)] = np.nan
        return -np.nansum(table * log_p)

    result = scipy.optimize.minimize_scalar(objective, bounds=(-1, 1), method="bounded", options={"xatol": 1e-10})
    return float(result.x)


def _cor_smooth(r, eig_tol=1e-12):
    """psych::cor.smooth: rescale the eigenvalues of a correlation matrix that is not positive definite."""
    values, vectors = np.linalg.eigh(r)
    if values.min() >= np.finfo(float).eps:
        return r
    warnings.warn("Matrix was not positive definite, smoothing was done")
    values = np.where(values < eig_tol, 100 * eig_tol, values)
    values = values * len(values) / values.sum()
    smoothed = vectors @ np.diag(values) @ vectors.T
    d = np.sqrt(np.diag(smoothed))
    return smoothed / np.outer(d, d)


def _drop_constant_items(data):
    """psych's delete = TRUE: items without variance are dropped with a warning."""
    sd = data.std()
    bad = [column for column in data.columns if not sd[column] > 0]
    for column in bad:
        warnings.warn(f"Item = {column} had no variance and was deleted")
    return data.drop(columns=bad)


def tetrachoric(x, correct=0.5, smooth=True):
    """
    Tetrachoric correlations of dichotomous items, as psych::tetrachoric(x).

    Each item is shifted so its smallest value is 0 and its threshold is
    tau = -qnorm(mean), the proportion of the higher value taken over the
    item's non-missing cases (psych's global = TRUE). For each pair the
    2 x 2 table of complete cases is built, empty cells are replaced by
    correct (when the table holds more than one case) and rho maximises
    the bivariate normal likelihood of the table. A pair whose complete
    cases do not show both values of both items gets NaN, as in psych.
    The matrix is smoothed (psych::cor.smooth) when it is not positive
    definite and has no missing correlations.

    Differences from psych: the optimum is found to 1e-10, whereas R's
    optimize stops at about 1e-4, so values agree with psych to about
    1e-4. The x/y form, weights and the 2 x 2 table input are not ported.

    Parameters:
    x (pandas.DataFrame): Dichotomous items (two values each).
    correct (float, optional): Value put in empty cells; 0 for none. Defaults to 0.5.
    smooth (bool, optional): Smooth a non positive definite matrix. Defaults to True.

    Returns:
    dict: rho (DataFrame of correlations), tau (Series of thresholds) and
    n_obs (number of rows), psych's rho, tau and n.obs.

    Examples:
    >>> result = tetrachoric(df_ocean[["E1", "E2", "E3"]].gt(3).astype(int))
    >>> result["rho"]
    """
    data = _drop_constant_items(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"))
    names = list(data.columns)
    values = (data - data.min()).to_numpy(dtype=float)
    if np.nanmax(values) > 1:
        raise ValueError("Tetrachoric correlations require dictomous data")
    tau = -scipy.stats.norm.ppf(np.nanmean(values, axis=0))
    n = len(names)
    rho = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            pair = values[:, [i, j]]
            pair = pair[~np.isnan(pair).any(axis=1)]
            value = np.nan
            if len(pair) > 2 and all(len(np.unique(pair[:, k])) == 2 for k in (0, 1)):
                table = np.array([[np.sum((pair[:, 0] == a) & (pair[:, 1] == b)) for b in (0, 1)] for a in (0, 1)],
                                 dtype=float)
                if table.sum() > 1 and table.min() == 0 and correct > 0:
                    table[table == 0] = correct
                value = _ml_correlation(table, [tau[i]], [tau[j]])
            else:
                warnings.warn(f"For i = {i + 1} j = {j + 1}  No variance for either i or j   rho set to NA")
            rho[i, j] = rho[j, i] = value
    if np.isnan(rho).any():
        warnings.warn("some correlations are missing, smoothing turned off")
    elif smooth:
        rho = _cor_smooth(rho)
    return {"rho": pd.DataFrame(rho, index=names, columns=names), "tau": pd.Series(tau, index=names),
            "n_obs": len(data)}


def polychoric(x, smooth=True, global_=True, correct=0.5, max_cat=8):
    """
    Polychoric correlations of ordinal items, as psych::polychoric(x).

    The items are shifted so the smallest value over all items is 1. Each
    item's thresholds are qnorm of its cumulative category proportions
    (over its non-missing cases). For each pair the table of complete
    cases is built over all categories and rho maximises the bivariate
    normal likelihood of the table with the items' own thresholds
    (global = TRUE; cells of probability zero are left out, and correct
    is not used). When the items do not have the same number of response
    alternatives psych sets global = FALSE, with a warning, and so does
    this function: empty rows and columns of the pair's table are
    dropped, empty cells get correct / n and the thresholds come from the
    table's margins. The matrix is smoothed (psych::cor.smooth) when it
    is not positive definite and has no missing correlations.

    Differences from psych: the optimum is found to 1e-10, whereas R's
    optimize stops at about 1e-4, so values agree with psych to about
    1e-4. The x/y form and weights are not ported.

    Parameters:
    x (pandas.DataFrame): Ordinal items with integer categories.
    smooth (bool, optional): Smooth a non positive definite matrix. Defaults to True.
    global_ (bool, optional): Use each item's own thresholds (psych's global). Defaults to True.
    correct (float, optional): Continuity correction of empty cells when global_ is False. Defaults to 0.5.
    max_cat (int, optional): Largest number of categories allowed. Defaults to 8.

    Returns:
    dict: rho (DataFrame of correlations), tau (DataFrame of thresholds, one
    row per item) and n_obs (number of rows), psych's rho, tau and n.obs.

    Examples:
    >>> result = polychoric(df_ocean[["E1", "E2", "E3", "E4", "E5"]])
    >>> result["rho"]
    """
    data = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
    n_values = int(np.nanmax(data.to_numpy(dtype=float)) - np.nanmin(data.to_numpy(dtype=float)) + 1)
    if n_values > max_cat:
        raise ValueError(f"You have more than {max_cat} categories for your items, polychoric is probably not needed")
    data = _drop_constant_items(data)
    names = list(data.columns)
    values = data.to_numpy(dtype=float)
    values = values - np.nanmin(values) + 1
    maxima = np.nanmax(values, axis=0)
    if maxima.min() != maxima.max():
        global_ = False
        warnings.warn("The items do not have an equal number of response alternatives, global set to FALSE.")
    categories = int(maxima.max())
    frequencies = np.array([[np.sum(values[:, k] == c) for k in range(len(names))] for c in range(1, n_values + 1)],
                           dtype=float)
    frequencies = frequencies / frequencies.sum(axis=0)
    tau = scipy.stats.norm.ppf(np.minimum(np.cumsum(frequencies, axis=0)[:-1], 1))
    n = len(names)
    rho = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            pair = values[:, [i, j]]
            pair = pair[~np.isnan(pair).any(axis=1)]
            table = np.array([[np.sum((pair[:, 0] == a) & (pair[:, 1] == b)) for b in range(1, categories + 1)]
                              for a in range(1, categories + 1)], dtype=float)
            total = table.sum()
            value = np.nan
            if total > 0:
                table = table / total
                if global_:
                    value = _ml_correlation(table, tau[:, i], tau[:, j])
                else:
                    table = table[table.any(axis=1)][:, table.any(axis=0)]
                    if correct > 0 and (table == 0).any():
                        table[table == 0] = correct / total
                    if min(table.shape) >= 2:
                        row_cuts = scipy.stats.norm.ppf(np.cumsum(table.sum(axis=1))[:-1])
                        column_cuts = scipy.stats.norm.ppf(np.cumsum(table.sum(axis=0))[:-1])
                        value = _ml_correlation(table, row_cuts, column_cuts)
            rho[i, j] = rho[j, i] = value
    if np.isnan(rho).any():
        warnings.warn("some correlations are missing, smoothing turned off")
    elif smooth:
        rho = _cor_smooth(rho)
    return {"rho": pd.DataFrame(rho, index=names, columns=names),
            "tau": pd.DataFrame(tau.T, index=names, columns=[str(k) for k in range(1, n_values)]),
            "n_obs": len(data)}


def polyserial(x, y):
    """
    Polyserial correlations between continuous variables and ordinal
    variables, as psych::polyserial(x, y) (the two-step estimator, not ML):
    r = cor(x, y) * sqrt((n - 1) / n) * sd(y) / sum(dnorm(tau)), with the
    thresholds tau from the cumulative category proportions of y over the
    categories min(y)..max(y) of all y variables, clipped to [-1, 1].

    Parameters:
    x (pandas.DataFrame or array-like): Continuous variable(s).
    y (pandas.DataFrame or array-like): Ordinal variable(s) with integer categories.

    Returns:
    pandas.DataFrame: One row per y variable and one column per x variable,
    as psych.

    Examples:
    >>> polyserial(df_ocean[["E1", "E2"]].sum(axis=1).to_frame("E"), df_ocean[["N1", "N2"]])
    """
    x = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
    y = pd.DataFrame(y).apply(pd.to_numeric, errors="coerce")
    y_values = y.to_numpy(dtype=float)
    levels = np.arange(np.nanmin(y_values), np.nanmax(y_values) + 1)
    n_cases = len(y)
    counts = np.array([[np.sum(y[column] == level) for level in levels] for column in y.columns], dtype=float)
    cumulative = np.cumsum(counts / counts.sum(axis=1, keepdims=True), axis=1)
    stau = scipy.stats.norm.pdf(scipy.stats.norm.ppf(cumulative[:, :-1])).sum(axis=1)
    rxy = pd.DataFrame({ycol: [x[xcol].corr(y[ycol]) for xcol in x.columns] for ycol in y.columns}, index=x.columns)
    rps = rxy.T * np.sqrt((n_cases - 1) / n_cases)
    rps = rps.mul(y.std().to_numpy() / stau, axis=0)
    return rps.clip(-1, 1)


def _biserial_pair(x, y, i, j):
    """psych:::biserialc for one continuous (or dichotomous) x and one dichotomous y."""
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    levels = np.unique(y)
    if len(levels) != 2:
        if len(np.unique(x)) != 2:
            warnings.warn(f"For x = {i} y = {j} neither x nor y is dichotomous  biserial is not possible.")
            return np.nan
        x, y = y, x
        levels = np.unique(y)
    proportions = np.array([np.mean(y == level) for level in levels])
    zp = scipy.stats.norm.pdf(scipy.stats.norm.ppf(proportions[1]))
    r = (x[y == levels[1]].mean() - x[y == levels[0]].mean()) * proportions.prod() / (zp * x.std(ddof=1))
    if abs(r) > 1:
        warnings.warn(f"For x = {i} y = {j} x seems to be dichotomous, not continuous")
        r = float(np.sign(r))
    return float(r)


def biserial(x, y):
    """
    Biserial correlations between continuous variables and dichotomous
    variables, as psych::biserial(x, y): for each pair of complete cases
    r = (mean of x at the higher y - mean of x at the lower y) * p * q /
    (dnorm(qnorm(p)) * sd(x)), with p the proportion of the higher y value,
    clipped to [-1, 1]. When y is not dichotomous but x is, the two are
    swapped, as in psych.

    Difference from psych: when neither variable is dichotomous psych
    warns but still returns a number computed from the wrong variable;
    this function returns NaN.

    Parameters:
    x (pandas.DataFrame or array-like): Continuous variable(s).
    y (pandas.DataFrame or array-like): Dichotomous variable(s).

    Returns:
    pandas.DataFrame: One row per y variable and one column per x variable,
    as psych.

    Examples:
    >>> biserial(df_ocean[["E1", "E2"]].sum(axis=1).to_frame("E"), df_ocean[["N1"]].gt(3).astype(int))
    """
    x = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
    y = pd.DataFrame(y).apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame([[_biserial_pair(x[xcol].to_numpy(float), y[ycol].to_numpy(float), a + 1, b + 1)
                          for a, xcol in enumerate(x.columns)] for b, ycol in enumerate(y.columns)],
                        index=y.columns, columns=x.columns)


def report_choric_serial(x, y=None, file=None, w=10, h=10, type="tetrachoric", **kwargs):
    """
    Tetrachoric, polychoric, polyserial or biserial correlations with a
    corrplot and an Excel report, as rwf::report_choric_serial.

    type "tetrachoric" and "polychoric" correlate the columns of x (y is
    not used) and return their result dict; the corrplot of rho is shown
    or written to file.pdf and file.xlsx gets the sheets R, R Squared,
    Tau, Observations and Call. type "polyserial" and "biserial" return
    the DataFrame of psych::polyserial(x, y) / psych::biserial(x, y)
    (one row per y variable); for "biserial" its corrplot and an Excel
    sheet named "biserial" are written. As in rwf (whose condition reads
    type=="biserial"||type=="biserial") nothing is plotted or written for
    "polyserial".

    Parameters:
    x (pandas.DataFrame): Items (tetrachoric, polychoric) or continuous variables (polyserial, biserial).
    y (pandas.DataFrame, optional): Ordinal (polyserial) or dichotomous (biserial) variables. Defaults to None.
    file (str, optional): Output filename without extension. Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10.
    type (str, optional): "tetrachoric" (default), "polychoric", "polyserial" or "biserial".
    **kwargs: Passed to tetrachoric, polychoric, polyserial or biserial.

    Returns:
    dict or pandas.DataFrame: tetrachoric/polychoric result (rho, tau, n_obs, call) or the correlation table.

    Examples:
    >>> report_choric_serial(df_ocean[["E1", "E2", "E3"]].gt(3).astype(int), type="tetrachoric")
    >>> report_choric_serial(df_ocean[["E1", "E2", "E3"]], type="polychoric", file="polychoric")
    >>> report_choric_serial(x=df_ocean[["E1"]], y=df_ocean[["N1"]].gt(3).astype(int), type="biserial")
    """
    if type == "tetrachoric":
        result = tetrachoric(x, **kwargs)
    elif type == "polychoric":
        result = polychoric(x, **kwargs)
    elif type == "polyserial":
        result = polyserial(x, y, **kwargs)
    elif type == "biserial":
        result = biserial(x, y, **kwargs)
    else:
        raise ValueError("type must be 'tetrachoric', 'polychoric', 'polyserial' or 'biserial'")

    try:
        from .functions_excel import excel_matrix, report_dataframe
    except ImportError:
        from functions_excel import excel_matrix, report_dataframe

    if type == "biserial":
        report_pdf(plot_corrplot(result, title=type), w=w, h=h, file=file, title=type, print_plot=(file is None))
        report_dataframe(result, sheet=type, file=file)

    if type in ("tetrachoric", "polychoric"):
        arguments = ", ".join(f"{key} = {value!r}" for key, value in kwargs.items())
        result["call"] = f"{type}(x{', ' + arguments if arguments else ''})"
        report_pdf(plot_corrplot(result["rho"], title=type), w=w, h=h, file=file, title=type,
                   print_plot=(file is None))
        if file is not None:
            filename = f"{file}.xlsx"
            if os.path.exists(filename):
                os.remove(filename)
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                tau = result["tau"]
                excel_matrix(result["rho"], writer, "R", conditional_formatting=True)
                excel_matrix(result["rho"] * result["rho"], writer, "R Squared", conditional_formatting=True)
                excel_matrix(tau.to_frame("tau") if isinstance(tau, pd.Series) else tau, writer, "Tau")
                excel_matrix(pd.DataFrame({"Observations": [result["n_obs"]]}), writer, "Observations")
                excel_matrix(pd.DataFrame({"call": [result["call"]]}), writer, "Call")
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    import matplotlib
    matplotlib.use("Agg")
    import os

    df_insurance = load_dataset("insurance")
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
    ordinal = pd.cut(latent, bins=[-np.inf, -1, 0, 1, np.inf], labels=[1, 2, 3, 4]).astype(int)
    dichotomous = (latent > 0).astype(int)
    print("polyserial (expect high, since ordinal derives from latent):\n", polyserial(continuous, ordinal))
    print("biserial (expect high, since dichotomous derives from latent):\n", biserial(continuous, dichotomous))

    print("\n" + "=" * 80, "\nreport_choric_serial\n", "=" * 80, sep="")
    items = pd.DataFrame({f"i{k}": pd.cut(latent + np.random.normal(size=500), bins=[-np.inf, -1, 0, 1, np.inf],
                                           labels=[1, 2, 3, 4]).astype(int) for k in range(1, 5)})
    print(report_choric_serial(items, type="polychoric")["rho"])
    print(report_choric_serial(items.gt(2).astype(int), type="tetrachoric")["rho"])
