# -*- coding: utf-8 -*-
"""
Plots and summaries for checking distributional assumptions (normality,
outliers) on numeric data.

Fixed a real bug in plot_normality_diagnostics: `plt.subplots(num_plots,
4, ...)` returns a 1-D axes array when num_plots == 1 (a single-column
input), so `axes[i, 0]`-style 2-D indexing crashed with IndexError --
fixed via squeeze=False, the same fix already applied to plot_multiplot
in functions_plot.py.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import contextlib
import io
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from plotnine import (
    ggplot, aes, geom_histogram, geom_boxplot, geom_abline, stat_qq, lims,
    theme, labs, coord_flip, geom_point, geom_text, geom_hline,
)

try:  # plotnine's theme_bw with ggplot2's left-aligned title
    from ._rcompat import theme_bw
except ImportError:
    from _rcompat import theme_bw
try:
    from ._rcompat import gg_text_size, r_boxplot_stats, r_color, r_density, r_hist, r_number, r_ppoints, r_round
    from .functions_environment import write_txt
    from .functions_excel import excel_critical_value
    from .functions_statistical import compute_standard
    from .functions_strings import output_separator, str_aes
    from .functions_plot import report_pdf
except ImportError:
    from _rcompat import gg_text_size, r_boxplot_stats, r_color, r_density, r_hist, r_number, r_ppoints, r_round
    from functions_environment import write_txt
    from functions_excel import excel_critical_value
    from functions_statistical import compute_standard
    from functions_strings import output_separator, str_aes
    from functions_plot import report_pdf
##########################################################################################
# CAPTIONS AND INPUTS (internal, as rwf's plots print them)
##########################################################################################
def _r_text(x):
    """R's paste() of round(x, 2): "NA" for a missing value."""
    return "NA" if not np.isfinite(x) else r_number(r_round(x, 2))


def _summary_caption(values, variable=None):
    """rwf's caption: Variable (optional), Observations, Mean, SD and Median, each on a new line."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    lines = ([f"Variable={variable}"] if variable is not None else []) + [
        f"Observations={n}",
        "Mean=" + (_r_text(np.mean(values)) if n else "NA"),
        "SD=" + (_r_text(np.std(values, ddof=1)) if n > 1 else "NA"),
        "Median=" + (_r_text(np.median(values)) if n else "NA")]
    return "".join("\n" + line for line in lines)


def _as_frame(df):
    """A data frame of the input; a vector is one column named after its Series name or "data"."""
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pd.Series):
        return df.to_frame(name=df.name if df.name is not None else "data")
    return pd.DataFrame({"data": np.asarray(df)})
##########################################################################################
# PLOT HISTOGRAM
##########################################################################################
def plot_histogram(df, bins=30, title="", base_size=10, xlims=None, fill="gray25", color="gray50", ylab="Count",
                   pb=False):
    """
    One histogram per numeric column, as rwf::plot_histogram, each captioned
    with its observations, mean, SD and median.

    Parameters:
    df (pandas.DataFrame, Series or array-like): Data; non-numeric columns are skipped.
    bins (int, optional): Number of bins. Defaults to 30.
    title (str, optional): Plot title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.
    xlims (tuple, optional): x-axis limits. Defaults to None.
    fill (str, optional): Bar fill. Defaults to "gray25".
    color (str, optional): Bar outline. Defaults to "gray50".
    ylab (str, optional): y-axis label. Defaults to "Count".
    pb (bool, optional): rwf's progress bar; accepted and ignored.

    Returns:
    dict: {column: plotnine.ggplot}. As rwf, a single numeric column is
    keyed and labelled "".

    Examples:
    >>> plot_histogram(df_insurance[["age", "bmi"]])
    """
    numeric = _as_frame(df).select_dtypes(include=np.number)
    names = [str(c) for c in numeric.columns] if numeric.shape[1] != 1 else [""]
    plots = {}
    for name, column in zip(names, numeric.columns):
        values = numeric[column]
        plot = (ggplot(pd.DataFrame({"data": values}), aes(x="data"))
                + geom_histogram(bins=bins, fill=r_color(fill), color=r_color(color), na_rm=True)
                + theme_bw(base_size=base_size)
                + labs(title=title, x=name, y=ylab, caption=_summary_caption(values.dropna())))
        if xlims is not None:
            plot = plot + lims(x=xlims)
        plots[name] = plot
    return plots
##########################################################################################
# PLOT QQ
##########################################################################################
def plot_qq(df, title="", base_size=10, pb=False):
    """
    One normal QQ plot per numeric column, as rwf::plot_qq: the line runs
    through the quartiles (R's qqline), and the caption gives the variable,
    observations, mean, SD and median of its finite values.

    Parameters:
    df (pandas.DataFrame, Series or array-like): Data; non-numeric columns are skipped.
    title (str, optional): Plot title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.
    pb (bool, optional): rwf's progress bar; accepted and ignored.

    Returns:
    dict: {column: plotnine.ggplot}.

    Examples:
    >>> plot_qq(df_insurance[["age", "bmi"]])
    """
    plots = {}
    for column, series in _as_frame(df).items():
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        y, x = np.quantile(values, [0.25, 0.75]), stats.norm.ppf([0.25, 0.75])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        plots[str(column)] = (ggplot(pd.DataFrame({"resids": values}), aes(sample="resids"))
                              + stat_qq(alpha=.1, quantiles=r_ppoints(len(values)))  # ggplot2's ppoints(n)
                              + geom_abline(slope=slope, intercept=y[0] - slope * x[0])
                              + theme_bw(base_size=base_size)
                              + labs(title=title, caption=_summary_caption(values, variable=column)))
    return plots
##########################################################################################
# PLOT BOXPLOT
##########################################################################################
def plot_boxplot(df, title="", base_size=10):
    """
    Horizontal boxplots of the numeric columns, as rwf::plot_boxplot.

    Parameters:
    df (pandas.DataFrame, Series or array-like): Data; non-numeric columns are skipped.
    title (str, optional): Plot title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.

    Returns:
    plotnine.ggplot, captioned "Observations= n" (the number of rows).

    Examples:
    >>> plot_boxplot(df_insurance[["age", "bmi"]])
    """
    numeric = _as_frame(df).select_dtypes(include=np.number)
    long = numeric.melt(var_name="variable", value_name="value")
    long = long[np.isfinite(long["value"].to_numpy(dtype=float))]
    long["variable"] = pd.Categorical(long["variable"].astype(str), categories=[str(c) for c in numeric.columns])
    return (ggplot(long, aes(x="variable", y="value"))
            + geom_boxplot()
            + labs(title=title, y="", x="", caption=f"Observations= {len(numeric)}")
            + theme_bw(base_size=base_size)
            + coord_flip())
##########################################################################################
# PLOT NORMALITY DIAGNOSTICS
##########################################################################################
def plot_normality_diagnostics(df, breaks=None, title="", file=None, w=10, h=10, pb=False):
    """
    For each numeric column, a page of four normality plots, as rwf's: a
    histogram (Sturges' bins by default), the density, a boxplot and a
    normal QQ plot with R's quartile line, subtitled with the variable,
    observations, mean, SD and median.

    Parameters:
    df (pandas.DataFrame, Series or array-like): Data; non-numeric columns are skipped.
    breaks (int or str, optional): Histogram bins. Defaults to None (Sturges).
    title (str, optional): Title of every page. Defaults to "".
    file (str, optional): PDF filename without extension (report_pdf). Defaults to None.
    w, h (float, optional): Page width and height in inches. Default to 10.
    pb (bool, optional): rwf's progress bar; accepted and ignored.

    Returns:
    dict: {column: matplotlib.figure.Figure}; columns with fewer than 3
    values or no variance are skipped, with a message, as rwf.

    Examples:
    >>> plot_normality_diagnostics(df_insurance[["age", "bmi"]], title="Normality")
    """
    figures = {}
    for column, series in _as_frame(df).select_dtypes(include=np.number).items():
        values = series.dropna().to_numpy(dtype=float)
        if len(values) <= 2 or np.var(values) == 0:
            print(f"Graph not produced for {column} due to sample size")
            continue
        fig, axes = plt.subplots(2, 2, figsize=(w, h))
        # R's base graphics: hist() (pretty() breaks, right-closed classes), density(), boxplot() (Tukey's
        # hinges) and qqnorm() (ppoints) with qqline(); par(adj = .01) puts titles and labels on the left
        edges, counts = r_hist(values, "Sturges" if breaks is None else breaks)
        axes[0, 0].hist(edges[:-1], bins=edges, weights=counts, color="lightgray", edgecolor="black")
        axes[0, 0].set_ylabel("Frequency", loc="bottom")
        axes[0, 0].set_title("Histogram", loc="left")
        grid, density, bandwidth = r_density(values)
        axes[0, 1].plot(grid, density, color="black")
        axes[0, 1].set_xlabel(f"N = {len(values)}   Bandwidth = {bandwidth:.4g}", loc="left")  # formatC()
        axes[0, 1].set_ylabel("Density", loc="bottom")
        axes[0, 1].set_title("Density Function", loc="left")
        box = r_boxplot_stats(values)
        five = box["stats"]
        axes[1, 0].bxp([{"whislo": five[0], "q1": five[1], "med": five[2], "q3": five[3], "whishi": five[4],
                         "fliers": box["out"]}],
                       flierprops={"marker": "o", "markerfacecolor": "none"},
                       medianprops={"color": "black", "linewidth": 3})  # R's medlwd = 3
        axes[1, 0].set_title("Boxplot", loc="left")
        axes[1, 1].scatter(stats.norm.ppf(r_ppoints(len(values))), np.sort(values),
                           facecolors="none", edgecolors="black")
        y, x = np.quantile(values, [0.25, 0.75]), stats.norm.ppf([0.25, 0.75])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        axes[1, 1].axline((x[0], y[0]), slope=slope, color="black")
        axes[1, 1].set_xlabel("Theoretical Quantiles", loc="left")
        axes[1, 1].set_ylabel("Sample Quantiles", loc="bottom")
        axes[1, 1].set_title("Normal Q-Q Plot", loc="left")
        fig.suptitle(title)
        caption = _summary_caption(values, variable=str_aes(str(column))).lstrip("\n")
        fig.text(0.01, 0.01, caption, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        figures[str(column)] = fig
    if file is not None and figures:
        report_pdf(plotlist=list(figures.values()), file=file, title=title, w=w, h=h, print_plot=False)
    return figures
##########################################################################################
# OUTLIERS
##########################################################################################
def outlier_summary(vector):
    """
    Proportion of z-score outliers at the 95%/99%/99.9% thresholds.

    Parameters:
    vector (array-like): Numeric values.

    Returns:
    pandas.DataFrame: One row, columns abs_z_1.96, abs_z_2.58, abs_z_3.29
    (percentage of the non-missing cases beyond each threshold, as rwf's
    text, e.g. "5.26 %").

    Examples:
    >>> import numpy as np
    >>> vector = np.random.normal(size=1000)
    >>> outlier_summary(vector)
    """
    vector = np.asarray(vector, dtype=float)
    zvariable = (vector - np.nanmean(vector)) / np.nanstd(vector, ddof=1)  # R: mean, sd with na.rm = TRUE
    ncases = int(np.sum(~np.isnan(zvariable)))
    # R's paste(round(100 * share, 2), "%"): R's rounding and number text, and a space before "%"
    percents = {f"abs_z_{cut}": f"{r_number(r_round(100 * np.sum(np.abs(zvariable) >= cut) / ncases, 2))} %"
                for cut in (1.96, 2.58, 3.29)}
    return pd.DataFrame({name: [value] for name, value in percents.items()})
##########################################################################################
# REMOVE OUTLIERS
##########################################################################################
def remove_outliers(vector, probs=(0.25, 0.75), na_rm=True, **kwargs):
    """
    Replace Tukey (IQR-rule) outliers of a numeric vector with NaN, as rwf:
    values more than 1.5 times the distance between the two quantiles
    below the first or above the second.

    Parameters:
    vector (array-like): Numeric values.
    probs (sequence of float, optional): Lower/upper quantiles defining
        the box. Defaults to (0.25, 0.75).
    na_rm (bool, optional): Ignore missing values when computing the
        quantiles (True, default); as R's quantile(), False raises if any
        value is missing.
    **kwargs: R's quantile() arguments; type (1 to 9, default 7) is
        supported.

    Returns:
    numpy.ndarray: Same length as vector, outliers replaced with NaN.

    Examples:
    >>> import numpy as np
    >>> vector = np.random.normal(size=1000)
    >>> remove_outliers(vector)
    """
    vector = np.asarray(vector, dtype=float)
    if not na_rm and np.isnan(vector).any():
        raise ValueError("missing values and NaN's not allowed if 'na_rm' is False")
    methods = {1: "inverted_cdf", 2: "averaged_inverted_cdf", 3: "closest_observation",
               4: "interpolated_inverted_cdf", 5: "hazen", 6: "weibull", 7: "linear", 8: "median_unbiased",
               9: "normal_unbiased"}
    qnt = np.nanquantile(vector, probs, method=methods[kwargs.get("type", 7)])
    H = 1.5 * (qnt[1] - qnt[0])
    y = vector.copy()
    y[vector < (qnt[0] - H)] = np.nan
    y[vector > (qnt[1] + H)] = np.nan
    return y
##########################################################################################
# PLOT OUTLIER
##########################################################################################
def _row_ids(df):
    """R's row.names(): "1".."n" for a default index, otherwise the index labels."""
    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        return [str(i + 1) for i in range(len(df))]
    return [str(i) for i in df.index]


def plot_outlier(df, method="mean", title="", base_size=10, pb=False):
    """
    Dot plot of each numeric column with outliers coloured and labelled by
    row id, plus dashed lines at the centre and the outlier bounds (port of
    rwf::plot_outlier). Unlike rwf, the plots are returned without being drawn.

    Methods:
    "mean": more than 2 standard deviations from the mean.
    "median": more than 2 * MAD / 0.6745 from the median (MAD unscaled).
    "boxplot": outside Q1 - 1.5 IQR or Q3 + 1.5 IQR.

    Parameters:
    df (pandas.DataFrame or pandas.Series): Data; non-numeric columns are
        dropped. A Series is named after its .name (R: the argument's name).
    method (str, optional): "mean" (default), "median" or "boxplot".
    title (str, optional): Plot title.
    base_size (float, optional): Base font size. Defaults to 10.
    pb (bool, optional): Print progress. Defaults to False.

    Returns:
    dict: column -> plotnine.ggplot, with data columns id, obs and Outlier
    (missing values dropped), as in rwf.

    Examples:
    >>> plot_outlier(df=df_blood_pressure[["bp_before", "bp_after"]], method="boxplot")
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "df")
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"df": np.asarray(df)})
    df = df[[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]]
    ids = _row_ids(df)
    plots = {}
    for counter, column in enumerate(df.columns, start=1):
        if pb:
            print(f"plot_outlier {counter}/{df.shape[1]}")
        vector = df[column].to_numpy(dtype=float)
        observed = vector[~np.isnan(vector)]
        median_point = float(np.median(observed))
        mean_point = float(np.mean(observed))
        std = float(np.std(observed, ddof=1))
        mad = float(np.median(np.abs(median_point - observed)))
        midp = median_point
        with np.errstate(invalid="ignore"):
            if method == "mean":
                outlier = np.abs(vector - mean_point) > 2 * std
                midp, lower, upper = mean_point, mean_point - 2 * std, mean_point + 2 * std
            elif method == "median":
                outlier = np.abs(vector - median_point) > 2 * (mad / 0.6745)
                lower, upper = median_point - 2 * (mad / 0.6745), median_point + 2 * (mad / 0.6745)
            elif method == "boxplot":
                q1, q3 = np.quantile(observed, [0.25, 0.75])
                iqr = q3 - q1
                outlier = (vector < q1 - 1.5 * iqr) | (vector > q3 + 1.5 * iqr)
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                raise ValueError('method must be "mean", "median" or "boxplot"')
        keep = ~np.isnan(vector)
        data = pd.DataFrame({"id": np.array(ids)[keep], "obs": vector[keep], "Outlier": outlier[keep]})
        n_outliers = int(data["Outlier"].sum())
        caption = (f"\nMethod={method}\nObservations={len(data)}"
                   f"\nMean={r_number(r_round(float(data['obs'].mean()), 2))}"
                   f"\nSD={r_number(r_round(float(data['obs'].std(ddof=1)), 2))}"
                   f"\nMedian={r_number(r_round(float(data['obs'].median()), 2))}")
        plots[column] = (
            ggplot(data, aes(x="id", y="obs", label="id"))
            + geom_point(aes(colour="Outlier"))
            + geom_text(data=data[data["Outlier"]], size=gg_text_size(2.7), nudge_x=0.6, color="black")
            + labs(x=f"Observation ID \n Outliers: {n_outliers}", y=str(column), title=str(title), caption=caption)
            + theme_bw(base_size=base_size)
            + theme(legend_position="none")
            + geom_hline(yintercept=midp, colour="black", linetype="dashed")
            + geom_hline(yintercept=lower, colour="black", linetype="dashed")
            + geom_hline(yintercept=upper, colour="black", linetype="dashed")
            + coord_flip())
    return plots
##########################################################################################
# NORMALITY TESTS
##########################################################################################
def _anderson_darling(x):
    """nortest::ad.test."""
    x = np.sort(x)
    n = len(x)
    z = (x - np.mean(x)) / np.std(x, ddof=1)
    h = (2 * np.arange(1, n + 1) - 1) * (stats.norm.logcdf(z) + stats.norm.logcdf(-z)[::-1])
    a = -n - np.mean(h)
    aa = (1 + 0.75 / n + 2.25 / n ** 2) * a
    if aa < 0.2:
        p = 1 - np.exp(-13.436 + 101.14 * aa - 223.73 * aa ** 2)
    elif aa < 0.34:
        p = 1 - np.exp(-8.318 + 42.796 * aa - 59.938 * aa ** 2)
    elif aa < 0.6:
        p = np.exp(0.9177 - 4.279 * aa - 1.38 * aa ** 2)
    elif aa < 10:
        p = np.exp(1.2937 - 5.709 * aa + 0.0186 * aa ** 2)
    else:
        p = 3.7e-24
    return {"statistic": a, "p": p, "method": "Anderson-Darling normality test"}


def _cramer_von_mises(x):
    """nortest::cvm.test."""
    x = np.sort(x)
    n = len(x)
    p_values = stats.norm.cdf((x - np.mean(x)) / np.std(x, ddof=1))
    w = 1 / (12 * n) + np.sum((p_values - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2)
    ww = (1 + 0.5 / n) * w
    if ww < 0.0275:
        p = 1 - np.exp(-13.953 + 775.5 * ww - 12542.61 * ww ** 2)
    elif ww < 0.051:
        p = 1 - np.exp(-5.903 + 179.546 * ww - 1515.29 * ww ** 2)
    elif ww < 0.092:
        p = np.exp(0.886 - 31.62 * ww + 10.897 * ww ** 2)
    elif ww < 1.1:
        p = np.exp(1.111 - 34.242 * ww + 12.832 * ww ** 2)
    else:
        p = 7.37e-10
    return {"statistic": w, "p": p, "method": "Cramer-von Mises normality test"}


def _swilk_poly(cc, x):
    """poly() of R's swilk.c (algorithm AS 181.2): cc[0] + cc[1] x + ..., summed in R's order."""
    value = cc[0]
    if len(cc) > 1:
        p = x * cc[-1]
        for j in range(len(cc) - 2, 0, -1):
            p = (p + cc[j]) * x
        value += p
    return value


def _shapiro_wilk(x):
    """
    stats::shapiro.test, ported line by line from R's swilk.c (algorithm AS R94,
    in double precision), so W and p are R's whatever the SciPy version (SciPy
    before 1.18 gave p values that differ from R's in the 7th digit).
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n < 3 or n > 5000:
        raise ValueError("sample size must be between 3 and 5000")
    rng = x[-1] - x[0]
    if rng == 0:
        raise ValueError("all 'x' values are identical")
    if rng < 1e-10:
        x = x / rng
    x = [float(v) for v in x]

    # coefficients a[1..n/2] (a[0] is unused, as in the 1-based C code)
    nn2 = n // 2
    a = [0.0] * (nn2 + 1)
    an = float(n)
    if n == 3:
        a[1] = 0.70710678
    else:
        an25 = an + .25
        quantiles = stats.norm.ppf((np.arange(1, nn2 + 1) - 0.375) / an25)
        summ2 = 0.
        for i in range(1, nn2 + 1):
            a[i] = float(quantiles[i - 1])
            summ2 += a[i] * a[i]
        summ2 *= 2.
        ssumm2 = math.sqrt(summ2)
        rsn = 1. / math.sqrt(an)
        a1 = _swilk_poly((0., .221157, -.147981, -2.07119, 4.434685, -2.706056), rsn) - a[1] / ssumm2
        if n > 5:
            i1 = 3
            a2 = -a[2] / ssumm2 + _swilk_poly((0., .042981, -.293762, -1.752461, 5.682633, -3.582633), rsn)
            fac = math.sqrt((summ2 - 2. * (a[1] * a[1]) - 2. * (a[2] * a[2]))
                            / (1. - 2. * (a1 * a1) - 2. * (a2 * a2)))
            a[2] = a2
        else:
            i1 = 2
            fac = math.sqrt((summ2 - 2. * (a[1] * a[1])) / (1. - 2. * (a1 * a1)))
        a[1] = a1
        for i in range(i1, nn2 + 1):
            a[i] /= -fac

    range_x = x[n - 1] - x[0]
    if range_x < 1e-19:
        raise ValueError("ifault=6. This should not happen")
    sx = x[0] / range_x
    sa = -a[1]
    i, j = 1, n - 1
    while i < n:  # (swilk.c also checks the sort order here, which shapiro.test ignores)
        sx += x[i] / range_x
        i += 1
        if i != j:
            sa += (1. if i > j else -1.) * a[min(i, j)]
        j -= 1

    # W as the squared correlation between the data and the coefficients
    sa /= n
    sx /= n
    ssa = ssx = sax = 0.
    j = n - 1
    for i in range(n):
        asa = (1. if i > j else -1.) * a[1 + min(i, j)] - sa if i != j else -sa
        xsx = x[i] / range_x - sx
        ssa += asa * asa
        ssx += xsx * xsx
        sax += asa * xsx
        j -= 1
    # w1 is 1 - W, calculated to avoid excessive rounding error for W very near 1
    ssassx = math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1. - w1

    if n == 3:  # exact p value
        pw = 1.90985931710274 * (math.asin(math.sqrt(w)) - 1.04719755119660)
        return {"statistic": w, "p": max(pw, 0.), "method": "Shapiro-Wilk normality test"}
    y = math.log(w1)
    xx = math.log(an)
    if n <= 11:
        gamma = _swilk_poly((-2.273, .459), an)
        if y >= gamma:
            return {"statistic": w, "p": 1e-99, "method": "Shapiro-Wilk normality test"}
        y = -math.log(gamma - y)
        m = _swilk_poly((.544, -.39978, .025054, -6.714e-4), an)
        s = math.exp(_swilk_poly((1.3822, -.77857, .062767, -.0020322), an))
    else:
        m = _swilk_poly((-1.5861, -.31082, -.083751, .0038915), xx)
        s = math.exp(_swilk_poly((-.4803, -.082676, .0030302), xx))
    pw = float(stats.norm.sf(y, loc=m, scale=s))
    return {"statistic": w, "p": pw, "method": "Shapiro-Wilk normality test"}


def _shapiro_francia(x):
    """nortest::sf.test."""
    x = np.sort(x)
    n = len(x)
    a = 3 / 8  # sf.test calls ppoints(n, a = 3/8)
    y = stats.norm.ppf((np.arange(1, n + 1) - a) / (n + 1 - 2 * a))
    w = np.corrcoef(x, y)[0, 1] ** 2
    u = np.log(n)
    v = np.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sig = 1.0308 - 0.26758 * (v + 2 / u)
    z = (np.log(1 - w) - mu) / sig
    return {"statistic": w, "p": stats.norm.sf(z), "method": "Shapiro-Francia normality test"}


def _jarque_bera_robust(x):
    """rwf's robust Jarque-Bera test (Gel and Gastwirth, 2008), as DescTools::JarqueBeraTest(robust = TRUE)."""
    n = len(x)
    deviation = x - np.mean(x)
    j = np.sqrt(np.pi / 2) * np.mean(np.abs(x - np.median(x)))
    statistic = n / 6 * (np.mean(deviation ** 3) / j ** 3) ** 2 + n / 64 * (np.mean(deviation ** 4) / j ** 4 - 3) ** 2
    return {"statistic": statistic, "df": 2.0, "p": stats.chi2.sf(statistic, 2), "method": "Robust Jarque Bera Test"}


def _lilliefors(x):
    """nortest::lillie.test."""
    x = np.sort(x)
    n = len(x)
    p_values = stats.norm.cdf((x - np.mean(x)) / np.std(x, ddof=1))
    i = np.arange(1, n + 1)
    k = max(np.max(i / n - p_values), np.max(p_values - (i - 1) / n))
    kd, nd = (k, n) if n <= 100 else (k * (n / 100) ** 0.49, 100)
    p = np.exp(-7.01256 * kd ** 2 * (nd + 2.78019) + 2.99587 * kd * np.sqrt(nd + 2.78019)
               - 0.122119 + 0.974598 / np.sqrt(nd) + 1.67997 / nd)
    if p > 0.1:
        kk = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * k
        if kk <= 0.302:
            p = 1.0
        elif kk <= 0.5:
            p = 2.76773 - 19.828315 * kk + 80.709644 * kk ** 2 - 138.55152 * kk ** 3 + 81.218052 * kk ** 4
        elif kk <= 0.9:
            p = -4.901232 + 40.662806 * kk - 97.490286 * kk ** 2 + 94.029866 * kk ** 3 - 32.355711 * kk ** 4
        elif kk <= 1.31:
            p = 6.198765 - 19.558097 * kk + 23.186922 * kk ** 2 - 12.234627 * kk ** 3 + 2.423045 * kk ** 4
        else:
            p = 0.0
    return {"statistic": k, "p": p, "method": "Lilliefors (Kolmogorov-Smirnov) normality test"}


def _kolmogorov_limit_upper(x, tol=1e-6):
    """1 - Kolmogorov limiting CDF at x, computed as R's C_pkolmogorov_two_limit(x, lower = FALSE)."""
    if x <= 0:
        return 1.0
    if x < 1:
        k_max = int(np.sqrt(2 - np.log(tol)))
        z = -(np.pi / 2 * np.pi / 4) / (x * x)
        w = np.log(x)
        lower = sum(np.exp(k * k * z - w) for k in range(1, k_max, 2)) * np.sqrt(2 * np.pi)
        return 1 - lower
    z = -2 * x * x
    s, k, new = 1.0, 1, 0.0
    while True:
        old = new
        new += 2 * s * np.exp(z * k * k)
        s, k = -s, k + 1
        if abs(old - new) <= tol:
            return new


def _kolmogorov_smirnov(x):
    """stats::ks.test(x, "pnorm", mean(x), sd(x)), two-sided."""
    n = len(x)
    ties = len(np.unique(x)) < n
    exact = n < 100 and not ties
    cdf = stats.norm.cdf(np.sort(x), loc=np.mean(x), scale=np.std(x, ddof=1)) - np.arange(n) / n
    statistic = max(np.max(cdf), np.max(1 / n - cdf))
    if statistic <= 0:
        p = 1.0
    elif statistic > 1:
        p = 0.0
    elif exact:
        p = stats.kstwo.sf(statistic, n)
    else:
        p = _kolmogorov_limit_upper(np.sqrt(n) * statistic)
    method = ("Exact" if exact else "Asymptotic") + " one-sample Kolmogorov-Smirnov test"
    return {"statistic": statistic, "p": min(1.0, max(0.0, p)), "method": method,
            "alternative": "two-sided", "ties": ties}


def _pearson_chi_square(x, n_classes):
    """nortest::pearson.test(x, n.classes, adjust = TRUE)."""
    n = len(x)
    num = np.floor(1 + n_classes * stats.norm.cdf(x, loc=np.mean(x), scale=np.std(x, ddof=1))).astype(int)
    count = np.bincount(num[(num >= 1) & (num <= n_classes)], minlength=n_classes + 1)[1:]
    expected = n / n_classes
    statistic = np.sum((count - expected) ** 2 / expected)
    df = n_classes - 2 - 1
    return {"statistic": statistic, "df": float(df), "p": stats.chi2.sf(statistic, df),
            "method": "Pearson chi-square normality test", "n.classes": n_classes}


_NORMALITY_INSTRUCTIONS = {
    "shapiro": "Shapiro-Wilk Composite null hypothesis: any normal distribution",
    "anderson": "Anderson-Darling Composite null hypothesis: any normal distribution",
    "crammer": "Cramer-von-Mises Composite null hypothesis: any normal distribution",
    "shapiro_francia": "Shapiro-Francia Composite null hypothesis: any normal distribution",
    "jarque": "Jarque-Bera Composite null hypothesis: any normal distribution",
    "lilliefors": "Lilliefors Composite null hypothesis: any normal distribution",
    "kolmogorov": "Kolmogorov-Smirnov Exact null hypothesis: fully specified normal distribution",
    "pearson": "Pearson X2 Tests weaker null hypothesis: any distribution with the same probabilities for the given class intervals",
}


def report_normality_tests(df, file=None):
    """
    Eight normality tests on each numeric column, after z-standardising it
    (port of rwf::report_normality_tests): Shapiro-Wilk, Anderson-Darling,
    Cramer-von Mises, Shapiro-Francia, robust Jarque-Bera, Lilliefors,
    Kolmogorov-Smirnov and Pearson chi-square. Columns with fewer than 8 or
    more than 4999 values, or no variance, are skipped with a message. The
    table is printed, and with file also written to <file>.log and
    <file>.xlsx (p <= 0.05 highlighted).

    rwf returns nothing; this returns the table as well.

    Parameters:
    df (pandas.DataFrame or array-like): Data.
    file (str, optional): Output filename without extension.

    Returns:
    pandas.DataFrame: Columns variable, n, statistic, df, p, method,
    alternative, n.classes, instruction; eight rows per tested column.

    Examples:
    >>> report_normality_tests(df=df_insurance[["age", "bmi", "charges"]])
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "df")
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"df": np.asarray(df)})
    rows = []

    def add(test, variable, n, instruction):
        rows.append({"variable": variable, "n": n, "statistic": test["statistic"], "df": test.get("df", np.nan),
                     "p": test["p"], "method": test["method"], "alternative": test.get("alternative", np.nan),
                     "n.classes": test.get("n.classes", np.nan), "instruction": instruction})

    for variable in df.columns:
        vector = pd.to_numeric(df[variable], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vector)
        if 7 < n < 5000 and np.var(vector, ddof=1) != 0:
            z = np.asarray(compute_standard(vector, type="z"), dtype=float)
            add(_shapiro_wilk(z), variable, n, _NORMALITY_INSTRUCTIONS["shapiro"])
            add(_anderson_darling(z), variable, n, _NORMALITY_INSTRUCTIONS["anderson"])
            add(_cramer_von_mises(z), variable, n, _NORMALITY_INSTRUCTIONS["crammer"])
            add(_shapiro_francia(z), variable, n, _NORMALITY_INSTRUCTIONS["shapiro_francia"])
            add(_jarque_bera_robust(z), variable, n, _NORMALITY_INSTRUCTIONS["jarque"])
            add(_lilliefors(z), variable, n, _NORMALITY_INSTRUCTIONS["lilliefors"])
            ks = _kolmogorov_smirnov(z)
            instruction = _NORMALITY_INSTRUCTIONS["kolmogorov"]
            if ks["ties"]:
                instruction += " (ties present, p value is approximate)"
            add(ks, variable, n, instruction)
            add(_pearson_chi_square(z, int(np.ceil(2 * n ** (2 / 5)))), variable, n, _NORMALITY_INSTRUCTIONS["pearson"])
        else:
            print("NORMALITY INDICES NOT CALCULATED DUE TO OUT OF BOUNDS SAMPLE SIZE FOR", variable)
    result = pd.DataFrame(rows, columns=["variable", "n", "statistic", "df", "p", "method", "alternative",
                                         "n.classes", "instruction"])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        output_separator("NORMALITY TESTS", output=result, instruction="")
    write_txt(buffer.getvalue().rstrip("\n"), file=file)
    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(result, writer, sheet="Normality Tests", critical={"p": "<=0.05"})
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    import os
    import matplotlib
    matplotlib.use("Agg")

    personality = load_dataset("personality")

    print("=" * 80, "\nplot_histogram / plot_qq / plot_boxplot\n", "=" * 80, sep="")
    for name, plot in plot_histogram(personality.iloc[:, 1:6]).items():
        plot.save(f"plot_histogram_{name}.png", verbose=False)
    for name, plot in plot_qq(personality.iloc[:, 1:6]).items():
        plot.save(f"plot_qq_{name}.png", verbose=False)
    plot_boxplot(personality.iloc[:, 1:6]).save("plot_boxplot_example.png", verbose=False)
    print("saved plot_histogram_*.png, plot_qq_*.png, plot_boxplot_example.png")

    print("\n" + "=" * 80, "\nplot_normality_diagnostics\n", "=" * 80, sep="")
    np.random.seed(0)
    vector = np.random.normal(size=1000)
    plot_normality_diagnostics(vector, title="Normality Diagnostics (single vector)",
                                breaks=30, file="plot_normality_diagnostics_vector")
    print("saved plot_normality_diagnostics_vector_*.pdf")

    df_norm = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=["A", "B"])
    plot_normality_diagnostics(df_norm, title="Normality Diagnostics (2 columns)",
                                file="plot_normality_diagnostics_df")
    print("saved plot_normality_diagnostics_df_*.pdf")

    print("\n" + "=" * 80, "\nouter_summary / remove_outliers\n", "=" * 80, sep="")
    print(outlier_summary(vector))
    cleaned_na = remove_outliers(vector)
    print(f"remove_outliers (na_rm=True): {np.isnan(cleaned_na).sum()} values replaced with NaN")
    cleaned_type6 = remove_outliers(vector, type=6)
    print(f"remove_outliers (quantile type 6): {np.isnan(cleaned_type6).sum()} values replaced with NaN")
