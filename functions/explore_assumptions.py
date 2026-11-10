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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from plotnine import (
    ggplot, aes, geom_histogram, geom_qq, geom_qq_line, geom_boxplot,
    theme_bw, theme, labs, facet_wrap, coord_flip, geom_point, geom_text, geom_hline,
)
try:
    from ._rcompat import r_number
    from .functions_environment import write_txt
    from .functions_excel import excel_critical_value
    from .functions_statistical import compute_standard
    from .functions_strings import output_separator
except ImportError:
    from _rcompat import r_number
    from functions_environment import write_txt
    from functions_excel import excel_critical_value
    from functions_statistical import compute_standard
    from functions_strings import output_separator
##########################################################################################
# PLOT HISTOGRAM
##########################################################################################
def plot_histogram(df, base_size=10, title="Histogram", bins=10):
    """
    Faceted histogram, one panel per numeric column.

    Parameters:
    df (pandas.DataFrame): Numeric columns to plot.
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Plot title. Defaults to "Histogram".
    bins (int, optional): Number of histogram bins. Defaults to 10.

    Returns:
    plotnine.ggplot

    Examples:
    >>> import pandas as pd
    >>> personality = pd.read_csv("data/personality.csv")
    >>> plot_histogram(personality.iloc[:, 1:6])
    """
    df = df.melt()
    gp = (ggplot(data=df, mapping=aes(x="value"))
          + geom_histogram(bins=bins, na_rm=True, colour="#FFFFFF")
          + labs(title=title, y="Frequency", x="")
          + theme_bw(base_size=base_size)
          + theme(panel_spacing=0.025)
          + facet_wrap('~variable', scales='free'))
    return gp
##########################################################################################
# PLOT QQ
##########################################################################################
def plot_qq(df, base_size=10, title="QQ"):
    """
    Faceted normal Q-Q plot, one panel per numeric column.

    Parameters:
    df (pandas.DataFrame): Numeric columns to plot.
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Plot title. Defaults to "QQ".

    Returns:
    plotnine.ggplot

    Examples:
    >>> plot_qq(personality.iloc[:, 1:6])
    """
    df = df.melt()
    gp = (ggplot(data=df)
          + geom_qq(aes(sample="value"))
          + geom_qq_line(aes(sample="value"))
          + labs(title=title)
          + theme_bw(base_size=base_size)
          + theme(panel_spacing=0.025)
          + facet_wrap('~variable', scales='free'))
    return gp
##########################################################################################
# PLOT BOXPLOT
##########################################################################################
def plot_boxplot(df, base_size=10, title="Boxplot"):
    """
    Boxplot of every numeric column, flipped horizontal.

    Parameters:
    df (pandas.DataFrame): Data (non-numeric columns are dropped).
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Plot title. Defaults to "Boxplot".

    Returns:
    plotnine.ggplot

    Examples:
    >>> plot_boxplot(personality)
    """
    df = df.select_dtypes(include=np.number)
    df = df.melt()
    gp = (ggplot(df, aes(x="variable", y="value"))
          + geom_boxplot()
          + labs(title=title, x="", y="")
          + coord_flip()
          + theme_bw(base_size=base_size)
          + theme(panel_spacing=0.025))
    return gp
##########################################################################################
# PLOT NORMALITY DIAGNOSTICS
##########################################################################################
def plot_normality_diagnostics(df, breaks='sturges', title='', file=None, w=10, h=10):
    """
    4-panel normality diagnostics (histogram, density, boxplot, Q-Q) per
    numeric column, via matplotlib/seaborn.

    Parameters:
    df (array-like or pandas.DataFrame): Numeric data. A 1-D array/Series
        is treated as a single column named "data".
    breaks (str or int, optional): Histogram bin spec, forwarded to
        seaborn.histplot's `bins`. Defaults to "sturges".
    title (str, optional): Figure suptitle. Defaults to "".
    file (str, optional): If given, saves the figure to this path.
        Defaults to None.
    w, h (float, optional): Figure width, and height per row. Defaults to 10, 10.

    Returns:
    None. Displays (and optionally saves) the figure as a side effect.

    Examples:
    >>> import numpy as np
    >>> vector = np.random.normal(size=1000)
    >>> plot_normality_diagnostics(vector, title="Normality Diagnostics", breaks=30)
    >>> df = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=["A", "B"])
    >>> plot_normality_diagnostics(df, title="Normality Diagnostics")
    """
    data_name = df.columns if isinstance(df, pd.DataFrame) else 'data'
    df = pd.DataFrame(df)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 1:
        df.columns = [data_name] if not isinstance(data_name, pd.Index) else df.columns

    num_plots = len(df.columns)
    fig, axes = plt.subplots(num_plots, 4, figsize=(w, h * num_plots), squeeze=False)

    for i, col in enumerate(df.columns):
        vector = df[col].dropna()

        if len(vector) > 2 and np.var(vector) != 0:
            sns.histplot(vector, bins=breaks, kde=False, ax=axes[i, 0])
            axes[i, 0].set_title('Histogram')

            sns.kdeplot(vector, ax=axes[i, 1])
            axes[i, 1].set_title('Density Function')

            sns.boxplot(x=vector, ax=axes[i, 2])
            axes[i, 2].set_title('Boxplot')

            stats.probplot(vector, dist="norm", plot=axes[i, 3])
            axes[i, 3].get_lines()[1].set_color('red')
            axes[i, 3].set_title('QQ Plot')

            axes[i, 0].set_xlabel('')
            axes[i, 1].set_xlabel('')
            axes[i, 2].set_xlabel('')

            fig.suptitle(title, y=1.05)
            plt.subplots_adjust(hspace=0.4)

            if file:
                fig.savefig(file)
        else:
            print(f"Graph not produced for {col} due to sample size")
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
    (percentage of cases beyond each threshold, as strings with a "%" suffix).

    Examples:
    >>> import numpy as np
    >>> vector = np.random.normal(size=1000)
    >>> outlier_summary(vector)
    """
    zvariable = (vector - np.mean(vector)) / np.std(vector)
    outlier95 = np.abs(zvariable) >= 1.96
    outlier99 = np.abs(zvariable) >= 2.58
    outlier999 = np.abs(zvariable) >= 3.29
    ncases = len(vector[~np.isnan(zvariable)])
    percent95 = round(100 * len(outlier95[outlier95 == True]) / ncases, 2)
    percent99 = round(100 * len(outlier99[outlier99 == True]) / ncases, 2)
    percent999 = round(100 * len(outlier999[outlier999 == True]) / ncases, 2)
    result = pd.DataFrame({
        'abs_z_1.96': [f"{percent95}%"],
        'abs_z_2.58': [f"{percent99}%"],
        'abs_z_3.29': [f"{percent999}%"]
    })
    return result
##########################################################################################
# REMOVE OUTLIERS
##########################################################################################
def remove_outliers(vector, probs=[0.25, 0.75], na_rm=True, **kwargs):
    """
    Flag or drop Tukey (IQR-rule) outliers from a numeric vector.

    Parameters:
    vector (array-like): Numeric values.
    probs (list of float, optional): Lower/upper quantiles defining the
        IQR box. Defaults to [0.25, 0.75].
    na_rm (bool, optional): If True (default), out-of-range values are
        replaced with NaN (same length as input). If False, they're
        dropped (shorter output).

    Returns:
    numpy.ndarray

    Examples:
    >>> import numpy as np
    >>> vector = np.random.normal(size=1000)
    >>> remove_outliers(vector)
    >>> remove_outliers(vector, na_rm=False)
    """
    qnt = np.quantile(vector, probs)
    H = 1.5 * (qnt[1] - qnt[0])
    y = np.copy(vector)
    if na_rm:
        y[vector < (qnt[0] - H)] = np.nan
        y[vector > (qnt[1] + H)] = np.nan
    else:
        y = vector[(vector >= (qnt[0] - H)) & (vector <= (qnt[1] + H))]
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
                   f"\nMean={r_number(round(float(data['obs'].mean()), 2))}"
                   f"\nSD={r_number(round(float(data['obs'].std(ddof=1)), 2))}"
                   f"\nMedian={r_number(round(float(data['obs'].median()), 2))}")
        plots[column] = (
            ggplot(data, aes(x="id", y="obs", label="id"))
            + geom_point(aes(colour="Outlier"))
            + geom_text(data=data[data["Outlier"]], size=7, nudge_x=0.6, color="black")
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
            shapiro = stats.shapiro(z)
            add({"statistic": shapiro.statistic, "p": shapiro.pvalue, "method": "Shapiro-Wilk normality test"},
                variable, n, _NORMALITY_INSTRUCTIONS["shapiro"])
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
    import os
    import matplotlib
    matplotlib.use("Agg")

    personality = pd.read_csv("data/personality.csv") if os.path.exists("data/personality.csv") \
        else pd.read_csv("../data/personality.csv")

    print("=" * 80, "\nplot_histogram / plot_qq / plot_boxplot\n", "=" * 80, sep="")
    plot_histogram(personality.iloc[:, 1:6]).save("plot_histogram_example.png", verbose=False)
    plot_qq(personality.iloc[:, 1:6]).save("plot_qq_example.png", verbose=False)
    plot_boxplot(personality.iloc[:, 1:6]).save("plot_boxplot_example.png", verbose=False)
    print("saved plot_histogram_example.png, plot_qq_example.png, plot_boxplot_example.png")

    print("\n" + "=" * 80, "\nplot_normality_diagnostics\n", "=" * 80, sep="")
    np.random.seed(0)
    vector = np.random.normal(size=1000)
    plot_normality_diagnostics(vector, title="Normality Diagnostics (single vector)",
                                breaks=30, file="plot_normality_diagnostics_vector.png")
    print("saved plot_normality_diagnostics_vector.png")

    df_norm = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=["A", "B"])
    plot_normality_diagnostics(df_norm, title="Normality Diagnostics (2 columns)",
                                file="plot_normality_diagnostics_df.png")
    print("saved plot_normality_diagnostics_df.png")

    print("\n" + "=" * 80, "\nouter_summary / remove_outliers\n", "=" * 80, sep="")
    print(outlier_summary(vector))
    cleaned_na = remove_outliers(vector)
    print(f"remove_outliers (na_rm=True): {np.isnan(cleaned_na).sum()} values replaced with NaN")
    cleaned_drop = remove_outliers(vector, na_rm=False)
    print(f"remove_outliers (na_rm=False): {len(vector) - len(cleaned_drop)} values dropped")
