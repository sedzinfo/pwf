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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from plotnine import (
    ggplot, aes, geom_histogram, geom_qq, geom_qq_line, geom_boxplot,
    theme_bw, theme, labs, facet_wrap, coord_flip,
)
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
          + theme(subplots_adjust={'hspace': 0.1, 'wspace': 0.1})
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
          + theme(subplots_adjust={'hspace': 0.1, 'wspace': 0.1})
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
          + theme(subplots_adjust={'hspace': 0.1, 'wspace': 0.1}))
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
