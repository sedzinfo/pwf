# -*- coding: utf-8 -*-
"""
Python port of R rwf::EXPLORE_TIME_SERIES.R.

Notes on the port:
  - R's `ts` object (value + start/frequency) has no direct Python
    equivalent, so every function here accepts a plain array-like /
    pandas Series and uses a numeric 0..n-1 time index, matching what
    R's own `time(df)` reduces to conceptually.
  - `plot_acf`'s confidence bands replicate R's `Rmisc::CI()` exactly:
    a t-distribution CI of the *mean* of the ACF/covariance/PACF values
    themselves — not the usual +-1.96/sqrt(n) significance bands. This
    is what the R source actually computes, kept as-is for parity.
  - `ts_smoothing`'s "friedman" (R's `supsmu`) and "splines" (R's
    `smooth.spline`) have no exact Python equivalent; both are
    approximated (lowess and a smoothing spline respectively) — see
    each branch's comment.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, acovf
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from scipy.interpolate import UnivariateSpline
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_smooth, geom_hline, facet_grid,
    labs, theme_bw, theme,
)
##########################################################################################
# PLOTS
##########################################################################################
def plot_ts(df, base_size=10, ylab="Count", title=""):
    """
    Line plot for a time series: a line + semi-transparent points, with
    an overlaid linear trend line (via geom_smooth(method="lm")). The
    returned plot can be extended with additional layers.

    Parameters:
    df (array-like or pandas.Series): The time series to plot. Values are
        plotted against a plain 0..n-1 time index.
    base_size (int, optional): Base font size for theme_bw(). Defaults to 10.
    ylab (str, optional): Y-axis label. Defaults to "Count".
    title (str, optional): Plot title. Defaults to "".

    Returns:
    plotnine.ggplot: line + point plot with an lm trend line, captioned
    with the number of observations.

    Examples:
    >>> import numpy as np
    >>> ts_data = np.random.normal(100, 10, 120) + np.linspace(0, 20, 120)
    >>> p = plot_ts(ts_data, title="Example series")
    """
    series = pd.Series(df).reset_index(drop=True)
    data = pd.DataFrame({'date': np.arange(len(series)), 'value': series.values})
    p = (ggplot(data, aes(x='date', y='value'))
         + geom_line(alpha=.5)
         + geom_point(alpha=.1, size=1)
         + theme_bw(base_size=base_size)
         + labs(title=title, x="Time index", y=ylab, caption=f"Observations {len(data)}")
         + geom_smooth(method='lm'))
    return p
##########################################################################################
# PLOT ACF
##########################################################################################
def _ci_of_mean(values, ci=0.95):
    """95% CI of the mean of `values`, via t-distribution (R's Rmisc::CI())."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_val = scipy.stats.t.ppf((1 + ci) / 2, df=n - 1)
    return mean - t_val * se, mean, mean + t_val * se  # lower, mean, upper


def plot_acf(df, lag_max=None, base_size=10, title=""):
    """
    Faceted autocorrelation (ACF), autocovariance, and partial
    autocorrelation (PACF) plot, with dashed 95% CI lines computed from
    the distribution of each function's own values (matching R's
    Rmisc::CI-based bands, not the usual significance bands). NaNs are
    dropped before computing.

    Note: to match R's `stats::acf(type="correlation"/"covariance")`,
    which include the lag-0 term, and `type="partial"`, which does not,
    the Correlation/Covariance facets start at lag 0 while the Partial
    Correlation facet starts at lag 1 — the same index-vs-lag offset
    already present in the R source.

    Parameters:
    df (array-like): The time series to analyse.
    lag_max (int, optional): Maximum number of lags. Defaults to len(df) - 1.
    base_size (int, optional): Base font size for theme_bw(). Defaults to 10.
    title (str, optional): Plot title. Defaults to "".

    Returns:
    plotnine.ggplot: 3 free-scale facets (Correlation, Covariance,
    Partial.Correlation), each with 2 dashed blue CI bound lines and a
    dashed black mean line.

    Examples:
    >>> import numpy as np
    >>> ts_data = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
    >>> p = plot_acf(ts_data, base_size=12)
    """
    series = pd.Series(df).dropna().to_numpy()
    if lag_max is None:
        lag_max = len(series) - 1
    lag_max = min(lag_max, len(series) - 1)

    correlation = acf(series, nlags=lag_max, fft=False)
    covariance = acovf(series, nlag=lag_max)
    # statsmodels prepends an artificial 1.0 "lag 0" pacf isn't defined for; drop it to match R's pacf.
    partial = pacf(series, nlags=lag_max)[1:]

    frames = []
    for name, values in [('Correlation', correlation), ('Covariance', covariance),
                          ('Partial.Correlation', partial)]:
        frames.append(pd.DataFrame({'index': np.arange(1, len(values) + 1), 'type': name, 'value': values}))
    autocorrelation = pd.concat(frames, ignore_index=True)
    type_order = ['Correlation', 'Covariance', 'Partial.Correlation']
    autocorrelation['type'] = pd.Categorical(autocorrelation['type'], categories=type_order)

    ci_rows = []
    for name, values in [('Correlation', correlation), ('Covariance', covariance),
                          ('Partial.Correlation', partial)]:
        lower, mean, upper = _ci_of_mean(values)
        ci_rows += [
            {'type': name, 'line': 'upper', 'ci': upper},
            {'type': name, 'line': 'lower', 'ci': lower},
            {'type': name, 'line': 'mean', 'ci': mean},
        ]
    ci_df = pd.DataFrame(ci_rows)
    ci_df['type'] = pd.Categorical(ci_df['type'], categories=type_order)

    p = (ggplot(autocorrelation, aes(x='index', y='value', color='type'))
         + geom_line()
         + geom_point(alpha=.1)
         + geom_hline(ci_df[ci_df['line'] == 'upper'], aes(yintercept='ci'),
                       alpha=.5, colour='blue', linetype='dashed')
         + geom_hline(ci_df[ci_df['line'] == 'lower'], aes(yintercept='ci'),
                       alpha=.5, colour='blue', linetype='dashed')
         + geom_hline(ci_df[ci_df['line'] == 'mean'], aes(yintercept='ci'),
                       alpha=.5, colour='black', linetype='dashed')
         + labs(title=title, x="Lag", y="")
         + facet_grid('type ~ .', scales='free')
         + theme_bw(base_size=base_size)
         + theme(legend_position='none'))
    return p
##########################################################################################
# SMOOTHING
##########################################################################################
def ts_smoothing(df, start=.01, stop=2, step=.001, title="", type="kernel", frequency=1):
    """
    Plot a time series with an overlaid family of smoothed curves swept
    across a bandwidth/span range, drawn in matplotlib (a side-effect
    plot, like R's base-graphics original — this function returns None).

    For bandwidth-based methods ("kernel", "lowess", "friedman",
    "splines", "default"), a sequence of values from `start` to `stop`
    is swept and each curve drawn in a different rainbow color, useful
    for visually picking a smoothing level. "polynomial" and "linear"
    ignore the sweep and fit regression-based trend lines instead.

    Parameters:
    df (array-like): The time series to smooth.
    start (float, optional): Start of the bandwidth/span sequence. Defaults to 0.01.
    stop (float, optional): End of the bandwidth/span sequence. Defaults to 2.
    step (float, optional): Increment between values. Defaults to 0.001.
    title (str, optional): Appended to the plot title. Defaults to "".
    type (str, optional): One of:
        - "kernel": Gaussian-weighted moving average, bandwidth = std of
          the Gaussian weights (approximates R's ksmooth(..., "normal")).
        - "lowess": LOWESS via statsmodels, bandwidth = span fraction
          (must be in (0, 1]).
        - "friedman": approximated with LOWESS (span in (0, 1)) — Python
          has no equivalent to R's supsmu() Friedman super-smoother.
        - "splines": smoothing spline via scipy.interpolate.UnivariateSpline;
          bandwidth maps loosely to R's `spar` penalty, not numerically
          equivalent.
        - "default": centered running-mean filter, window = round(bandwidth).
        - "polynomial": fits a centred cubic trend with and without
          seasonal cos/sin terms (period = `frequency`).
        - "linear": fits and draws a simple linear trend.
    frequency (int, optional): Seasonal period used by "polynomial" (R's
        ts frequency attribute has no Python equivalent, so it's passed
        explicitly). Defaults to 1 (no seasonality).

    Returns:
    None. Produces a matplotlib plot as a side effect.

    Examples:
    >>> import numpy as np
    >>> ts_data = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
    >>> ts_smoothing(ts_data, start=.01, stop=2, step=.01, type="lowess")
    """
    series = np.asarray(df, dtype=float)
    if len(series) == 0 or np.isnan(series[0]):
        return None

    t = np.arange(len(series), dtype=float)
    fig, ax = plt.subplots()
    ax.plot(t, series, '-', color='black', linewidth=1)
    ax.set_xlabel("Time index")
    ax.set_title(f"{type.capitalize()} Smoothing {title}")

    if type in ("default", "kernel", "lowess", "friedman", "splines"):
        sweep = np.arange(start, stop, step)
        n = len(sweep)
        colors = plt.cm.rainbow(np.linspace(0, max(1, n - 1) / n, n))
        for i, bandwidth in enumerate(sweep):
            try:
                if type == "default":
                    window = max(1, round(bandwidth))
                    smoothed = pd.Series(series).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
                    ax.plot(t, smoothed, color=colors[i], linewidth=1)
                elif type == "kernel":
                    if bandwidth <= 0:
                        continue
                    weights = np.exp(-0.5 * ((t[:, None] - t[None, :]) / bandwidth) ** 2)
                    smoothed = (weights @ series) / weights.sum(axis=1)
                    ax.plot(t, smoothed, color=colors[i], linewidth=1)
                elif type == "lowess":
                    if not (0 < bandwidth <= 1):
                        continue
                    smoothed = sm_lowess(series, t, frac=bandwidth, return_sorted=False)
                    ax.plot(t, smoothed, color=colors[i], linewidth=1)
                elif type == "friedman":
                    if not (0 < bandwidth < 1):
                        continue
                    smoothed = sm_lowess(series, t, frac=bandwidth, return_sorted=False)
                    ax.plot(t, smoothed, color=colors[i], linewidth=1)
                elif type == "splines":
                    if bandwidth <= 0:
                        continue
                    spline = UnivariateSpline(t, series, s=bandwidth * len(series))
                    ax.plot(t, spline(t), color=colors[i], linewidth=1)
            except Exception:
                continue

    if type == "polynomial":
        wk = t / frequency - (t / frequency).mean()
        wk2, wk3 = wk ** 2, wk ** 3
        cs, sn = np.cos(2 * np.pi * wk), np.sin(2 * np.pi * wk)
        X1 = np.column_stack([np.ones_like(wk), wk, wk2, wk3])
        X2 = np.column_stack([np.ones_like(wk), wk, wk2, wk3, cs, sn])
        fit1 = X1 @ np.linalg.lstsq(X1, series, rcond=None)[0]
        fit2 = X2 @ np.linalg.lstsq(X2, series, rcond=None)[0]
        ax.plot(t, fit1, linewidth=1)
        ax.plot(t, fit2, linewidth=1)

    if type == "linear":
        slope, intercept = np.polyfit(t, series, 1)
        ax.plot(t, intercept + slope * t, linewidth=1, color='#400c0c')

    return None
##########################################################################################
# MOVING AVERAGE
##########################################################################################
def compute_moving_average(df, w):
    """
    Centred moving average: replaces every value in every numeric column
    with the mean of a symmetric window of 2w+1 rows (w rows before, the
    row itself, w rows after). Windows are clipped near the edges.

    Note: faithfully replicates an off-by-one quirk in the R source —
    the very last row of the series is never included in *any* row's
    average window, even for rows far from the boundary, because the
    window-clipping filter uses a strict `< max_row` bound instead of
    `<= max_row`. This is preserved as-is rather than "fixed", to match
    R's actual output.

    Parameters:
    df (pandas.DataFrame): Data frame whose numeric columns are smoothed.
    w (int): Half-window size (total window width = 2w + 1).

    Returns:
    pandas.DataFrame: Same shape and columns as `df`, values replaced by
    their centred moving average.

    Examples:
    >>> import pandas as pd
    >>> compute_moving_average(df=pd.DataFrame({'x': range(20)}), w=2)
    """
    df = pd.DataFrame(df)
    result = df.astype(float)
    max_row = len(df)
    for col in df.columns:
        values = df[col].to_numpy()
        for row_index in range(max_row):
            idx = np.arange(row_index - w, row_index + w + 1)
            idx = idx[(idx >= 0) & (idx < max_row - 1)]
            result.iloc[row_index, result.columns.get_loc(col)] = values[idx].mean()
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    np.random.seed(42)
    n = 200
    t = np.arange(n)
    ts_data = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n)

    print("=" * 80, "\nplot_ts\n", "=" * 80, sep="")
    p_ts = plot_ts(ts_data, title="Example series")
    p_ts.save("plot_ts.png", verbose=False)
    print("saved plot_ts.png")

    print("\n" + "=" * 80, "\nplot_acf\n", "=" * 80, sep="")
    p_acf = plot_acf(ts_data, lag_max=40, title="Example ACF")
    p_acf.save("plot_acf.png", verbose=False)
    print("saved plot_acf.png")

    print("\n" + "=" * 80, "\nts_smoothing\n", "=" * 80, sep="")
    for smoothing_type in ["default", "kernel", "lowess", "friedman", "splines", "polynomial", "linear"]:
        plt.close("all")
        ts_smoothing(ts_data, start=.01, stop=2, step=.05, title="Example",
                     type=smoothing_type, frequency=12)
        fname = f"ts_smoothing_{smoothing_type}.png"
        plt.savefig(fname)
        print(f"saved {fname}")

    print("\n" + "=" * 80, "\ncompute_moving_average\n", "=" * 80, sep="")
    df_small = pd.DataFrame({"x": range(10)})
    print(compute_moving_average(df_small, w=2))
