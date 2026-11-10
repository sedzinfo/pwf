# -*- coding: utf-8 -*-
"""
Python port of R rwf::EXPLORE_TIME_SERIES.R.

plot_ts, plot_acf and compute_moving_average match rwf
(tests/test_explore_time_series.py):
  - A series without its own time index is indexed 1..n, as R's time()
    of a plain vector; a pandas Series keeps a non-default index.
  - plot_acf's data are R's acf (correlation and covariance with lag 0)
    and pacf (Durbin-Levinson from the biased autocovariances), and its
    dashed lines are Rmisc::CI of each function's own values (the
    t-interval of their mean, as rwf computes, not significance bands).
  - compute_moving_average averages the rows within w of each row,
    including the last row.

ts_smoothing draws, as rwf, one curve per smoothing value from start to
stop (inclusive, R's seq): "default" is R's two-sided filter (missing at
the ends), "kernel" R's ksmooth with the normal kernel, "lowess"
Cleveland's lowess as R's. R's supsmu ("friedman") and smooth.spline
("splines") have no Python equivalents; they are approximated by lowess
and scipy's smoothing spline.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from scipy.interpolate import UnivariateSpline
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_smooth, geom_hline, facet_grid,
    labs, theme_bw, theme,
)
##########################################################################################
# TIME INDEX
##########################################################################################
def _time(df):
    """R's time(): the Series' own index when it has one, else 1..n."""
    if isinstance(df, pd.Series) and not isinstance(df.index, pd.RangeIndex):
        return df.index.to_numpy()
    return np.arange(1, len(df) + 1)


def _r_seq(start, stop, step):
    """R's seq(start, stop, by = step)."""
    n = int(np.floor((stop - start) / step + 1e-10))
    return start + np.arange(n + 1) * step
##########################################################################################
# PLOTS
##########################################################################################
def plot_ts(df, base_size=10, ylab="Count", title=""):
    """
    Time series plot, as rwf::plot_ts: the values against time as a line
    and faint points, with a linear trend and its confidence band.

    Parameters:
    df (array-like or pandas.Series): The series; plotted against its index when it is a Series with one,
        else against 1..n (R's time() of a plain vector).
    base_size (int, optional): Base font size. Defaults to 10.
    ylab (str, optional): Y-axis label. Defaults to "Count".
    title (str, optional): Plot title. Defaults to "".

    Returns:
    plotnine.ggplot: The plot; its data has rwf's columns date and value.

    Examples:
    >>> plot_ts(np.random.normal(100, 10, 120) + np.linspace(0, 20, 120), title="Example series")
    """
    data = pd.DataFrame({"date": _time(df), "value": np.asarray(df, dtype=float)})
    return (ggplot(data, aes(x="date", y="value"))
            + geom_line(alpha=.5)
            + geom_point(alpha=.1, size=1)
            + theme_bw(base_size=base_size)
            + labs(title=title, x="Time index", y=ylab, caption=f"Observations {len(data)}")
            + geom_smooth(method="lm"))
##########################################################################################
# PLOT ACF
##########################################################################################
_ACF_TYPES = ["Correlation", "Covariance", "Partial.Correlation"]


def _durbin_levinson(r):
    """Partial autocorrelations at lags 1..len(r)-1 from autocorrelations r (lag 0 first), as R's pacf."""
    lags = len(r) - 1
    partial = np.zeros(lags)
    phi = np.zeros(lags)
    for k in range(1, lags + 1):
        previous = phi[:k - 1].copy()
        numerator = r[k] - np.sum(previous * r[k - 1:0:-1]) if k > 1 else r[1]
        denominator = 1 - np.sum(previous * r[1:k]) if k > 1 else 1.0
        phi_kk = numerator / denominator
        phi[:k - 1] = previous - phi_kk * previous[::-1]
        phi[k - 1] = phi_kk
        partial[k - 1] = phi_kk
    return partial


def _acf_values(series, lag_max):
    """R's acf(type = "correlation"), acf(type = "covariance") and acf(type = "partial")."""
    lag_max = min(lag_max, len(series) - 1)
    correlation = acf(series, nlags=lag_max, fft=False)
    return {"Correlation": correlation, "Covariance": acovf(series, nlag=lag_max),
            "Partial.Correlation": _durbin_levinson(correlation)}


def _ci(values, ci=0.95):
    """Rmisc::CI: upper, mean and lower of the t-interval of the mean."""
    x = np.asarray(values, dtype=float)
    mean = x.mean()
    error = scipy.stats.t.ppf((1 + ci) / 2, df=len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return mean + error, mean, mean - error


def _acf_ci(series, lag_max):
    """rwf's dashed lines of plot_acf: Rmisc::CI of each function's values."""
    series = pd.Series(series).dropna().to_numpy(dtype=float)
    rows = [dict(zip(["upper", "mean", "lower"], _ci(values)), type=name)
            for name, values in _acf_values(series, lag_max).items()]
    return pd.DataFrame(rows)[["type", "upper", "mean", "lower"]]


def plot_acf(df, lag_max=None, base_size=10, title=""):
    """
    Autocorrelation, autocovariance and partial autocorrelation of a
    series in three free-scale facets, as rwf::plot_acf, each with dashed
    lines at the mean (black) and the 95% t-interval of the mean (blue)
    of the function's own values (Rmisc::CI). The series index runs from
    1 in every facet, so the correlation and covariance (which include
    lag 0) start at index 1 for lag 0, while the partial correlation
    starts at index 1 for lag 1, as in rwf.

    Parameters:
    df (array-like): The series (missing values are dropped).
    lag_max (int, optional): Largest lag. Defaults to the series length (R caps it at n - 1).
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Plot title. Defaults to "".

    Returns:
    plotnine.ggplot: The plot; its data has rwf's columns index, type and value.

    Examples:
    >>> plot_acf(np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200))
    """
    series = pd.Series(df).dropna().to_numpy(dtype=float)
    lag_max = len(series) if lag_max is None else lag_max
    frames = [pd.DataFrame({"index": np.arange(1, len(values) + 1), "type": name, "value": values})
              for name, values in _acf_values(series, lag_max).items()]
    autocorrelation = pd.concat(frames, ignore_index=True)
    autocorrelation["type"] = pd.Categorical(autocorrelation["type"], categories=_ACF_TYPES)
    lines = _acf_ci(series, lag_max)
    lines["type"] = pd.Categorical(["Correlation", "Covariance", "Partial.Correlation"], categories=_ACF_TYPES)
    return (ggplot(autocorrelation, aes(x="index", y="value", color="type", fill="type"))
            + geom_line()
            + geom_hline(lines, aes(yintercept="upper"), alpha=.5, colour="blue", linetype="dashed")
            + geom_hline(lines, aes(yintercept="mean"), alpha=.5, colour="black", linetype="dashed")
            + geom_hline(lines, aes(yintercept="lower"), alpha=.5, colour="blue", linetype="dashed")
            + geom_point(alpha=.1)
            + labs(title=title, x="Lag", y="")
            + facet_grid("type ~ .", scales="free")
            + theme_bw(base_size=base_size)
            + theme(legend_position="none"))
##########################################################################################
# SMOOTHING
##########################################################################################
def _r_filter(x, k):
    """R's filter(x, rep(1 / k, k)) with sides = 2: centred running mean, missing where incomplete."""
    n = len(x)
    offset = k // 2
    result = np.full(n, np.nan)
    for i in range(n):
        start = i + offset - k + 1
        if start >= 0 and i + offset < n:
            result[i] = x[start:i + offset + 1].mean()
    return result


def _ksmooth(t, y, bandwidth):
    """R's ksmooth(t, y, "normal", bandwidth): Gaussian weights with quartiles at +-bandwidth / 4."""
    points = np.linspace(t.min(), t.max(), max(100, len(t)))
    sd = 0.3706506 * bandwidth
    values = np.full(len(points), np.nan)
    for k, x0 in enumerate(points):
        near = np.abs(t - x0) < 4 * sd
        if near.any():
            weights = np.exp(-0.5 * ((t[near] - x0) / sd) ** 2)
            values[k] = np.sum(weights * y[near]) / np.sum(weights)
    return points, values


def ts_smoothing(df, start=.01, stop=2, step=.001, title="", type="kernel", frequency=1):
    """
    A series with a family of smoothed curves, one per smoothing value
    from start to stop, as rwf::ts_smoothing (matplotlib, drawn as a side
    effect; returns None). "polynomial" and "linear" draw trend fits
    instead.

    Parameters:
    df (array-like): The series.
    start, stop, step (float, optional): Smoothing values, R's seq(start, stop, step). Defaults to .01, 2, .001.
    title (str, optional): Added to the plot title. Defaults to "".
    type (str, optional): "kernel" (R's ksmooth, normal kernel, default), "default" (running mean of
        round(value) points, R's filter), "lowess" (span value), "friedman" (span in (0, 1), lowess
        standing in for R's supsmu), "splines" (scipy smoothing spline standing in for smooth.spline),
        "polynomial" (cubic trend with and without a cos/sin cycle) or "linear" (linear trend).
    frequency (int, optional): Cycle length used by "polynomial" (R takes it from the ts). Defaults to 1.

    Returns:
    None

    Examples:
    >>> ts_smoothing(np.sin(np.linspace(0, 20, 200)), start=.05, stop=.5, step=.05, type="lowess")
    """
    series = np.asarray(df, dtype=float)
    if len(series) == 0 or np.isnan(series[0]):
        return None
    t = np.asarray(_time(df), dtype=float)
    fig, ax = plt.subplots()
    ax.plot(t, series, "-", color="black", linewidth=1)
    ax.set_xlabel("Time index")
    ax.set_title(f"{type.capitalize()} Smoothing {title}")
    if type in ("default", "kernel", "lowess", "friedman", "splines"):
        sweep = _r_seq(start, stop, step)
        n = len(sweep)
        colors = plt.cm.hsv(np.linspace(0, max(1, n - 1) / n, n + 1))
        for i, value in enumerate(sweep):
            color = colors[i + 1]
            if type == "default" and round(value) >= 1:
                ax.plot(t, _r_filter(series, int(round(value))), color=color, linewidth=1)
            elif type == "kernel" and value > 0:
                ax.plot(*_ksmooth(t, series, value), color=color, linewidth=1)
            elif type == "lowess" and 0 < value:
                fitted = sm_lowess(series, t, frac=min(value, 1), it=3, delta=0.01 * np.ptp(t), return_sorted=False)
                ax.plot(t, fitted, color=color, linewidth=1)
            elif type == "friedman" and 0 < value < 1:
                ax.plot(t, sm_lowess(series, t, frac=value, return_sorted=False), color=color, linewidth=1)
            elif type == "splines" and value > 0:
                ax.plot(t, UnivariateSpline(t, series, s=value * len(series))(t), color=color, linewidth=1)
    if type == "polynomial":
        wk = t / frequency - (t / frequency).mean()
        cs, sn = np.cos(2 * np.pi * wk), np.sin(2 * np.pi * wk)
        X1 = np.column_stack([np.ones_like(wk), wk, wk ** 2, wk ** 3])
        X2 = np.column_stack([X1, cs, sn])
        ax.plot(t, X1 @ np.linalg.lstsq(X1, series, rcond=None)[0], linewidth=1)
        ax.plot(t, X2 @ np.linalg.lstsq(X2, series, rcond=None)[0], linewidth=1)
    if type == "linear":
        slope, intercept = np.polyfit(t, series, 1)
        ax.plot(t, intercept + slope * t, linewidth=1, color="#400c0c")
    return None
##########################################################################################
# MOVING AVERAGE
##########################################################################################
def compute_moving_average(df, w):
    """
    Centred moving average of every column, as
    rwf::compute_moving_average: each row becomes the mean of the rows
    from w before to w after it (fewer near the ends).

    Parameters:
    df (pandas.DataFrame): Numeric columns.
    w (int): Rows on each side.

    Returns:
    pandas.DataFrame: The averages, same shape as df.

    Examples:
    >>> compute_moving_average(df=pd.DataFrame({"x": range(20)}), w=2)
    """
    df = pd.DataFrame(df)
    result = df.astype(float)
    max_row = len(df)
    for column in df.columns:
        values = df[column].to_numpy(dtype=float)
        result[column] = [values[max(0, i - w):min(max_row, i + w + 1)].mean() for i in range(max_row)]
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    np.random.seed(42)
    t = np.arange(200)
    ts_data = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, 200)
    print(plot_ts(ts_data, title="Example series").data.head())
    print(plot_acf(ts_data, lag_max=40).data.head())
    for smoothing_type in ["default", "kernel", "lowess", "friedman", "splines", "polynomial", "linear"]:
        ts_smoothing(ts_data, start=.05, stop=2, step=.25, title="Example", type=smoothing_type, frequency=12)
        plt.close("all")
    print(compute_moving_average(pd.DataFrame({"x": range(10)}), w=2))
