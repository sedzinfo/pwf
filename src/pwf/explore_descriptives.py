# -*- coding: utf-8 -*-
"""
Descriptive statistics, frequency tables and cross-tabulations (port of
rwf EXPLORE_DESCRIPTIVES.R), plus the Python-only flatten and
describe_by_mean helpers. rwf's plot_* functions print their plots; these
return them undrawn, as elsewhere in pwf.

Fixed a real bug: describe_by_mean used pandas' private/internal
`DataFrame._append`, which was removed in pandas 3.0 (the public
`.append()` method was removed in pandas 2.0, and `_append` was never a
stable API) -- replaced with `pd.concat()`.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_text, geom_rect, geom_col, scale_size_continuous,
    labs, theme, theme_bw, element_text, element_blank, coord_fixed, coord_flip,
)
try:
    from ._rcompat import r_label, r_levels, r_make_names, r_recycle, resolve_columns
    from .functions_excel import report_dataframe
    from .functions_statistical import compute_kurtosis, compute_skewness, compute_standard_error
    from .functions_strings import str_aes, str_wrap
except ImportError:
    from _rcompat import r_label, r_levels, r_make_names, r_recycle, resolve_columns
    from functions_excel import report_dataframe
    from functions_statistical import compute_kurtosis, compute_skewness, compute_standard_error
    from functions_strings import str_aes, str_wrap
##########################################################################################
# FLATTEN
##########################################################################################
def flatten(l):
    """
    Flatten a list of lists into a single flat list.

    Parameters:
    l (list of list): Nested list.

    Returns:
    list: Flattened list.

    Examples:
    >>> flatten([[1, 2], [3], [4, 5]])
    """
    return [item for sublist in l for item in sublist]
##########################################################################################
# DESCRIBE BY MEAN
##########################################################################################
def describe_by_mean(df, factorname=[]):
    """
    Mean/min/max/SD of every numeric column, grouped separately by each
    of several factor columns.

    Parameters:
    df (pandas.DataFrame): Data containing the factor and numeric columns.
    factorname (list of str, optional): Grouping column names. Each is
        summarized independently (not crossed). Defaults to [].

    Returns:
    pandas.DataFrame: Columns Statistic, Factor, Level, then one column
    per numeric variable in `df`.

    Examples:
    >>> import pandas as pd
    >>> titanic = load_dataset("titanic")
    >>> describe_by_mean(df=titanic, factorname=["survived", "pclass", "sex"])
    """
    result = pd.DataFrame()
    factor = []
    for category in factorname:
        df_mean = df.groupby(category).mean(numeric_only=True)
        df_min = df.groupby(category).min(numeric_only=True)
        df_max = df.groupby(category).max(numeric_only=True)
        df_std = df.groupby(category).std(numeric_only=True)
        df_mean = df_mean.assign(Statistic=["Mean"] * df_mean.shape[0])
        df_min = df_min.assign(Statistic=["Min"] * df_min.shape[0])
        df_max = df_max.assign(Statistic=["Max"] * df_max.shape[0])
        df_std = df_std.assign(Statistic=["SD"] * df_std.shape[0])
        descriptives = pd.concat([df_mean, df_min, df_max, df_std])
        descriptives.reset_index(inplace=True)
        descriptives.rename(columns={list(descriptives)[0]: 'Level'}, inplace=True)
        factor.append([category] * descriptives.shape[0])
        result = pd.concat([result, descriptives], ignore_index=True)
    result.insert(0, "Factor", flatten(factor))
    result.insert(0, "Statistic", result.pop("Statistic"))
    result = result.sort_values(by=["Statistic", "Factor", "Level"])
    return result
##########################################################################################
# DESCRIPTIVES
##########################################################################################
_DESCRIPTIVES_COMMENT = {
    "factor": "independent variable",
    "levels": "levels of independent variable\n\ndiscrete variables observed in independent variable",
    "variable": "dependent variable\n\ncontinous variable",
    "n": "sample size",
    "mean": "measure of central tendency\n\nmean",
    "sd": "measure of dispersion\nstandard deviation\n\nlow values indicate low dispersion of observations from the mean",
    "median": "measure of central tendency\n\nmedian\n\nvalue separating lower half from higher half of ordered observations",
    "trimmed": "measure of central tendency\n\ntrimmed mean\n\nmean after dropping .1 of minimum and maximum values in data",
    "mad": "measure of dispersion\n\nmedian absolute deviation\n\nlow values indicate low dispersion of observations from the median",
    "min": "minimum value observed",
    "max": "maximum value observed",
    "range": "measure of dispersion\n\nrange\n\ndifference between minimum and maximum value",
    "skew": "skewness\nnegative values indicate a left skewed distribution\n\npositive values indicate a right skewed distribution\n\ndeviations from 0 indicate deviation from normality",
    "kurtosis": "kurtosis\n\nnegative values generally indicate flat peak\n\npositive values generally indicate sharp peak\n\ndeviations from 0 indicate deviation from normality",
    "se": "measure of dispersion\n\nstandard error\n\nlow values indicate low dispersion of observations from the mean",
    "IQR": "measure of dispersion\n\ninterquantile range\n\nthe range around the median where 50% of observations fall",
    "Q0.1": "percentile\n\n10% of observations fall bellow this value",
    "Q0.25": "percentile\n\n25% of observations fall bellow this value",
    "Q0.5": "percentile\n\n50% of observations fall bellow this value",
    "Q0.75": "percentile\n\n75% of observations fall bellow this value",
    "Q0.9": "percentile\n\n90% of observations fall bellow this value",
}


def _trimmed_mean(x, trim):
    """R's mean(x, trim = trim) for x without missing values."""
    n = len(x)
    if n == 0:
        return np.nan
    if trim >= 0.5:
        return float(np.median(x))
    lo = int(np.floor(n * trim))
    return float(np.mean(np.sort(x)[lo:n - lo]))


def _mad(x):
    """stats::mad(x): 1.4826 * median absolute deviation from the median."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _psych_describe(values):
    """psych::describe(x, skew, ranges, IQR, quant = c(.1, .25, .5, .75, .9), trim = .1, type = 3)."""
    x = pd.Series(values).dropna().to_numpy(dtype=float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    quantiles = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]) if n else [np.nan] * 5
    q25, q75 = (np.quantile(x, [0.25, 0.75]) if n else (np.nan, np.nan))
    return {
        "n": n,
        "mean": float(np.mean(x)) if n else np.nan,
        "sd": sd,
        "median": float(np.median(x)) if n else np.nan,
        "trimmed": _trimmed_mean(x, 0.1),
        "mad": _mad(x),
        "min": float(np.min(x)) if n else np.inf,
        "max": float(np.max(x)) if n else -np.inf,
        "range": float(np.max(x) - np.min(x)) if n else -np.inf,
        "skew": compute_skewness(x),
        "kurtosis": compute_kurtosis(x),
        "se": sd / np.sqrt(n) if n else np.nan,
        "IQR": q75 - q25,
        "Q0.1": quantiles[0], "Q0.25": quantiles[1], "Q0.5": quantiles[2],
        "Q0.75": quantiles[3], "Q0.9": quantiles[4],
    }


def compute_descriptives(df, dv, iv=None, file=None):
    """
    Descriptive statistics of continuous variables, optionally per level of
    grouping variables (port of rwf::compute_descriptives, which uses
    psych::describe).

    Parameters:
    df (pandas.DataFrame): Input data.
    dv (str, int or list): Variables to describe, as names or 1-based positions.
    iv (str, int or list, optional): Grouping variables; each is used
        separately (not crossed). Rows missing the dv or the iv are dropped,
        and a pair with fewer than 2 complete rows is skipped.
    file (str, optional): Excel filename without extension.

    Returns:
    pandas.DataFrame: One row per variable (and level): [factor,] variable,
    [levels,] n, mean, sd, median, trimmed (10% trimmed mean), mad, min, max,
    range, skew, kurtosis, se, IQR, Q0.1, Q0.25, Q0.5, Q0.75, Q0.9.

    Examples:
    >>> compute_descriptives(df=df_insurance, dv=["age", "bmi", "charges"])
    >>> compute_descriptives(df=df_insurance, dv=[1, 7], iv=[2, 4])
    """
    rows = []
    if iv is None:
        for variable in resolve_columns(df, dv):
            rows.append({"variable": variable, **_psych_describe(df[variable])})
    else:
        for variable in resolve_columns(df, dv):
            for factor in resolve_columns(df, iv):
                temp = df[[variable, factor]].dropna()
                if len(temp) > 1:
                    for level in r_levels(temp[factor]):
                        values = temp.loc[temp[factor] == level, variable]
                        rows.append({"factor": factor, "variable": variable, "levels": r_label(level),
                                     **_psych_describe(values)})
    result = pd.DataFrame(rows)
    report_dataframe(result, file=file, sheet="Descriptives", comment=_DESCRIPTIVES_COMMENT)
    return result
##########################################################################################
# AGGREGATE
##########################################################################################
def _numeric_columns(df, exclude=()):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def compute_aggregate(df, iv, file=None):
    """
    Descriptive statistics of every numeric column for each combination of
    the grouping variables, in long format (port of rwf::compute_aggregate).
    Missing values in a grouping variable form their own group, last.

    Parameters:
    df (pandas.DataFrame): Input data; every numeric column not in iv is summarised.
    iv (str, int or list): Grouping variables, as names or 1-based positions.
    file (str, optional): Excel filename without extension.

    Returns:
    pandas.DataFrame: Columns statistic, the grouping variables, then one per
    numeric variable. statistic runs through mean, SD, median, mad,
    "trimmed mean" (50% trim, i.e. the median, as in rwf), N, min, max,
    range, skewness, kurtosis, IQR, SE.

    Examples:
    >>> compute_aggregate(df=df_insurance, iv=["sex", "smoker"])
    """
    factornames = resolve_columns(df, iv)
    columns = _numeric_columns(df, exclude=factornames)
    statistics = [
        ("mean", lambda x: np.mean(x) if len(x) else np.nan),
        ("SD", lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan),
        ("median", lambda x: np.median(x) if len(x) else np.nan),
        ("mad", _mad),
        ("trimmed mean", lambda x: _trimmed_mean(x, 0.5)),
        ("N", len),
        ("min", lambda x: np.min(x) if len(x) else np.inf),
        ("max", lambda x: np.max(x) if len(x) else -np.inf),
        ("range", lambda x: np.max(x) - np.min(x) if len(x) else -np.inf),
        ("skewness", compute_skewness),
        ("kurtosis", compute_kurtosis),
        ("IQR", lambda x: np.subtract(*np.quantile(x, [0.75, 0.25])) if len(x) else np.nan),
        ("SE", compute_standard_error),
    ]
    groups = list(df.groupby(factornames, dropna=False, sort=True))
    blocks = []
    for name, stat in statistics:
        rows = []
        for keys, piece in groups:
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = {"statistic": name, **dict(zip(factornames, keys))}
            for column in columns:
                row[column] = float(stat(piece[column].dropna().to_numpy(dtype=float)))
            rows.append(row)
        blocks.append(pd.DataFrame(rows))
    result = pd.concat(blocks, ignore_index=True)
    report_dataframe(result, file=file, sheet="Descriptives")
    return result
##########################################################################################
# FREQUENCIES
##########################################################################################
def compute_frequencies(df, file=None):
    """
    Frequency, proportion and percent of every value of every column, most
    frequent first within each column (port of rwf::compute_frequencies).
    Missing values are not counted.

    Parameters:
    df (pandas.DataFrame): Columns to tabulate.
    file (str, optional): Excel filename without extension.

    Returns:
    pandas.DataFrame: Columns variable, Observation, Frequency, Proportion, Percent.

    Examples:
    >>> compute_frequencies(df=df_insurance[["sex", "children", "smoker", "region"]])
    """
    blocks = []
    for variable in df.columns:
        values = df[variable].dropna()
        levels = r_levels(df[variable])
        if not levels:
            continue
        counts = [int((values == level).sum()) for level in levels]
        total = sum(counts)
        block = pd.DataFrame({"variable": variable, "Observation": [r_label(level) for level in levels],
                              "Frequency": counts, "Proportion": [c / total for c in counts]})
        blocks.append(block.sort_values("Frequency", ascending=False, kind="mergesort"))
    frequency = (pd.concat(blocks, ignore_index=True) if blocks
                 else pd.DataFrame(columns=["variable", "Observation", "Frequency", "Proportion"]))
    frequency["Percent"] = frequency["Proportion"] * 100
    report_dataframe(frequency, file=file, sheet="Frequency")
    return frequency
##########################################################################################
# RESPONSE FREQUENCIES
##########################################################################################
def response_frequency(df, max=10, uniqueitems=None, type="percent", file=None):
    """
    How often each response option was chosen, per variable, for ordinal or
    Likert items (port of rwf::response_frequency). Values outside
    uniqueitems count as missing. Returns None when the items look
    continuous: more than max unique values, or a value range wider than max.

    Parameters:
    df (pandas.DataFrame): Items as columns.
    max (int, optional): Largest number of response options allowed. Defaults to 10.
    uniqueitems (sequence, optional): All valid responses, e.g. range(1, 6).
        Defaults to the values found in df.
    type (str, optional): "frequency", "proportion", "percent" (default) or
        "all" (all three stacked).
    file (str, optional): Excel filename without extension.

    Returns:
    pandas.DataFrame or None: Columns type, variable (as R's make.names
    spells it), one per response option, miss, responses.

    Examples:
    >>> response_frequency(df_ocean[["N1", "N2"]], uniqueitems=range(1, 6), type="all")
    """
    df = pd.DataFrame(df)
    numbers = df.to_numpy(dtype=float)
    min_item, max_item = np.nanmin(numbers), np.nanmax(numbers)
    if uniqueitems is None:
        uniqueitems = list(pd.unique(pd.Series(numbers.ravel())))
    uniqueitems = list(uniqueitems)
    items = sorted(v for v in uniqueitems if not pd.isna(v))
    first_levels = df.iloc[:, 0].dropna().nunique()
    if max_item - min_item > max or first_levels > max or len(uniqueitems) > max:
        return None
    n_cases = len(df)
    frequency = pd.DataFrame([[int((df[c] == item).sum()) for item in items] for c in df.columns],
                             columns=[r_label(item) for item in items])
    responses = frequency.sum(axis=1)
    variables = r_make_names(df.columns)

    def block(kind):
        if kind == "frequency":
            values, miss = frequency, n_cases - responses
        elif kind == "proportion":
            values, miss = frequency.div(responses, axis=0), 1 - responses / n_cases
        else:
            values, miss = frequency.div(responses, axis=0) * 100, 100 - (responses / n_cases) * 100
        out = pd.concat([pd.DataFrame({"type": kind.capitalize(), "variable": variables}), values], axis=1)
        out["miss"] = miss
        out["responses"] = responses
        return out

    kinds = ["frequency", "proportion", "percent"] if type == "all" else [type]
    result = pd.concat([block(k) for k in kinds], ignore_index=True)
    if file is not None:
        report_dataframe(result, file=file, sheet="Frequency")
    return result
##########################################################################################
# CROSSTABLE
##########################################################################################
def _pairwise_combinations(df, factor_index, combinations):
    """Variable pairs: given combinations (first two columns), or every unordered pair in factor_index."""
    if combinations is not None:
        combinations = pd.DataFrame(combinations)
        return [(str(a), str(b)) for a, b in zip(combinations.iloc[:, 0], combinations.iloc[:, 1])]
    names = resolve_columns(df, factor_index)
    pairs, seen = [], set()
    for second in names:  # expand.grid: the first column varies fastest
        for first in names:
            key = tuple(sorted((first, second)))
            if first != second and key not in seen:
                seen.add(key)
                pairs.append((first, second))
    return pairs


def _table(df, f1, f2):
    """R's table(df[, f1], df[, f2]) as a long frame (first variable fastest)."""
    levels1, levels2 = r_levels(df[f1]), r_levels(df[f2])
    complete = df[[f1, f2]].dropna()
    counts = complete.groupby([f1, f2]).size()
    rows = [(l1, l2, int(counts.get((l1, l2), 0))) for l2 in levels2 for l1 in levels1]
    return pd.DataFrame(rows, columns=["l1", "l2", "Frequency"])


def compute_crosstable(df, factor_index=None, combinations=None):
    """
    Frequencies and percentages of every combination of levels for pairs of
    categorical variables (port of rwf::compute_crosstable).

    Parameters:
    df (pandas.DataFrame): Input data.
    factor_index (list, optional): Variables (names or 1-based positions);
        every unordered pair of them is tabulated.
    combinations (pandas.DataFrame, optional): Pairs to tabulate, one per
        row, in columns index1 and index2. Takes precedence over factor_index.

    Returns:
    pandas.DataFrame: Columns f1, f2, l1, l2, Frequency, Percent (of the
    pair's complete cases). Pairs without complete cases are dropped.

    Examples:
    >>> compute_crosstable(df=df_insurance, factor_index=["sex", "smoker", "region"])
    """
    blocks = []
    for f1, f2 in _pairwise_combinations(df, factor_index, combinations):
        table = _table(df, f1, f2)
        total = table["Frequency"].sum()
        if total > 0:
            table["Percent"] = table["Frequency"] / total * 100
            table["l1"] = table["l1"].map(r_label)
            table["l2"] = table["l2"].map(r_label)
            blocks.append(pd.concat([pd.DataFrame({"f1": [f1] * len(table), "f2": [f2] * len(table)}), table], axis=1))
    return pd.concat(blocks, ignore_index=True)


_R_SHAPES = {0: "s", 1: "o", 2: "^", 15: "s", 16: "o", 17: "^", 18: "D", 19: "o", 20: "."}


def plot_crosstable(df, factor_index=None, combinations=None, shape=16, angle=0, base_size=10, title="", pb=False):
    """
    Bubble plot of the cross-tabulation of each pair of categorical variables:
    point size and label give the frequency (port of rwf::plot_crosstable).
    Unlike rwf, the plots are returned without being drawn.

    Parameters:
    df (pandas.DataFrame): Input data.
    factor_index (list, optional): Variables (names or 1-based positions);
        every unordered pair is plotted.
    combinations (pandas.DataFrame, optional): Pairs to plot (columns index1,
        index2). Takes precedence over factor_index.
    shape (int or str, optional): Point shape, an R pch number or a
        matplotlib marker. Defaults to 16 (filled circle).
    angle (float, optional): Rotation of the x tick labels. Defaults to 0.
    base_size (float, optional): Base font size. Defaults to 10.
    title (str, optional): Plot title.
    pb (bool, optional): Print progress. Defaults to False.

    Returns:
    dict: "first_second" -> plotnine.ggplot, with data columns variable1,
    variable2, Frequency.

    Examples:
    >>> plot_crosstable(df=df_insurance, factor_index=["sex", "smoker", "region"])
    """
    plots = {}
    pairs = _pairwise_combinations(df, factor_index, combinations)
    for counter, (f1, f2) in enumerate(pairs, start=1):
        if pb:
            print(f"plot_crosstable {counter}/{len(pairs)}")
        table = _table(df, f1, f2)
        total = int(table["Frequency"].sum())
        if total == 0:
            continue
        levels1 = [r_label(v) for v in r_levels(df[f1])]
        levels2 = [r_label(v) for v in r_levels(df[f2])]
        data = pd.DataFrame({
            "variable1": pd.Categorical(table["l1"].map(r_label), categories=levels1),
            "variable2": pd.Categorical(table["l2"].map(r_label), categories=levels2),
            "Frequency": table["Frequency"],
        })
        plots[f"{f1}_{f2}"] = (
            ggplot(data, aes(x="variable1", y="variable2"))
            + geom_point(aes(size="Frequency", color="variable1"), shape=_R_SHAPES.get(shape, shape))
            + scale_size_continuous(range=(1, 30))
            + labs(x=str_aes(f1), y=str_aes(f2))
            + geom_text(aes(label="Frequency"))
            + theme_bw(base_size=base_size)
            + theme(axis_text_x=element_text(rotation=angle, ha="right" if angle else "center"),
                    legend_position="none")
            + labs(title=title, caption=f"Observations: {total}"))
    return plots
##########################################################################################
# MOSAIC
##########################################################################################
def plot_mosaic(df, factor_index=None, base_size=10, title="", pb=False):
    """
    Mosaic plot for every ordered pair of categorical variables: bar width is
    the share of each level of the first variable, bar height the share of
    the second variable within it (port of rwf::plot_mosaic). Rows missing
    either variable are dropped. Unlike rwf, the plots are returned without
    being drawn.

    Parameters:
    df (pandas.DataFrame): Input data.
    factor_index (list, optional): Variables (names or 1-based positions).
        Defaults to every column.
    base_size (float, optional): Base font size. Defaults to 10.
    title (str, optional): Placed above "<first> By <second>" in each title.
    pb (bool, optional): Print progress. Defaults to False.

    Returns:
    dict: "first second" -> plotnine.ggplot, with data columns v1, v2, Freq,
    marginVar1, var2height, var1Center as in rwf.

    Examples:
    >>> plot_mosaic(df=df_insurance, factor_index=["sex", "smoker"])
    """
    plots = {}
    names = resolve_columns(df, factor_index)
    for counter, first in enumerate(names, start=1):
        if pb:
            print(f"plot_mosaic {counter}/{len(names)}")
        for second in names:
            if first == second:
                continue
            temp = df[[first, second]].dropna()
            levels1 = [r_label(v) for v in r_levels(temp[first])]
            levels2 = [r_label(v) for v in r_levels(temp[second])]
            n_levels1 = len(levels1)
            if len(levels1) < 2:
                levels1.append("Second Level is Not Available")
            if len(levels2) < 2:
                levels2.append("Second Level is Not Available")
            v1, v2 = temp[first].map(r_label), temp[second].map(r_label)
            n = len(temp)
            margin = [float((v1 == l1).sum()) / n for l1 in levels1]
            rows = [(l1, l2, float(((v1 == l1) & (v2 == l2)).sum()) / n) for l2 in levels2 for l1 in levels1]
            data = pd.DataFrame(rows, columns=["v1", "v2", "Freq"])
            data["marginVar1"] = r_recycle(margin, len(data))
            with np.errstate(invalid="ignore", divide="ignore"):
                data["var2height"] = data["Freq"] / data["marginVar1"]
            starts = [0.0] + list(np.cumsum(margin)[:max(n_levels1 - 1, 0)])
            length = max(len(starts), len(margin))
            centers = [s + m / 2 for s, m in zip(r_recycle(starts, length), r_recycle(margin, length))]
            data["var1Center"] = r_recycle(centers, len(data))
            data["v1"] = pd.Categorical(data["v1"], categories=levels1)
            data["v2"] = pd.Categorical(data["v2"], categories=levels2)

            rects = data.copy()
            rects["xmin"] = rects["var1Center"] - rects["marginVar1"] / 2
            rects["xmax"] = rects["var1Center"] + rects["marginVar1"] / 2
            heights = rects["var2height"].fillna(0)
            # stacked like ggplot's position_stack: the first level of v2 on top
            rects["ymax"] = heights[::-1].groupby(rects["v1"][::-1], observed=False).cumsum()[::-1]
            rects["ymin"] = rects["ymax"] - heights
            labels = data.drop_duplicates("v1")
            plots[f"{first} {second}"] = (
                ggplot(data)
                + geom_rect(rects, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="v2"),
                            color="white")
                + geom_text(labels, aes(x="var1Center", y=0.5, label="v1"), angle=90)
                + labs(x="Proportion", y="Proportion", title=f"{title}\n{str_aes(first)} By {str_aes(second)}",
                       fill=first, caption=f"Observations: {n}")
                + theme_bw(base_size=base_size)
                + theme(legend_title=element_blank(), panel_grid_major=element_blank(),
                        panel_grid_minor=element_blank())
                + coord_fixed())
    return plots
##########################################################################################
# RESPONSE FREQUENCY PLOTS
##########################################################################################
def plot_response_frequencies(df, factor_index=None, base_size=10, title="", width=100, reorder=False):
    """
    Horizontal bar chart of the count of each value, one per variable (port
    of rwf::plot_response_frequencies). Missing values are dropped and
    variables without values are skipped. Unlike rwf, the plots are returned
    without being drawn.

    Parameters:
    df (pandas.DataFrame): Input data.
    factor_index (list, optional): Variables (names or 1-based positions).
        Defaults to every column.
    base_size (float, optional): Base font size. Defaults to 10.
    title (str, optional): Placed before the variable name in each title.
    width (int, optional): Wrap width of the variable name. Defaults to 100.
    reorder (bool, optional): Order bars by count instead of by value.

    Returns:
    dict: variable -> plotnine.ggplot, with data columns tempdata (the value)
    and Freq.

    Examples:
    >>> plot_response_frequencies(df=df_ocean[["N1", "N2"]])
    """
    plots = {}
    for variable in resolve_columns(df, factor_index):
        values = df[variable].dropna()
        levels = r_levels(df[variable].dropna())
        if not levels:
            continue
        labels = [r_label(level) for level in levels]
        counts = [int((values == level).sum()) for level in levels]
        data = pd.DataFrame({"tempdata": pd.Categorical(labels, categories=labels), "Freq": counts})
        if reorder:
            order = [labels[i] for i in np.argsort(counts, kind="mergesort")]
            data["tempdata"] = data["tempdata"].cat.reorder_categories(order)
        plots[variable] = (
            ggplot(data, aes(x="tempdata", y="Freq"))
            + geom_col()
            + labs(x="", y="Count", title=f"{title} {str_wrap(variable, width=width)}",
                   caption=f"Observations:{sum(counts)}")
            + coord_flip()
            + theme_bw(base_size=base_size))
    return plots
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    import os

    print("=" * 80, "\nflatten\n", "=" * 80, sep="")
    print(flatten([[1, 2], [3], [4, 5]]))

    print("\n" + "=" * 80, "\ndescribe_by_mean\n", "=" * 80, sep="")
    titanic = load_dataset("titanic")
    result = describe_by_mean(df=titanic, factorname=["survived", "pclass", "sex"])
    print(result)

    df_insurance = load_dataset("insurance")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n" + "=" * 80, "\ncompute_descriptives\n", "=" * 80, sep="")
        print(compute_descriptives(df=df_insurance, dv=["age", "bmi", "charges"]))
        print(compute_descriptives(df=df_insurance, dv=[1, 7], iv=[2, 4]))
        print("\n" + "=" * 80, "\ncompute_aggregate\n", "=" * 80, sep="")
        print(compute_aggregate(df=df_insurance, iv=["sex", "smoker"]))
        print("\n" + "=" * 80, "\ncompute_frequencies\n", "=" * 80, sep="")
        print(compute_frequencies(df=df_insurance[["sex", "children", "smoker", "region"]]))
        print("\n" + "=" * 80, "\nresponse_frequency\n", "=" * 80, sep="")
        print(response_frequency(df_insurance[["children"]], uniqueitems=range(0, 6), type="all"))
        print("\n" + "=" * 80, "\ncompute_crosstable\n", "=" * 80, sep="")
        print(compute_crosstable(df=df_insurance, factor_index=["sex", "smoker", "region"]))
        print("\n" + "=" * 80, "\nplot_crosstable / plot_mosaic / plot_response_frequencies\n", "=" * 80, sep="")
        print(list(plot_crosstable(df=df_insurance, factor_index=["sex", "smoker", "region"])))
        print(list(plot_mosaic(df=df_insurance, factor_index=["sex", "smoker"])))
        print(list(plot_response_frequencies(df=df_insurance, factor_index=["children", "region"])))

