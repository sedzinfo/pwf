# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_RECODE.R (flatten_list, swap, dummy_arrange,
drop_levels). Outputs match rwf, checked against R reference output in
tests/test_functions_recode.py.

Differences from R that cannot be avoided:
  - R lists become Python dicts (named list) or lists/tuples (unnamed list).
  - swap() on a pandas Categorical maps the categories; R's swap() fails on a
    factor ("adding class factor to an invalid object").
  - dummy_arrange() of a single response value: R drops the one-column data
    frame to a character vector; the port returns a numpy array of the same
    "1"/"0" strings.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd

try:
    from ._rcompat import r_label, r_levels, resolve_columns
except ImportError:
    from _rcompat import r_label, r_levels, resolve_columns
##########################################################################################
# FLATTEN LIST
##########################################################################################
def _as_data_frame(x):
    """R's data.frame(x, stringsAsFactors = FALSE) for one list element."""
    if isinstance(x, pd.DataFrame):
        return x.reset_index(drop=True)
    if isinstance(x, dict):
        if all(np.ndim(v) == 0 for v in x.values()):
            return pd.DataFrame({k: [v] for k, v in x.items()})
        return pd.DataFrame(x)
    if isinstance(x, pd.Series):
        return pd.DataFrame({"x" if x.name is None else x.name: x.to_numpy()})
    values = np.atleast_1d(np.asarray(x, dtype=object) if isinstance(x, list) else x)
    if values.ndim == 2:
        return pd.DataFrame(values, columns=[f"X{i + 1}" for i in range(values.shape[1])])
    return pd.DataFrame({"x": list(values)})


def flatten_list(mydata):
    """
    Flatten a two-dimensional list into a data frame (rwf's
    plyr::ldply(lapply(mydata, data.frame))): each element is turned into a
    data frame and the pieces are stacked row-wise, filling columns an element
    lacks with missing values.

    Parameters:
    mydata (dict or list): The elements, each coercible to a data frame (a
        DataFrame, a dict of columns, or a vector, which becomes a column
        named "x" as in R's data.frame(x)). A dict is R's named list, a list
        or tuple an unnamed one.

    Returns:
    pandas.DataFrame: All elements combined row-wise. For a dict, a leading
    ".id" column holds the key each row came from; an unnamed list gets no
    ".id" column, as in plyr::ldply.

    Examples:
    >>> flatten_list({'a': {'x': [1, 2], 'y': ['p', 'q']}, 'b': {'x': [3, 4]}})
    >>> flatten_list([{'x': [1, 2]}, {'x': [3, 4], 'z': 5}])
    >>> flatten_list({'a': [1, 2], 'b': [3, 4]})
    """
    named = isinstance(mydata, dict)
    items = mydata.items() if named else enumerate(mydata)
    frames = []
    for name, x in items:
        d = _as_data_frame(x)
        if named:
            d.insert(0, ".id", str(name))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
##########################################################################################
# SWAP
##########################################################################################
def swap(vector):
    """
    Reverse-score a vector: each value is mapped to its mirror among the
    observed levels (rwf: rev(levels(f))[f] with f = factor(vector)). The
    smallest observed value becomes the largest, the second smallest the
    second largest, and so on. Useful for reverse-scoring Likert items.

    Parameters:
    vector (array-like): Numeric or text vector to reverse-score. Missing
        values stay missing. Text is ordered as R's factor() orders it
        (case-insensitive, lower case first).

    Returns:
    numpy.ndarray: Same length as `vector`, values reverse-mapped; the dtype
    of `vector` is kept (integers with missing values become floats).

    Examples:
    >>> swap(list(range(1, 11)) + [1, 2, 3])
    >>> swap(['b', 'B', 'a', 'A', 'b'])
    """
    series = pd.Series(vector)
    levels = r_levels(series)
    mapping = dict(zip(levels, levels[::-1]))
    result = series.map(lambda v: mapping.get(v, np.nan) if pd.notna(v) else np.nan)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return pd.Categorical(result, categories=series.cat.categories).to_numpy()
    original = np.asarray(vector)
    if original.dtype.kind in "iub" and result.isna().any():
        return result.to_numpy(dtype=float)
    if original.dtype.kind in "iufb":
        return result.to_numpy(dtype=original.dtype)
    return result.to_numpy(dtype=object)
##########################################################################################
# DUMMY ARRANGE
##########################################################################################
def _r_sort_key(value):
    return (value.casefold(), value.swapcase())


def dummy_arrange(vector):
    """
    Dummy-code a multiple response vector into a binary data frame: each
    element is split on "," and every response value seen anywhere becomes a
    column marking whether that element contains it. Single-value responses
    are also accepted.

    As in rwf, pieces are not trimmed of spaces ("A, B" gives a column " B"),
    only of tabs and newlines; empty pieces, missing values and "Inf"/"-Inf"
    are not responses.

    Parameters:
    vector (array-like): Elements holding one or more comma-separated
        response values. Numbers are written as R writes them (2.0 -> "2").

    Returns:
    pandas.DataFrame: One row per element of `vector`, one column per unique
    response value, sorted as R's sort() orders text (case-insensitive,
    "10" before "2"). As in rwf the cells are the strings "1" (selected) and
    "0" (not selected). When there is only one response value, R drops the
    data frame to a vector: the port returns a numpy array of "1"/"0" too.

    Examples:
    >>> dummy_arrange(['Agree,Hi', 'All', 'Hi,All,Agree', 'Hi'])
    >>> dummy_arrange([1, 3, 2, 4, 1])
    >>> dummy_arrange(['2,10', '1', '10,3'])
    """
    rows = []
    for v in pd.Series(vector, dtype=object):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            rows.append([])
            continue
        pieces = [p.strip("\t\r\n") for p in r_label(v).split(",")]
        rows.append([p for p in pieces if p not in ("", "Inf", "-Inf")])
    values = sorted({p for row in rows for p in row}, key=_r_sort_key)
    result = pd.DataFrame({value: ["1" if value in row else "0" for row in rows] for value in values},
                          columns=values)
    if result.shape[1] == 1:
        return result.iloc[:, 0].to_numpy(dtype=object)
    return result
##########################################################################################
# DROP LEVELS
##########################################################################################
def drop_levels(df, factor_index=None, minimum_frequency=5):
    """
    Collapse rare factor levels into "Other" and drop unused levels: in each
    processed column every level whose frequency is at or below
    `minimum_frequency` (unused levels included) is renamed "Other", then the
    unused levels are dropped.

    Parameters:
    df (pandas.DataFrame): Data frame with one or more categorical columns.
    factor_index (int, str or list, optional): Columns to process, as 1-based
        positions (R's names(df)[factor_index]) or names. If None (default),
        every categorical column is processed. As in rwf, a column that is
        not categorical has no levels, so all of its values become missing
        (a categorical with no categories).
    minimum_frequency (int, optional): Levels with a frequency less than or
        equal to this value are collapsed into "Other". Defaults to 5.

    Returns:
    pandas.DataFrame: A copy of df with rare levels renamed "Other" (added
    after the existing levels) and all unused levels dropped.

    Examples:
    >>> import pandas as pd
    >>> factor1 = pd.Categorical(['A'] * 10 + ['B'] * 10, categories=['A', 'B', 'C', 'D'])
    >>> df = pd.DataFrame({'numeric1': range(1, 21), 'factor1': factor1, 'factor2': factor1})
    >>> drop_levels(df=df, minimum_frequency=9)
    >>> drop_levels(df=df, minimum_frequency=10)
    >>> drop_levels(df=df, factor_index=2, minimum_frequency=10)
    """
    df = df.copy()
    if factor_index is None:
        factor_columns = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    else:
        factor_columns = resolve_columns(df, factor_index)

    for col in factor_columns:
        series = df[col]
        counts = series.value_counts(sort=False)
        rare_levels = set(counts[counts <= minimum_frequency].index)
        levels = list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype) else []
        new_levels = list(dict.fromkeys(levels + ["Other"]))
        values = [v if v in new_levels else np.nan for v in series]
        values = ["Other" if (isinstance(v, str) or pd.notna(v)) and v in rare_levels else v for v in values]
        recoded = pd.Series(pd.Categorical(values, categories=new_levels), index=series.index)
        df[col] = recoded.cat.remove_unused_categories()
    return df
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nflatten_list\n", "=" * 80, sep="")
    print(flatten_list({'a': {'x': [1, 2], 'y': ['p', 'q']}, 'b': {'x': [3, 4]}}))
    print(flatten_list([{'x': [1, 2]}, {'x': [3, 4], 'z': 5}]))

    print("\n" + "=" * 80, "\nswap\n", "=" * 80, sep="")
    print(swap(list(range(1, 11)) + [1, 2, 3]))

    print("\n" + "=" * 80, "\ndummy_arrange\n", "=" * 80, sep="")
    print(dummy_arrange(['Agree,Hi', 'All', 'Hi,All,Agree', 'Hi']))
    print(dummy_arrange([1, 3, 2, 4, 1]))

    print("\n" + "=" * 80, "\ndrop_levels\n", "=" * 80, sep="")
    factor1 = pd.Categorical(['A'] * 10 + ['B'] * 10, categories=['A', 'B', 'C', 'D'])
    df = pd.DataFrame({'numeric1': range(1, 21), 'factor1': factor1, 'factor2': factor1})
    print(drop_levels(df=df, minimum_frequency=9)['factor1'].cat.categories)
    print(drop_levels(df=df, minimum_frequency=10)['factor1'].cat.categories)
