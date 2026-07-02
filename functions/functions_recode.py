# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_RECODE.R.

Deviations from the R originals, by design:
  - dummy_arrange's R implementation dynamically creates new data frame
    columns inside a loop via `mydata[r, value] <- value` (valid R
    data-frame assignment behavior: naming a not-yet-existing column
    creates it). This is reimplemented as a straightforward vectorized
    multi-hot encoding (one column per unique response value across all
    split pieces), which produces the identical result without the
    loop-driven column-growth mechanism.
  - drop_levels' `factor_index` is 0-based here, not R's 1-based column
    indices, matching normal Python indexing conventions for a
    caller-supplied argument (unlike a value *returned* for
    cross-referencing, where this project has kept 1-based indexing
    elsewhere for R parity).

Reuses change_data_type/remove_nc from functions.py rather than
duplicating them, since dummy_arrange needs their exact behavior.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd

try:
    from .functions import change_data_type, remove_nc
except ImportError:
    from functions import change_data_type, remove_nc
##########################################################################################
# FLATTEN LIST
##########################################################################################
def flatten_list(mydata):
    """
    Flatten a list (or dict) of table-like objects into a single data
    frame, tagging each row with which top-level element it came from.

    Parameters:
    mydata (dict or list): Each value/element must be coercible to a
        pandas.DataFrame.

    Returns:
    pandas.DataFrame: All elements combined row-wise, with a leading
    ".id" column holding the dict key (or 1-based position if a list)
    each row came from.

    Examples:
    >>> flatten_list({'a': {'x': [1, 2]}, 'b': {'x': [3, 4]}})
    >>> flatten_list([{'x': [1, 2]}, {'x': [3, 4]}])
    """
    items = mydata.items() if isinstance(mydata, dict) else enumerate(mydata, start=1)
    frames = []
    for name, x in items:
        d = pd.DataFrame(x)
        d.insert(0, '.id', name)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
##########################################################################################
# SWAP
##########################################################################################
def swap(vector):
    """
    Reverse-score a vector by mirroring each value across its observed
    range: the smallest observed value maps to the largest, the second
    smallest to the second largest, and so on — based on the *unique*
    observed values, not an assumed fixed scale. Useful for
    reverse-scoring Likert items.

    Parameters:
    vector (array-like): Vector to reverse-score.

    Returns:
    numpy.ndarray: Same length and dtype as `vector`, values reverse-mapped.

    Examples:
    >>> swap(list(range(1, 11)) + [1, 2, 3])
    """
    vector = np.asarray(vector)
    levels = np.unique(vector)
    mapping = dict(zip(levels, levels[::-1]))
    return np.array([mapping[v] for v in vector], dtype=vector.dtype)
##########################################################################################
# DUMMY ARRANGE
##########################################################################################
def dummy_arrange(vector):
    """
    Dummy-code a multiple-response vector into a binary (multi-hot) data
    frame: splits each element on "," and, for every unique response
    value seen anywhere, creates a 0/1 column indicating whether that
    row contained it. Single-value (no comma) responses are also
    accepted.

    Parameters:
    vector (array-like of str): Each element holds one or more
        comma-separated response values.

    Returns:
    pandas.DataFrame: One row per element of `vector`, one column per
    unique response value (sorted alphabetically), values 1 (selected)
    or 0 (not selected).

    Examples:
    >>> import numpy as np
    >>> vector3 = np.random.choice([1, 2, 3, 4], 10)
    >>> dummy_arrange(vector3)
    >>> vector4 = np.random.choice(list("ABC"), 10)
    >>> dummy_arrange(vector4)
    """
    series = pd.Series(vector).astype(str).str.strip()
    split = series.str.split(",", expand=True)
    split = split.apply(lambda col: col.str.strip())
    split = remove_nc(split, value=np.nan)
    split = change_data_type(split, type="character")

    values = pd.unique(split.to_numpy().ravel())
    unique_values = sorted(v for v in values if pd.notna(v) and v != 'nan')

    result = pd.DataFrame({
        value: split.eq(value).any(axis=1).astype(int)
        for value in unique_values
    }, index=split.index)
    return result[sorted(result.columns)]
##########################################################################################
# DROP LEVELS
##########################################################################################
def drop_levels(df, factor_index=None, minimum_frequency=5):
    """
    Collapse rare (and unused) categories into "Other" for the
    categorical ("factor") columns of a data frame, then drop any
    now-unused categories.

    Parameters:
    df (pandas.DataFrame): Data frame with one or more category-dtype columns.
    factor_index (list of int, optional): 0-based column indices to
        process. If None (default), every category-dtype column is
        processed. Note: 0-based, unlike R's 1-based column indices.
    minimum_frequency (int, optional): Categories with a count at or
        below this value (including categories with zero occurrences)
        are renamed to "Other". Defaults to 5.

    Returns:
    pandas.DataFrame: Same structure as df, with rare/unused categories
    renamed to "Other" and all now-unused categories dropped.

    Examples:
    >>> import pandas as pd
    >>> factor1 = pd.Categorical(['A']*10 + ['B']*10, categories=['A','B','C','D'])
    >>> df = pd.DataFrame({'numeric1': range(20), 'factor1': factor1})
    >>> drop_levels(df=df, minimum_frequency=9)
    >>> drop_levels(df=df, minimum_frequency=10)
    """
    df = df.copy()
    if factor_index is None:
        factor_columns = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    else:
        factor_columns = [df.columns[i] for i in factor_index]

    for col in factor_columns:
        series = df[col]
        levels = list(series.cat.categories)
        counts = series.value_counts().reindex(levels, fill_value=0)
        rare_levels = counts[counts <= minimum_frequency].index.tolist()

        new_levels = levels + (["Other"] if "Other" not in levels else [])
        series = series.cat.set_categories(new_levels)
        series[series.isin(rare_levels)] = "Other"
        series = series.cat.remove_unused_categories()
        df[col] = series
    return df
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80, "\nflatten_list\n", "=" * 80, sep="")
    print(flatten_list({'a': {'x': [1, 2]}, 'b': {'x': [3, 4]}}))

    print("\n" + "=" * 80, "\nswap\n", "=" * 80, sep="")
    print(swap(list(range(1, 11)) + [1, 2, 3]))

    print("\n" + "=" * 80, "\ndummy_arrange\n", "=" * 80, sep="")
    vector1 = [",".join(np.random.choice(["Agree", "Hi", "All"], np.random.randint(1, 4), replace=False)) for _ in range(10)]
    vector2 = [",".join(str(v) for v in np.random.choice([1, 2, 3, 4], np.random.randint(1, 5), replace=False)) for _ in range(10)]
    vector3 = np.random.choice([1, 2, 3, 4], 10)
    vector4 = np.random.choice(list("ABC"), 10)
    print("vector1:", vector1)
    print(dummy_arrange(vector1))
    print()
    print("vector2:", vector2)
    print(dummy_arrange(vector2))
    print()
    print(dummy_arrange(vector3))
    print()
    print(dummy_arrange(vector4))

    print("\n" + "=" * 80, "\ndrop_levels\n", "=" * 80, sep="")
    factor1 = pd.Categorical(['A'] * 10 + ['B'] * 10, categories=['A', 'B', 'C', 'D'])
    factor2 = pd.Categorical(['A'] * 10 + ['B'] * 10, categories=['A', 'B', 'C', 'D'])
    numeric1 = list(range(20))
    df = pd.DataFrame({'numeric1': numeric1, 'factor1': factor1, 'factor2': factor2})
    print(df['factor1'].value_counts())
    print(drop_levels(df=df, minimum_frequency=9)['factor1'].value_counts())
    print(drop_levels(df=df, minimum_frequency=10)['factor1'].value_counts())
