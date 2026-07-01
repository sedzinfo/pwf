# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS.R (generic dataframe/vector utilities).

Deviations from the R originals, by design:
  - c_bind takes name=data keyword arguments instead of unevaluated
    expressions (R's `substitute()`-based automatic naming from the call
    site has no Python equivalent) — dotnames() is folded into this and
    dropped as a separate function.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
from itertools import combinations as _combinations
import numpy as np
import pandas as pd
##########################################################################################
# ROUND DATAFRAME
##########################################################################################
def round_dataframe(df, digits=0, type="round"):
    """
    Round or transform every numeric column in a data frame, leaving
    non-numeric columns unchanged.

    Parameters:
    df (pandas.DataFrame): Data frame with a mix of numeric and
        non-numeric columns.
    digits (int, optional): Decimal places, used only with type="round"
        or type="tenth". Defaults to 0.
    type (str, optional): One of:
        - "round": round to `digits` decimal places (default).
        - "ceiling": round up to the nearest integer.
        - "floor": round down to the nearest integer.
        - "tenth": divide by 10, then round to `digits` decimal places.

    Returns:
    pandas.DataFrame: Same structure as df with numeric columns
    transformed according to `type`.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1.234, 2.567], 'label': ['a', 'b']})
    >>> round_dataframe(df, digits=1)
    >>> round_dataframe(df, type="ceiling")
    >>> round_dataframe(df * 10 if False else df, type="floor")
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    if type == "round":
        df[numeric_cols] = df[numeric_cols].round(digits)
    elif type == "ceiling":
        df[numeric_cols] = np.ceil(df[numeric_cols])
    elif type == "floor":
        df[numeric_cols] = np.floor(df[numeric_cols])
    elif type == "tenth":
        df[numeric_cols] = (df[numeric_cols] / 10).round(digits)
    return df
##########################################################################################
# CHANGE DATA TYPE OF COLUMNS IN DATA FRAME
##########################################################################################
def change_data_type(df, type):
    """
    Convert all columns in a data frame to a specified data type.
    Whitespace (tabs, carriage returns, newlines — not plain spaces, to
    match R's `whitespace="[\\t\\r\\n]"`) is trimmed when converting to
    "character" or "numeric".

    Parameters:
    df (pandas.DataFrame): Data frame whose columns will be converted.
    type (str): One of:
        - "character": convert all columns to string, trimming
          leading/trailing tab/CR/newline.
        - "numeric": convert all columns to numeric (via string with the
          same trimming); non-numeric strings become NaN.
        - "factor": convert all columns to pandas "category" dtype.
        - "factor_character": convert only category-dtype columns to
          string; other columns unchanged.
        - "character_factor": convert only object/string columns to
          "category"; other columns unchanged.

    Returns:
    pandas.DataFrame: Same shape as df with column types converted.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
    >>> change_data_type(df, "character")
    >>> change_data_type(df, "factor")
    >>> change_data_type(change_data_type(df, "factor"), "factor_character")
    """
    df = df.copy()
    if type == "character":
        df = df.apply(lambda col: col.astype(str).str.strip('\t\r\n'))
    elif type == "numeric":
        df = df.apply(lambda col: pd.to_numeric(col.astype(str).str.strip('\t\r\n'), errors='coerce'))
    elif type == "factor":
        df = df.apply(lambda col: col.astype('category'))
    elif type == "factor_character":
        df = df.apply(lambda col: col.astype(str) if col.dtype.name == 'category' else col)
    elif type == "character_factor":
        df = df.apply(lambda col: col.astype('category') if col.dtype == object else col)
    return df
##########################################################################################
# RBIND ALL
##########################################################################################
def rbind_all(df1, df2):
    """
    Row-bind two data frames that don't share the same columns. Columns
    present in one input but absent in the other are added and filled
    with NaN before binding.

    Parameters:
    df1 (pandas.DataFrame)
    df2 (pandas.DataFrame)

    Returns:
    pandas.DataFrame: All rows from df1 followed by all rows from df2,
    with the union of both column sets (NaN where a column didn't exist
    in the original input). Original index values are preserved unless
    they would produce duplicates, in which case a default integer
    index is used.

    Examples:
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
    >>> rbind_all(df1, df2)
    """
    df1 = df1.copy()
    df2 = df2.copy()
    all_cols = list(df1.columns) + [c for c in df2.columns if c not in df1.columns]
    for c in df2.columns:
        if c not in df1.columns:
            df1[c] = np.nan
    for c in df1.columns:
        if c not in df2.columns:
            df2[c] = np.nan
    df1 = df1[all_cols]
    df2 = df2[all_cols]

    combined_index = list(df1.index) + list(df2.index)
    result = pd.concat([df1, df2], axis=0)
    if len(set(combined_index)) == len(combined_index):
        result.index = combined_index
    else:
        result = result.reset_index(drop=True)
    return result
##########################################################################################
# REMOVE VALUES THAT CANNOT BE CALCULATED
##########################################################################################
def remove_nc(df, value=np.nan, remove_rows=False, aggressive=False, remove_cols=False, remove_zero_variance=False):
    """
    Clean a data frame by replacing non-computable values (NaN, +-Inf,
    and empty strings) with a chosen replacement, then optionally drop
    rows or columns that still contain missing values or have zero
    variance.

    Parameters:
    df (pandas.DataFrame): Data frame to clean.
    value: Replacement value for all non-computable entries. Defaults to NaN.
    remove_rows (bool, optional): If True, remove rows containing NaN
        after replacement, per `aggressive`. Defaults to False.
    aggressive (bool, optional): Only used when remove_rows=True.
        True removes a row if *any* value is NaN; False removes a row
        only if *all* values are NaN. Defaults to False.
    remove_cols (bool, optional): If True, drop columns where *all*
        values are NaN. Defaults to False.
    remove_zero_variance (bool, optional): Only used when
        remove_cols=True. If True, also drop columns with only one
        unique non-missing value. Defaults to False.

    Returns:
    pandas.DataFrame: Cleaned data frame.

    Examples:
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'x': [1.0, np.inf, np.nan], 'y': [1, 1, 1]})
    >>> remove_nc(df, value=np.nan)
    >>> remove_nc(df, remove_cols=True, remove_zero_variance=True)
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf, ""], value)
    df = df.fillna(value)

    if remove_rows:
        if aggressive:
            df = df[~df.isna().any(axis=1)]
        else:
            df = df[df.notna().any(axis=1)]

    if remove_cols:
        df = df.loc[:, df.notna().any(axis=0)]
        if remove_zero_variance:
            df = df.loc[:, df.nunique(dropna=True) > 1]

    return df
##########################################################################################
# REPLACE NA WITH PREVIOUS CELLS
##########################################################################################
def replace_na_with_previous(vector):
    """
    Last observation carried forward (LOCF) imputation: replaces each
    NaN with the most recent preceding non-NaN value. If the first
    element is NaN, it's replaced with the first non-NaN value found
    anywhere in the vector.

    Parameters:
    vector (array-like or pandas.Series): Values that may contain NaN.

    Returns:
    Same type as `vector` (a pandas.Series if given one, else a numpy
    array), with NaN values replaced by the preceding non-NaN element.

    Examples:
    >>> import numpy as np
    >>> replace_na_with_previous([np.nan, 1.0, np.nan, 3.0, np.nan])
    """
    result = pd.Series(vector).copy()
    if len(result) > 0 and pd.isna(result.iloc[0]):
        non_na = result.dropna()
        if len(non_na) > 0:
            result.iloc[0] = non_na.iloc[0]
    result = result.ffill()
    return result if isinstance(vector, pd.Series) else result.to_numpy()
##########################################################################################
# PAD DATA FRAME WITH NA ROWS
##########################################################################################
def padNA(df, rowsneeded, first=True):
    """
    Pad a data frame to `rowsneeded` rows by appending (or prepending)
    NaN-filled rows. Internal helper used by c_bind.

    Parameters:
    df (pandas.DataFrame): Data frame to pad.
    rowsneeded (int): Target row count; must be >= len(df).
    first (bool, optional): If True (default), NaN rows are appended at
        the bottom; if False, prepended at the top.

    Returns:
    pandas.DataFrame: `rowsneeded` rows, same columns as df.

    Examples:
    >>> import pandas as pd
    >>> padNA(pd.DataFrame({'x': [1, 2]}), rowsneeded=5)
    """
    df = pd.DataFrame(df)
    n_pad = rowsneeded - len(df)
    if n_pad < 0:
        raise ValueError("rowsneeded must be >= the number of existing rows")
    pad = pd.DataFrame(np.nan, index=range(n_pad), columns=df.columns)
    frames = [df, pad] if first else [pad, df]
    return pd.concat(frames, ignore_index=True)
##########################################################################################
# COLUMN-BIND OF UNEQUAL LENGTHS
##########################################################################################
def c_bind(first=True, **kwargs):
    """
    Column-bind any number of data frames or vectors side by side,
    padding shorter inputs with NaN rows so all columns reach the same
    length. Each input's columns are prefixed with its keyword name to
    avoid duplicate column names.

    Unlike R's c_bind, names come from explicit name=data keyword
    arguments rather than R's substitute()-based automatic naming of
    unevaluated expressions (no Python equivalent) — dotnames() is
    folded into this and dropped as a separate function.

    Parameters:
    **kwargs: name=data pairs. Each value is a pandas.DataFrame,
        pandas.Series, or array-like.
    first (bool, optional): If True (default), NaN padding rows are
        appended at the bottom of shorter inputs; if False, prepended.

    Returns:
    pandas.DataFrame: One column per column across all inputs, padded
    with NaN rows to the length of the longest input. Column names
    follow "<name>" for single-column inputs and "<name>_<original>"
    for multi-column inputs.

    Examples:
    >>> import numpy as np
    >>> c_bind(a=np.random.normal(size=10), b=np.random.normal(size=13))
    """
    if not kwargs:
        raise ValueError("c_bind requires at least one name=data keyword argument")

    frames = {}
    for name, obj in kwargs.items():
        if isinstance(obj, pd.DataFrame):
            d = obj.copy()
            d.columns = [f"{name}_{c}" for c in d.columns]
        elif isinstance(obj, pd.Series):
            d = pd.DataFrame({name: obj.to_numpy()})
        else:
            d = pd.DataFrame({name: np.asarray(obj)})
        frames[name] = d

    nrows = max(len(d) for d in frames.values())
    padded = [padNA(d, nrows, first=first) for d in frames.values()]
    return pd.concat(padded, axis=1)
##########################################################################################
# COMBINATIONS
##########################################################################################
def comparison_combinations(df, all_orders=True):
    """
    Generate a data frame of all pairwise combinations of column names.

    Parameters:
    df (pandas.DataFrame): Data frame whose column names will be combined.
    all_orders (bool, optional): If True (default), both orderings of
        each pair are included (n*(n-1) rows for n columns); if False,
        only unique unordered pairs (n*(n-1)/2 rows).

    Returns:
    pandas.DataFrame: Two string columns "X1"/"X2", one row per pair.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=['X1', 'X2', 'X3', 'X4'])
    >>> comparison_combinations(df)
    >>> comparison_combinations(df, all_orders=False)
    """
    cols = list(df.columns)
    pairs = list(_combinations(cols, 2))
    result = pd.DataFrame(pairs, columns=['X1', 'X2'])
    if all_orders:
        reversed_pairs = pd.DataFrame({'X1': result['X2'], 'X2': result['X1']})
        result = pd.concat([result, reversed_pairs], ignore_index=True)
        result = result.sort_values(['X1', 'X2']).reset_index(drop=True)
    return result
##########################################################################################
# MINIMUM MAXIMUM INDEX OF A VECTOR
##########################################################################################
def min_max_index(vector):
    """
    Positions of the minimum and maximum values in a vector. Ties return
    all tied positions.

    Parameters:
    vector (array-like): Numeric vector.

    Returns:
    dict: {'max_index': 1-based positions of the max value,
           'min_index': 1-based positions of the min value} (both numpy
    arrays), matching R's 1-based `which()` indexing.

    Examples:
    >>> min_max_index([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> min_max_index([1, 6, 3, 4, 6, 4, 3, 2, 1])
    """
    arr = np.asarray(vector)
    max_val, min_val = arr.max(), arr.min()
    return {
        'max_index': np.flatnonzero(arr == max_val) + 1,
        'min_index': np.flatnonzero(arr == min_val) + 1,
    }
##########################################################################################
# GET SCRIPT DIRECTORY
##########################################################################################
def get_script_directory():
    """
    Return the directory of the currently running script, with a
    trailing slash. Falls back to the current working directory when
    there's no running script file (e.g. an interactive REPL) — the
    Python equivalent of R's RStudio/Rscript/getwd() fallback chain,
    minus the RStudio-specific branch (no equivalent in Python).

    Returns:
    str: Directory path ending in "/".

    Examples:
    >>> get_script_directory()
    """
    main_module = sys.modules.get('__main__')
    if main_module is not None and hasattr(main_module, '__file__'):
        return os.path.dirname(os.path.abspath(main_module.__file__)) + '/'
    return os.getcwd() + '/'
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80, "\nround_dataframe\n", "=" * 80, sep="")
    df_mixed = pd.DataFrame({'mpg': [21.234, 22.876, 18.5], 'cyl': ['4', '6', '8']})
    print(round_dataframe(df_mixed, digits=1))
    print()
    print(round_dataframe(df_mixed, type="ceiling"))
    print()
    print(round_dataframe(df_mixed * 10 if False else df_mixed, digits=2, type="tenth"))

    print("\n" + "=" * 80, "\nchange_data_type\n", "=" * 80, sep="")
    df_types = pd.DataFrame({'mpg': [21.0, 22.8], 'cyl': [4, 6]})
    print(change_data_type(df_types, "character").dtypes)
    df_factor = change_data_type(df_types, "factor")
    print(df_factor.dtypes)
    print(change_data_type(df_factor, "factor_character").dtypes)

    print("\n" + "=" * 80, "\nrbind_all\n", "=" * 80, sep="")
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[10, 11])
    df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]}, index=[20, 21])
    print("non-overlapping index (preserved):")
    print(rbind_all(df1, df2))
    print()
    df1_dup = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2_dup = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
    print("overlapping index (falls back to default integer index):")
    print(rbind_all(df1_dup, df2_dup))

    print("\n" + "=" * 80, "\nremove_nc\n", "=" * 80, sep="")
    df_nc = pd.DataFrame({'x': [1.0, np.inf, np.nan, -np.inf, 5.0], 'y': [1, 1, 1, 1, 1]})
    print(remove_nc(df_nc, value=np.nan))
    print()
    print(remove_nc(df_nc, remove_rows=True, aggressive=True))
    print()
    print(remove_nc(df_nc, remove_cols=True, remove_zero_variance=True))

    print("\n" + "=" * 80, "\nreplace_na_with_previous\n", "=" * 80, sep="")
    print(replace_na_with_previous([np.nan, 1.0, np.nan, 3.0, np.nan]))

    print("\n" + "=" * 80, "\npadNA\n", "=" * 80, sep="")
    print(padNA(pd.DataFrame({'x': [1, 2]}), rowsneeded=5))

    print("\n" + "=" * 80, "\nc_bind\n", "=" * 80, sep="")
    print(c_bind(a=np.random.normal(size=5), b=np.random.normal(size=8)))

    print("\n" + "=" * 80, "\ncomparison_combinations\n", "=" * 80, sep="")
    df_combos = pd.DataFrame(columns=['X1', 'X2', 'X3', 'X4'])
    print(comparison_combinations(df_combos))
    print()
    print(comparison_combinations(df_combos, all_orders=False))

    print("\n" + "=" * 80, "\nmin_max_index\n", "=" * 80, sep="")
    print(min_max_index([1, 2, 3, 4, 5, 4, 3, 2, 1]))
    print(min_max_index([1, 6, 3, 4, 6, 4, 3, 2, 1]))

    print("\n" + "=" * 80, "\nget_script_directory\n", "=" * 80, sep="")
    print(get_script_directory())
