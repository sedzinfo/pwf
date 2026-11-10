# -*- coding: utf-8 -*-
"""
Quick data-quality summaries for a DataFrame: null/NaN/empty-string
counts, per-column descriptives, and dtype checks.

Fixed a real bug in cdf(): its Excel-export branch referenced an
undefined function (critical_value_excel) and an undefined variable
(dataframes), and called the removed pandas ExcelWriter.save() method --
it would crash immediately whenever a filename was passed. Replaced with
plain DataFrame.to_excel() calls and writer.close().
"""
##########################################################################################
# LOAD
##########################################################################################
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
try:
    from ._rcompat import r_mixedsort_key as _r_mixedsort_key, r_number as _r_number
    from .functions_excel import excel_critical_value
except ImportError:
    from _rcompat import r_mixedsort_key as _r_mixedsort_key, r_number as _r_number
    from functions_excel import excel_critical_value
##########################################################################################
# OBSERVED
##########################################################################################
def observed(vector):
    """
    Count of non-NaN values in a vector.

    Parameters:
    vector (array-like): Numeric values.

    Returns:
    int

    Examples:
    >>> import numpy as np
    >>> observed(np.array([1, 2, np.nan, 4]))
    """
    return np.count_nonzero(~np.isnan(vector))
##########################################################################################
# CHECK INTEGER
##########################################################################################
def is_int(vector):
    """
    True if `vector` is a Python int.

    Parameters:
    vector: Any value.

    Returns:
    bool

    Examples:
    >>> is_int(5)
    >>> is_int(5.0)
    """
    if type(vector) == int:
        return True


def check_integer(vector):
    """
    True if every element of `vector` is a Python int.

    Parameters:
    vector (iterable): Values to check.

    Returns:
    bool

    Examples:
    >>> check_integer([1, 2, 3])
    >>> check_integer([1, 2.5, 3])
    """
    for i in vector:
        if not is_int(i):
            return False
    return True
##########################################################################################
# SHORT CHECK DATAFRAME
##########################################################################################
def short_check(df):
    """
    Performs a quick summary check of the given DataFrame (`df`) to provide basic statistics including the number of rows,
    columns, empty strings, null values, and non-null values.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    Returns:
    -------
    result : pandas.DataFrame
        A DataFrame containing the following statistics for the provided DataFrame:
        - COLUMNS: The number of columns in the DataFrame.
        - ROWS: The number of rows in the DataFrame.
        - EMPTY_STRINGS: The total count of empty string values in the DataFrame.
        - NULLS: The total count of null (NaN) values in the DataFrame.
        - NOT_NULLS: The total count of non-null (non-NaN) values in the DataFrame.

    Notes:
    ------
    - The function calculates the number of rows and columns using the shape of the DataFrame.
    - It counts null values using `isnull()`, empty strings using the comparison (`df == ""`), 
      and non-null values using `count()`.
    """
    dimensions=np.array(df.shape)
    rows=dimensions[0]
    collumns=dimensions[1]
    nulls=df.isnull().sum().sum()
    not_nulls=df.count().sum()
    empty_strings=(df=="").sum().sum()
    result=pd.DataFrame({'COLUMNS':[collumns],
                         'ROWS':[rows],
                         'EMPTY_STRINGS':[empty_strings],
                         'NULLS':[nulls],
                         'NOT_NULLS':[not_nulls]})
    return result
##########################################################################################
# CHECK DATAFRAME (OPTIMISED)
##########################################################################################
def _r_column_kind(s):
    """Classify a pandas column as the R vector type it corresponds to."""
    dtype = s.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "ordered" if dtype.ordered else "factor"
    if pd.api.types.is_bool_dtype(dtype):
        return "logical"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "POSIXct"
    observed_values = s.dropna()
    if len(observed_values) and all(isinstance(v, datetime.date) and not isinstance(v, datetime.datetime)
                                    for v in observed_values):
        return "Date"
    if len(observed_values) and all(isinstance(v, (bool, np.bool_)) for v in observed_values):
        return "logical"
    return "character"


_R_TYPES = {  # kind -> (MODE, TYPE, CLASS)
    "factor": ("numeric", "integer", "factor"),
    "ordered": ("numeric", "integer", "ordered, factor"),
    "logical": ("logical", "logical", "logical"),
    "integer": ("numeric", "integer", "integer"),
    "double": ("numeric", "double", "numeric"),
    "POSIXct": ("numeric", "double", "POSIXct, POSIXt"),
    "Date": ("numeric", "double", "Date"),
    "character": ("character", "character", "character"),
}


def _r_text_sort_key(text):
    """Approximates R's locale collation: case-insensitive, lower case first."""
    return (text.casefold(), text.swapcase())


def _r_value_text(value, kind):
    if kind == "POSIXct":
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if kind == "Date":
        return value.isoformat()
    if kind in ("integer", "double", "logical"):
        return _r_number(value)
    return str(value)


@np.errstate(invalid="ignore")  # mean/sd of Inf and -Inf is NaN, as in R
def _cdff_column(s, digits, nuniques):
    kind = _r_column_kind(s)
    is_na = s.isna()
    observed_values = s[~is_na]
    has_data = len(observed_values) > 0
    is_num = kind in ("integer", "double")
    numbers = observed_values.to_numpy(dtype=float) if is_num else None

    nan_count = 0
    if kind == "double" and isinstance(s.dtype, pd.Float64Dtype):  # nullable floats keep NaN apart from NA
        array = s.array
        nan_count = int(np.isnan(array._data[~array._mask]).sum())

    col_min = col_max = np.nan
    if has_data:
        if is_num:
            col_min, col_max = _r_number(observed_values.min()), _r_number(observed_values.max())
        elif kind == "ordered":
            present = [c for c in s.cat.categories if c in set(observed_values)]
            col_min, col_max = str(present[0]), str(present[-1])
        elif kind in ("Date", "POSIXct"):
            col_min = _r_value_text(observed_values.min(), kind)
            col_max = _r_value_text(observed_values.max(), kind)

    if kind in ("integer", "double"):
        finite = int(np.isfinite(numbers).sum())
    elif kind == "character":
        finite = 0
    else:
        finite = len(observed_values)

    def rounded(value):
        return round(float(value), digits) if pd.notna(value) else np.nan

    mode, r_type, r_class = _R_TYPES[kind]
    result = {
        "EMPTY": int((observed_values.astype(str) == "").sum()) if kind in ("character", "factor", "ordered") else 0,
        "null": 0,
        "na": int(is_na.sum()),
        "NOT_NA": len(observed_values),
        "NAN": nan_count,
        "INF": int(np.isinf(numbers).sum()) if is_num else 0,
        "FIN": finite,
        "RANGE": int(s.nunique(dropna=False)),
        "MEAN": rounded(np.mean(numbers)) if is_num and has_data else np.nan,
        "MEDIAN": rounded(np.median(numbers)) if is_num and has_data else np.nan,
        "SD": rounded(np.std(numbers, ddof=1)) if is_num and len(numbers) > 1 else np.nan,
        "MIN": col_min,
        "MAX": col_max,
        "MODE": mode,
        "TYPE": r_type,
        "CLASS": r_class,
        "FACTOR": kind in ("factor", "ordered"),
    }
    if nuniques > 0:
        n_unique = result["RANGE"]
        if n_unique > nuniques:
            result["UNIQUES"] = f"{n_unique} Uniques"
        else:
            values = observed_values.unique()
            if kind in ("factor", "ordered"):
                values = [c for c in s.cat.categories if c in set(values)]
            elif kind == "character":
                values = sorted(map(str, values), key=_r_text_sort_key)
            else:
                values = sorted(values)
            result["UNIQUES"] = ", ".join(_r_value_text(v, kind) for v in values)
        levels = list(s.cat.categories) if kind in ("factor", "ordered") else []
        result["LEVELS"] = (f"{len(levels)} Levels" if len(levels) > nuniques
                            else ", ".join(str(level) for level in levels))
    return result


def cdff(df, name_length=None, digits=2, nuniques=0, parralel=False, file=None):
    """
    Column-by-column diagnostic summary of a data frame: missing, empty,
    NaN, infinite and finite counts, number of distinct values, mean,
    median, sd, min, max and R type (port of rwf::cdff, the fast version of
    rwf::cdf).

    Columns are described with R's types: integer, double, logical,
    character, factor / ordered factor (pandas categorical), Date
    (datetime.date values) and POSIXct (datetime64). In a plain numpy float
    column a NaN is pandas' missing value, so it counts under na, not NAN;
    NAN is only filled for nullable Float64 columns, which keep the two
    apart, as R does.

    Parameters:
    df (pandas.DataFrame): Data to inspect.
    name_length (int, optional): Maximum characters kept of NAMES, MIN and
        MAX. Defaults to a third of pandas' display width (R: a third of
        getOption("width")).
    digits (int, optional): Decimals for MEAN, MEDIAN and SD. Defaults to 2.
    nuniques (int, optional): If > 0, adds UNIQUES (the distinct values, or
        "N Uniques" when there are more than nuniques) and, when any column is
        a factor, LEVELS. Defaults to 0.
    parralel (bool, optional): Summarise columns in parallel threads.
        Defaults to False. (Spelling as in rwf.)
    file (str, optional): Excel filename without extension; writes sheets
        "variables" and "summary". Defaults to None.

    Returns:
    dict: {"summary": one-row DataFrame with COLLUMNS, ROWS, TOTAL, EMPTY,
    null, NAN, na, INF, FIN, FACTOR; "check": one row per column with NAMES,
    EMPTY, null, na, NOT_NA, NAN, INF, FIN, RANGE, MEAN, MEDIAN, SD, MIN,
    MAX, MODE, TYPE, CLASS, FACTOR and optionally UNIQUES, LEVELS}.

    Examples:
    >>> cdff(df_titanic)
    >>> cdff(df_titanic, nuniques=3)
    """
    return _check_dataframe(df, _cdff_column, name_length, digits, nuniques, parralel, file)


def _check_dataframe(df, column_fn, name_length, digits, nuniques, parralel, file):
    """The common body of cdf and cdff: one row per column from column_fn, the summary row and the file."""
    if name_length is None:
        name_length = pd.get_option("display.width") / 3
    name_length = int(name_length)
    columns = [df.iloc[:, i] for i in range(df.shape[1])]
    if parralel:
        with ThreadPoolExecutor() as pool:
            rows = list(pool.map(lambda s: column_fn(s, digits, nuniques), columns))
    else:
        rows = [column_fn(s, digits, nuniques) for s in columns]
    check = pd.DataFrame(rows)
    check.insert(0, "NAMES", [str(c) for c in df.columns])
    if nuniques > 0 and len(check) and (check["LEVELS"] == "").all():
        check = check.drop(columns="LEVELS")

    summary = pd.DataFrame([{
        "COLLUMNS": df.shape[1],
        "ROWS": df.shape[0],
        "TOTAL": df.shape[1] * df.shape[0],
        "EMPTY": int(check["EMPTY"].sum()) if len(check) else 0,
        "null": int(check["null"].sum()) if len(check) else 0,
        "NAN": int(check["NAN"].sum()) if len(check) else 0,
        "na": int(check["na"].sum()) if len(check) else 0,
        "INF": int(check["INF"].sum()) if len(check) else 0,
        "FIN": int(check["FIN"].sum()) if len(check) else 0,
        "FACTOR": int(check["FACTOR"].sum()) if len(check) else 0,
    }])

    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(check, writer, sheet="variables")
            excel_critical_value(summary, writer, sheet="summary")

    for col in ["NAMES", "MIN", "MAX"]:
        if len(check):
            check[col] = check[col].map(lambda v: v[:name_length] if isinstance(v, str) else np.nan)
    return {"summary": summary, "check": check}
##########################################################################################
# CHECK DATAFRAME (the original rwf::cdf)
##########################################################################################
_R_EPOCH = datetime.date(1970, 1, 1)


def _cdf_column(s, digits, nuniques):
    """
    rwf::cdf for one column: cdff's statistics except MIN and MAX, which are the smallest and
    largest value of a double column (Inf and -Inf when it has none, R's min/max of nothing; a
    Date or date-time as R's number of days or seconds) and otherwise the first and last of the
    distinct values' text labels in gtools::mixedsort order.
    """
    result = _cdff_column(s, digits, nuniques)
    kind = _r_column_kind(s)
    observed_values = s.dropna()
    if kind == "double":
        result["MIN"] = _r_number(observed_values.min()) if len(observed_values) else "Inf"
        result["MAX"] = _r_number(observed_values.max()) if len(observed_values) else "-Inf"
    elif kind in ("Date", "POSIXct"):
        if kind == "Date":
            numbers = [(v - _R_EPOCH).days for v in observed_values]
        else:
            numbers = [pd.Timestamp(v).timestamp() for v in observed_values]
        result["MIN"] = _r_number(min(numbers)) if numbers else "Inf"
        result["MAX"] = _r_number(max(numbers)) if numbers else "-Inf"
    else:
        if kind in ("factor", "ordered", "character"):
            labels = [str(v) for v in observed_values.unique()]
        else:
            labels = [_r_value_text(v, kind) for v in observed_values.unique()]
        labels = sorted(set(labels), key=_r_mixedsort_key)
        result["MIN"] = labels[0] if labels else np.nan
        result["MAX"] = labels[-1] if labels else np.nan
    return result


def cdf(df, name_length=None, digits=2, nuniques=0, parralel=False, file=None):
    """
    Column-by-column diagnostic summary of a data frame, as rwf::cdf (the
    original version of cdff): missing, empty, NaN, infinite and finite
    counts, number of distinct values, mean, median, sd, min, max and R
    type of every column, and a summary row.

    It differs from cdff in MIN and MAX: a double column gives its
    smallest and largest number (Inf and -Inf when it has no values, as
    R's min and max of nothing), a Date or date-time R's underlying
    number, and every other column (integer, text, factor, logical) the
    first and last of its distinct values as text in gtools::mixedsort
    order.

    Parameters:
    df (pandas.DataFrame): Data to inspect.
    name_length (int, optional): Maximum characters kept of NAMES, MIN and MAX. Defaults to a third of
        pandas' display width (R: a third of getOption("width")).
    digits (int, optional): Decimals for MEAN, MEDIAN and SD. Defaults to 2.
    nuniques (int, optional): If > 0, adds UNIQUES and, when any column is a factor, LEVELS. Defaults to 0.
    parralel (bool, optional): Summarise columns in parallel threads. Defaults to False. (Spelling as in rwf.)
    file (str, optional): Excel filename without extension; writes sheets "variables" and "summary".
        Defaults to None.

    Returns:
    dict: {"summary": one-row DataFrame, "check": one row per column}, as cdff.

    Examples:
    >>> cdf(df_titanic)
    >>> cdf(df_titanic, nuniques=3)
    """
    return _check_dataframe(df, _cdf_column, name_length, digits, nuniques, parralel, file)
##########################################################################################
# CHECK DATAFRAME ROWS
##########################################################################################
def check_rows(df, filename=None):
    """
    cdf() of the rows instead of the columns (cdf of the transposed data).
    Python-only helper.

    Parameters:
    df (pandas.DataFrame): Data to analyze.
    filename (str, optional): Excel filename without extension. Defaults to None.

    Returns:
    dict: As cdf().

    Examples:
    >>> check_rows(df_titanic.iloc[:5, :5])
    """
    return cdf(df.transpose(), file=filename)
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print(observed(np.array([1, 2, np.nan, 4])))
    print(is_int(5), is_int(5.0))
    print(check_integer([1, 2, 3]), check_integer([1, 2.5, 3]))
    titanic = pd.read_csv("data/titanic.csv") if os.path.exists("data/titanic.csv") \
        else pd.read_csv("../data/titanic.csv")
    print(short_check(df=titanic))
    print(cdf(df=titanic)["check"])
    print(cdff(df=titanic, nuniques=3)["check"])
    print(check_rows(titanic.iloc[:5, :5])["check"])
