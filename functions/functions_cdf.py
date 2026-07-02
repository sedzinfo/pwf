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
import pandas as pd
import numpy as np
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
# CHECK DATAFRAME
##########################################################################################
def cdf(df,width=10000,filename=""):
    """
    Performs a thorough check of the given DataFrame (`df`) and provides various statistics such as NULL, NAN, 
    EMPTY_STRINGS, INFINITE, NOT_NULL, UNIQUE, MIN, MAX, MEAN, MEDIAN, RANGE, and SD for each column. Optionally,
    the results can be saved to an Excel file. Additionally, the function temporarily adjusts pandas display settings 
    to show the full content of the DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.
        
    width : int, optional, default=10000
        The width setting to be used for pandas display. This determines the column width when displaying the DataFrame.
        
    filename : str, optional, default=""
        If provided, the function saves the results to an Excel file with two sheets: "SUMMARY" and "CHECK". If not provided, 
        the results are not saved to a file.

    Returns:
    -------
    check : pandas.DataFrame
        A DataFrame containing statistical information for each column in `df` including NULLs, NANs, 
        unique values, MIN, MAX, etc.
        
    short_check_result : pandas.DataFrame
        A DataFrame returned by the `short_check` function, which provides additional insights into the DataFrame's structure.
        
    Notes:
    ------
    - The function temporarily modifies pandas display settings to allow for full viewing of the DataFrame columns and rows.
    - The original display settings are restored after execution.
    - This function uses helper functions `is_nan` and `is_inf` to check for NaN and infinite values in the DataFrame.
    - If `filename` is provided, results are written to an Excel file using `xlsxwriter`.
    """
    
    # Save the original display options
    original_display_options = {
        'max_columns': pd.get_option('display.max_columns'),
        'max_rows': pd.get_option('display.max_rows'),
        'max_colwidth': pd.get_option('display.max_colwidth'),
        'width': pd.get_option('display.width'),
        'max_seq_item': pd.get_option('display.max_seq_item')
    }
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.width', width)
    pd.set_option('display.max_seq_item', None)
    def is_nan(x):
        return x != x
    def is_inf(x):
        if isinstance(x,(int, float)):
            return np.isinf(x)
        else:
            return False
    short_check_result=short_check(df)
    sdf=pd.DataFrame({"NULL":df.isnull().sum(),
                      "NAN":df.map(is_nan).sum(),
                      "EMPTY_STRINGS":(df=="").sum(),
                      "INFINITE":df.map(is_inf).sum(),
                      "NOT_NULL":df.count(),
                      "UNIQUE":df.nunique(),
                      "MIN":df.min(axis=0,skipna=True,numeric_only=True),
                      "MAX":df.max(axis=0,skipna=True,numeric_only=True),
                      "MEAN":df.mean(axis=0,skipna=True,numeric_only=True),
                      "MEDIAN":df.median(axis=0,skipna=True,numeric_only=True),
                      "RANGE":df.max(axis=0,skipna=True,numeric_only=True)-df.min(axis=0,skipna=True,numeric_only=True),
                      "SD":df.std(axis=0,skipna=True,ddof=1,numeric_only=True),
                      "TYPE":df.dtypes
                      })
    check=sdf.reset_index()
    if filename !="":
        writer=pd.ExcelWriter(filename,engine='xlsxwriter')
        short_check_result.to_excel(writer,sheet_name="SUMMARY",index=False)
        check.to_excel(writer,sheet_name="CHECK",index=False)
        writer.close()
    print(short_check_result)
    print(check)
    pd.set_option('display.max_columns', original_display_options['max_columns'])
    pd.set_option('display.max_rows', original_display_options['max_rows'])
    pd.set_option('display.max_colwidth', original_display_options['max_colwidth'])
    pd.set_option('display.width', original_display_options['width'])
    pd.set_option('display.max_seq_item', original_display_options['max_seq_item'])
    return check, short_check_result
##########################################################################################
# CHECK DATAFRAME ROWS
##########################################################################################
def check_rows(df,filename=""):
    """
    Same statistics as cdf(), but computed per row instead of per column
    (via transposing df first).

    Parameters:
    df (pandas.DataFrame): Data to analyze.
    filename (str, optional): If given, writes an Excel report. Defaults to "".

    Returns:
    tuple: Same as cdf() -- (check, short_check_result).

    Examples:
    >>> import pandas as pd
    >>> titanic = pd.read_csv("data/titanic.csv")
    >>> check_rows(titanic.iloc[:5, :5])
    """
    dataframes=cdf(df.transpose(),filename=filename)
    return dataframes
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os

    print("=" * 80, "\nobserved / is_int / check_integer\n", "=" * 80, sep="")
    print(observed(np.array([1, 2, np.nan, 4])))
    print(is_int(5), is_int(5.0))
    print(check_integer([1, 2, 3]), check_integer([1, 2.5, 3]))

    titanic = pd.read_csv("data/titanic.csv") if os.path.exists("data/titanic.csv") \
        else pd.read_csv("../data/titanic.csv")

    print("\n" + "=" * 80, "\nshort_check\n", "=" * 80, sep="")
    print(short_check(df=titanic))

    print("\n" + "=" * 80, "\ncdf\n", "=" * 80, sep="")
    check, short_check_result = cdf(df=titanic.iloc[:, :6])

    print("\n" + "=" * 80, "\ncheck_rows\n", "=" * 80, sep="")
    check_r, short_r = check_rows(titanic.iloc[:5, :5])
    print(check_r)

