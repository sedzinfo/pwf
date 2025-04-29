#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:32:36 2017
@author: Dimitrios Zacharatos
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
    return np.count_nonzero(~np.isnan(vector))
# observed(np.empty((10,)))
##########################################################################################
# CHECK INTEGER
##########################################################################################
def is_int(vector):
    if type(vector) == int:
        return True
def check_integer(vector):
    for i in vector:
        if not is_int(i):
            return False
    return True
# is_int(np.empty((10,)))
# check_integer(np.empty((10,)))
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
# short_check(df=df_admission)
# short_check(df=df_automotive)
# short_check(df=df_blood_pressure)
# short_check(df=df_crop_yield)
# short_check(df=df_difficile)
# short_check(df=df_insurance)
# short_check(df=df_responses)
# short_check(df=df_sexual_comp)
# short_check(df=df_personality)
# short_check(df=df_titanic)
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
        critical_value_excel(pd.DataFrame(dataframes[0]),writer,"SUMMARY")
        critical_value_excel(pd.DataFrame(dataframes[1]),writer,"CHECK")
        writer.save()
    print(short_check_result)
    print(check)
    pd.set_option('display.max_columns', original_display_options['max_columns'])
    pd.set_option('display.max_rows', original_display_options['max_rows'])
    pd.set_option('display.max_colwidth', original_display_options['max_colwidth'])
    pd.set_option('display.width', original_display_options['width'])
    pd.set_option('display.max_seq_item', original_display_options['max_seq_item'])
    return check, short_check_result

# cdf(df=df_admission)
# cdf(df=df_automotive)
# cdf(df=df_blood_pressure)
# cdf(df=df_crop_yield)
# cdf(df=df_difficile)
# cdf(df=df_insurance)
# cdf(df=df_responses)
# cdf(df=df_sexual_comp)
# cdf(df=df_personality)
# cdf(df=df_titanic)
# cdf(df=df_ocean)
##########################################################################################
# CHECK DATAFRAME ROWS
##########################################################################################
def check_rows(df,filename=""):
    dataframes=cdf(df.transpose(),filename=filename)
    return dataframes
# check_rows(personality)
# check_rows(df)
# check_rows(titanic)

