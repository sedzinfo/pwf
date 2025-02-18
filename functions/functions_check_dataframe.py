#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:32:36 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
get_path = robjects.r('rstudioapi::getActiveDocumentContext()$path')
file_path = str(get_path[0]).replace(os.path.basename(str(get_path[0])),"").rstrip("/")
file_directory = os.path.dirname(file_path) or os.getcwd()
sys.path.insert(1,file_path)
from __init__ import *
from functions import *
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
# check_integer(np.empty((10,)))
##########################################################################################
# SHORT CHECK DATAFRAME
##########################################################################################
def short_check(df):
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
# short_check(personality)
# short_check(df)
# short_check(titanic)
##########################################################################################
# CHECK DATAFRAME
##########################################################################################
def check(df,filename=""):
    def is_nan(x):
        return x != x
    def is_inf(x):
        if isinstance(x,(int, float)):
            return np.isinf(x)
        else:
            return False
    short_check_result=short_check(df)
    sdf=pd.DataFrame({"NULL":df.isnull().sum(),
                      "NAN":df.applymap(is_nan).sum(),
                      "EMPTY_STRINGS":(df=="").sum(),
                      "INFINITE":df.applymap(is_inf).sum(),
                      "NOT_NULL":df.count(),
                      "UNIQUE":df.nunique(),
                      "MIN":df.min(axis=0,skipna=True,numeric_only=True),
                      "MAX":df.max(axis=0,skipna=True,numeric_only=True),
                      "MEAN":df.mean(axis=0,skipna=True,numeric_only=True),
                      "MEDIAN":df.median(axis=0,skipna=True,numeric_only=True),
                      "RANGE":df.max(axis=0,skipna=True,numeric_only=True)-df.min(axis=0,skipna=True,numeric_only=True),
                      "SD":df.std(axis=None,skipna=True,ddof=1,numeric_only=True),
                      "TYPE":df.dtypes
                      })
    check=sdf.reset_index()
    if filename !="":
        writer=pd.ExcelWriter(filename,engine='xlsxwriter')
        critical_value_excel(pd.DataFrame(dataframes[0]),writer,"SUMMARY")
        critical_value_excel(pd.DataFrame(dataframes[1]),writer,"CHECK")
        writer.save()
    return check, short_check_result
# check(personality)
# check(df)
# check(titanic)
##########################################################################################
# CHECK DATAFRAME ROWS
##########################################################################################
def check_rows(df,filename=""):
    dataframes=check(df.transpose(),filename=filename)
    dataframes[1]=dataframes[1].dropna(axis=1,how='all')
    return dataframes
# check_rows(personality)
# check_rows(df)
# check_rows(titanic)

