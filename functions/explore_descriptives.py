#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:05:57 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd

path_script = os.getcwd()
path_root = path_script.replace('\\functions', '')

sys.path.insert(1,file_path)
from __init__ import *
from functions import *
##########################################################################################
# 
##########################################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]
##########################################################################################
# 
##########################################################################################
def describe_by_mean(df,factorname=[]):
    result=pd.DataFrame()
    factor=[]
    for category in factorname:
        df_mean=df.groupby(category).mean(numeric_only=True)
        df_min=df.groupby(category).min(numeric_only=True)
        df_max=df.groupby(category).max(numeric_only=True)
        df_std=df.groupby(category).std(numeric_only=True)
        df_mean=df_mean.assign(Statistic=["Mean"]*df_mean.shape[0])
        df_min=df_min.assign(Statistic=["Min"]*df_min.shape[0])
        df_max=df_max.assign(Statistic=["Max"]*df_max.shape[0])
        df_std=df_std.assign(Statistic=["SD"]*df_std.shape[0])
        descriptives=pd.concat([df_mean,df_min,df_max,df_std])
        descriptives.reset_index(inplace=True)
        descriptives.rename(columns={list(descriptives)[0]:'Level'},inplace=True)
        factor.append([category]*descriptives.shape[0])
        result=result._append(descriptives)
    result.insert(0,"Factor",flatten(factor))
    result.insert(0,"Statistic",result.pop("Statistic"))
    result=result.sort_values(by=["Statistic","Factor","Level"])
    return(result)
# describe_by_mean(df=titanic,factorname=['Survived','Pclass','Sex','Embarked'])







