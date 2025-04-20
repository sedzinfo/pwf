# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:20:04 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
import itertools
from scipy.stats import levene, bartlett
import functions_excel as fe
##########################################################################################
# 
##########################################################################################
def report_ttests(df,dv,iv,paired=False,alternative="two-sided"):
    result_iterative=pd.DataFrame()
    for factor in iv:
        levels=df[factor].value_counts().index.values
        combinations=list(itertools.combinations(levels,2))
        for levels in combinations:
            variable_1=df[df[factor]==levels[0]][dv]
            variable_2=df[df[factor]==levels[1]][dv]
            # df_test=pd.concat(variable_1,variable_2,axis=0).dropna()
            levene_result=levene(variable_1,variable_2)
            bartlett_result=bartlett(variable_1,variable_2)
            ttest=pd.DataFrame(pg.ttest(variable_1,
                                        variable_2,
                                        paired=paired,
                                        alternative=alternative,
                                        correction=True))
            descriptives=pd.DataFrame({"Paired":[paired],
                                       "alternative":[alternative],
                                       "IV":[factor],
                                       "V1":[levels[0]],
                                       "V2":[levels[1]],
                                       "Mean_V1":[variable_1.mean()],
                                       "Mean_V2":[variable_2.mean()],
                                       "Variance_V1":[variable_1.var()],
                                       "Variance_V2":[variable_2.var()],
                                       "Levene":[levene_result[0]],
                                       "Levene_p":[levene_result[1]],
                                       "Bartlett":[bartlett_result[0]],
                                       "Bartlett_p":[bartlett_result[1]]},
                                        index=ttest.index)
            result=pd.concat([descriptives,ttest],axis=1)
            result_iterative=pd.concat([result_iterative,result],axis=0)
    return result_iterative

# ttest(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"])
##########################################################################################
# LEVENE
##########################################################################################
from scipy.stats import levene
import pandas as pd
import itertools

def report_levene(df, dv, iv):
    result_iterative = pd.DataFrame()
    for factor in iv:
        levels = df[factor].unique()
        combinations = list(itertools.combinations(levels, 2))
        for level1, level2 in combinations:
            group1 = df[df[factor] == level1][dv]
            group2 = df[df[factor] == level2][dv]
            stat, p = levene(group1, group2)
            result = pd.DataFrame({
                "IV": [factor],
                "V1": [level1],
                "V2": [level2],
                "Levene": [stat],
                "Levene_p": [p]
            })
            result_iterative = pd.concat([result_iterative, result], ignore_index=True)
    return result_iterative

report_levene(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"])
##########################################################################################
# BARTLETT
##########################################################################################
from scipy.stats import bartlett
import pandas as pd
import itertools

def report_bartlett(df, dv, iv):
    result_iterative = pd.DataFrame()
    for factor in iv:
        levels = df[factor].unique()
        combinations = list(itertools.combinations(levels, 2))
        for level1, level2 in combinations:
            group1 = df[df[factor] == level1][dv]
            group2 = df[df[factor] == level2][dv]
            stat, p = bartlett(group1, group2)
            result = pd.DataFrame({
                "IV": [factor],
                "V1": [level1],
                "V2": [level2],
                "Bartlett": [stat],
                "Bartlett_p": [p]
            })
            result_iterative = pd.concat([result_iterative, result], ignore_index=True)
    return result_iterative
  
report_bartlett(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"])





