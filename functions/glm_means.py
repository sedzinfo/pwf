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

path_script = os.getcwd()
path_root = path_script.replace('\\functions', '')

sys.path.insert(1,file_path)
from __init__ import *
from functions import *
##########################################################################################
# LOAD
##########################################################################################
import pandas as pd
import numpy as np
import pingouin as pg
import itertools
from scipy.stats import levene, bartlett
import functions_excel as fe

df=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
personality=pd.read_csv('/opt/pyrepo/data/personality.csv')
titanic=pd.read_csv('/opt/pyrepo/data/titanic.csv')
##########################################################################################
# 
##########################################################################################
def ttest(df,dv,iv,paired=False,alternative="two-sided"):
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

result_titanic=ttest(df=titanic,dv="Age",iv=["Embarked","Sex"])
result_before=ttest(df=df,dv="bp_before",iv=["sex","agegrp"])
result_after=ttest(df=df,dv="bp_after",iv=["sex","agegrp"])

writer_critical_value_excel=pd.ExcelWriter('/opt/pyrepo/output/xlsxwriter_critical_value_excel.xlsx',engine='xlsxwriter')
fe.critical_value_excel(result_titanic,writer_critical_value_excel,"DATA",comment="Test Comment")
writer_critical_value_excel._save()






