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
def report_ttests(df,dv,iv,paired=False,alternative="two-sided",correction=False):
    """  
    
    Parameters
    ----------
    x : array_like
     First set of observations.
    y : array_like or float
     Second set of observations. If ``y`` is a single value, a one-sample
     T-test is computed against that value (= "mu" in the t.test R function).
    paired : boolean
     Specify whether the two observations are related (i.e. repeated measures) or independent.
    alternative : string
     Defines the alternative hypothesis, or tail of the test. Must be one of
     "two-sided" (default), "greater" or "less". Both "greater" and "less" return one-sided
     p-values. "greater" tests against the alternative hypothesis that the mean of ``x``
     is greater than the mean of ``y``.
     correction : string or boolean
     For unpaired two sample T-tests, specify whether or not to correct for
     unequal variances using Welch separate variances T-test. If 'auto', it
     will automatically uses Welch T-test when the sample sizes are unequal,
     as recommended by Zimmerman 2004.
    r : float
     Cauchy scale factor for computing the Bayes Factor.
     Smaller values of r (e.g. 0.5), may be appropriate when small effect
     sizes are expected a priori; larger values of r are appropriate when
     large effect sizes are expected (Rouder et al 2009).
     The default is 0.707 (= :math:`\sqrt{2} / 2`).
    confidence : float
     Confidence level for the confidence intervals (0.95 = 95%)
    
    * ``'T'``: T-value
    * ``'dof'``: degrees of freedom
    * ``'alternative'``: alternative of the test
    * ``'p-val'``: p-value
    * ``'CI95%'``: confidence intervals of the difference in means
    * ``'cohen-d'``: Cohen d effect size
    * ``'BF10'``: Bayes Factor of the alternative hypothesis
    * ``'power'``: achieved power of the test ( = 1 - type II error)
    """  
    
    
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
                                        correction=correction,
                                        r=0.707))
            descriptives=pd.DataFrame({"Paired":[paired],
                                       "alternative":[alternative],
                                       "IV":[factor],
                                       "L1":[levels[0]],
                                       "L2":[levels[1]],
                                       "Mean_L1":[variable_1.mean()],
                                       "Mean_L2":[variable_2.mean()],
                                       "Var_L1":[variable_1.var()],
                                       "Var_L2":[variable_2.var()],
                                       "sd_L1":[variable_1.std()],
                                       "sd_L2":[variable_2.std()],
                                       "pooled_sd":[np.sqrt(((variable_1.std()**2)+(variable_2.std()**2))/2)],
                                       "Levene":[levene_result[0]],
                                       "L_p":[levene_result[1]],
                                       "Bartlett":[bartlett_result[0]],
                                       "B_p":[bartlett_result[1]]},
                                        index=ttest.index)
            result=pd.concat([descriptives,ttest],axis=1)
            result_iterative=pd.concat([result_iterative,result],axis=0)
    return result_iterative

# report_ttests(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"]).round(2)
##########################################################################################
# LEVENE
##########################################################################################
from scipy.stats import levene
import pandas as pd
import itertools

def report_levene_bartlett(df, dv, iv):
    result_iterative = pd.DataFrame()
    for factor in iv:
        levels = df[factor].unique()
        combinations = list(itertools.combinations(levels, 2))
        for level1, level2 in combinations:
            group1 = df[df[factor] == level1][dv]
            group2 = df[df[factor] == level2][dv]
            stat_l, p_l = levene(group1, group2)
            stat_b, p_b = bartlett(group1, group2)
            result = pd.DataFrame({
                "IV": [factor],
                "L1": [level1],
                "L2": [level2],
                "Levene": [stat_l],
                "Levene_p": [p_l],
                "Bartlett": [stat_b],
                "Bartlett_p": [p_b]
            })
            result_iterative = pd.concat([result_iterative, result], ignore_index=True)
    return result_iterative

# report_levene_bartlett(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"])

