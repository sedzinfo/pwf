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
##########################################################################################
# LOAD
##########################################################################################
import pandas as pd
import researchpy as rp
df=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")
df.drop('person',axis=1,inplace= True)

# Recoding value from numeric to string
df['dose'].replace({1:'placebo',2:'low',3:'high'},inplace= True)
df.info()

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

stats.f_oneway(df['libido'][df['dose']=='high'],
               df['libido'][df['dose']=='low'],
               df['libido'][df['dose']=='placebo'])

model=ols('libido~C(dose)',data=df).fit()
aov_table=sm.stats.anova_lm(model,typ=2)
aov_table

def anova_table(aov):
    aov['mean_sq']=aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq']=aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq']=(aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols=['sum_sq','df','mean_sq','F','PR(>F)','eta_sq','omega_sq']
    aov=aov[cols]
    return aov

anova_table(aov_table)
stats.shapiro(model.resid)


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)

normality_plot,stat=stats.probplot(model.resid,plot= plt,rvalue= True)
ax.set_title("Probability plot of model residual's",fontsize= 20)
ax.set

plt.show()

stats.levene(df['libido'][df['dose']=='high'],
             df['libido'][df['dose']=='low'],
             df['libido'][df['dose']=='placebo'])


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)

ax.set_title("Box Plot of Libido by Dosage",fontsize= 20)
ax.set

data=[df['libido'][df['dose']=='placebo'],
      df['libido'][df['dose']=='low'],
      df['libido'][df['dose']=='high']]

ax.boxplot(data,
           labels= ['Placebo','Low','High'],
           showmeans= True)

plt.xlabel("Drug Dosage")
plt.ylabel("Libido Score")

plt.show()

import statsmodels.stats.multicomp as mc

comp=mc.MultiComparison(df['libido'],df['dose'])
post_hoc_res=comp.tukeyhsd()
print(post_hoc_res.summary())

post_hoc_res.plot_simultaneous(ylabel= "Drug Dose",xlabel= "Score Difference")
allpairtest(statistical_test_method,method= "correction_method")

import statsmodels.stats.multicomp as mc

comp=mc.MultiComparison(df['libido'],df['dose'])
tbl,a1,a2=comp.allpairtest(stats.ttest_ind,method= "bonf")

print(tbl)

allpairtest(statistical_test_method,method= "correction_method")

import statsmodels.stats.multicomp as mc

comp=mc.MultiComparison(df['libido'],df['dose'])
tbl,a1,a2=comp.allpairtest(stats.ttest_ind,method= "")

print(tbl)

