# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:45:39 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
# sys.path.insert(1,'/opt/pyrepo/functions/')
# from __init__ import *
sys.path.insert(1,'/opt/pyrepo/functions/functions.py')
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer,calculate_bartlett_sphericity,calculate_kmo
from sklearn.decomposition import FactorAnalysis
import numpy as np
##########################################################################################
# 
##########################################################################################
df=pd.read_csv("/opt/pyrepo/data/personality.csv")
chi_square_value,p_value=calculate_bartlett_sphericity(df)
kmo_all,kmo_model=calculate_kmo(df)
kmo_variables=pd.DataFrame(list(zip(df.columns.to_list(),kmo_all)),columns=['Name','KMO'])
# print({"chi_square_value":chi_square_value,"p_value":p_value})
# print(kmo_all)
# print(df.columns.to_list())

fa=FactorAnalyzer(rotation=None)
result=fa.fit(df)
eigenvalues=fa.get_eigenvalues()
communalities=fa.get_communalities()
loadings=fa.loadings_
factor_variance=fa.get_factor_variance()
uniquinesses=fa.get_uniquenesses()
sufficiency=fa.sufficiency(df.shape[0])
fa.transform(df) # get factor scores for new dataset

index=list(range(1,df.shape[1]+1))

df_eigenvalues=pd.DataFrame({"index":index,"eigen1":eigenvalues[0],"eigen2":eigenvalues[1]})
df_communalities=pd.DataFrame({"index":index,"communalities":communalities})
df_uniqueness=pd.DataFrame({"index":index,"uniquinesses":uniquinesses})
df_loadings=pd.DataFrame(loadings)
df_loadings.insert(0,"index",index)
df_factor_variance=pd.DataFrame(factor_variance)
df_loadings=df_loadings.add_prefix("loading_")

correlations=df.corr()
residual_correlations=pd.DataFrame(np.dot(fa.loadings_,fa.loadings_.T))
residual_correlations.columns=correlations.columns
residual_correlations.index=correlations.index
residuals=correlations.abs()-residual_correlations.abs()

print(residuals)





