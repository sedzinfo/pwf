#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:32:36 2025
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import pingouin as pg
alpha = pg.cronbach_alpha(data=df_personality.iloc[0:10])
alpha


import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# Activate automatic pandas conversion
pandas2ri.activate()

# Import R's psych package
ro.r('library(psych)')

# Assume df_subset is your pandas dataframe
# Send pandas dataframe to R
ro.globalenv['df_subset'] = df_subset

# Run psych::alpha
alpha_result = ro.r('alpha(df_subset)')

# Print R output
print(alpha_result)
