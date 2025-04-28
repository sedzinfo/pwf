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
##########################################################################################
# 
##########################################################################################
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
from rpy2.robjects import pandas2ri

# Import psych package
psych = importr('psych')

# Example data (optional: if you want to simulate)
# Let's create some dummy data in R
robjects.r('''
library(psych)
set.seed(123)
df <- data.frame(matrix(rnorm(100), nrow=20))
''')

# Run alpha on the R dataframe
alpha_result = psych.alpha(robjects.r('df'))

pandas2ri.activate()

# If you want to see what keys the result has
print(list(alpha_result.names))

# Then you can access specific parts
print(alpha_result.rx2('total'))

alpha_result.do_slot
alpha_result.do_slot_assign
alpha_result.from_iterable
alpha_result.from_length
alpha_result.from_memoryview

# alpha_result is an R list
# Let's print the names of the list elements
print(alpha_result)

# Access parts, for example:
raw_alpha = alpha_result.rx2('total').rx2('raw_alpha')[0]
print("Cronbach's alpha:", raw_alpha)

# If you want **all** results as a Python dictionary:
total = alpha_result.rx2('total')
keys = total.names
values = [total.rx2(k) for k in keys]

alpha_dict = {k: float(v[0]) if len(v) == 1 else list(v) for k, v in zip(keys, values)}

print(alpha_dict)

# Save to a dataframe (optional)
df_alpha = pd.DataFrame.from_dict(alpha_dict, orient='index', columns=['Value'])
print(df_alpha)

# Then save to CSV if you want
df_alpha.to_csv('alpha_output.csv')
