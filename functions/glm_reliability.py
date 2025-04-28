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
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


psych = importr('psych')

robjects.globalenv['df'] = df_personality.iloc[:,0:5]
alpha_result = psych.alpha(robjects.r('df'),check_keys=True)

pandas2ri.activate()

# If you want to see what keys the result has
print(list(alpha_result.names))

x01=alpha_result.rx2('total')
x02=alpha_result.rx2('alpha.drop')
x03=alpha_result.rx2('item.stats')
x04=alpha_result.rx2('response.freq')
x05=alpha_result.rx2('keys')
x06=alpha_result.rx2('scores')
x07=alpha_result.rx2('nvar')
x08=alpha_result.rx2('boot.ci')
x09=alpha_result.rx2('boot')
x10=alpha_result.rx2('feldt')
x11=alpha_result.rx2('Unidim')
x12=alpha_result.rx2('var.r')
x13=alpha_result.rx2('Fit')
x14=alpha_result.rx2('call')
x15=alpha_result.rx2('title')
x16=alpha_result.rx2('feldt')

with(ro.default_converter+pandas2ri.converter).context():
  result_total = ro.conversion.get_conversion().rpy2py(x01)
with(ro.default_converter+pandas2ri.converter).context():
  result_alpha_drop = ro.conversion.get_conversion().rpy2py(x02)
with(ro.default_converter+pandas2ri.converter).context():
  result_item_stats = ro.conversion.get_conversion().rpy2py(x03)

result_response_frequency=pd.DataFrame(x04)
result_keys=pd.DataFrame(x05)
result_scores=pd.DataFrame(x06)

result_alpha_drop
result_total
result_item_stats
result_response_frequency


print(alpha_result.rx2('total'))
print(alpha_result.rx2('alpha.drop'))
print(alpha_result.rx2('item.stats'))
print(alpha_result.rx2('response.freq'))
print(alpha_result.rx2('keys'))
print(alpha_result.rx2('scores'))
print(alpha_result.rx2('nvar'))
print(alpha_result.rx2('boot.ci'))
print(alpha_result.rx2('boot'))
print(alpha_result.rx2('feldt'))
print(alpha_result.rx2('Unidim'))
print(alpha_result.rx2('var.r'))
print(alpha_result.rx2('Fit'))
print(alpha_result.rx2('call'))
print(alpha_result.rx2('title'))

alpha_result.rx2('total')
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
