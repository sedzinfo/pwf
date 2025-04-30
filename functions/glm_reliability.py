#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:32:36 2025
@author: Dimitrios Zacharatos
"""
##########################################################################################
# 
##########################################################################################
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import pandas as pd

pandas2ri.activate()

psych=importr('psych')

round(df_personality.corr(),1)

pd.DataFrame(df_ocean.columns)
df=df_ocean.loc[:, df_ocean.columns.str.startswith("E")]
df.corr()
df_ocean.head()

ro.globalenv['df']=df_ocean.loc[:, df_ocean.columns.str.startswith("O")]
ro.globalenv['df']=df_ocean.loc[:, df_ocean.columns.str.startswith("C")]
ro.globalenv['df']=df_ocean.loc[:, df_ocean.columns.str.startswith("A")]
ro.globalenv['df']=df_ocean.loc[:, df_ocean.columns.str.startswith("E")]
ro.globalenv['df']=df_ocean.loc[:, df_ocean.columns.str.startswith("N")]

traits=['O','C','E','A','N']
keys_dict = {"O":[1,-1,1,-1,1,-1,1,1,1,1],
             "C":[1,-1,1,-1,1,-1,1,-1,1,1],
             "E":[1,-1,1,-1,1,-1,1,-1,1,-1],
             "A":[-1,1,-1,1,-1,1,-1,1,1,1],
             "N":[1,-1,1,-1,1,1,1,1,1,1]}

alpha_result={}

for trait in traits:
    keys="NULL"
    keys=keys_dict[trait]
    str_with_parentheses = "(" + ", ".join(str(x) for x in keys) + ")"
    psych_string='psych::alpha(df,'+'keys=c'+str_with_parentheses+',n.iter=10)'
    df_trait=df_ocean.loc[:, df_ocean.columns.str.startswith(trait)]
    ro.globalenv['df']=df_trait
    result=ro.r(psych_string)
    alpha_result[trait]=result

trait='N'

print(alpha_result[trait])

alpha_result=ro.r('psych::alpha(df,check.keys=TRUE,n.iter=10)')

# If you want to see what keys the result has
print(list(alpha_result.names))
print(alpha_result.names)
print(alpha_result)

x01=alpha_result[trait].rx2('total')
x02=alpha_result[trait].rx2('alpha.drop')
x03=alpha_result[trait].rx2('item.stats')
x04=alpha_result[trait].rx2('response.freq')
x05=alpha_result[trait].rx2('keys')
x06=alpha_result[trait].rx2('scores')
x07=alpha_result[trait].rx2('nvar')
x08=alpha_result[trait].rx2('boot.ci')
x09=alpha_result[trait].rx2('boot')
x10=alpha_result[trait].rx2('feldt')
x11=alpha_result[trait].rx2('Unidim')
x12=alpha_result[trait].rx2('var.r')
x13=alpha_result[trait].rx2('Fit')
x14=alpha_result[trait].rx2('call')
x15=alpha_result[trait].rx2('title')
x16=alpha_result[trait].rx2('feldt')

with(ro.default_converter+pandas2ri.converter).context():
  result_total = ro.conversion.get_conversion().rpy2py(x01)
with(ro.default_converter+pandas2ri.converter).context():
  result_alpha_drop = ro.conversion.get_conversion().rpy2py(x02)
with(ro.default_converter+pandas2ri.converter).context():
  result_item_stats = ro.conversion.get_conversion().rpy2py(x03)

col_names=[f"{i+1}" for i in range(x04.shape[1] - 1)] + ['miss']
result_response_frequency=pd.DataFrame(x04,columns=col_names)
result_keys=pd.DataFrame(x05)
result_scores=pd.DataFrame(x06)
result_boot_ci=pd.DataFrame(pd.DataFrame(x08).T)
result_boot_ci.columns=['ci_lower','ci','ci_upper']
result_boot=pd.DataFrame(x09)
result_nvar=pd.DataFrame(x07,columns=["nvar"])

item_names=pd.DataFrame(result_item_stats.index,columns=["item"])

result_total_summary=pd.concat([result_nvar,
                                result_total.reset_index(drop=True),
                                result_boot_ci.reset_index(drop=True),
                                result_keys.reset_index(drop=True)],
                                axis=1)
result_items=pd.concat([item_names,
                        result_alpha_drop.reset_index(drop=True),
                        result_item_stats.reset_index(drop=True),
                        result_response_frequency],
                        axis=1)
result_items.index=result_item_stats.index

result_total_summary
result_items

result_scores



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

alpha_result.rx2('call')

alpha_result.rx2('call')

alpha_result.rx2('total')
alpha_result.do_slot_assign
alpha_result.from_iterable
alpha_result.from_length
alpha_result.from_memoryview

print(alpha_result)


