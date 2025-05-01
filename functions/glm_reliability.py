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

# alpha_result=ro.r('psych::alpha(df,check.keys=TRUE,n.iter=10)')

alpha_result={}

for trait in traits:
    print(trait)
    keys="NULL"
    keys=keys_dict[trait]
    str_with_parentheses = "(" + ", ".join(str(x) for x in keys) + ")"
    psych_string='psych::alpha(df,'+'keys=c'+str_with_parentheses+',n.iter=10)'
    df_trait=df_ocean.loc[:, df_ocean.columns.str.startswith(trait)]
    ro.globalenv['df']=df_trait
    result=ro.r(psych_string)
    alpha_result[trait]=result

trait='N'

for trait in traits:
    print(alpha_result[trait])

print(alpha_result[trait].names)
print(alpha_result[trait])

result_total_summary_list = []
result_items_list = []

for trait in traits:
    print(f"Trait: {trait}")
    r_total=alpha_result[trait].rx2('total')
    r_alpha_drop=alpha_result[trait].rx2('alpha.drop')
    r_item_stats=alpha_result[trait].rx2('item.stats')
    r_response_freq=alpha_result[trait].rx2('response.freq')
    r_keys=alpha_result[trait].rx2('keys')
    r_scores=alpha_result[trait].rx2('scores')
    r_nvar=alpha_result[trait].rx2('nvar')
    r_boot_ci=alpha_result[trait].rx2('boot.ci')
    r_boot=alpha_result[trait].rx2('boot')
    r_fieldt=alpha_result[trait].rx2('feldt')
    r_unidim=alpha_result[trait].rx2('Unidim')
    r_var_r=alpha_result[trait].rx2('var.r')
    r_fit=alpha_result[trait].rx2('Fit')
    r_call=alpha_result[trait].rx2('call')
    r_title=alpha_result[trait].rx2('title')

    with(ro.default_converter+pandas2ri.converter).context():
      result_total = ro.conversion.get_conversion().rpy2py(r_total)
    with(ro.default_converter+pandas2ri.converter).context():
      result_alpha_drop = ro.conversion.get_conversion().rpy2py(r_alpha_drop)
    with(ro.default_converter+pandas2ri.converter).context():
      result_item_stats = ro.conversion.get_conversion().rpy2py(r_item_stats)

    col_names=[f"{i+1}" for i in range(r_response_freq.shape[1] - 1)] + ['miss']
    result_response_frequency=pd.DataFrame(r_response_freq,columns=col_names)
    result_keys=pd.DataFrame(r_keys)
    result_scores=pd.DataFrame(r_scores)
    result_boot_ci=pd.DataFrame(pd.DataFrame(r_boot_ci).T)
    result_boot_ci.columns=['ci_lower','ci','ci_upper']
    result_boot=pd.DataFrame(r_boot)
    result_nvar=pd.DataFrame(r_var_r,columns=["nvar"])

    item_names=pd.DataFrame(result_item_stats.index,columns=["item"])

    temp_result_total_summary=pd.concat([pd.DataFrame({'trait':[trait]*len(result_nvar)}),
                                         result_nvar,
                                         result_total.reset_index(drop=True),
                                         result_boot_ci.reset_index(drop=True),
                                         result_keys.reset_index(drop=True)],
                                         axis=1)
    temp_result_items=pd.concat([pd.DataFrame({'trait':[trait]*len(item_names)}),
                                 item_names,
                                 result_alpha_drop.reset_index(drop=True),
                                 result_item_stats.reset_index(drop=True),
                                 result_response_frequency],
                                 axis=1)
    temp_result_items.index=result_item_stats.index
    
    result_total_summary_list.append(temp_result_total_summary)
    result_items_list.append(temp_result_items)
    
    result_total_summary=pd.concat(result_total_summary_list, ignore_index=True, sort=False)
    result_items=pd.concat(result_items_list, ignore_index=True, sort=False)






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


