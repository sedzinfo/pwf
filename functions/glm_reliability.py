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
from factor_analyzer import FactorAnalyzer

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

# alpha_result=ro.r('psych::alpha(df,check.keys=TRUE,n.iter=10)')

comments = {"raw_alpha": """Crombach's alpha
alpha based on covariances
Lambda 3=(n)/(n-1)(1-tr(Vx)/(Vx) = (n)/(n-1)(Vx-tr(Vx)/Vx=alpha
0.91-1.00 Excellent (however check for multicolinearity problems)
0.81-0.90 Good
0.71-0.80 Good and Acceptable
0.61-0.70 Acceptable
0.56-0.60 Marginally Acceptable
0.01-0.55 Unacceptable""",
            "std_alpha": "standardized alpha based on correlations",
            "G6(smc)": """Guttman's Lambda 6 reliability
squared multiple correlation
considers the amount of variance in each item that can be accounted for the linear regression of all of the other items
lambda 6=1-sum(e^2)/Vx = 1-sum(1-r^2(smc))/Vx
if equal item loadings alpha > G6
if unequal item loadings alpha < G6
if there is a general factor alpha < G6""",
            "average_r": "average interitem correlation",
            "median_r": "median interitem correlation",
            "S/N": """signal to noise ratio index of the quality of the test that is linear with the number of items and the average correlation
S/N = n r/(1 - r)""",
            "ase": "alpha standard error",
            "mean": """for total statistics: mean of the scale formed by averaging or summing the items (depending upon the cumulative option)
                       for item statistics: mean of each item""",
            "sd": """for total statistics: standard deviation of the total score
                     for item statistics: standard deviation of each item""",
            "alpha_drop": "A data frame with all of the above for the case of each item being removed one by one.",
            "n": "number of complete cases for the item",
            "raw.r": "correlation of each item with the total score, not corrected for item overlap",
            "std.r": "correlation of each item with the total score (not corrected for item overlap) if the items were all standardized",
            "r.cor": "item whole correlation corrected for item overlap and scale reliability",
            "r.drop": "item whole correlation for this item against the scale without this item",
            "response_freq": "the frequency of each item response (if less than 20)",
            "scores": """scores are by default the average response for all items that a participant took If cumulative=TRUE, then these are sum scores.""",
            "ci_lower": "bootstrap confidence interval lower bound 2.5%",
            "ci": "bootstrap confidence interval 50%",
            "ci_upper": "bootstrap confidence interval lower bound 97.5%",
            "miss": "proportion of non answered items",
            "alpha se": "alpha standard error",
            "med.r": "median interitem correlation",
            "unidim": "index of unidimensionality",
            "kaiser_criterion": "number of eigenvalues > 1"
}

def report_alpha(df, keys, n_iter=10):
    
    alpha_result={}
    fa_result={}
    
    for trait in traits:
        keys="NULL"
        keys=keys_dict[trait]
        str_with_parentheses="(" + ", ".join(str(x) for x in keys) + ")"
        psych_string='psych::alpha(df,'+'keys=c'+str_with_parentheses+',n.iter=10)'
        df_trait=df_ocean.loc[:,df_ocean.columns.str.startswith(trait)]
        fa_result[trait]=FactorAnalyzer(rotation="varimax")
        fa_result[trait].fit(df_trait)
        ro.globalenv['df']=df_trait
        result=ro.r(psych_string)
        alpha_result[trait]=result

    result_total_summary_list=[]
    result_items_list=[]

    for trait in traits:
        print(f"Trait: {trait}")
        eigen_values,vectors=fa_result[trait].get_eigenvalues()
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
        result_var=pd.DataFrame(r_var_r,columns=["nvar"])
        result_eigen=pd.DataFrame([[np.sum(eigen_values>1)]],columns=["kaiser"])

        item_names=pd.DataFrame(result_item_stats.index,columns=["item"])

        temp_result_total_summary=pd.concat([pd.DataFrame({'trait':[trait]*len(result_var)}),
                                             result_var,
                                             result_eigen,
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
        
    return(result_total_summary,result_items)
        
keys_dict={"O":[1,-1,1,-1,1,-1,1,1,1,1],
           "C":[1,-1,1,-1,1,-1,1,-1,1,1],
           "E":[1,-1,1,-1,1,-1,1,-1,1,-1],
           "A":[-1,1,-1,1,-1,1,-1,1,1,1],
           "N":[1,-1,1,-1,1,1,1,1,1,1]}

total,items=report_alpha(df_ocean,keys_dict)

traits=['O','C','E','A','N']

for trait in traits:
    print(alpha_result[trait])
    print(alpha_result[trait].names)
    print(alpha_result[trait].rx2('total'))
    print(alpha_result[trait].rx2('alpha.drop'))
    print(alpha_result[trait].rx2('item.stats'))
    print(alpha_result[trait].rx2('response.freq'))
    print(alpha_result[trait].rx2('keys'))
    print(alpha_result[trait].rx2('scores'))
    print(alpha_result[trait].rx2('nvar'))
    print(alpha_result[trait].rx2('boot.ci'))
    print(alpha_result[trait].rx2('boot'))
    print(alpha_result[trait].rx2('feldt'))
    print(alpha_result[trait].rx2('Unidim'))
    print(alpha_result[trait].rx2('var.r'))
    print(alpha_result[trait].rx2('Fit'))
    print(alpha_result[trait].rx2('call'))
    print(alpha_result[trait].rx2('title'))


output_file=path_root+'/output/total.xlsx'
if os.path.exists(output_file):
   os.remove(output_file)
output_dir = os.path.dirname(output_file)   
os.makedirs(output_dir, exist_ok=True)
ge=pd.ExcelWriter(output_file,engine='xlsxwriter')
generic_format_excel(df=total,writer=ge,sheetname="total",comments=comments)
generic_format_excel(df=items,writer=ge,sheetname="items",comments=comments)
ge._save()
ge.close()

os.remove(output_file)






