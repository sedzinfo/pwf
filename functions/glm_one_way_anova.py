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
import researchpy as rp
##########################################################################################
# KRUSKALL WALLIS TEST WITH EFFECT SIZE
##########################################################################################
import pandas as pd
import numpy as np
import scipy.stats as stats
import patsy

def compute_kruskal_wallis_test(formula, df):
    # Use patsy to parse the formula
    y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    x = y.iloc[:, 0].values
    g = X.iloc[:, 1]  # assuming one-hot encoding; drop Intercept
    if 'Intercept' in X.columns:
        g = X.iloc[:, 1]
    else:
        g = X.iloc[:, 0]
    
    group_labels = df[formula.split('~')[1].strip()].astype('category')
    k = group_labels.nunique()
    n = len(x)
    
    r = stats.rankdata(x)
    ranks_by_group = pd.Series(r).groupby(group_labels).agg(['sum', 'count'])

    h = np.sum((ranks_by_group['sum'] ** 2) / ranks_by_group['count'])
    H = (12 * h / (n * (n + 1))) - 3 * (n + 1)

    # Correction for ties
    ties = pd.Series(x).value_counts()
    tie_correction = 1 - np.sum(ties**3 - ties) / (n**3 - n)
    H /= tie_correction

    df_k = k - 1
    p = stats.chi2.sf(H, df_k)

    # Effect sizes
    etasq = (H - k + 1) / (n - k)
    epsilonsq = H / ((n ** 2 - 1) / (n + 1))

    result = pd.DataFrame([{
        "formula": formula,
        "method": "Kruskal-Wallis rank sum test",
        "etasq": etasq,
        "epsilonsq": epsilonsq,
        "H": H,
        "df": df_k,
        "p": p
    }])

    return result

compute_kruskal_wallis_test(formula="bp_before~agegrp", df=df_blood_pressure)
##########################################################################################
# ONE WAY TEST WITH SS AND MS
##########################################################################################
import numpy as np
import pandas as pd
from scipy import stats

def compute_one_way_test(formula, df, var_equal=True):
    y_var, g_var = formula.replace(" ", "").split("~")
    y = df[y_var]
    g = df[g_var].astype("category")

    k = g.nunique()
    n_i = df.groupby(g_var)[y_var].count()
    m_i = df.groupby(g_var)[y_var].mean()
    v_i = df.groupby(g_var)[y_var].var()
    w_i = n_i / v_i
    sum_w_i = w_i.sum()
    n = len(y)
    df_effect = k - 1

    if var_equal:
        df_error = n - k
        ss_effect = np.sum(n_i * (m_i - y.mean()) ** 2)
        ss_error = np.sum((n_i - 1) * v_i)
        ms_effect = ss_effect / df_effect
        ms_error = ss_error / df_error
        method = "Assuming homoscedasticity"
    else:
        tmp = np.sum(((1 - (w_i / sum_w_i)) ** 2) / (n_i - 1)) / (k ** 2 - 1)
        df_error = 1 / (3 * tmp)
        m = np.sum(w_i * m_i) / sum_w_i
        ms_effect = np.sum(w_i * (m_i - m) ** 2)
        ms_error = df_effect * (1 + 2 * (k - 2) * tmp)
        ss_effect = ms_effect * df_effect
        ss_error = ms_error * df_error
        method = "Assuming heteroscedasticity"

    ss_total = ss_effect + ss_error
    statistic = ms_effect / ms_error
    p = stats.f.sf(statistic, df_effect, df_error)

    etasq = ss_effect / ss_total
    partial_etasq = ss_effect / (ss_effect + ss_error)
    omegasq = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    partial_omegasq = (df_effect * (ms_effect - ms_error)) / (df_effect * ms_effect + (n - df_effect) * ms_error)
    cohens_f = np.sqrt(etasq / (1 - etasq))
    lambda_ = cohens_f * (df_effect + df_error + 1)
    power = stats.ncf.sf(stats.f.isf(0.05, df_effect, df_error), df_effect, df_error, lambda_)

    result = pd.DataFrame([{
        "formula": formula,
        "method": method,
        "ss_effect": ss_effect,
        "ss_error": ss_error,
        "ms_effect": ms_effect,
        "ms_error": ms_error,
        "etasq": etasq,
        "partial.etasq": partial_etasq,
        "omegasq": omegasq,
        "partial.omegasq": partial_omegasq,
        "cohens.f": cohens_f,
        "power": power,
        "statistic": statistic,
        "df_effect": df_effect,
        "df_error": df_error,
        "p": p
    }])

    return result

compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=True)
compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=False)
##########################################################################################
# GAMES HOWELL
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations

def compute_games_howell(y, x, digits=4):
    # Remove missing values
    mask = ~pd.isnull(x) & ~pd.isnull(y)
    x = pd.Series(x)[mask].astype('category')
    y = pd.Series(y)[mask]

    group_labels = x.cat.categories
    groups = {level: y[x == level] for level in group_labels}

    n = pd.Series({level: len(g) for level, g in groups.items()})
    means = pd.Series({level: g.mean() for level, g in groups.items()})
    variances = pd.Series({level: g.var(ddof=1) for level, g in groups.items()})
    df_total = sum(n) - len(n)
    descriptives = pd.DataFrame({'n': n, 'mean': means, 'variance': variances}).round(digits)

    pair_names = [f"{a}:{b}" for a, b in combinations(group_labels, 2)]

    t_vals, df_vals, p_vals = [], [], []

    for a, b in combinations(group_labels, 2):
      na, nb = n[a], n[b]
      ma, mb = means[a], means[b]
      va, vb = variances[a], variances[b]
      diff = abs(ma - mb)
      se = np.sqrt(va / na + vb / nb)

      t = diff / se
      df_num = (va / na + vb / nb) ** 2
      df_denom = ((va / na) ** 2 / (na - 1)) + ((vb / nb) ** 2 / (nb - 1))
      df_ = df_num / df_denom

      p = stats.t.sf(t * np.sqrt(2), df_)*2
      t_vals.append(t)
      df_vals.append(df_)
      p_vals.append(p)

    result_df = pd.DataFrame({'t': t_vals,'df': df_vals,'p': p_vals},index=pair_names).round(digits)

    return {"descriptives": descriptives, "posthoc": result_df}


compute_games_howell(y=df_blood_pressure["bp_before"], x=df_blood_pressure["agegrp"])
##########################################################################################
# ETA PARTIAL ETA OMEGA PARTIAL OMEGA FOR AOV
##########################################################################################
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def compute_aov_es(model, ss="I"):
    # Get the model dataframe
    model_df = model.model.data.frame
    n_total = model_df.shape[0]

    # Select appropriate summary table
    if ss == "I":
        ss1 = anova_lm(model, typ=1)
        ss1 = ss1.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'mean_sq': 'Mean Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss1 = ss1[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        ss1 = ss1.reset_index()

        summary_aov = ss1
    elif ss == "II":
        ss2 = anova_lm(model, typ=2)
        ss2['Mean Sq'] = ss2['sum_sq'] / ss2['df']
        ss2 = ss2.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss2 = ss2[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        ss2 = ss2.reset_index()

        summary_aov = ss2
        
    elif ss == "III":
        ss3 = anova_lm(model, typ=3)
        ss3['Mean Sq'] = ss3['sum_sq'] / ss3['df']
        ss3 = ss3.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss3 = ss3[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        ss3 = ss3.reset_index()

        order = ['Treatment', 'Residual', 'Intercept']
        ss3 = ss3.set_index('index').loc[order].reset_index()
        summary_aov = ss3.iloc[0:].copy()
        intercept = ss3.iloc[2:3]
    
    residual_row = summary_aov[summary_aov["index"].str.contains("Residual")].index[0]
    ms_effect = summary_aov.loc[:residual_row - 1, 'Mean Sq'].values
    ms_error = summary_aov.loc[residual_row, 'Mean Sq']
    df_effect = summary_aov.loc[:residual_row - 1, 'Df'].values
    df_error = summary_aov.loc[residual_row, 'Df']
    ss_effect = summary_aov.loc[:residual_row - 1, 'Sum Sq'].values
    ss_error = summary_aov.loc[residual_row, 'Sum Sq']
    ss_total = np.full_like(ss_effect, np.sum(summary_aov.loc[:, 'Sum Sq']), dtype=np.float64)

    omega = np.abs((ss_effect - df_effect * ms_error) / (ss_total + ms_error))
    partial_omega = np.abs((df_effect * (ms_effect - ms_error)) / (ss_effect + (n_total - df_effect) * ms_error))
    eta = ss_effect / ss_total
    partial_eta = ss_effect / (ss_effect + ss_error)
    cohens_f = np.sqrt(partial_eta / (1 - partial_eta))
    epsilon = (ss_effect - df_effect * ms_error) / ss_total

    comparison_names = summary_aov.loc[:residual_row - 1, 'index'].str.strip()

    result = pd.DataFrame({
        'call': str(model.model.formula),
        'ss': ss,
        'comparisons': comparison_names,
        'etasq': eta,
        'partial_etasq': partial_eta,
        'omegasq': omega,
        'partial_omegasq': partial_omega,
        'epsilonsq': epsilon,
        'cohens_f': cohens_f
    })

    summary_aov = summary_aov.rename(columns={'index': 'comparisons'})
    summary_aov['call'] = str(model.model.formula)
    summary_aov['ss'] = ss

    # if ss == "III":
    #     intercept['call'] = str(model.model.formula)
    #     intercept['ss'] = ss
    #     summary_aov = pd.concat([intercept.rename(columns={'index': 'comparisons'}), summary_aov], ignore_index=True)

    merged = pd.merge(summary_aov, result, how='outer', on=['call', 'ss', 'comparisons'])
    return merged

one_way_between=smf.ols('uptake~Treatment', data=df_co2).fit()
factorial_between=smf.ols('uptake~Treatment*Type', data=df_co2).fit()
compute_aov_es(model=one_way_between,ss="I")
compute_aov_es(model=one_way_between,ss="II")
compute_aov_es(model=one_way_between,ss="III")
compute_aov_es(model=factorial_between,ss="I")
compute_aov_es(model=factorial_between,ss="II")
compute_aov_es(model=factorial_between,ss="III")






