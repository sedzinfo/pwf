# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:45:39 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
import os

sys.path.insert(1,'C://Users//dzach//Documents//GitHub//pwf//functions')
from __init__ import *
sys.path.insert(1,'/opt/pyrepo/functions/functions.py')
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer,calculate_bartlett_sphericity,calculate_kmo
from sklearn.decomposition import FactorAnalysis
import numpy as np
df=pd.read_csv('C:/Users/dzach/Documents/GitHub/pwf/data/personality.csv')
##########################################################################################
# SCREE PLOT
##########################################################################################
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_hline, geom_line, geom_point, scale_x_continuous,
    theme_bw, labs, annotate, theme, element_blank, element_text
)

def plot_scree(df, base_size=15, title="", color=("#5F2C91", "#5E912C")):
    """
    Scree plot displaying the Kaiser and Jolliffe criteria for factor extraction using plotnine (ggplot2 for Python).

    Parameters:
    - df: pandas DataFrame
    - base_size: base font size
    - title: plot title
    - color: tuple of colors for lines and point outlines

    Returns:
    - A plotnine.ggplot object representing the scree plot.
    """
    # Calculate eigenvalues of the correlation matrix
    corr_matrix = df.corr(method='pearson')
    eigenvalues = np.linalg.eigvals(corr_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Create a DataFrame for plotting
    eigenvalues_df = pd.DataFrame({
        'x': np.arange(1, len(eigenvalues) + 1),
        'eigenvalues': eigenvalues
    })
    
    # Kaiser and Jolliffe criteria
    kaiser = np.sum(eigenvalues > 1)
    jolliffe = np.sum(eigenvalues > 0.7)
    
    # Create the scree plot using plotnine
    plot = (ggplot(eigenvalues_df, aes(x='x', y='eigenvalues')) +
            geom_hline(yintercept=1, color=color[0]) +
            geom_hline(yintercept=0.7, color=color[1]) +
            geom_line(color=color[0]) +
            geom_point(size=base_size / 4, color=color[0]) +
            scale_x_continuous(breaks=eigenvalues_df['x'].tolist()) +
            theme_bw(base_size=base_size) +
            labs(x="Index", y="Eigenvalue", title=f"Scree plot {title}") +
            annotate("text",
                     x=eigenvalues_df['x'].max(),
                     y=eigenvalues_df['eigenvalues'].max(),
                     label=f"Top line: Kaiser criterion: {kaiser}\nBottom line: Jolliffe criterion: {jolliffe}",
                     ha='right', va='top', size=base_size) +
            theme(legend_title=element_blank(),
                  legend_position='bottom',
                  axis_title_x=element_blank(),
                  text=element_text(size=base_size)))
                  
    return plot
  
scree_plot = plot_scree(df, base_size=15, title="")
scree_plot.show()
##########################################################################################
# REPORT EFA
##########################################################################################
def report_efa(df,n_factors=3,rotation='promax',method='minres',
               use_smc=True,is_corr_matrix=False,bounds=(0.005, 1),
               impute='median',svd_method='randomized',rotation_kwargs=None):
    
    chi_square_value,p_value=calculate_bartlett_sphericity(df)
    kmo_all,kmo_model=calculate_kmo(df)
    kmo_variables=pd.DataFrame(list(zip(df.columns.to_list(),kmo_all)),columns=['Name','KMO'])
    
    fa=FactorAnalyzer(n_factors=n_factors,rotation=rotation,method=method,
                      use_smc=use_smc,is_corr_matrix=is_corr_matrix,
                      bounds=bounds,impute=impute,svd_method=svd_method,
                      rotation_kwargs=rotation_kwargs)
    model=fa.fit(df)
    eigenvalues=fa.get_eigenvalues()
    communalities=fa.get_communalities()
    loadings=model.loadings_
    
    cut_off = 0.3
    df_loadings=pd.DataFrame(loadings,
                             index=df.columns,
                             columns=[f'Factor {i+1}' 
                             for i in range(loadings.shape[1])])

    loadings_cut=df_loadings.apply(lambda x: x.map(lambda v: v if abs(v) > cut_off else ''))
    
  
    factor_variance=model.get_factor_variance()
    uniquinesses=model.get_uniquenesses()
    sufficiency=model.sufficiency(df.shape[0])
    
    index=list(range(1,df.shape[1]+1))
    df_eigenvalues=pd.DataFrame({"index":index,"eigen1":eigenvalues[0],"eigen2":eigenvalues[1]})
    df_communalities=pd.DataFrame({"index":index,"communalities":communalities})
    df_uniqueness=pd.DataFrame({"index":index,"uniquinesses":uniquinesses})
    df_loadings.insert(0,"index",index)
    df_factor_variance=pd.DataFrame(factor_variance)
    correlations=df.corr()
    residual_correlations=pd.DataFrame(np.dot(fa.loadings_,fa.loadings_.T))
    residual_correlations.columns=correlations.columns
    residual_correlations.index=correlations.index
    residuals=correlations.abs()-residual_correlations.abs()
    
    return (df_eigenvalues,df_communalities,
            df_uniqueness,df_loadings,df_factor_variance,
            residual_correlations,correlations,residuals)
  
report_efa(df)
##########################################################################################
# SORT LOADINGS
##########################################################################################
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate pandas2ri
pandas2ri.activate()

# Import the psych package from R
psych = importr('psych')

# Sample Data (using a small part of the mtcars dataset for illustration)
data = {
    'F1': [0.7, -0.4, 0.6, -0.8],
    'F2': [0.5, -0.2, 0.8, -0.1],
    'F3': [-0.3, 0.7, -0.1, 0.6]
}
df_loadings = pd.DataFrame(data, index=['Item 1', 'Item 2', 'Item 3', 'Item 4'])

# Perform Factor Analysis
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(df_loadings)

# Get the loadings
loadings = fa.loadings_

# Convert loadings to DataFrame and prepare for R
loadings_df = pd.DataFrame(loadings, 
                           columns=[f'Factor {i+1}' for i in range(loadings.shape[1])],
                           index=df_loadings.index)

# Convert the loadings DataFrame to an R DataFrame
loadings_r = pandas2ri.py2rpy(loadings_df)

# Use the fa.sort function from the psych package in R
sorted_loadings_r = psych.fa_sort(loadings_r)

# Convert the sorted loadings back to a pandas DataFrame
sorted_loadings_df = pandas2ri.rpy2py(sorted_loadings_r)

# Display the sorted and filtered loadings
print("Sorted and Filtered Loadings:")
print(sorted_loadings_df)




