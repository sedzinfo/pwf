# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:45:39 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# SCREE PLOT
##########################################################################################
import pandas as pd
import numpy as np
from plotnine import (ggplot,aes,geom_hline,geom_line,geom_point,theme_bw,
                      scale_x_continuous,labs,annotate,theme,element_blank,
                      element_text)
def plot_scree(df,base_size=15,title="",color=("#5F2C91","#5E912C")):
    """
    Scree plot displaying the Kaiser and Jolliffe criteria for factor extraction using plotnine (ggplot2 for Python).

    Parameters:
    - df: pandas DataFrame
    - base_size: base font size
    - title: plot title
    - color: tuple of colors for lines and point outlines

    Returns:
    - A plotnine.ggplot object representing the scree plot.
    
    Example:
    --------
    scree_plot=plot_scree(df,base_size=15,title="")
    scree_plot.show()
    
    """
    # Calculate eigenvalues of the correlation matrix
    corr_matrix=df.corr(method='pearson')
    eigenvalues=np.linalg.eigvals(corr_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues=np.sort(eigenvalues)[::-1]
    
    # Create a DataFrame for plotting
    eigenvalues_df=pd.DataFrame({
        'x': np.arange(1,len(eigenvalues) + 1),
        'eigenvalues': eigenvalues
    })
    
    # Kaiser and Jolliffe criteria
    kaiser=np.sum(eigenvalues>1)
    jolliffe=np.sum(eigenvalues>0.7)
    
    # Create the scree plot using plotnine
    plot=(ggplot(eigenvalues_df,aes(x='x',y='eigenvalues')) +
            geom_hline(yintercept=1,color=color[0]) +
            geom_hline(yintercept=0.7,color=color[1]) +
            geom_line(color=color[0]) +
            geom_point(size=base_size / 4,color=color[0]) +
            scale_x_continuous(breaks=eigenvalues_df['x'].tolist()) +
            theme_bw(base_size=base_size) +
            labs(x="Index",y="Eigenvalue",title=f"Scree plot {title}") +
            annotate("text",
                     x=eigenvalues_df['x'].max(),
                     y=eigenvalues_df['eigenvalues'].max(),
                     label=f"Top line: Kaiser criterion: {kaiser}\nBottom line: Jolliffe criterion: {jolliffe}",
                     ha='right',va='top',size=base_size) +
            theme(legend_title=element_blank(),
                  legend_position='bottom',
                  axis_title_x=element_blank(),
                  text=element_text(size=base_size)))
                  
    return plot
  
# scree_plot=plot_scree(df_personality,base_size=15,title="")
# scree_plot.show()
##########################################################################################
# REPORT EFA
##########################################################################################
from factor_analyzer import FactorAnalyzer,calculate_bartlett_sphericity,calculate_kmo
def report_efa(df,output_file,n_factors=3,rotation='promax',method='minres',
               use_smc=True,is_corr_matrix=False,bounds=(0.005,1),
               impute='median',svd_method='randomized',rotation_kwargs=None):
    
    """
    Perform Exploratory Factor Analysis (EFA) on a given dataset and generate relevant statistics,
    plots,and an Excel report.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the dataset for EFA.

    n_factors : int,default=3
        Number of factors to extract.

    rotation : str,default='promax'
        Rotation method to apply in the factor analysis.

    method : str,default='minres'
        Method used to perform factor analysis.

    use_smc : bool,default=True
        If True,use squared multiple correlations (SMC) as initial communality estimates.

    is_corr_matrix : bool,default=False
        Indicates if the input dataset is already a correlation matrix.

    bounds : tuple,default=(0.005,1)
        Bounds for the communalities during factor extraction.

    impute : str,default='median'
        Strategy to handle missing values in the dataset.

    svd_method : str,default='randomized'
        Method for Singular Value Decomposition (SVD).

    rotation_kwargs : dict,optional
        Additional arguments for the rotation method.

    output_file : str,
        Path to the output Excel file where the results will be saved.

    Returns:
    --------
    result : dict
        Dictionary containing:
            - Eigen_Communality_Uniqueness: DataFrame with eigenvalues,communalities,and uniqueness.
            - Loadings: Factor loadings DataFrame.
            - Variance: DataFrame with factor variance statistics.
            - Residual Correlations: Reproduced correlations from factor analysis.
            - Correlations: Original correlation matrix.
            - Residuals: Residual correlations between actual and reproduced matrices.

    Notes:
    ------
    - Generates a scree plot for visualizing eigenvalues.
    - Outputs key residual statistics,such as RMSR and proportions of large residuals.
    - Saves multiple sheets in an Excel file summarizing analysis results.

    Example:
    --------
    result=report_efa(df=my_data,n_factors=5,rotation='varimax',method='ml')
    """

    comments_s1={'A1': "The names are not properly sorted",
                 'B1': "Eigenvector 1",
                 'C1': "Eigenvector 2",
                 'D1': "Communality",
                 'E1': "Uniqueness"}


    index=list(range(1,df.shape[1]+1))
    
    scree_plot=plot_scree(df,base_size=15,title="")
    scree_plot.show()
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
    factor_variance=fa.get_factor_variance()
    params=fa.get_params()
    loadings=model.loadings_
    
    cut_off=0.3
    df_loadings=pd.DataFrame(loadings,
                             index=df.columns,
                             columns=[f'Factor {i+1}' 
                             for i in range(loadings.shape[1])])

    loadings_cut=df_loadings.apply(lambda x: x.map(lambda v: v if abs(v) > cut_off else ''))

    factor_variance=model.get_factor_variance()
    uniquinesses=model.get_uniquenesses()
    sufficiency=model.sufficiency(df.shape[0])

    df_eigenvalues=pd.DataFrame({"names":df.columns,"eigen1":eigenvalues[0],"eigen2":eigenvalues[1]},index=df.columns)
    df_communalities=pd.DataFrame({"names":df.columns,"communality":communalities},index=df.columns)
    df_uniqueness=pd.DataFrame({"names":df.columns,"uniquiness":uniquinesses},index=df.columns)
    df_factor_variance=pd.DataFrame(factor_variance)
    correlations=df.corr()
    correlations_reproduced=pd.DataFrame(np.dot(fa.loadings_,fa.loadings_.T),
                                       index=correlations.index,
                                       columns=correlations.columns)
    correlations_residual=correlations.abs()-correlations_reproduced.abs()
    
    upper_triangle_residuals=correlations_residual.where(np.triu(np.ones(correlations_residual.shape),k=1).astype(bool))
    lower_triangle_residuals=correlations_residual.where(np.tril(np.ones(correlations_residual.shape),k=-1).astype(bool))
    lower_triangle_array=lower_triangle_residuals.to_numpy().flatten()
    lower_triangle_array=lower_triangle_array[~np.isnan(lower_triangle_array)]
    n_large_residuals=(np.abs(lower_triangle_array)>0.05).sum()
    
    prop_large_resid=n_large_residuals/len(correlations_residual)
    rmsr=np.sqrt((lower_triangle_array**2).mean())
    
    residual_statistics=pd.DataFrame({"residual_statistics":["Root Mean Squared Residual",
                                                             "Number of absolute residuals > 0.05",
                                                             "Proportion of absolute residuals > 0.05"],
                                      "value":[rmsr,n_large_residuals,prop_large_resid],
                                      "critical":[np.nan,np.nan,.5],
                                      "formula":["sqrt(mean(residuals^2))",
                                      "abs(residuals)>0.05",
                                      "numberLargeResiduals/nrow(residuals)"]})
    
    eigencu=pd.concat([df_eigenvalues.set_index("names"),
                       df_communalities.set_index("names"),
                       df_uniqueness.set_index("names")],
                       axis=1)
                         
    writer=pd.ExcelWriter(path=output_file,engine='xlsxwriter')
    matrix_excel(df=eigencu,writer=writer,sheetname="Eigen Communality Uniqueness",comments=comments_s1)
    matrix_excel(df=df_loadings,writer=writer,sheetname="Loadings",comments=None)
    matrix_excel(df=correlations,writer=writer,sheetname="Correlation",comments=None)
    matrix_excel(df=correlations_reproduced,writer=writer,sheetname="Correlation Reproduced",comments=None)
    matrix_excel(df=correlations_residual,writer=writer,sheetname="Correlation Residual",comments=None)
    matrix_excel(df=df_factor_variance,writer=writer,sheetname="Factor Variance",comments=None)
    generic_format_excel(df=residual_statistics,writer=writer,sheetname="Residual Statistics",comments=None)
    
    writer._save()
    writer.close()
    
    result={"Eigen_Communality_Uniqueness":eigencu,
            "Loadings":df_loadings,
            "Variance":df_factor_variance,
            "Residual Correlations":correlations_reproduced,
            "Correlations":correlations,
            "Residuals":correlations_residual}

    return result



# result=report_efa(df=df_personality,n_factors=10,rotation='promax',method='minres',
#                   use_smc=True,is_corr_matrix=False,bounds=(0.005,1),
#                   impute='median',svd_method='randomized',rotation_kwargs=None,
#                   output_file=path_script+'/output/efa.xlsx')
##########################################################################################
# REPORT EFA
##########################################################################################
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_bar, facet_wrap, coord_flip, labs, theme_minimal, scale_fill_gradient2
from factor_analyzer import FactorAnalyzer

# Sample data: Generate a correlation matrix and synthetic dataset
# correlation_matrix = np.array([
#     [1, 0.8, 0.8, 0.1, 0.1, 0.1],
#     [0.8, 1, 0.8, 0.1, 0.1, 0.1],
#     [0.8, 0.8, 1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 1, 0.8, 0.8],
#     [0.1, 0.1, 0.1, 0.8, 1, 0.8],
#     [0.1, 0.1, 0.1, 0.8, 0.8, 1]
# ])

# Generate synthetic data based on the correlation matrix
# data = np.random.multivariate_normal(mean=np.zeros(correlation_matrix.shape[0]),
#                                      cov=correlation_matrix, size=10000)
# df = pd.DataFrame(data, columns=[f"Var{i+1}" for i in range(correlation_matrix.shape[0])])
# 
# Perform factor analysis
# fa = FactorAnalyzer(n_factors=2, rotation="oblimin", method="principal")
# fa.fit(df)

# Extract loadings
# loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=["Factor1", "Factor2"])


def plot_loadings_bar(loadings):
    """
    Plot factor loadings as bar plots using plotnine (ggplot2 for Python).
    
    Parameters:
    - loadings: pandas DataFrame containing factor loadings.
    
    Returns:
    - A plotnine.ggplot object representing the factor loadings bar plot.
    
    Example:
    --------
    plot = plot_loadings_bar(loadings)
    plot.show()
    """
    loadings = loadings.reset_index().melt(id_vars=["index"], var_name="Factor", value_name="Loading")
    
    # Create the factor loadings bar plot using plotnine
    plot=(ggplot(loadings,aes(x="index",y="abs(Loading)",fill="Loading"))+
          geom_bar(stat="identity")+
          facet_wrap("~Factor",nrow=1)+
          coord_flip()+
          labs(title="Factor Loadings",x="Variables",y="Loading")+
          scale_fill_gradient2(low="#5E912C",mid="white",high="#5F2C91",midpoint=0)+
          theme_minimal())
    
    return plot


# result=plot_loadings_bar(loadings=df_loadings)
# result.show()







































