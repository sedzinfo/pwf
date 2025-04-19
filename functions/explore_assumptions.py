# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:36 2018
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd
##########################################################################################
# LOAD
##########################################################################################
from plotnine import ggplot,aes,geom_histogram,geom_qq,geom_qq_line,geom_boxplot
from plotnine import theme_bw,theme,labs,facet_wrap,coord_flip,ggsave
from scipy import stats
# personality=pd.read_csv(path_root+"\\data\\personality.csv")
# titanic=pd.read_csv(path_root+"\\data\\titanic.csv")
##########################################################################################
# PLOT HISTOGRAM
##########################################################################################
def plot_histogram(df,base_size=10,title="Histogram",bins=10):
    df=df.melt()
    gp=(ggplot(data=df,mapping=aes(x="value"))
        + geom_histogram(bins=bins,na_rm=True,colour="#FFFFFF")
        + labs(title=title,y="Frequency",x="")
        + theme_bw(base_size=base_size)
        + theme(subplots_adjust={'hspace':0.1,'wspace':0.1})
        + facet_wrap('~variable',scales='free'))
    return gp
# ph0=plot_histogram(personality.iloc[:,0:10])
# ph1=plot_histogram(df=pd.DataFrame(np.random.normal(loc=0,scale=1,size=1000000)),bins=100)
# ph0.show()
# ph1.show()
##########################################################################################
# PLOT QQ
##########################################################################################
def plot_qq(df,base_size=10,title="QQ"):
    df=df.melt()
    gp=(ggplot(data=df)
        + geom_qq(aes(sample="value"))
        + geom_qq_line(aes(sample="value"))
        + labs(title=title)
        + theme_bw(base_size=base_size)
        + theme(subplots_adjust={'hspace':0.1,'wspace':0.1})
        + facet_wrap('~variable',scales='free'))
    return gp
# pqq=plot_qq(personality.iloc[:,0:10])
# pqq.show()
##########################################################################################
# PLOT NORMALITY ASSUMPTIONS BASE PLOT
##########################################################################################
def plot_boxplot(df,base_size=10,title="Boxplot"):
    df=df.select_dtypes(include=np.number)
    df=df.melt()
    gp=(ggplot(df,aes(x="variable",y="value"))
      + geom_boxplot()
      + labs(title=title,x="",y="")
      + coord_flip()
      + theme_bw(base_size=base_size)
      + theme(subplots_adjust={'hspace':0.1,'wspace':0.1}))
    return gp
# pbp0=plot_boxplot(personality)
# pbp1=plot_boxplot(titanic)
# pbp0.show()
# pbp1.show()
##########################################################################################
# PLOT NORMALITY ASSUMPTIONS BASE PLOT
##########################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def plot_normality_diagnostics(df,breaks='sturges',title='',file=None,w=10,h=10):
    def string_aes(name):
        return str(name)
    
    data_name=df.columns if isinstance(df,pd.DataFrame) else 'data'
    df=pd.DataFrame(df)
    df=df.select_dtypes(include=[np.number])
    if df.shape[1]==1:
        df.columns=[data_name]

    num_plots=len(df.columns)
    fig,axes=plt.subplots(num_plots,4,figsize=(w,h * num_plots))

    for i,col in enumerate(df.columns):
        vector=df[col].dropna()

        if len(vector) > 2 and np.var(vector) != 0:
            sns.histplot(vector,bins=breaks,kde=False,ax=axes[i,0])
            axes[i,0].set_title('Histogram')

            sns.kdeplot(vector,ax=axes[i,1])
            axes[i,1].set_title('Density Function')

            sns.boxplot(x=vector,ax=axes[i,2])
            axes[i,2].set_title('Boxplot')

            stats.probplot(vector,dist="norm",plot=axes[i,3])
            axes[i,3].get_lines()[1].set_color('red')
            axes[i,3].set_title('QQ Plot')

            axes[i,0].set_xlabel('')
            axes[i,1].set_xlabel('')
            axes[i,2].set_xlabel('')

            fig.suptitle(title,y=1.05)
            plt.subplots_adjust(hspace=0.4)

            if file:
                fig.savefig(file)
        else:
            print(f"Graph not produced for {col} due to sample size")

    plt.show()

# Example usage:
# vector=np.random.normal(size=1000)
# df=pd.DataFrame(np.random.normal(size=(1000,2)),columns=['A','B'])
# plot_normality_diagnostics(vector,title="Normality Diagnostics",breaks=30,file="normality_diagnostics.png")
# plot_normality_diagnostics(df,title="Normality Diagnostics")
##########################################################################################
# OUTLIERS
##########################################################################################
import numpy as np
import pandas as pd

def outlier_summary(vector):
    zvariable=(vector - np.mean(vector)) / np.std(vector)
    outlier95=np.abs(zvariable) >= 1.96
    outlier99=np.abs(zvariable) >= 2.58
    outlier999=np.abs(zvariable) >= 3.29
    ncases=len(vector[~np.isnan(zvariable)])
    percent95=round(100 * len(outlier95[outlier95==True]) / ncases,2)
    percent99=round(100 * len(outlier99[outlier99==True]) / ncases,2)
    percent999=round(100 * len(outlier999[outlier999==True]) / ncases,2)
    result=pd.DataFrame({
        'abs_z_1.96': [f"{percent95}%"],
        'abs_z_2.58': [f"{percent99}%"],
        'abs_z_3.29': [f"{percent999}%"]
    })
    return result

# Example usage:
# vector=np.random.normal(size=1000)
# outlier_summary(vector)
##########################################################################################
# OUTLIERS
##########################################################################################
import numpy as np

def remove_outliers(vector,probs=[0.25,0.75],na_rm=True,**kwargs):
    qnt=np.quantile(vector,probs)
    H=1.5 * (qnt[1] - qnt[0])
    y=np.copy(vector)
    if na_rm:
        y[vector < (qnt[0] - H)]=np.nan
        y[vector > (qnt[1] + H)]=np.nan
    else:
        y=vector[(vector >= (qnt[0] - H)) & (vector <= (qnt[1] + H))]
    return y

# Example usage:
# vector=np.random.normal(size=1000)
# clean_vector=remove_outliers(vector)
# print(clean_vector)



















