# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:36 2018
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
sys.path.insert(1,'/opt/pyrepo/functions/')
from __init__ import *
from plotnine import ggplot, aes, geom_histogram, geom_qq, geom_qq_line, geom_boxplot
from plotnine import theme_bw, theme, labs, facet_wrap, coord_flip, ggsave
from scipy import stats
##########################################################################################
# PLOT HISTOGRAM
##########################################################################################
def plot_histogram(df,base_size=10,title="Histogram"):
    histp=list()
    for collumn_name in list(df.columns):
        gp=(ggplot(data=df,mapping=aes(x=collumn_name))
        + geom_histogram(bins=df[collumn_name].nunique(),binwidth=1,na_rm=True,colour="#FFFFFF")
        + labs(title=title,y="Frequency",x="")
        + theme_bw(base_size=base_size)
        + theme(subplots_adjust={'hspace':0.1,'wspace':0.1}))
        histp.append(gp)
    return histp
def plot_histogram(df,base_size=10,title="Histogram",bins=10):
    df=df.melt()
    gp=(ggplot(data=df,mapping=aes(x="value"))
        + geom_histogram(bins=bins,na_rm=True,colour="#FFFFFF")
        + labs(title=title,y="Frequency",x="")
        + theme_bw(base_size=base_size)
        + theme(subplots_adjust={'hspace':0.1,'wspace':0.1})
        + facet_wrap('~variable',scales='free'))
    return gp
plot_histogram(personality.iloc[:,0:4])
plot_histogram(df=pd.DataFrame(np.random.normal(loc=0,scale=1,size=1000000)),bins=100)
##########################################################################################
# PLOT QQ
##########################################################################################
def plot_qq(df,base_size=10,title="QQ"):
    qqp=list()
    for collumn_name in list(df.columns):
        gp=(ggplot(data=df)
        + geom_qq(aes(sample=collumn_name))
        + geom_qq_line(aes(sample=collumn_name))
        + labs(title=title)
        + theme_bw(base_size=base_size)
        + theme(subplots_adjust={'hspace':0.1,'wspace':0.1}))
        qqp.append(gp)
    return qqp
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
# plot_qq(personality.iloc[:,0:4])
##########################################################################################
# PLOT BOXPLOT
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
# print(plot_boxplot(personality))
# print(plot_boxplot(titanic))











