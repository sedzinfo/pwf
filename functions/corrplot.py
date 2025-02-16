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
sys.path.insert(1,'C:/Users/dzach/Documents/GitHub/pwf/functions')
from __init__ import *
from plotnine import ggplot, aes, scale_x_discrete, scale_y_discrete
from plotnine import geom_tile, geom_text, scale_fill_gradient2, theme_bw, theme, ggsave
from plotnine import element_text, element_blank
##########################################################################################
# PLOT CORRPLOT
##########################################################################################
def plot_corrplot(df,base_size=15):
    df.insert(0,"index",df.columns)
    variables=list(df["index"])
    df=df.melt(id_vars="index")
    df=df.round(2)
    df=df.sort_values(by=["value"],ascending=False)
    gp=(ggplot(df)
        + aes(y="index",x="variable",fill="value")
        + scale_x_discrete(limits=variables)
        + scale_y_discrete(limits=list(reversed(variables)))
        + geom_tile(color="white")
        + geom_text(aes(y="index",x="variable",label="value"),color="black",size=base_size)
        + scale_fill_gradient2(low="#ffbe00",mid="#ffffff",high="#0092ff",limits=[-1,1])
        + theme_bw(base_size=base_size)
        + theme(axis_text_x=element_text(angle=-45,vjust=1,hjust=0),
                axis_title_x=element_blank(),
                axis_title_y=element_blank(),
                axis_ticks_minor=element_blank(),
                panel_grid_minor=element_blank(),
                panel_grid_major=element_blank(),
                panel_border=element_blank(),
                panel_background=element_blank(),
                legend_position="right",
                legend_direction="vertical",
                legend_title=element_blank(),
                legend_text=element_text(hjust=1)))
    return gp
##########################################################################################
# 
##########################################################################################
import rpy2.robjects as robjects
robjects.r('library(rstudioapi)')
script_dir = robjects.r('dirname(rstudioapi::getActiveDocumentContext()$path)')

# plot_corrplot(personality.iloc[:,1:20].corr(),base_size=10)
# ggsave(plot=result,filename="correlation.png",path="/opt/pyrepo/output",width=10,height=10,units="in",dpi=1200)
# ggsave(plot=result,filename="correlation.pdf",path="/opt/pyrepo/output",width=10,height=10,units="in",dpi=1200)
# ggsave(plot=result,filename="correlation.jpg",path="/opt/pyrepo/output",width=10,height=10,units="in",dpi=1200)
# ggsave(plot=result,filename="correlation.svg",path="/opt/pyrepo/output",width=10,height=10,units="in",dpi=1200)







