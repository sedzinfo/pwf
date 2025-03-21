# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:36 2018
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import os
import sys

path_script = os.getcwd()
# path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if(path_script.find('functions')==-1):
  path_script=path_script+"\\GitHub\\pwf\\functions"
path_root=path_script.replace('\\functions', '')
os.chdir(path_script)

sys.path.insert(1,path_script)
from plotnine import ggplot, aes, scale_x_discrete, scale_y_discrete
from plotnine import geom_tile, geom_text, scale_fill_gradient2, theme_bw, theme, ggsave
from plotnine import element_text, element_blank
import matplotlib.pyplot as plt
import pandas as pd
##########################################################################################
# PLOT CORRPLOT
##########################################################################################
def plot_corrplot(df,base_size=15,title=""):
    """
    Generate a correlation plot using `plotnine` (ggplot2-like syntax) to visualize correlations in a given DataFrame.

    This function takes a DataFrame with numerical values and generates a heatmap-style correlation plot. 
    Each correlation value is represented by a tile, with the color gradient showing the correlation 
    between the variables. The function uses `ggplot2`-style syntax from the `plotnine` library.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing numerical values for correlation. The DataFrame must have at least 2 columns, 
        and the columns' names will be used as the variables for the plot. The correlation plot will be generated 
        between each pair of columns.

    base_size : int, optional, default=15
        The base size of the text in the plot. This controls the font size of the correlation values displayed 
        within each tile.

    Returns:
    --------
    plotnine.ggplot.ggplot
        A `ggplot` object representing the correlation plot. You can use `plt.show()` to display it.

    Example:
    --------
    import pandas as pd
    from plot_corrplot import plot_corrplot

    df = pd.DataFrame({
        'A': [1, 0.8, 0.6],
        'B': [0.8, 1, 0.7],
        'C': [0.6, 0.7, 1]
    })

    plot_corrplot(df)

    This will generate a plot showing the correlation between columns 'A', 'B', and 'C'.
    """

    row_count=len(df)
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
        + labs(title=title,caption=f"Number of Variables: {row_count}")
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
df=personality=pd.read_csv(path_root+"/data/personality.csv")
gp=plot_corrplot(personality.iloc[:,1:20].corr(),base_size=5,title="Personality")
gp.show()

output_path = os.path.join(path_root, "output")
os.makedirs(output_path, exist_ok=True)

gp.save(filename="correlation.png", path=output_path, width=20, height=20, units="cm", dpi=1200)


