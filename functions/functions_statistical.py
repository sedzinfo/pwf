##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd

path_script = os.getcwd()
path_root = path_script.replace('\\functions', '')

sys.path.insert(1,file_path)
from __init__ import *
from functions import *
##########################################################################################
# 
##########################################################################################
import numpy as np
from scipy.stats import norm

def compute_standard(vector,mean=0,sd=1,type="z",input="non_standard"):
    """
    Compute standard scores for a given vector.

    Parameters:
    -----------
    vector : list or numpy array
        Input data.
    mean : float, optional
        Mean for "uz" type (default is 0).
    sd : float, optional
        Standard deviation for "uz" type (default is 1).
    type : str, optional
        Type of standardization (default is "z").
        Options: "z", "uz", "sten", "t", "stanine", "center", "center_reversed",
                 "percent", "scale_zero_one", "normal_density", "cumulative_density", "all".
    input : str, optional
        Type of input data (default is "non_standard").
        Options: "standard" (z-scores), "non_standard" (raw scores).

    Returns:
    --------
    result : numpy array or pandas DataFrame
        Standardized scores based on the specified type.
    """
    vector=np.array(vector)

    if input == "non_standard":
        z = (vector - np.nanmean(vector))/np.nanstd(vector)
    elif input == "standard":
        z = vector

    if type=="z":
        result=z
    elif type=="uz":
        result=vector*sd+mean
    elif type=="sten":
        result=np.round((z*2)+5.5,0)
        result[result<1]=1
        result[result>10]=10
    elif type=="t":
        result=(z*10)+50
    elif type=="stanine":
        result=(z*2)+5
        result[result<1]=1
        result[result>9]=9
        result=np.round(result,0)
    elif type=="center":
        result=vector-np.nanmean(vector)
    elif type=="center_reversed":
        result=np.nanmean(vector)-vector
    elif type=="percent":
        result=(vector/np.nanmax(vector))*100
    elif type=="scale_zero_one":
        result=(vector-np.nanmin(vector))/(np.nanmax(vector)-np.nanmin(vector))
    elif type=="normal_density":
        result=(1/(np.sqrt(sd*np.pi)))*np.exp(-0.5*((vector-mean)/sd)**2)
    elif type=="cumulative_density":
        result=np.cumsum(vector)
    elif type=="all":
        import pandas as pd
        mydata=pd.DataFrame({"score": vector})
        mydata["z"]=compute_standard(mydata["score"],type="z",input=input)
        mydata["sten"]=compute_standard(mydata["score"],type="sten",input=input)
        mydata["t"]=compute_standard(mydata["score"],type="t",input=input)
        mydata["stanine"]=compute_standard(mydata["score"],type="stanine",input=input)
        mydata["percent"]=compute_standard(mydata["score"],type="percent",input=input)
        mydata["scale_0_1"]=compute_standard(mydata["score"],type="scale_zero_one",input=input)
        result=mydata.sort_values(by="z")
    else:
        raise ValueError("Invalid type specified.")
    return result

vector=list(range(100))

# Compute z-scores
z_scores = compute_standard(vector, type="z")
print("Z-scores:", z_scores)

# Compute sten scores
sten_scores = compute_standard(vector, type="sten")
print("Sten scores:", sten_scores)

# Compute all scores
all_scores = compute_standard(vector, type="all",input="non_standard")
print(all_scores)




