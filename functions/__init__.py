#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:20:31 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
import rpy2.robjects as robjects
get_path = robjects.r('rstudioapi::getActiveDocumentContext()$path')
file_path = str(get_path[0]).replace(os.path.basename(str(get_path[0])),"").rstrip("/")
file_directory = os.path.dirname(file_path) or os.getcwd()
sys.path.insert(1,file_path)

import pandas as pd
import numpy as np
import xlsxwriter
from plotnine import *

from functions import *
from functions_check_dataframe import *
from functions_excel import *
##########################################################################################
# DATA
##########################################################################################
df_blood_pressure=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
df_admission=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/admission.csv")
df_automotive=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/automotive_data.csv")
df_blood_pressure=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
df_crop_yield=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/crop_yield.csv")
df_difficile=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")
df_insurance=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/insurance.csv")
df_responses=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/responses.csv")
df_responses_state=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/responses_state.csv")
df_sexual_comp=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/sexual_comp.csv")
df_personality=pd.read_csv(file_directory+"/data/personality.csv")
df_titanic=pd.read_csv(file_directory+"/data/titanic.csv")
