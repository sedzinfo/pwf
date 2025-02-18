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
df=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
personality=pd.read_csv(file_directory+"/data/personality.csv")
titanic=pd.read_csv(file_directory+"/data/titanic.csv")

check(df)
