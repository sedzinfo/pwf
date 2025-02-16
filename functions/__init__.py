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
sys.path.insert(1,'/opt/pyrepo/functions/')
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
personality=pd.read_csv('/opt/pyrepo/data/personality.csv')
titanic=pd.read_csv('/opt/pyrepo/data/titanic.csv')


check(df)
