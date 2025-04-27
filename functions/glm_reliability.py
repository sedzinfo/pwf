#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:32:36 2025
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd

path_script=os.getcwd()
path_root=path_script.replace('\\functions', '')

sys.path.insert(1,file_path)
from __init__ import *
from functions import *
##########################################################################################
# LOAD
##########################################################################################
import pingouin as pg
