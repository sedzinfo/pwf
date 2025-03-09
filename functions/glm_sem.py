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


import semopy
import pandas as pd
desc = semopy.examples.political_democracy.get_model()
print(desc)

data = semopy.examples.political_democracy.get_data()
print(data.head())

mod = semopy.Model(desc)
res = mod.fit(data)
print(res)

ins = mod.inspect()
print(ins)



from semopy import Model
model = Model(desc)
opt_res = model.fit(data)
estimates = model.inspect()
