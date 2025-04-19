##########################################################################################
# LOAD
##########################################################################################
import os
import sys
import pandas as pd

path_script = os.getcwd()
# path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if(path_script.find('functions')==-1):
  path_script=path_script+"/GitHub/pwf/functions"
path_root=path_script.replace('/functions', '')
os.chdir(path_script)

sys.path.insert(1,path_script)


files = [f for f in os.listdir(path_script) if os.path.isfile(os.path.join(path_script, f))]
files = [f[:-3] if f.endswith('.py') else f for f in files]
files=sorted(files)
for f in files:
  print(f)

from explore_assumptions import *
from explore_descriptives import *
from functions_cdf import *
from functions_excel import *
from functions_generic import *
from functions_statistical import *
from functions_timestamp import *
from functions_train_test import *
from glm_efa import *
from glm_means import *
from glm_one_way_anova import *
from glm_reliability import *
from glm_sem import *
from helper import *
from plot_corrplot import *



##########################################################################################
# DATA
##########################################################################################
df_admission=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/admission.csv")
df_automotive=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/automotive_data.csv")
df_blood_pressure=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
df_crop_yield=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/crop_yield.csv")
df_difficile=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")
df_insurance=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/insurance.csv")
df_responses=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/responses.csv")
df_responses_state=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/responses_state.csv")
df_sexual_comp=pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/sexual_comp.csv")
df_personality=pd.read_csv(path_root+"/data/personality.csv")
df_titanic=pd.read_csv(path_root+"/data/titanic.csv")


##########################################################################################
# DELETE CACHE
##########################################################################################
import shutil

for root, dirs, files in os.walk(path_script):
    for d in dirs:
        if d == "__pycache__":
            pycache_path = os.path.join(root, d)
            shutil.rmtree(pycache_path)
            print(f"Deleted: {pycache_path}")

for root, dirs, files in os.walk(path_script):
    print("Root:", root)
    print("Dirs:", dirs)
    print("Files:", files)
    print("-" * 40)


        
