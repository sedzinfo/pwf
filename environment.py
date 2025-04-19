import os
cwd = os.getcwd()
print(cwd)


import os
import sys
path_script = os.getcwd()
path_root = path_script.replace('/functions', '')

sys.path.insert(1,path_root)
from functions import *


for mod in sys.modules:
    if 'functions' in mod:
        print(mod)


'functions' in sys.modules
print(dir(functions))
print(functions.__file__)
