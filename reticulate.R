################################################################################
# LOAD
################################################################################
rstudioapi::restartSession()
library(reticulate)
directory<-paste0(dirname(rstudioapi::getActiveDocumentContext()$path),"/")
# reticulate::py_run_string(paste0("path_script = '", directory, "'"))

system2("python", "--version")
system2("R", "--version")
reticulate::py_config()

# reticulate::virtualenv_remove(paste0(directory,"venvironment/"))
# reticulate::virtualenv_create(paste0(directory,"venvironment/"))
reticulate::use_virtualenv(paste0(directory,"venvironment/"))

reticulate::py_install("numpy",pip=TRUE)
reticulate::py_install("pandas",pip=TRUE)
reticulate::py_install("scikit-learn",pip=TRUE)
reticulate::py_install("dash",pip=TRUE)
reticulate::py_install("yfinance",pip=TRUE)
reticulate::py_install("get_all_tickers",pip=TRUE)
reticulate::py_install("linkedin",pip=TRUE)
reticulate::py_install("rpy2-rinterface",pip=TRUE)
reticulate::py_install("rpy2-robjects",pip=TRUE)
reticulate::py_install("rpy2",pip=TRUE)
reticulate::py_install("plotnine",pip=TRUE)
reticulate::py_install("matplotlib",pip=TRUE)
reticulate::py_install("seaborn",pip=TRUE)
reticulate::py_install("xlsxwriter",pip=TRUE)
reticulate::py_install("researchpy",pip=TRUE)
reticulate::py_install("semopy",pip=TRUE)
reticulate::py_install("graphviz",pip=TRUE)
reticulate::py_install("factor_analyzer",pip=TRUE)
reticulate::py_install("raven-gen",pip=TRUE)
reticulate::py_install("matrix",pip=TRUE)
reticulate::py_install("nltk",pip=TRUE)
reticulate::py_install("openpyxl",pip=TRUE)
reticulate::py_install("pingouin",pip=TRUE)
reticulate::py_install("seaborn",pip=TRUE)

# Use Python in R
np<-import("numpy")
array<-np$array(c(1,2,3))

list.files(reticulate::virtualenv_root(),recursive=TRUE,pattern="cacert.pem",full.names=TRUE)

