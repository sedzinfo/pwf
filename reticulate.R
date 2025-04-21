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

reticulate::py_install("numpy")
reticulate::py_install("pandas")
reticulate::py_install("scikit-learn")
reticulate::py_install("dash")
reticulate::py_install("yfinance")
reticulate::py_install("get_all_tickers")
reticulate::py_install("linkedin")
reticulate::py_install("rpy2")
reticulate::py_install("plotnine")
reticulate::py_install("matplotlib")
reticulate::py_install("seaborn")
reticulate::py_install("xlsxwriter")
reticulate::py_install("researchpy")
reticulate::py_install("semopy")
reticulate::py_install("graphviz")
reticulate::py_install("factor_analyzer")
reticulate::py_install("raven-gen")
reticulate::py_install("matrix")
reticulate::py_install("nltk")
reticulate::py_install("openpyxl")
reticulate::py_install("pingouin")
reticulate::py_install("seaborn")


# Use Python in R
np<-import("numpy")
array<-np$array(c(1,2,3))

list.files(reticulate::virtualenv_root(), recursive = TRUE, pattern = "cacert.pem", full.names = TRUE)
