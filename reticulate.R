################################################################################
# LOAD
################################################################################
library(reticulate)
directory<-paste0(dirname(rstudioapi::getActiveDocumentContext()$path),"/")

system2("python", "--version")
system2("R", "--version")

# virtualenv_remove(paste0(directory,"venvironment/"))
# virtualenv_create(paste0(directory,"venvironment/"))
use_virtualenv(paste0(directory,"venvironment/"))

py_install("numpy")
py_install("pandas")
py_install("scikit-learn")
py_install("dash")
py_install("yfinance")
py_install("get_all_tickers")
py_install("linkedin")
py_install("rpy2")
py_install("plotnine")
py_install("matplotlib")
py_install("seaborn")
py_install("ggrepel")
py_install("xlsxwriter")
py_install("researchpy")
py_install("semopy")
py_install("graphviz")
py_install("factor_analyzer")
py_install("raven-gen")
py_install("matrix")
py_install("nltk")

# Check Python configuration
py_config()

# Use Python in R
np <- import("numpy")
array <- np$array(c(1, 2, 3))



Sys.getenv()
