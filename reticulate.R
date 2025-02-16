library(reticulate)
system2("python", "--version")
system2("R", "--version")

# virtualenv_create("pwd")
# virtualenv_remove("pwd")
use_virtualenv("pwd")

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

# Check Python configuration
py_config()

# Use Python in R
np <- import("numpy")
array <- np$array(c(1, 2, 3))

system2("python", "--version")





