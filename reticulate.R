library(reticulate)
# virtualenv_create("pystatistics")
use_virtualenv("pystatistics")

py_install("numpy")
py_install("pandas")
py_install("scikit-learn")
py_install("dash")
py_install("yfinance")
py_install("get_all_tickers")
py_install("linkedin")


# Check Python configuration
py_config()

# Use Python in R
np <- import("numpy")
array <- np$array(c(1, 2, 3))


system2("python", "--version")
















# Remove the existing virtual environment
virtualenv_remove("pystatistics")

# Create a new virtual environment
virtualenv_create("pystatistics")
