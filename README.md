# pwf

Python working functions — a personal toolkit of statistics, psychometrics, and ML/data-exploration helpers, ported from the [rwf](https://github.com/sedzinfo/rwf) R package.

## Install

```bash
pip install -e .
```

Heavier, module-specific dependencies (scipy, statsmodels, plotnine, rpy2, semopy, etc.) are grouped under the `full` extra:

```bash
pip install -e ".[full]"
```

## Usage

```python
import pwf.functions_generic as fg
import pwf.glm_efa as efa

df = fg.generate_normal(n=100, mean=0, sd=1)
```

Each module below can also be imported directly, e.g. `from pwf.explore_assumptions import plot_qq`.

## Modules

Every module is a port of an R file of [rwf](https://github.com/sedzinfo/rwf), which is the original:
functions, arguments and output columns keep rwf's names. Where a Python module ports one R file, the
R file is named in the module docstring. No R installation is needed.

### Explore

| Module | Contents |
| --- | --- |
| `explore_assumptions` | Normality diagnostics, outlier plots (`plot_outlier`), eight normality tests (`report_normality_tests`) |
| `explore_descriptives` | `compute_descriptives`, `compute_aggregate`, frequency and response tables, cross-tabulations, mosaic and bubble plots |
| `explore_time_series` | Time-series exploration and plotting helpers |

### Core Functions

| Module | Contents |
| --- | --- |
| `functions` | Data frame and vector utilities (`round_dataframe`, `remove_nc`, `dotnames`, ...) |
| `functions_cdf` | Data frame checks (`cdff`, `cdf`) |
| `functions_environment` | Runtime and file helpers (`get_fwd`, `combine_files`, `write_txt`, ...) |
| `functions_excel` | Excel export (`report_dataframe`, `excel_critical_value`, `excel_matrix`, ...) |
| `functions_generate_data` | Synthetic data generators |
| `functions_generic` | Data generation and manipulation helpers |
| `functions_keys` | Key/index lookup helpers |
| `functions_mathematical` | Mathematical utilities |
| `functions_matrix` | Matrix operations and diagnostics |
| `functions_plot` | Plot helpers (`report_pdf`, `plot_multiplot`, ...) |
| `functions_recode` | Value and category recoding |
| `functions_statistical` | Statistical helpers (`compute_standard`, `compute_adjustment`, ...) |
| `functions_strings` | String helpers |
| `functions_timestamp`, `functions_unix_time` | Date and time helpers |
| `functions_train_test` | ROC, confusion matrices (`plot_confusion_extended`), separability plots |
| `functions_train_test_full` | Cross-validation splits and cut-point performance |

### GLM And Modeling

| Module | Contents |
| --- | --- |
| `glm_one_way_anova` | Port of GLM_ANOVA.R: Kruskal-Wallis, Friedman, one-way tests, Tukey/Games-Howell (`compute_posthoc`), `report_oneway`, `report_factorial_anova` (between, within, mixed), `report_manova`, `compute_aov_es` |
| `glm_anova_plot` | ANOVA plots |
| `glm_means` | Pairwise t-tests and Wilcoxon tests (`report_ttests`, `report_wtests`), Cohen's d, Wilcoxon r |
| `glm_correlation` | Correlation reports |
| `glm_efa` | Exploratory factor analysis: scree plots, loadings, EFA report |
| `glm_validity` | `compute_efa_cfa`: EFA and CFA of questionnaire dimensions |
| `glm_sem` | CFA/SEM: `cfa`, `sem`, `report_cfa`, path diagrams, `simulate_cfa_fit` |
| `glm_reliability` | Cronbach's alpha (`report_alpha`), MTMM plots, variance components (`random_anova`, `extract_components`), Shrout-Fleiss reliability |
| `glm_hlr`, `glm_lda`, `glm_linear_regression`, `glm_logistic_regression` | Regression and discriminant analysis reports |
| `glm_irt`, `glm_irt_t`, `glm_irt_u` | Item response theory (`glm_irt` and parts of `glm_irt_t` still call R through rpy2) |

### Machine Learning

| Module | Contents |
| --- | --- |
| `ml_nlp` | Text cleaning, stopwords, part-of-speech tagging, text similarity, spelling statistics |
| `ml_xgboost` | `report_xgboost`, `plot_trees_xgboost` |

### Engines

`_sem` (a lavaan-compatible CFA/SEM engine: ML with listwise deletion or FIML, standard errors,
standardized solutions, lavaan's fit measures, modification indices, factor scores) and `_efa`
(a `psych::fa` compatible maximum likelihood EFA with oblimin rotation) are used by the modules above.

### Utilities

| Module | Contents |
| --- | --- |
| `helper` | Moment/cumulant conversions (`mc2mnc`, `cov2corr`, ...) |
| `download_nltk_data` | Script that downloads all NLTK data |
| `plot_corrplot` | Correlation matrix plotting |

## Data

`data/` contains the example datasets of rwf as CSV (titanic, insurance, blood pressure, personality,
ocean, admission, ...), used in the docstrings, examples and tests.

## Tests

The tests check pwf against rwf itself. `tests/reference/make_reference.R` runs rwf (and lavaan,
psych, mixlm, xgboost) on the same data and saves the results as CSV; the tests compare pwf's output
with them. Running the tests needs only Python; regenerating the references needs R with rwf:

```bash
pip install -e ".[full,dev]"
pytest
make reference RSCRIPT="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"   # regenerate the R references
```

## Documentation

The docstrings (description, Parameters, Returns, Examples) are the reference documentation. To build
it as a website with [pdoc](https://pdoc.dev):

```bash
make docs        # writes docs/
make docs-serve  # serves it locally
```

![Stars](https://img.shields.io/github/stars/sedzinfo/pwf)
![Watchers](https://img.shields.io/github/watchers/sedzinfo/pwf)
![Repo Size](https://img.shields.io/github/repo-size/sedzinfo/pwf)
![Open Issues](https://img.shields.io/github/issues/sedzinfo/pwf)
![Forks](https://img.shields.io/github/forks/sedzinfo/pwf)
![Last Commit](https://img.shields.io/github/last-commit/sedzinfo/pwf)
![Contributors](https://img.shields.io/github/contributors/sedzinfo/pwf)
![License](https://img.shields.io/github/license/sedzinfo/pwf)
![Release](https://img.shields.io/github/v/release/sedzinfo/pwf)
![Workflow Status](https://img.shields.io/github/actions/workflow/status/sedzinfo/pwf/main.yml)
