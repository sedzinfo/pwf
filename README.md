# pwf

Python working functions: statistics, psychometrics and machine-learning reports. pwf is a port of
the R package [rwf](https://github.com/sedzinfo/rwf), which is the original: functions, arguments and
output columns keep rwf's names, and the test suite checks pwf's results against rwf's (and against the
R packages rwf uses: lavaan, psych, car, MASS, nlme, mixlm, irr, pROC, thurstonianIRT). No R
installation is needed to use pwf.

## Install

```bash
pip install -e .            # core: numpy, pandas, scipy, statsmodels, plotnine, ...
pip install -e ".[full]"    # plus text analysis (nltk, spylls), xgboost and path diagrams (networkx, pydot)
pip install -e ".[dev]"     # plus pytest and pdoc
```

## Usage

```python
from pwf.datasets import load_dataset
from pwf.glm_correlation import report_correlation
from pwf.glm_sem import cfa

df_insurance = load_dataset("insurance")
report_correlation(df_insurance[["age", "bmi", "charges"]], scatterplot=False)["r_lower"]
```

Every module also runs as a script with worked examples, e.g. `python -m pwf.glm_logistic_regression`.

## Modules

### Explore

| Module | Contents |
| --- | --- |
| `explore_assumptions` | Outlier plots (`plot_outlier`), normality tests (`report_normality_tests`), Q-Q plots |
| `explore_descriptives` | `compute_descriptives`, `compute_aggregate`, frequency and response tables, cross tables, mosaic plots |
| `explore_time_series` | `plot_ts`, `plot_acf`, `ts_smoothing`, `compute_moving_average` |

### GLM and psychometrics

| Module | Contents |
| --- | --- |
| `glm_means` | t and Wilcoxon tests (`report_ttests`, `report_wtests`), Cohen's d, Wilcoxon r |
| `glm_one_way_anova` | Port of GLM_ANOVA.R: Kruskal-Wallis, Friedman, one-way tests, post hoc tests, `report_oneway`, `report_factorial_anova`, `report_manova`, `compute_aov_es` |
| `glm_anova_plot` | `plot_oneway`, `plot_interaction`, `plot_oneway_diagnostics` |
| `glm_correlation` | `report_correlation` (psych::corr.test), `tetrachoric`, `polychoric`, `polyserial`, `biserial`, `report_choric_serial`, power curves |
| `glm_linear_regression` | `report_regression` (R's lm view of a statsmodels fit), `plot_scatterplot` |
| `glm_logistic_regression` | `report_logistic` (R's glm.fit replica), `plot_logistic_model`, `output_compare_model_logistic` |
| `glm_lda` | `lda` (MASS::lda) and `report_lda` |
| `glm_hlr` | `report_hlr`: multilevel model comparisons (nlme) |
| `glm_efa` | Exploratory factor analysis: `fa` (psych::fa), `report_efa`, scree plots |
| `glm_sem` | CFA and SEM (`cfa`, `sem`, `report_cfa`, path diagrams, `simulate_cfa_fit`) |
| `glm_validity` | `compute_efa_cfa` |
| `glm_reliability` | `report_alpha`, variance components (`random_anova`, `extract_components`), MTMM plots |
| `glm_irt_t` | Thurstonian IRT: comparison matrices, recoding, ICCs, MAP scoring, `fit_tirt`, `extract_tirt_params`, `check_heywood` |
| `glm_irt_u` | Unidimensional IRT: theta, ability, item information |
| `glm_irt` | mirt models (still calls R through rpy2: `pip install rpy2` and R with mirt) |

### Machine learning

| Module | Contents |
| --- | --- |
| `ml_nlp` | Text cleaning, stopwords, part-of-speech tagging, text similarity, spelling statistics |
| `ml_xgboost` | `report_xgboost`, `plot_trees_xgboost` |
| `functions_train_test` | ROC curves, confusion matrices, cut-point search, k-fold and train/test/validation splits, scaling and dummy coding |

### Utilities

| Module | Contents |
| --- | --- |
| `datasets` | `load_dataset("insurance")`: rwf's datasets (`df_insurance`, ...) and example data |
| `functions` | Data frame and vector helpers (`round_dataframe`, `change_data_type`, `remove_nc`, `dotnames`, ...) |
| `functions_cdf` | Data frame checks (`cdf`, `cdff`) |
| `functions_environment` | `get_fwd`, `combine_files`, `write_txt`, ... |
| `functions_excel` | Formatted Excel output (`report_dataframe`, `excel_critical_value`, `excel_matrix`, `excel_confusion_matrix`) |
| `functions_generate_data` | Synthetic data, correlation matrices, missing values |
| `functions_keys`, `functions_recode`, `functions_matrix`, `functions_mathematical`, `functions_statistical`, `functions_strings`, `functions_unix_time`, `functions_plot` | The matching rwf helper files |

### Engines (private modules)

| Module | Contents |
| --- | --- |
| `_sem` | lavaan-compatible CFA/SEM: ML with listwise deletion or FIML, standard errors, standardized solutions, fit measures, modification indices, factor scores |
| `_ordinal` | lavaan-compatible models of ordered indicators: thresholds, polychoric correlations and their asymptotic covariance, WLSMV and ULSMV with labels and linear constraints, robust standard errors and fit measures |
| `_efa` | psych::fa (minres and ml, oblimin rotation, factor statistics) |
| `_rcompat` | R behaviour the ports depend on (column positions, `as.character`, `make.names`, `p.adjust`, `abbreviate`, `mixedsort`, ...) |

### Python-only extras

`functions_generic` (older generators), `functions_timestamp`, `helper` (statsmodels' moment conversions),
`plot_corrplot` (heat map of a data frame's correlations; see `glm_correlation.plot_corrplot` for rwf's),
`download_nltk_data`. `functions_train_test_full` is a deprecated alias of `functions_train_test`.

## Layout

```
src/pwf/          the package (one module per rwf R file) and src/pwf/data (the datasets)
tests/            pytest suite; tests/reference holds rwf's outputs as CSV
tests/reference/make_reference.R   regenerates them (needs R with rwf, lavaan, psych, thurstonianIRT, ...)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
make reference RSCRIPT="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"   # regenerate the R references
```

## Documentation

The docstrings (description, Parameters, Returns, Examples) are the reference documentation; build them
as a website with [pdoc](https://pdoc.dev): `make docs` (writes docs/) or `make docs-serve`.

## License

GPL-3.0-or-later, as rwf (pwf translates GPL R code from lavaan, psych, GPArotation, mixlm and nortest).

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
