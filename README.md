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

### Explore

| Module | Contents |
| --- | --- |
| `explore_assumptions` | Normality/outlier diagnostics: histograms, QQ plots, boxplots |
| `explore_descriptives` | Descriptive summaries (`describe_by_mean`, `flatten`) |
| `explore_time_series` | Time-series exploration and plotting helpers |

### Core Functions

| Module | Contents |
| --- | --- |
| `functions` | Legacy/all-in-one convenience wrapper of assorted helpers |
| `functions_cdf` | CDF/observed-value checks and helpers |
| `functions_environment` | Runtime/environment helpers |
| `functions_excel` | Excel export formatting via `xlsxwriter` |
| `functions_generate_data` | Synthetic data generators |
| `functions_generic` | Data generation and manipulation: missing data, random/correlated matrices, dummy columns, factor simulation |
| `functions_keys` | Key/index lookup helpers |
| `functions_mathematical` | Mathematical utilities |
| `functions_matrix` | Matrix operations and matrix diagnostics |
| `functions_plot` | Generic plotting helpers |
| `functions_recode` | Value and category recoding helpers |
| `functions_statistical` | General statistical helpers (`compute_standard`, ...) |
| `functions_strings` | String-cleaning and text utilities |
| `functions_timestamp` | Timestamp/date decomposition helpers |
| `functions_train_test` | Train/test evaluation: ROC, confusion matrix, separability plots |
| `functions_train_test_full` | Extended train/test and model-evaluation helpers |
| `functions_unix_time` | Unix timestamp conversion helpers |

### GLM And Model Helpers

| Module | Contents |
| --- | --- |
| `glm_anova_plot` | ANOVA-focused plotting helpers |
| `glm_correlation` | Correlation and association helpers |
| `glm_efa` | Exploratory factor analysis: scree plots, loadings, EFA report |
| `glm_hlr` | Hierarchical linear regression helpers |
| `glm_irt` | Item-response-theory utilities |
| `glm_irt_t` | IRT tooling (T-parameterization variants) |
| `glm_irt_u` | IRT tooling (U-parameterization variants) |
| `glm_lda` | Linear discriminant analysis helpers |
| `glm_linear_regression` | OLS/linear regression helpers |
| `glm_logistic_regression` | Logistic regression helpers |
| `glm_means` | t-tests, Levene/Bartlett homogeneity tests |
| `glm_one_way_anova` | One-way ANOVA, Kruskal-Wallis, Games-Howell post hoc |
| `glm_reliability` | Reliability/Cronbach's alpha reporting |
| `glm_sem` | Structural equation modeling via `semopy` *(work in progress)* |

### Utility Modules

| Module | Contents |
| --- | --- |
| `helper` | Moment/cumulant conversions (`mc2mnc`, `cov2corr`, ...) |
| `nltk` | NLTK corpus download helper |
| `plot_corrplot` | Correlation matrix plotting |

Some modules are still experimental or carry known issues (for example: `glm_sem`, `glm_reliability`, `helper`, `nltk`, and `plot_corrplot`). See git history/issues for current status.

## Data

`data/` contains example CSV datasets (titanic, insurance, blood pressure, personality, ocean, ...) used across the module docstrings and examples, mirroring the datasets shipped with the `rwf` R package.

## Development

```bash
pip install -e ".[dev]"
pytest
```
