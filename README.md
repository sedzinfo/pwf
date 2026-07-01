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

| Module | Contents |
| --- | --- |
| `functions_generic` | Data generation and manipulation: missing data, random/correlated matrices, dummy columns, factor simulation |
| `functions_statistical` | General statistical helpers (`compute_standard`, ...) |
| `functions_cdf` | CDF/observed-value checks and helpers |
| `functions_timestamp` | Timestamp/date decomposition helpers |
| `functions_excel` | Excel export formatting via `xlsxwriter` |
| `functions_train_test` | Train/test evaluation: ROC, confusion matrix, separability plots |
| `explore_descriptives` | Descriptive summaries (`describe_by_mean`, `flatten`) |
| `explore_assumptions` | Normality/outlier diagnostics: histograms, QQ plots, boxplots |
| `helper` | Moment/cumulant conversions (`mc2mnc`, `cov2corr`, ...) |
| `glm_means` | t-tests, Levene/Bartlett homogeneity tests |
| `glm_one_way_anova` | One-way ANOVA, Kruskal-Wallis, Games-Howell post hoc |
| `glm_efa` | Exploratory factor analysis: scree plots, loadings, EFA report |
| `glm_reliability` | Reliability/Cronbach's alpha reporting |
| `glm_sem` | Structural equation modeling via `semopy` *(work in progress)* |
| `plot_corrplot` | Correlation matrix plotting |
| `nltk` | NLTK corpus download helper |

`glm_sem`, `glm_reliability`, `helper`, `nltk`, and `plot_corrplot` currently have known issues (leftover scratch code, version-incompatible dependencies, or hardcoded paths) — see git history for status.

## Data

`data/` contains example CSV datasets (titanic, insurance, blood pressure, personality, ocean, ...) used across the module docstrings and examples, mirroring the datasets shipped with the `rwf` R package.

## Development

```bash
pip install -e ".[dev]"
pytest
```
