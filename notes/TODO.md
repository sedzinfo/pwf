# pwf: what is left (20/11/2026, updated 23/11)

Goal: pwf is a full rwf in Python (same functions, arguments, outputs and numbers; rwf is the original).

Where things stand: all 189 of rwf's exported functions are in pwf, and each one has a test against
rwf's own output. The suite passed on Windows on 20/11 (501 tests, with 3 skipped because they need R;
see below), and on Linux on 23/11 (701 tests, with R).
The history of the work is in `notes/handoff_*.md`.

The list below is what still differs from rwf, in a suggested order.

## 1. Options rwf offers that pwf refuses (pwf raises NotImplementedError)

rwf passes these on to psych, lavaan or ez, so an R user can use them; pwf stops with a message.

- [x] **EFA methods and rotations** (done 23/11; `_efa.py`, see `notes/handoff_2026-11-23.md`)
  - Every `fm` but `"minrank"`, the common rotations, and `oblique.scores` now match psych.
  - Still refused: `fm = "minrank"` (psych needs the Rcsdp solver for it), the target, bifactor,
    biquartimin, cluster, varimin and special rotations, and score methods other than psych's default
    `"regression"`.
- [x] **SEM with observed predictors** (done 23/11; `_sem.py`, see `notes/handoff_2026-11-23.md`): regressions on
  and of observed variables, `fixed.x` (and `fixed_x=False`), lavaan's dummy latent variables, `auto.cov.y`.
  - Still refused: `simulate_data` with observed regressions, and higher-order factors.
- [x] **SEM mean structure with complete data** (done 23/11): `meanstructure = TRUE` with complete data, and
  intercepts in the syntax (`y ~ 1`, `y ~ 0*1`, a free latent mean `F ~ 1`, labelled intercepts).
- [ ] **Multi-group SEM** (`group =`, and the equality constraints across groups that measurement
  invariance uses).
- [ ] **Inequality and non-linear constraints** (`a > 0`, `a == b*c`). Labels, linear `==` constraints and
  `:=` definitions work since 20/11.
- [ ] **Categorical (WLSMV) CFA in the reports**
  - pwf has the engine (`_ordinal.OrdinalCFA`, which `fit_tirt` uses), but there is no public
    `cfa(..., ordered = ...)`.
  - `report_cfa`, `plot_cfa` and `plot_cfa_gg` accept only ML fits; rwf's accept any lavaan fit.
  - The ordinal engine also refuses regressions.
- [ ] **FIML fit measures**: with `missing = "ml"` the classic GFI family (`gfi_lisrel`, `agfi_lisrel`, `pgfi`)
  and lavaan's robust ("FIML corrected") CFI, TLI and RMSEA come out missing. CRMR and `gfi` match since 23/11.
  - Close to lavaan's GFI-LISREL: the per-pattern sum of the h1 expected information at the model's Sigma
    (1.3e-5 off); lavaan's FIML weight differs in a detail not found yet. The ordinal engine's GFI family, CN and MFI are missing too.
- [ ] **`report_factorial_anova`**: `within_covariates`, `between_covariates`, `diff` and `reverse_diff`,
  and sums of squares other than type 3.

## 2. Stand-ins that could become exact ports

- [ ] `ts_smoothing(type = "friedman")`: R's `supsmu` is approximated by lowess.
- [ ] `ts_smoothing(type = "splines")`: R's `smooth.spline` is approximated by scipy's smoothing spline.
- [ ] **IRT (`report_irt`, `plot_irt_onefactor`)**
  - These still run R's mirt through rpy2, so they need R, mirt and rpy2. They match rwf exactly.
  - A native port would be a large job (mirt's EM estimation, M2, Q3, item fit). It was left out on purpose
    earlier; start only if wanted.
  - Running these tests on Windows: install rpy2 into a separate folder with `pip install --target`, set
    `PYTHONPATH` to it, set `R_HOME`, set `RPY2_CFFI_MODE=ABI`, and put R's `bin/x64` on PATH.
- [ ] `tag_pos`: uses NLTK's tagger, not OpenNLP. It gives the same Penn tags, and agrees on about 90% of words.

## 3. Graphics details not yet compared with ggplot2

- [ ] **Point sizes**
  - plotnine draws a point (size + stroke) × √π points wide.
  - ggplot2 works from a font size of size × 2.845 + stroke × 1.89 points.
  - The two have never been compared, so points may be a little smaller or larger than rwf's. If they
    differ, add a `gg_point_size` helper beside `gg_text_size` and `gg_linewidth`.
- [ ] **Default line widths**: plotnine's are about 17% thinner than ggplot2's. Explicit widths already go
  through `gg_linewidth`.
- [ ] **Plot margin of `plot_cfa_gg`**: ggplot2's 20 pt margin isn't set, because plotnine's margins are
  fractions of the figure.

## 4. Known differences that stay (accepted, documented in the code)

- Anything random differs from R, because R and numpy draw different numbers:
  - the folds of `k_fold` and `k_sample`;
  - simulated data;
  - bootstraps;
  - the "spring" layout.
- `psych::fa` names its factors after random starts in R, so tests compare loadings by position.
- `psych::fa` rotates from 20 starts (the identity and 19 random ones from R's random numbers) and keeps the
  one with most loadings below .15; pwf rotates from the identity only. Usually the two give the same
  solution, but quartimax, geomin and simplimax can land on another one (seen on mtcars). rwf's result then
  changes from run to run unless a seed is set. pwf equals psych's `n.rotations = 1` to 1e-11.
- `fm = "ols"`: psych leaves the factor columns unnamed. pwf calls them X1, X2, ..., as rwf's tables show
  them. With one factor, rwf's `model_loadings` shows a long name built from its own code instead.
- psych's alpha factoring (`fm = "alpha"`) with one factor fails ("non-conformable arguments") unless the
  first eigenvalue is between 1 and 4, and then its loadings are not scaled. This is psych's `diag()` of a
  single number. pwf copies both.
- References for `fm = "wls"`, `"gls"`, `"minchi"` and `"old.min"` must come from R on the reference BLAS
  (Windows, or Linux with `LD_PRELOAD` of libblas.so.3 and liblapack.so.3). psych optimizes these with the
  ML gradient, so some fits are ill-conditioned: R on OpenBLAS moves 3.4e-4 on one of the test models.
- `report_hlr` drops the slope model when its random effects are perfectly correlated. rwf drops it when
  nlme stops at its iteration limit. The two agree on 252 of 300 test data sets.
- Diagnostic panels with tied leverages: R's last-bit noise in the QR reorders the ties.
- `plot_scatterplot` uses seaborn's jointplot instead of ggMarginal, so it has no point-count legend.
  `plot_multiplot` pastes its plots as images.
- `plot_confusion`'s x axis is at the bottom, because plotnine can't put an axis on top.
- `report_xgboost`: a Python Booster doesn't keep the training parameters and call that R keeps, so pass them.
- Excel files: only invisible cells differ.
  - R writes "" in A1 and in empty cells.
  - R writes Inf as #NUM!, where pandas writes the text "inf". This could be fixed by writing error cells.
- Console output keeps Python's print, not R's. For example, `output_separator` prints `TEXT` where R
  prints `[1] "TEXT"`. `report_cfa`'s log has pwf's own summary, not lavaan's `summary()`.
- `str_trim_df` does not copy rwf's broken cases (see section 5).

## 5. rwf bugs (fix in rwf; pwf copies them for now, so update pwf and its references afterwards)

- `report_irt`: `model_call` is the text "NULL", because `call_to_string()` gets `model@Call`.
- `report_regression(file = ...)` is very slow: over 10 minutes on 1,338 rows, from openxlsx styling.
- Durbin-Watson sheet comments use keys (r, dw, p) that match no column.
- `plot_cfa_gg(what = "eq")` fails on a model without labels, because lavaan then has no label column.
- `plot_irt_onefactor` selects items with `grep()`, so "V1" also takes V10 and V11.
- `report_regression`'s `labs(title = "")` blanks the titles of the six diagnostic panels.
- `increase_index(1, k)` returns three rows, and `cfa_icc_index(n, 1)` returns twice n indices: their
  `for (i in 2:n)` loops run twice when n is 1.
- `generate_comparisons_matrix(1)` and `rank_to_binary(x, 1)` fail, because of `1:0` loops.
- `str_split`:
  - strings that split into different numbers of parts shift the rows, with only a warning;
  - a one-part split gets the column name
    "matrix.unlist.str_split...byrow...TRUE..ncol...length.str_split..1....".
- `str_split_df(type = "column")` fails: only the misspelled "collumn" works.
- `k_sample`:
  - folds of 1 or 2 rows fail;
  - the printed "Validation:" count is the test set's size;
  - in a fold of one row j, `sample(j)` takes rows 1..j. pwf does not copy this one.
- `report_irt`'s Q3 sheet shows #NUM! (Inf) for the last item's column minimum and maximum.
- Not copied, because rwf's result is wrong:
  - `plot_logistic_model` with an outcome not named "outcome";
  - `str_trim_df`, which turns numbers into text and NA into "NA", drops the columns of an all-numeric data
    frame, and garbles long text.

## 6. Housekeeping

- [ ] Stale module docstrings:
  - [x] `glm_linear_regression.py` (`plot_scatterplot`'s `method`/`formula`) and `glm_reliability.py`
    (`responseFrequency()`) fixed 23/11.
  - [ ] Check the other module docstrings.
- [x] Old files in the repo root from the earlier project (`environment.py`, `examples/examples.py`,
  `reticulate.R`, `sublime/`): removed 23/11.
- [ ] Old names kept for compatibility:
  - `getfwp` (now `get_fwd`);
  - the module `functions_train_test_full`;
  - the Excel helpers' `writer`/`sheetname`/`comments`/`decimals` arguments (with a DeprecationWarning).
  - Decided 23/11: remove them at the first release.
- [x] pwf-only modules whose names overlap rwf's ports (`functions_generic.py`, with its own
  `generate_correlation_matrix`, and `plot_corrplot.py`): decided 23/11 to keep them.
- [ ] CI runs without R, so the 3 IRT tests never run there. An optional job with R and mirt would cover them.
- [ ] No release or tag yet (`pyproject.toml` says 0.2.0).
