# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_HLR.R.

Deviations from the R original, by design:
  - R's nlme::gls/nlme::lme are replaced with statsmodels: the fixed-effects-
    only baseline model uses statsmodels.formula.api.ols (equivalent to a
    gls() with no correlation/variance structure), and the two- and
    three-level random-effects models use statsmodels.formula.api.mixedlm,
    both fit with method="ML" (reml=False) to match R's method="ML".
  - R's `anova(base, random_intercept, random_intercept_predictor, ...)`
    (nlme:::anova.lme, an ML likelihood-ratio test cascade across nested
    models) has no ready-made statsmodels equivalent, so it's rebuilt from
    scratch as _anova_lme(): degrees of freedom come from each model's own
    parameter count (fixed effects + upper-triangular random-effects
    covariance parameters + 1 residual variance), and each row's L.Ratio/
    p-value compare it against the *previous* row via 2*(logLik_i -
    logLik_(i-1)) against a chi-square with the corresponding df
    difference — matching nlme's sequential (not all-pairs) comparison.
  - Column names are wrapped in patsy's Q("...") quoting function in every
    formula string. R identifiers may contain literal dots (e.g. R's own
    "pooled.stratum"); patsy's formula mini-language treats a bare "."
    specially, so an unquoted dotted name fails to parse. Q() sidesteps
    this (and also handles spaces/other awkward characters) uniformly.
  - `corlist`/`factorlist` are 1-based column indices, matching R's and
    this port's own established convention elsewhere (e.g. min_max_index,
    off_diagonal_index) rather than Python's native 0-based indexing.
  - plyr::rbind.fill (bind rows, filling missing columns with NA) is just
    pandas.concat(sort=False) — pandas already fills unmatched columns
    with NaN by default.
  - report_dataframe (R rwf::FUNCTIONS_EXCEL.R) is reimplemented locally
    as _report_dataframe rather than imported from functions_excel.py,
    which currently has unrelated top-level scratch code that breaks a
    plain `import functions_excel`; excel_critical_value itself is
    imported lazily (only when file is not None) to avoid that at
    module-import time.
  - Preserved R quirk: `temp` (the per-outcome complete-case frame actually
    passed to every fitted model) is built from just corlist + factorlist +
    predictor — random_effect is never added to it explicitly, in either
    the R original or this port. random_effect is only usable if it's
    already present in temp, i.e. it equals `predictor` (as in R's own
    docstring example) or its column is also listed in `factorlist`.
    Otherwise fitting raises a plain KeyError/R "object not found" error
    in both languages — not fixed here since it's exactly how the R
    source behaves, not a Python-specific translation gap.
  - Caveat inherited from the statistics, not the translation: R's example
    passes predictor="case" and random_effect="case" — the exact same
    column used as both the fixed-effect predictor and the grouping factor
    for the random effect, with only 2 groups. That aliases the fixed
    effect with the between-group random-intercept variance so the model
    isn't identified; nlme::lme may settle on a boundary (near-zero
    variance) solution, while statsmodels' MixedLM optimizer can instead
    diverge to a non-finite log-likelihood for the "random_intercept" or
    "random_intercept_predictor" step. This isn't a bug in this port —
    those two steps aren't wrapped in try/except in R either (only the
    random-slope step is) — so it surfaces here exactly as it would if
    nlme itself hit a similar degeneracy: a numerically nonsensical row in
    the output rather than a crash. Don't reuse the same column as both
    predictor and random_effect unless that's genuinely intended.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
##########################################################################################
# ANOVA-STYLE MODEL COMPARISON (internal helper, mirrors nlme:::anova.lme)
##########################################################################################
def _model_df(fit):
    """Number of estimated parameters for an OLS or MixedLM fit, including variance components."""
    if hasattr(fit, "k_fe"):
        k_re = fit.k_re
        return fit.k_fe + k_re * (k_re + 1) // 2 + 1
    return len(fit.params) + 1


def _anova_lme(fits):
    """
    Sequential ML likelihood-ratio comparison across nested models,
    mirroring nlme:::anova.lme's output columns (Model, df, AIC, BIC,
    logLik, Test, L.Ratio, p.value).
    """
    dfs = [_model_df(f) for f in fits]
    logliks = [f.llf for f in fits]
    aics = [f.aic for f in fits]
    bics = [f.bic for f in fits]

    n = len(fits)
    test = [np.nan] + [f"{i} vs {i + 1}" for i in range(1, n)]
    l_ratio, p_value = [np.nan], [np.nan]
    for i in range(1, n):
        ddf = dfs[i] - dfs[i - 1]
        lr = 2 * (logliks[i] - logliks[i - 1])
        l_ratio.append(lr)
        p_value.append(chi2.sf(lr, ddf) if ddf > 0 else np.nan)

    return pd.DataFrame({
        "Model": range(1, n + 1), "df": dfs, "AIC": aics, "BIC": bics, "logLik": logliks,
        "Test": test, "L.Ratio": l_ratio, "p.value": p_value,
    })
##########################################################################################
# WRITE DATAFRAME TO EXCEL (internal helper, mirrors rwf::report_dataframe)
##########################################################################################
def _report_dataframe(df, file, sheet, critical=None):
    if file is None:
        return df
    try:
        from .functions_excel import excel_critical_value
    except ImportError:
        from functions_excel import excel_critical_value
    writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
    excel_critical_value(df, writer, sheetname=sheet, critical=critical)
    writer._save()
    writer.close()
    return df
##########################################################################################
# REPORT HLR
##########################################################################################
def report_hlr(df, corlist, factorlist, predictor, random_effect, file=None, sheet="report"):
    """
    Hierarchical (multilevel) linear regression report: for each outcome
    in `corlist`, fits a cascade of nested ML models —
    fixed-intercept-only, random-intercept, random-intercept +
    fixed-effect predictor, and (if it fits) random-intercept + random
    slope for `predictor` — and reports a sequential likelihood-ratio
    test comparing each model to the previous one.

    Parameters:
    df (pandas.DataFrame): Data containing the outcome, factorlist, and
        predictor/random_effect columns.
    corlist (int or sequence of int): 1-based column index/indices of
        the numeric outcome(s) to model.
    factorlist (int or sequence of int): 1-based column index/indices
        of columns used only to determine complete cases (dropped
        alongside NAs in the outcome/predictor before fitting) — they
        are not otherwise used in any fitted model, matching the R
        original's behavior verbatim. If `random_effect` isn't already
        `predictor` itself, its column must be included here, since it
        is otherwise never added to the working frame (see module
        docstring) and fitting will raise a KeyError.
    predictor (str): Column name of the fixed-effect predictor.
    random_effect (str): Column name of the grouping factor for the
        random effect(s).
    file (str, optional): Output filename (without extension) for an
        Excel report. If None (default), no file is written.
    sheet (str, optional): Excel sheet name. Defaults to "report".

    Returns:
    pandas.DataFrame: One row per fitted model per outcome, with
    columns dv, model, fixed, random, Model, df, AIC, BIC, logLik,
    Test, L.Ratio, p.value.

    Examples:
    >>> import statsmodels.api as sm
    >>> infert = sm.datasets.get_rdataset("infert", "datasets").data
    >>> report_hlr(df=infert, corlist=8, factorlist=1, predictor="case", random_effect="case")
    """
    corlist = [corlist] if np.isscalar(corlist) else list(corlist)
    factorlist = [factorlist] if np.isscalar(factorlist) else list(factorlist)

    anova_comparisons = []
    for i in corlist:
        dv = df.columns[i - 1]
        cols = [dv] + [df.columns[j - 1] for j in factorlist] + [predictor]
        temp = df[cols].dropna()

        fbaseline = f'Q("{dv}") ~ 1'
        fpredictor = f'Q("{dv}") ~ Q("{predictor}")'
        frandom_intercept = f'~1 | {random_effect}'
        frandom_slope = f'~{predictor} | {random_effect}'

        base = smf.ols(fbaseline, data=temp).fit()
        random_intercept = smf.mixedlm(fbaseline, data=temp, groups=temp[random_effect],
                                        re_formula="~1").fit(reml=False)
        random_intercept_predictor = smf.mixedlm(fpredictor, data=temp, groups=temp[random_effect],
                                                  re_formula="~1").fit(reml=False)

        fits = [base, random_intercept, random_intercept_predictor]
        model_names = ["base", "random_intercept", "random_intercept_predictor"]
        fixed = [fbaseline, fbaseline, fpredictor]
        random = [np.nan, frandom_intercept, frandom_intercept]

        try:
            random_intercept_slope = smf.mixedlm(fpredictor, data=temp, groups=temp[random_effect],
                                                  re_formula=f'~Q("{predictor}")').fit(reml=False)
            if np.isfinite(random_intercept_slope.llf):
                fits.append(random_intercept_slope)
                model_names.append("random_intercept_slope")
                fixed.append(fpredictor)
                random.append(frandom_slope)
        except Exception:
            pass

        anova_table = _anova_lme(fits)
        anova_table.insert(0, "random", random)
        anova_table.insert(0, "fixed", fixed)
        anova_table.insert(0, "model", model_names)
        anova_table.insert(0, "dv", dv)
        anova_comparisons.append(anova_table)

    anova_comparisons = pd.concat(anova_comparisons, ignore_index=True, sort=False)
    return _report_dataframe(anova_comparisons, file=file, sheet=sheet, critical={"p.value": "<0.05"})
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import warnings
    import statsmodels.api as sm

    infert = sm.datasets.get_rdataset("infert", "datasets").data

    print("=" * 80, "\nreport_hlr (literal R docstring example)\n", "=" * 80, sep="")
    print("predictor and random_effect are both 'case' here, exactly as in the R\n"
          "docstring — see the module-level caveat: this aliases the fixed effect\n"
          "with the random intercept (only 2 groups), so expect a degenerate row.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report_hlr(df=infert, corlist=8, factorlist=1, predictor="case", random_effect="case")
    print(result)

    print("\n" + "=" * 80, "\nreport_hlr (well-posed example: distinct predictor/random_effect)\n",
          "=" * 80, sep="")
    print("random_effect='stratum' is only usable because column 7 (stratum) is\n"
          "also listed in factorlist, so it's retained in the per-outcome working\n"
          "frame — see the module-level note on this preserved R quirk.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result2 = report_hlr(df=infert, corlist=2, factorlist=7, predictor="case", random_effect="stratum")
    print(result2)
