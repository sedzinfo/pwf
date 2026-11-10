# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_HLR.R.

report_hlr matches rwf (tests/test_glm_hlr.py): nlme's gls and lme fitted
by maximum likelihood become statsmodels' ols and mixedlm(reml=False),
whose log-likelihoods agree with nlme's, and nlme::anova.lme's sequential
likelihood-ratio table is rebuilt (df counts the fixed effects, the
random-effect covariances and the residual variance; AIC and BIC use it).
The random-slope model is left out when it does not converge, as rwf's
try() leaves it out when nlme fails. The fixed, random and call columns
hold R's deparsed formulas and rwf's calls.

Notes kept from rwf:
  - corlist and factorlist are 1-based column positions.
  - The rows used are the complete cases of the outcome, factorlist and
    predictor; random_effect is not added, so it must be the predictor or
    be listed in factorlist.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

try:
    from .functions_excel import report_dataframe
except ImportError:
    from functions_excel import report_dataframe

# rwf's model calls as nlme's anova.lme deparses them (lines pasted with " ")
_CALLS = {
    "base": 'nlme::gls(model = formula(fbaseline), data = temp, method = "ML")',
    "random_intercept": 'lme.formula(fixed = formula(fbaseline), data = temp, random = frandom_intercept,     '
                        'method = "ML")',
    "random_intercept_predictor": 'lme.formula(fixed = fpredictor, data = temp, random = frandom_intercept,     '
                                  'method = "ML")',
    "random_intercept_slope": 'lme.formula(fixed = fpredictor, data = temp, random = frandom_slope,     '
                              'method = "ML")',
}
##########################################################################################
# ANOVA OF NESTED MODELS (internal, mirrors nlme:::anova.lme)
##########################################################################################
def _model_df(fit):
    """nlme's parameter count: fixed effects, random-effect covariances and the residual variance."""
    if hasattr(fit, "k_fe"):
        return fit.k_fe + fit.k_re * (fit.k_re + 1) // 2 + 1
    return len(fit.params) + 1


def _anova_lme(fits, names):
    """nlme::anova.lme of ML fits: AIC, BIC, logLik and each model's likelihood ratio against the previous one."""
    dfs = np.array([_model_df(f) for f in fits])
    logliks = np.array([f.llf for f in fits])
    n_obs = np.array([f.nobs for f in fits])
    l_ratio = [np.nan] + [2 * abs(logliks[i] - logliks[i - 1]) for i in range(1, len(fits))]
    p_value = [np.nan] + [chi2.sf(l_ratio[i], abs(dfs[i] - dfs[i - 1])) for i in range(1, len(fits))]
    return pd.DataFrame({
        "call": [_CALLS[name] for name in names], "Model": np.arange(1, len(fits) + 1), "df": dfs,
        "AIC": -2 * logliks + 2 * dfs, "BIC": -2 * logliks + np.log(n_obs) * dfs, "logLik": logliks,
        "Test": [""] + [f"{i} vs {i + 1}" for i in range(1, len(fits))], "L.Ratio": l_ratio, "p.value": p_value,
    })


def _r_name(name):
    """A column name as R deparses it: backquoted unless syntactic."""
    return name if re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", name) else f"`{name}`"
##########################################################################################
# REPORT HLR
##########################################################################################
def report_hlr(df, corlist, factorlist, predictor, random_effect, file=None, sheet="report"):
    """
    Multilevel model comparison, as rwf::report_hlr: for each outcome in
    corlist, maximum-likelihood fits of the intercept-only model, a
    random intercept per random_effect, the random intercept with the
    fixed predictor, and (when it converges) a random intercept and
    slope of the predictor, compared in sequence by likelihood-ratio
    tests (nlme::anova.lme).

    Parameters:
    df (pandas.DataFrame): Data.
    corlist (int or list of int): 1-based positions of the outcomes.
    factorlist (int or list of int): 1-based positions of columns whose missing values drop rows
        (list random_effect here unless it is the predictor).
    predictor (str): Name of the fixed predictor.
    random_effect (str): Name of the grouping variable.
    file (str, optional): Excel filename without extension. Defaults to None.
    sheet (str, optional): Excel sheet name. Defaults to "report".

    Returns:
    pandas.DataFrame: dv, model, fixed, random, call, Model, df, AIC, BIC,
    logLik, Test, L.Ratio and p.value for every model of every outcome.

    Examples:
    >>> import statsmodels.api as sm
    >>> sleepstudy = sm.datasets.get_rdataset("sleepstudy", "lme4").data
    >>> report_hlr(df=sleepstudy, corlist=1, factorlist=3, predictor="Days", random_effect="Subject")
    """
    corlist = [corlist] if np.isscalar(corlist) else list(corlist)
    factorlist = [factorlist] if np.isscalar(factorlist) else list(factorlist)
    comparisons = []
    for i in corlist:
        dv = df.columns[i - 1]
        columns = list(dict.fromkeys([dv] + [df.columns[j - 1] for j in factorlist] + [predictor]))
        temp = df[columns].dropna()
        groups = temp[random_effect]
        baseline, with_predictor = f'Q("{dv}") ~ 1', f'Q("{dv}") ~ Q("{predictor}")'
        fixed_baseline, fixed_predictor = f"{_r_name(dv)} ~ 1", f"{_r_name(dv)} ~ {_r_name(predictor)}"
        random_intercept, random_slope = f"~1 | {_r_name(random_effect)}", \
            f"~{_r_name(predictor)} | {_r_name(random_effect)}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits = [smf.ols(baseline, data=temp).fit(),
                    smf.mixedlm(baseline, data=temp, groups=groups, re_formula="~1").fit(reml=False),
                    smf.mixedlm(with_predictor, data=temp, groups=groups, re_formula="~1").fit(reml=False)]
            names = ["base", "random_intercept", "random_intercept_predictor"]
            fixed = [fixed_baseline, fixed_baseline, fixed_predictor]
            random = [np.nan, random_intercept, random_intercept]
            try:
                slope = smf.mixedlm(with_predictor, data=temp, groups=groups,
                                    re_formula=f'~Q("{predictor}")').fit(reml=False)
                if slope.converged and np.isfinite(slope.llf):
                    fits.append(slope)
                    names.append("random_intercept_slope")
                    fixed.append(fixed_predictor)
                    random.append(random_slope)
            except Exception:
                pass
        table = _anova_lme(fits, names)
        table.insert(0, "random", random)
        table.insert(0, "fixed", fixed)
        table.insert(0, "model", names)
        table.insert(0, "dv", dv)
        comparisons.append(table)
    result = pd.concat(comparisons, ignore_index=True)
    report_dataframe(result, file=file, sheet=sheet, type="critical_value", critical={"p.value": "<0.05"})
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os

    path = "tests/reference/sleepstudy.csv" if os.path.exists("tests/reference/sleepstudy.csv") \
        else "../tests/reference/sleepstudy.csv"
    sleepstudy = pd.read_csv(path)
    print(report_hlr(df=sleepstudy, corlist=[1, 4], factorlist=3, predictor="Days", random_effect="Subject"))
