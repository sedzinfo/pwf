# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_HLR.R.

report_hlr matches rwf (tests/test_glm_hlr.py): nlme's gls fitted by
maximum likelihood becomes statsmodels' ols, nlme's lme fits become
_lme_fit (lme4's profiled deviance), whose log-likelihoods agree with
nlme's, and nlme::anova.lme's sequential likelihood-ratio table is rebuilt
(df counts the fixed effects, the random-effect covariances and the
residual variance; AIC and BIC use it). The fixed, random and call columns
hold R's deparsed formulas and rwf's calls.

statsmodels' mixedlm isn't used: when the ML random-effects covariance is
singular (a variance of 0 or a correlation of +-1) its optimizer can stop
far from the maximum (logLik -280.0 instead of -252.58 for the Fatigue
slope model in the tests) or return an infinite log-likelihood (rwf's
infert example), and whether it reports convergence depends on the scipy
version. Where a random intercept's variance is 0 at the maximum, nlme
stops just short of it, so its logLik can be ~1e-7 lower than pwf's and a
likelihood ratio near 0 is optimizer noise (its p-value moves with the
ratio's square root).

The random intercept and slope model: rwf's try() leaves it out when
nlme::lme fails, which is when nlminb reaches its iteration limit.
- That mostly happens when the ML random effects are perfectly correlated
  (a correlation of +-1, both variances above 0). nlme's log-Cholesky
  parameters can only reach that at infinity, and the likelihood flattens
  too slowly for nlminb to stop.
- Where the maximum has a variance of 0 instead, the likelihood flattens
  fast and nlme finishes. The tests' Fatigue slope model is the first
  kind, rwf's infert example the second.
nlminb's path can't be ported, so pwf leaves the model out when its ML
random effects are perfectly correlated. On 300 simulated data sets this
matched rwf's choice on 252. It left out 145 of the 147 models nlme failed
on, and also 46 that nlme finished, mostly where the correlation adds
little likelihood.

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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.optimize import minimize
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


def _lme_fit(y, x, z, groups):
    """
    ML fit of the linear mixed model y = x beta + z b + e, with random effects b ~ N(0, Psi) per group.

    Minimises lme4's profiled deviance over theta, the lower-triangular
    Cholesky factor of Psi relative to the residual variance. It needs no
    inverse of Psi, so a singular Psi is reached at a finite theta.
    - The Psi whose theta has a last diagonal element of 0 are exactly the
      singular ones.
    - Those with a zero row j have a variance of 0 for random effect j.
    The maximum is one of these when fixing those elements at 0 lowers the
    log-likelihood by less than 1e-6. On simulated data, the loss from
    making Psi singular was either below 1e-10 or above 1e-4.

    Returns:
    tuple: the fit (llf, nobs, k_fe, k_re, as _anova_lme reads them) and whether its random effects are
    perfectly correlated (Psi singular at the maximum, with every variance above 0).
    """
    n, p = x.shape
    q = z.shape[1]
    groups = pd.factorize(groups)[0]
    ztz, ztx, zty = (np.zeros((groups.max() + 1,) + shape) for shape in [(q, q), (q, p), (q,)])
    np.add.at(ztz, groups, np.einsum("ni,nj->nij", z, z))
    np.add.at(ztx, groups, np.einsum("ni,nj->nij", z, x))
    np.add.at(zty, groups, z * y[:, None])
    rows, cols = np.tril_indices(q)  # theta's last element is the last diagonal one

    def deviance(theta):
        lam = np.zeros((q, q))
        lam[rows, cols] = theta
        m = np.eye(q) + lam.T @ ztz @ lam  # per group: I + lam' Z'Z lam
        b, c = lam.T @ ztx, zty @ lam  # per group: lam' Z'X and lam' Z'y
        m_b, m_c = np.linalg.solve(m, b), np.linalg.solve(m, c[..., None])[..., 0]
        xwx = x.T @ x - np.einsum("gki,gkj->ij", b, m_b)
        xwy = x.T @ y - np.einsum("gki,gk->i", b, m_c)
        rss = y @ y - np.einsum("gk,gk->", c, m_c) - xwy @ np.linalg.solve(xwx, xwy)
        return np.linalg.slogdet(m)[1].sum() + n * (1 + np.log(2 * np.pi * rss / n))

    options = {"gtol": 1e-8, "maxiter": 10000}
    free = minimize(deviance, np.eye(q)[rows, cols], method="BFGS", options=options)

    def restricted(varied):
        """The maximum log-likelihood with theta's elements outside varied fixed at 0."""
        def masked(theta_varied):
            theta = np.zeros(len(rows))
            theta[varied] = theta_varied
            return deviance(theta)
        return -(minimize(masked, free.x[varied], method="BFGS", options=options).fun if varied.any()
                 else masked([])) / 2

    singular = restricted((rows != q - 1) | (cols != q - 1))
    zero_variance = max(restricted(rows != j) for j in range(q))
    llf = max(-free.fun / 2, singular, zero_variance)
    correlated = llf - singular < 1e-6 and llf - zero_variance >= 1e-6
    return SimpleNamespace(llf=llf, nobs=n, k_fe=p, k_re=q), correlated
##########################################################################################
# REPORT HLR
##########################################################################################
def report_hlr(df, corlist, factorlist, predictor, random_effect, file=None, sheet="report"):
    """
    Multilevel model comparison, as rwf::report_hlr: for each outcome in
    corlist, maximum-likelihood fits of the intercept-only model, a
    random intercept per random_effect, the random intercept with the
    fixed predictor, and (unless its random effects are perfectly
    correlated) a random intercept and slope of the predictor, compared in
    sequence by likelihood-ratio tests (nlme::anova.lme).

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
        y, groups = temp[dv].to_numpy(float), temp[random_effect]
        intercept = np.ones((len(temp), 1))
        with_predictor = np.asarray(patsy.dmatrix(f'Q("{predictor}")', temp))  # intercept and predictor
        fixed_baseline, fixed_predictor = f"{_r_name(dv)} ~ 1", f"{_r_name(dv)} ~ {_r_name(predictor)}"
        random_intercept, random_slope = f"~1 | {_r_name(random_effect)}", \
            f"~{_r_name(predictor)} | {_r_name(random_effect)}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits = [smf.ols(f'Q("{dv}") ~ 1', data=temp).fit(),
                    _lme_fit(y, intercept, intercept, groups)[0],
                    _lme_fit(y, with_predictor, intercept, groups)[0]]
            names = ["base", "random_intercept", "random_intercept_predictor"]
            fixed = [fixed_baseline, fixed_baseline, fixed_predictor]
            random = [np.nan, random_intercept, random_intercept]
            try:
                slope, correlated = _lme_fit(y, with_predictor, with_predictor, groups)
                if not correlated and np.isfinite(slope.llf):
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
