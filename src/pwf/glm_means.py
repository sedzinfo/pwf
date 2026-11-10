# -*- coding: utf-8 -*-
"""
Pairwise group-comparison reports (port of rwf GLM_MEANS.R): t-tests and
Wilcoxon rank-sum tests over every pairwise combination of factor levels,
Cohen's d and the Wilcoxon effect size r, plus a Python-only standalone
Levene/Bartlett report.

rwf is the original: function names, arguments and output columns follow
rwf, and results are checked against rwf in tests/test_glm_means.py.
stats::t.test and stats::wilcox.test are re-implemented here
(_t_test_two_sample, _wilcox_test_two_sample) so that statistics, p values,
confidence intervals and method labels match R, including wilcox.test's
continuity correction, tie correction, exact distribution and the
root-finding behind its confidence interval.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import functools
import itertools
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import levene, bartlett
try:
    from ._rcompat import r_label as _level_label, r_levels as _factor_levels, resolve_columns as _resolve_columns
    from .functions_excel import report_dataframe
    from .functions_statistical import compute_adjustment
except ImportError:
    from _rcompat import r_label as _level_label, r_levels as _factor_levels, resolve_columns as _resolve_columns
    from functions_excel import report_dataframe
    from functions_statistical import compute_adjustment
##########################################################################################
# HELPERS
##########################################################################################
_COHENS_D_COMMENT = ("measure of effect size\n\ncohen's d\n\nVery small\t0.01\t\tSawilowsky (2009)\n"
                     "Small\t\t0.20\t\tCohen (1988)\nMedium\t\t0.50\t\tCohen (1988)\n"
                     "Large\t\t0.80\t\tCohen (1988)\nVery large\t1.20\t\tSawilowsky (2009)\n"
                     "Huge\t\t12.0\t\tSawilowsky (2009)")


def _parse_formula(formula):
    """Split "dv ~ group" into ("dv", "group")."""
    dv, group = (part.strip() for part in formula.split("~"))
    return dv, group


def _t_test_two_sample(x, y, alternative="two.sided", mu=0, var_equal=False, conf_level=0.95):
    """Two-sample stats::t.test(x, y, ...): statistic, df, p, conf_int, method."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if var_equal:
        df = nx + ny - 2
        v = ((nx - 1) * vx + (ny - 1) * vy) / df
        stderr = np.sqrt(v * (1 / nx + 1 / ny))
    else:
        stderrx, stderry = np.sqrt(vx / nx), np.sqrt(vy / ny)
        stderr = np.sqrt(stderrx ** 2 + stderry ** 2)
        df = stderr ** 4 / (stderrx ** 4 / (nx - 1) + stderry ** 4 / (ny - 1))
    tstat = (mx - my - mu) / stderr
    if alternative == "less":
        p = stats.t.cdf(tstat, df)
        cint = (-np.inf, tstat + stats.t.ppf(conf_level, df))
    elif alternative == "greater":
        p = stats.t.sf(tstat, df)
        cint = (tstat - stats.t.ppf(conf_level, df), np.inf)
    else:
        p = 2 * stats.t.cdf(-abs(tstat), df)
        q = stats.t.ppf(1 - (1 - conf_level) / 2, df)
        cint = (tstat - q, tstat + q)
    cint = tuple(mu + c * stderr for c in cint)
    method = "Welch Two Sample t-test" if not var_equal else " Two Sample t-test"
    return {"statistic": tstat, "parameter": df, "p_value": p, "conf_int": cint, "method": method}


@functools.lru_cache(maxsize=4096)
def _rank_sum_distribution(scores, m):
    """
    Exact null distribution of the sum of m scores drawn without replacement
    from `scores` (sorted tuple of non-negative ints), as (sums, probabilities)
    over every reachable sum. With scores 1..N this is the Wilcoxon rank-sum
    distribution; with (doubled) mid-ranks it is the conditional distribution
    given the ties (R's C_dpermdist2, used by stats:::.dwilcox).
    """
    total = sum(sorted(scores)[-m:]) if m else 0
    counts = np.zeros((m + 1, total + 1))
    counts[0, 0] = 1.0
    for i, score in enumerate(scores):
        top = min(i + 1, m)
        if score == 0:
            counts[1:top + 1] += counts[:top].copy()
        else:
            counts[1:top + 1, score:] += counts[:top, :total + 1 - score].copy()
    dist = counts[m]
    return np.flatnonzero(dist), dist[dist > 0] / dist.sum()


def _wilcox_null(m, n, z=None):
    """
    Null distribution of W for samples of size m and n, as (w, prob) arrays.
    z=None gives the untied distribution (R's dwilcox); otherwise z are the
    mid-ranks of the combined sample and the distribution is conditional on
    the ties (R's stats:::.dwilcox with z).
    """
    if z is None:
        scores = tuple(range(1, m + n + 1))
        f = 1
    else:
        f = 1 if np.all(z == np.floor(z)) else 2
        scores = tuple(sorted(int(round(f * v)) for v in z))
    sums, prob = _rank_sum_distribution(scores, m)
    return sums / f - m * (m + 1) / 2, prob


def _pwilcox(q, m, n, z=None, lower_tail=True):
    """R's pwilcox (z=None) or stats:::.pwilcox (tie-aware, z = mid-ranks)."""
    w, prob = _wilcox_null(m, n, z)
    if z is None:
        q = np.floor(q + 1e-7)
        lower = prob[w <= q].sum()
        return lower if lower_tail else prob[w > q].sum()
    y = prob[w < q + 1e-8].sum()
    if lower_tail:
        return min(y, 1.0)
    if abs(y - 1) < 0.015625:
        return prob[w >= q + 1e-8].sum()
    return 1 - y


def _qwilcox(p, m, n):
    """R's qwilcox for a lower-tail probability p <= 0.5 (untied distribution)."""
    w, prob = _wilcox_null(m, n)
    index = np.searchsorted(np.cumsum(prob), p - 10 * np.finfo(float).eps)
    return int(w[min(index, len(w) - 1)])


def _wilcox_exact_cint(x, y, z, alternative, conf_level):
    """stats:::.wilcox_test_two_cint_exact: exact Hodges-Lehmann interval."""
    n_x, n_y = len(x), len(y)
    alpha = 1 - conf_level
    diffs = np.sort(np.subtract.outer(x, y).ravel())
    toler = 10 * np.finfo(float).eps
    if z is None:
        level = alpha / 2 if alternative == "two.sided" else alpha
        qu = _qwilcox(level, n_x, n_y)
        if _pwilcox(qu, n_x, n_y) <= level + toler:
            qu += 1
        if qu == 0:
            return (-np.inf, np.inf)
        ql = n_x * n_y - qu
        if alternative == "two.sided":
            return (diffs[qu - 1], diffs[ql])
        if alternative == "greater":
            return (diffs[qu - 1], np.inf)
        return (-np.inf, diffs[ql])

    tails = {}

    def ptail(mu, lower=True):
        if (mu, lower) not in tails:
            ranks = stats.rankdata(np.concatenate([x - mu, y]))
            w = ranks[:n_x].sum() - n_x * (n_x + 1) / 2
            tails[(mu, lower)] = (_pwilcox(w, n_x, n_y, ranks) if lower
                                  else _pwilcox(w - 0.25, n_x, n_y, ranks, lower_tail=False))
        return tails[(mu, lower)]

    def lower_bound(a):
        a += toler
        if ptail(diffs[0] - 1, False) > a:
            return -np.inf
        for k in range(len(diffs) - 1):
            if ptail((diffs[k] + diffs[k + 1]) / 2, False) > a:
                return diffs[k]
        return diffs[-1]

    def upper_bound(a):
        a += toler
        if ptail(diffs[-1] + 1) > a:
            return np.inf
        for k in range(len(diffs) - 2, -1, -1):
            if ptail((diffs[k] + diffs[k + 1]) / 2) > a:
                return diffs[k + 1]
        return diffs[0]

    if alternative == "two.sided":
        return (lower_bound(alpha / 2), upper_bound(alpha / 2))
    if alternative == "greater":
        return (lower_bound(alpha), np.inf)
    return (-np.inf, upper_bound(alpha))


def _wilcox_test_two_sample(x, y, alternative="two.sided", mu=0, exact=None, correct=True,
                            conf_int=False, conf_level=0.95):
    """
    Two-sample stats::wilcox.test(x, y, ...) as in R >= 4.6: statistic W,
    p value, confidence interval and method label. The exact test is used by
    default when both samples have fewer than 50 values, with ties handled by
    the exact conditional distribution of the mid-ranks.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n_x, n_y = len(x), len(y)
    r = stats.rankdata(np.concatenate([x - mu, y]))
    statistic = r[:n_x].sum() - n_x * (n_x + 1) / 2
    ties = len(np.unique(r)) != len(r)
    if exact is None:
        exact = n_x < 50 and n_y < 50
    cint = None
    if exact:
        method = "Wilcoxon rank sum exact test"
        z = r if ties else None
        lower = _pwilcox(statistic, n_x, n_y, z)
        upper = _pwilcox(statistic - 0.25, n_x, n_y, z, lower_tail=False)
        p = {"two.sided": min(2 * lower, 2 * upper, 1.0), "greater": upper, "less": lower}[alternative]
        if conf_int:
            cint = _wilcox_exact_cint(x, y, z, alternative, conf_level)
        return {"statistic": statistic, "p_value": p, "conf_int": cint, "method": method}

    method = "Wilcoxon rank sum test" + (" with continuity correction" if correct else "")

    def continuity(z):
        if not correct:
            return 0.0
        return {"two.sided": np.sign(z) * 0.5, "greater": 0.5, "less": -0.5}[alternative]

    def sigma(ranks):
        _, nties = np.unique(ranks, return_counts=True)
        return np.sqrt((n_x * n_y / 12) * ((n_x + n_y + 1)
                       - np.sum(nties ** 3 - nties) / ((n_x + n_y) * (n_x + n_y - 1))))

    z = statistic - n_x * n_y / 2
    z = (z - continuity(z)) / sigma(r)
    if alternative == "two.sided":
        p = 2 * min(stats.norm.cdf(z), 1 - stats.norm.cdf(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    if conf_int:
        alpha = 1 - conf_level
        mumin, mumax = x.min() - y.max(), x.max() - y.min()

        def w_of(d):
            dr = stats.rankdata(np.concatenate([x - d, y]))
            dz = dr[:n_x].sum() - n_x * (n_x + 1) / 2 - n_x * n_y / 2
            return (dz - continuity(dz)) / sigma(dr)

        w_min, w_max = w_of(mumin), w_of(mumax)

        def root(zq):
            if w_min - zq <= 0:
                return mumin
            if w_max - zq >= 0:
                return mumax
            return optimize.brentq(lambda d: w_of(d) - zq, mumin, mumax, xtol=1e-4)

        if alternative == "two.sided":
            cint = (root(stats.norm.isf(alpha / 2)), root(stats.norm.ppf(alpha / 2)))
        elif alternative == "greater":
            cint = (root(stats.norm.isf(alpha)), np.inf)
        else:
            cint = (-np.inf, root(stats.norm.ppf(alpha)))
    return {"statistic": statistic, "p_value": p, "conf_int": cint, "method": method}


def _group_values(df, dv, iv, levels):
    values = df.loc[df[iv] == levels[0], dv].astype(float).to_numpy()
    other = df.loc[df[iv] == levels[1], dv].astype(float).to_numpy()
    return values, other


def _pairwise_report(df, dv, iv, run_test, file, comment):
    """Shared loop of report_ttests/report_wtests: one row per DV x IV x level pair."""
    rows = []
    for dependent in _resolve_columns(df, dv):
        for independent in _resolve_columns(df, iv):
            complete = df[df[[dependent, independent]].notna().all(axis=1)]
            for f1, f2 in itertools.combinations(_factor_levels(complete[independent]), 2):
                x, y = _group_values(complete, dependent, independent, (f1, f2))
                n1, n2 = len(x), len(y)
                mean1, mean2 = x.mean(), y.mean()
                sd1, sd2 = x.std(ddof=1), y.std(ddof=1)
                sd_pooled = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)
                cohen_d = abs(mean2 - mean1) / sd_pooled
                k_squared, p_bartlett = bartlett(x, y)
                test_columns, r = run_test(x, y, cohen_d)
                rows.append({"DV": dependent, "IV": independent,
                             "level1": _level_label(f1), "level2": _level_label(f2),
                             "n1": n1, "n2": n2, **test_columns,
                             "mean1": mean1, "mean2": mean2, "sd1": sd1, "sd2": sd2,
                             "sd_pooled": sd_pooled, "d": cohen_d, "r": r,
                             "k_squared[bartlett]": k_squared, "df[bartlett]": 1.0,
                             "p[bartlett]": p_bartlett})
    result = pd.DataFrame(rows)
    adjustment = compute_adjustment(0.05, len(result))
    result["bonferroni_p"] = adjustment["bonferroni"]
    result["significant"] = np.where(adjustment["bonferroni"] > result["p"], "TRUE", "FALSE")
    report_dataframe(result, file=file, sheet="t test", comment=comment,
                     critical={"p": "<0.05", "p[bartlett]": "<0.05"})
    return result
##########################################################################################
# COHENS D
##########################################################################################
def compute_cohens_d(formula, data):
    """
    Cohen's d for a two-level grouping factor, using the pooled standard
    deviation sqrt((sd1^2 + sd2^2) / 2) (port of rwf::compute_cohens_d).

    Parameters:
    formula (str): "dv ~ group"; group must have exactly 2 levels.
    data (pandas.DataFrame): Data containing both variables. Rows with a
        missing value in either are dropped.

    Returns:
    float: Absolute Cohen's d.

    Examples:
    >>> compute_cohens_d("bp_before ~ sex", df_blood_pressure)
    """
    dv, group = _parse_formula(formula)
    data = data[[dv, group]].dropna()
    levels = _factor_levels(data[group])
    if len(levels) != 2:
        raise ValueError("grouping factor must have exactly 2 levels")
    x, y = _group_values(data, dv, group, levels)
    sd_pooled = np.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2)
    return float(abs(x.mean() - y.mean()) / sd_pooled)
##########################################################################################
# T TEST
##########################################################################################
def report_ttests(df, dv, iv, file=None, alternative="two.sided", mu=0, var_equal=False, conf_level=0.95):
    """
    Pairwise t-tests of every dv across every pair of levels of every iv,
    with descriptives, Cohen's d, r, Bartlett's test and a Bonferroni
    adjustment (port of rwf::report_ttests).

    Parameters:
    df (pandas.DataFrame): Input data.
    dv (str, int or list): Dependent variable(s), as column names or
        1-based column positions (as in rwf).
    iv (str, int or list): Independent (grouping) variable(s), as names or
        1-based positions. Levels are compared in sorted order.
    file (str, optional): Excel output filename without extension.
    alternative (str, optional): "two.sided" (default), "less" or "greater".
    mu (float, optional): Hypothesised mean difference. Defaults to 0.
    var_equal (bool, optional): Pooled-variance t-test if True; Welch's
        test (default) if False. R's var.equal.
    conf_level (float, optional): Confidence level of the interval. R's conf.level.

    Returns:
    pandas.DataFrame: One row per comparison with DV, IV, level1, level2,
    n1, n2, t, df, p, CI_l, CI_u, alternative, method, mean1, mean2, sd1,
    sd2, sd_pooled, d, r, k_squared[bartlett], df[bartlett], p[bartlett],
    bonferroni_p and significant ("TRUE"/"FALSE").

    Examples:
    >>> report_ttests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    >>> report_ttests(df=df_blood_pressure, dv=4, iv=[2, 3], var_equal=True, file="ttest")
    """
    comment = {
        "DV": "dependent variable", "IV": "independent variable",
        "level1": "level 1", "level2": "level 2",
        "n1": "sample size for level 1", "n2": "sample size for level 2",
        "t": "t statistic", "df": "degrees of freedom for t statistic", "p": "p value",
        "CI_l": "confidence interval lower bound", "CI_u": "confidence interval upper bound",
        "alternative": "alternative hypothesis", "method": "",
        "mean1": "mean for level 1", "mean2": "mean for level 2",
        "sd1": "standard deviation for level 1", "sd2": "standard deviation for level 2",
        "sd_pooled": "pooled standard deviation\n\ncombined standard deviation of both levels",
        "d": _COHENS_D_COMMENT, "r": "measure of effect size\n\ncorrelation effect size",
        "k_squared[bartlett]": "bartlett test for homogeneity of variances\n\n",
        "df[bartlett]": "bartlett test for homogeneity of variances\n\n",
        "p[bartlett]": "bartlett test for homogeneity of variances\n\nsignificant values indicate heteroscedasticity",
        "bonferroni_p": "bonferroni adjusted critical value for a=0.05",
        "significant": "if TRUE result is significant after bonferroni adjustment",
    }

    def run_test(x, y, cohen_d):
        ttest = _t_test_two_sample(x, y, alternative=alternative, mu=mu, var_equal=var_equal,
                                   conf_level=conf_level)
        n1, n2 = len(x), len(y)
        r = cohen_d / np.sqrt(cohen_d ** 2 + (n1 + n2 - 2) * (1 / n1 + 1 / n2))
        return {"t": ttest["statistic"], "df": ttest["parameter"], "p": ttest["p_value"],
                "CI_l": ttest["conf_int"][0], "CI_u": ttest["conf_int"][1],
                "alternative": alternative, "method": ttest["method"]}, r

    return _pairwise_report(df, dv, iv, run_test, file, comment)
##########################################################################################
# WILCOXON EFFECT SIZE
##########################################################################################
def compute_wilcoxon_effect_size(formula, data, mu=0, exact=None, correct=True):
    """
    Wilcoxon rank-sum effect size r = |Z| / sqrt(N), with Z recovered from the
    two-sided wilcox.test p value (port of rwf::compute_wilcoxon_effect_size).

    Parameters:
    formula (str): "dv ~ group" with a two-level group.
    data (pandas.DataFrame): Data containing both variables.
    mu (float, optional): Location shift. Defaults to 0.
    exact (bool, optional): Exact p value; None (default) uses it when both
        groups have fewer than 50 values and there are no ties.
    correct (bool, optional): Continuity correction. Defaults to True.

    Returns:
    float: Effect size r.

    Examples:
    >>> compute_wilcoxon_effect_size("bp_before ~ agegrp",
    ...     df_blood_pressure[df_blood_pressure["agegrp"].isin(["30-45", "46-59"])])
    """
    dv, group = _parse_formula(formula)
    data = data[[dv, group]].dropna()
    levels = _factor_levels(data[group])
    if len(levels) != 2:
        raise ValueError("grouping factor must have exactly 2 levels")
    x, y = _group_values(data, dv, group, levels)
    wtest = _wilcox_test_two_sample(x, y, alternative="two.sided", mu=mu, exact=exact, correct=correct)
    z = stats.norm.isf(wtest["p_value"] / 2)
    return float(abs(z) / np.sqrt(len(data)))
##########################################################################################
# WILCOXON TEST
##########################################################################################
def report_wtests(df, dv, iv, file=None, alternative="two.sided", mu=0, exact=None, correct=True,
                  conf_level=0.95):
    """
    Pairwise Wilcoxon rank-sum tests of every dv across every pair of levels
    of every iv, with Hodges-Lehmann confidence intervals, descriptives,
    Cohen's d, the Wilcoxon effect size r, Bartlett's test and a Bonferroni
    adjustment (port of rwf::report_wtests).

    Parameters:
    df (pandas.DataFrame): Input data.
    dv (str, int or list): Dependent variable(s), as names or 1-based positions.
    iv (str, int or list): Independent (grouping) variable(s), as names or
        1-based positions. Levels are compared in sorted order.
    file (str, optional): Excel output filename without extension.
    alternative (str, optional): "two.sided" (default), "less" or "greater".
    mu (float, optional): Location shift. Defaults to 0.
    exact (bool, optional): Exact test; None (default) uses it when both
        groups have fewer than 50 values and there are no ties.
    correct (bool, optional): Continuity correction. Defaults to True.
    conf_level (float, optional): Confidence level. R's conf.level.

    Returns:
    pandas.DataFrame: One row per comparison with DV, IV, level1, level2,
    n1, n2, W, p, CI_l, CI_u, alternative, method, mean1, mean2, sd1, sd2,
    sd_pooled, d, r, k_squared[bartlett], df[bartlett], p[bartlett],
    bonferroni_p and significant ("TRUE"/"FALSE"). r is always based on the
    two-sided p value, whatever the alternative.

    Examples:
    >>> report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    >>> report_wtests(df=df_blood_pressure, dv=4, iv=[2, 3], alternative="less")
    """
    comment = {
        "DV": "dependent variable", "IV": "independent variable",
        "level1": "level 1", "level2": "level 2",
        "n1": "sample size for level 1", "n2": "sample size for level 2",
        "W": "wilcoxon statistic", "df": "degrees of freedom for t statistic", "p": "p value",
        "CI_l": "confidence interval lower bound", "CI_u": "confidence interval upper bound",
        "alternative": "alternative hypothesis", "method": "",
        "mean1": "mean for level 1", "mean2": "mean for level 2",
        "sd1": "standard deviation for level 1", "sd2": "standard deviation for level 2",
        "sd_pooled": "pooled standard deviation\n\ncombined standard deviation of both levels",
        "d": _COHENS_D_COMMENT, "r": "measure of effect size\n\ncorrelation effect size",
        "k_squared[bartlett]": "bartlett test for homogeneity of variances\n\n",
        "df[bartlett]": "bartlett test for homogeneity of variances\n\n",
        "p[bartlett]": "bartlett test for homogeneity of variances\n\nsignificant values indicate heteroscedasticity",
        "bonferroni_p": "bonferroni adjusted critical value for a=0.05",
        "significant": "if TRUE result is significant after bonferroni adjustment",
    }

    def run_test(x, y, cohen_d):
        wtest = _wilcox_test_two_sample(x, y, alternative=alternative, mu=mu, exact=exact,
                                        correct=correct, conf_int=True, conf_level=conf_level)
        two_sided = _wilcox_test_two_sample(x, y, alternative="two.sided", mu=mu, exact=exact,
                                            correct=correct)
        r = abs(stats.norm.isf(two_sided["p_value"] / 2)) / np.sqrt(len(x) + len(y))
        return {"W": wtest["statistic"], "p": wtest["p_value"],
                "CI_l": wtest["conf_int"][0], "CI_u": wtest["conf_int"][1],
                "alternative": alternative, "method": wtest["method"]}, r

    return _pairwise_report(df, dv, iv, run_test, file, comment)
##########################################################################################
# LEVENE
##########################################################################################
def report_levene_bartlett(df, dv, iv):
    """
    Run pairwise Levene and Bartlett homogeneity-of-variance tests across
    factor levels and return a reporting table. Python-only (no rwf
    counterpart).

    Parameters:
    df (pandas.DataFrame): Input data. Must contain dv and every column in iv.
    dv (str): Dependent variable column name.
    iv (list of str): Independent (grouping) variable column name(s). For
        each factor, every pairwise combination of its levels is tested.

    Returns:
    pandas.DataFrame: One row per pairwise level comparison, per factor,
    with IV, L1, L2, Levene, Levene_p, Bartlett, Bartlett_p.

    Examples:
    >>> report_levene_bartlett(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    """
    result_iterative = pd.DataFrame()
    for factor in iv:
        levels = df[factor].unique()
        combinations = list(itertools.combinations(levels, 2))
        for level1, level2 in combinations:
            variable_1 = df[df[factor] == level1][dv].astype(float)
            variable_2 = df[df[factor] == level2][dv].astype(float)
            stat_l, p_l = levene(variable_1, variable_2)
            stat_b, p_b = bartlett(variable_1, variable_2)
            result = pd.DataFrame({
                "IV": [factor],
                "L1": [level1],
                "L2": [level2],
                "Levene": [stat_l],
                "Levene_p": [p_l],
                "Bartlett": [stat_b],
                "Bartlett_p": [p_b]
            })
            result_iterative = pd.concat([result_iterative, result], ignore_index=True)
    return result_iterative
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    import os

    df_blood_pressure = load_dataset("blood_pressure")

    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print("=" * 80, "\ncompute_cohens_d\n", "=" * 80, sep="")
        print(compute_cohens_d("bp_before ~ sex", df_blood_pressure))

        print("\n" + "=" * 80, "\nreport_ttests\n", "=" * 80, sep="")
        print(report_ttests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"]).round(2))
        print(report_ttests(df=df_blood_pressure, dv=4, iv=[2, 3], var_equal=True).round(2))

        print("\n" + "=" * 80, "\ncompute_wilcoxon_effect_size\n", "=" * 80, sep="")
        print(compute_wilcoxon_effect_size(
            "bp_before ~ agegrp", df_blood_pressure[df_blood_pressure["agegrp"].isin(["30-45", "46-59"])]))

        print("\n" + "=" * 80, "\nreport_wtests\n", "=" * 80, sep="")
        print(report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"]).round(2))
        print(report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"], alternative="less").round(2))

        print("\n" + "=" * 80, "\nreport_levene_bartlett\n", "=" * 80, sep="")
        print(report_levene_bartlett(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"]))
