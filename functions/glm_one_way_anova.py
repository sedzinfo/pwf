# -*- coding: utf-8 -*-
"""
Port of rwf GLM_ANOVA.R (compute_kruskal_wallis_test, compute_friedman_test,
compute_one_way_test, compute_posthoc, report_oneway, report_factorial_anova,
report_manova, compute_aov_es) plus
the Python-only compute_games_howell. Original notes:

One-way/factorial ANOVA utilities: Kruskal-Wallis with effect sizes, a
from-scratch one-way F-test (Fisher or Welch), Games-Howell post-hoc
pairwise comparisons, and eta/omega/epsilon effect sizes for a fitted
statsmodels OLS ANOVA model (Type I/II/III sums of squares).

Fixed a real bug in compute_aov_es: its Type III branch hardcoded
`order = ["Treatment", "Residual", "Intercept"]` to reorder the ANOVA
table, which silently dropped every other effect term (e.g. "Type" and
"Treatment:Type") for any model with more than one factor -- confirmed
by running this file's own factorial-model test call. Rewritten to keep
every effect row (in their original order), then Residual, then
Intercept, so it generalizes to any model instead of assuming a factor
literally named "Treatment".
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import os
import re
from itertools import combinations, product
try:
    from ._rcompat import r_label, r_levels, r_round, resolve_columns
    from .explore_descriptives import compute_aggregate, compute_descriptives
    from .functions_excel import excel_critical_value
    from .functions_plot import report_pdf
    from .functions_statistical import compute_adjustment
    from .functions_strings import output_separator
    from .glm_anova_plot import plot_oneway, plot_oneway_diagnostics
except ImportError:
    from _rcompat import r_label, r_levels, r_round, resolve_columns
    from explore_descriptives import compute_aggregate, compute_descriptives
    from functions_excel import excel_critical_value
    from functions_plot import report_pdf
    from functions_statistical import compute_adjustment
    from functions_strings import output_separator
    from glm_anova_plot import plot_oneway, plot_oneway_diagnostics
##########################################################################################
# KRUSKALL WALLIS TEST WITH EFFECT SIZE
##########################################################################################
def compute_kruskal_wallis_test(formula, df, ci=False, conf_level=0.95, nboot=1000):
    """
    Kruskal-Wallis rank sum test with eta-squared and epsilon-squared effect
    sizes, optionally with percentile bootstrap confidence intervals for both
    (port of rwf::compute_kruskal_wallis_test). Rows with a missing value in
    either variable are dropped.

    Parameters:
    formula (str): "dv ~ group".
    df (pandas.DataFrame): Data containing the DV and group columns.
    ci (bool, optional): If True, adds bootstrap confidence intervals for
        etasq and epsilonsq. Defaults to False.
    conf_level (float, optional): Confidence level. R's conf.level. Defaults to 0.95.
    nboot (int, optional): Number of bootstrap resamples. Defaults to 1000.
        Resamples use numpy's global RNG, so np.random.seed() makes them
        reproducible (as set.seed() does in R; the draws differ from R's).

    Returns:
    pandas.DataFrame: One row with formula, method, etasq, epsilonsq, H, df,
    p; with ci=True, etasq_lower/etasq_upper follow etasq and
    epsilonsq_lower/epsilonsq_upper follow epsilonsq.

    Examples:
    >>> compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure)
    >>> np.random.seed(1)
    >>> compute_kruskal_wallis_test(formula="bp_before ~ agegrp", df=df_blood_pressure, ci=True)
    """
    dv, group = (part.strip() for part in formula.split("~"))
    data = df[[dv, group]].dropna()
    x = data[dv].to_numpy(dtype=float)
    g = data[group].to_numpy()

    def kruskal_h(x, g):
        n = len(x)
        r = stats.rankdata(x)
        rank_sums = pd.Series(r).groupby(g).agg(["sum", "count"])
        h = np.sum(rank_sums["sum"] ** 2 / rank_sums["count"])
        _, ties = np.unique(x, return_counts=True)
        return (12 * h / (n * (n + 1)) - 3 * (n + 1)) / (1 - np.sum(ties ** 3 - ties) / (n ** 3 - n))

    k = len(pd.unique(g))
    n = len(x)
    H = kruskal_h(x, g)
    df_k = k - 1
    p = stats.chi2.sf(H, df_k)
    etasq = (H - k + 1) / (n - k)
    epsilonsq = H / ((n ** 2 - 1) / (n + 1))
    result = {"formula": f"{dv} ~ {group}", "method": "Kruskal-Wallis rank sum test", "etasq": etasq}
    if ci:
        boot = np.empty((nboot, 2))
        for b in range(nboot):
            i = np.random.randint(0, n, size=n)
            k_boot = len(pd.unique(g[i]))
            H_boot = kruskal_h(x[i], g[i])
            boot[b] = ((H_boot - k_boot + 1) / (n - k_boot), H_boot / (n - 1))
        probs = [(1 - conf_level) / 2, 1 - (1 - conf_level) / 2]
        result["etasq_lower"], result["etasq_upper"] = np.nanquantile(boot[:, 0], probs)
        result["epsilonsq"] = epsilonsq
        result["epsilonsq_lower"], result["epsilonsq_upper"] = np.nanquantile(boot[:, 1], probs)
    else:
        result["epsilonsq"] = epsilonsq
    result.update({"H": H, "df": df_k, "p": p})
    return pd.DataFrame([result])
##########################################################################################
# ONE WAY TEST WITH SS AND MS
##########################################################################################
def compute_one_way_test(formula, df, var_equal=True):
    """
    One-way ANOVA F test with sums and mean squares and effect sizes (port of
    rwf::compute_one_way_test): Fisher's test assuming equal variances
    (var_equal=True) or Welch's test (var_equal=False). Rows missing either
    variable are dropped.

    Parameters:
    formula (str): "dv ~ group".
    df (pandas.DataFrame): Data containing both variables.
    var_equal (bool, optional): Assume equal group variances. R's var.equal.
        Defaults to True.

    Returns:
    pandas.DataFrame: One row with formula, method, ss_effect, ss_error,
    ms_effect, ms_error, etasq, partial.etasq, omegasq, partial.omegasq,
    cohens.f, power, statistic, df_effect, df_error, p.

    Examples:
    >>> compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=True)
    >>> compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=False)
    """
    y_var, g_var = (part.strip() for part in formula.split("~"))
    data = df[[y_var, g_var]].dropna()
    y = data[y_var].astype(float)
    grouped = y.groupby(data[g_var].map(r_label), sort=True)
    n_i, m_i, v_i = grouped.count(), grouped.mean(), grouped.var(ddof=1)
    k = len(n_i)
    w_i = n_i / v_i
    sum_w_i = w_i.sum()
    n = n_i.sum()
    df_effect = k - 1
    if var_equal:
        df_error = n - k
        ss_effect = float(np.sum(n_i * (m_i - y.mean()) ** 2))
        ss_error = float(np.sum((n_i - 1) * v_i))
        ms_effect = ss_effect / df_effect
        ms_error = ss_error / df_error
        method = "Assuming homoscedasticity"
    else:
        tmp = float(np.sum((1 - w_i / sum_w_i) ** 2 / (n_i - 1)) / (k ** 2 - 1))
        df_error = 1 / (3 * tmp)
        m = float(np.sum(w_i * m_i) / sum_w_i)
        ms_effect = float(np.sum(w_i * (m_i - m) ** 2))
        ms_error = df_effect * (1 + 2 * (k - 2) * tmp)
        ss_effect = ms_effect * df_effect
        ss_error = ms_error * df_error
        method = "Assuming heteroscedasticity"
    ss_total = ss_effect + ss_error
    statistic = ms_effect / ms_error
    p = stats.f.sf(statistic, df_effect, df_error)
    etasq = ss_effect / ss_total
    partial_etasq = ss_effect / (ss_effect + ss_error)
    omegasq = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    partial_omegasq = (df_effect * (ms_effect - ms_error)) / (df_effect * ms_effect + (df_error + 1) * ms_error)
    cohens_f = np.sqrt(etasq / (1 - etasq))
    power = stats.ncf.sf(stats.f.isf(0.05, df_effect, df_error), df_effect, df_error, cohens_f ** 2 * n)
    return pd.DataFrame([{
        "formula": f"{y_var} ~ {g_var}", "method": method,
        "ss_effect": ss_effect, "ss_error": ss_error, "ms_effect": ms_effect, "ms_error": ms_error,
        "etasq": etasq, "partial.etasq": partial_etasq, "omegasq": omegasq, "partial.omegasq": partial_omegasq,
        "cohens.f": cohens_f, "power": power, "statistic": statistic,
        "df_effect": df_effect, "df_error": df_error, "p": p,
    }])
##########################################################################################
# GAMES HOWELL
##########################################################################################
def compute_games_howell(y, x, digits=4):
    """
    Games-Howell pairwise post-hoc comparisons (does not assume equal
    variances or equal group sizes), via a Welch-style t approximation.

    Parameters:
    y (array-like): Continuous outcome values.
    x (array-like): Group labels, same length as y.
    digits (int, optional): Rounding for the returned tables. Defaults to 4.

    Returns:
    dict: {"descriptives": pandas.DataFrame (n, mean, variance per
    group), "posthoc": pandas.DataFrame (t, df, p per pairwise
    comparison, indexed "group_a:group_b")}.

    Examples:
    >>> compute_games_howell(y=df_blood_pressure["bp_before"], x=df_blood_pressure["agegrp"])
    """
    # Remove missing values
    mask = ~pd.isnull(x) & ~pd.isnull(y)
    x = pd.Series(x)[mask].astype('category')
    y = pd.Series(y)[mask]

    group_labels = x.cat.categories
    groups = {level: y[x == level] for level in group_labels}

    n = pd.Series({level: len(g) for level, g in groups.items()})
    means = pd.Series({level: g.mean() for level, g in groups.items()})
    variances = pd.Series({level: g.var(ddof=1) for level, g in groups.items()})
    descriptives = pd.DataFrame({'n': n, 'mean': means, 'variance': variances}).round(digits)

    pair_names = [f"{a}:{b}" for a, b in combinations(group_labels, 2)]

    t_vals, df_vals, p_vals = [], [], []

    for a, b in combinations(group_labels, 2):
      na, nb = n[a], n[b]
      ma, mb = means[a], means[b]
      va, vb = variances[a], variances[b]
      diff = abs(ma - mb)
      se = np.sqrt(va / na + vb / nb)

      t = diff / se
      df_num = (va / na + vb / nb) ** 2
      df_denom = ((va / na) ** 2 / (na - 1)) + ((vb / nb) ** 2 / (nb - 1))
      df_ = df_num / df_denom

      p = stats.t.sf(t * np.sqrt(2), df_)*2
      t_vals.append(t)
      df_vals.append(df_)
      p_vals.append(p)

    result_df = pd.DataFrame({'t': t_vals,'df': df_vals,'p': p_vals},index=pair_names).round(digits)

    return {"descriptives": descriptives, "posthoc": result_df}
##########################################################################################
# ETA PARTIAL ETA OMEGA PARTIAL OMEGA FOR AOV
##########################################################################################
def _r_model_label(text):
    """R spelling of a statsmodels/patsy term or formula: C(Treatment, Sum) -> Treatment."""
    return re.sub(r"C\(\s*([^,\)]+?)\s*(,[^\)]*)?\)", r"\1", text)


def compute_aov_es(model, ss="I"):
    """
    Eta, partial eta, omega, partial omega and epsilon squared and Cohen's f
    for every term of an ANOVA model, from its Type I, II or III sums of
    squares table (port of rwf::compute_aov_es). Type III needs sum-to-zero
    contrasts, e.g. C(factor, Sum) in the formula.

    Parameters:
    model (statsmodels RegressionResultsWrapper): A fitted
        statsmodels.formula.api.ols(...) model (R: an aov fit).
    ss (str, optional): "I" (default), "II" or "III".

    Returns:
    pandas.DataFrame: Columns call, ss, comparisons, Df, Sum Sq, Mean Sq,
    F value, Pr(>F), etasq, partial_etasq, omegasq, partial_omegasq,
    epsilonsq, cohens_f; one row per term, then (Intercept) (Type III) and
    Residuals, whose effect sizes are missing. Term names drop the C(...)
    wrapper, as R prints them.

    Examples:
    >>> import statsmodels.formula.api as smf
    >>> compute_aov_es(smf.ols("uptake ~ Treatment", data=df_co2).fit(), ss="I")
    >>> compute_aov_es(smf.ols("uptake ~ C(Treatment, Sum) * C(Type, Sum)", data=df_co2).fit(), ss="III")
    """
    if ss not in ("I", "II", "III"):
        raise ValueError('ss must be "I", "II" or "III"')
    table = anova_lm(model, typ={"I": 1, "II": 2, "III": 3}[ss])
    table = pd.DataFrame({
        "Df": table["df"].to_numpy(dtype=float),
        "Sum Sq": table["sum_sq"].to_numpy(dtype=float),
        "Mean Sq": (table["sum_sq"] / table["df"]).to_numpy(dtype=float),
        "F value": table["F"].to_numpy(dtype=float),
        "Pr(>F)": table["PR(>F)"].to_numpy(dtype=float),
    }, index=[{"Intercept": "(Intercept)", "Residual": "Residuals"}.get(t, _r_model_label(t)) for t in table.index])
    if ss == "III":
        intercept, table = table.iloc[:1], table.iloc[1:]
    effects, residual = table.iloc[:-1], table.iloc[-1]
    ss_total = table["Sum Sq"].sum()
    n_total = int(model.nobs)
    ms_error, ss_error = residual["Mean Sq"], residual["Sum Sq"]
    df_effect, ss_effect, ms_effect = effects["Df"], effects["Sum Sq"], effects["Mean Sq"]
    partial_eta = ss_effect / (ss_effect + ss_error)
    sizes = pd.DataFrame({
        "etasq": ss_effect / ss_total,
        "partial_etasq": partial_eta,
        "omegasq": (ss_effect - df_effect * ms_error) / (ss_total + ms_error),
        "partial_omegasq": (df_effect * (ms_effect - ms_error)) / (ss_effect + (n_total - df_effect) * ms_error),
        "epsilonsq": (ss_effect - df_effect * ms_error) / ss_total,
        "cohens_f": np.sqrt(partial_eta / (1 - partial_eta)),
    })
    ordered = pd.concat([effects, intercept, table.iloc[-1:]] if ss == "III" else [effects, table.iloc[-1:]])
    result = ordered.join(sizes)
    call = _r_model_label(model.model.formula)
    result.insert(0, "comparisons", result.index)
    result.insert(0, "ss", ss)
    result.insert(0, "call", call)
    return result.reset_index(drop=True)
##########################################################################################
# FRIEDMAN TEST WITH EFFECT SIZE
##########################################################################################
def _friedman_q(y):
    """Friedman chi-squared with tie correction, for a blocks x groups matrix."""
    n, k = y.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, y)
    ties = 0.0
    for row in y:
        _, counts = np.unique(row, return_counts=True)
        ties += np.sum(counts ** 3 - counts)
    return 12 * np.sum((ranks.sum(axis=0) - n * (k + 1) / 2) ** 2) / (n * k * (k + 1) - ties / (k - 1))


def compute_friedman_test(formula, df, ci=False, conf_level=0.95, nboot=1000):
    """
    Friedman rank sum test for a complete block design (repeated measures),
    with Kendall's W as effect size and optional percentile bootstrap
    confidence interval (port of rwf::compute_friedman_test). Blocks with a
    missing value in any group are dropped.

    Parameters:
    formula (str): "y ~ group | block".
    df (pandas.DataFrame): Long data, at most one row per block and group.
    ci (bool, optional): Add a bootstrap interval for kendall_w. Defaults to False.
    conf_level (float, optional): Confidence level. R's conf.level. Defaults to 0.95.
    nboot (int, optional): Bootstrap resamples, drawn with numpy's global RNG
        (np.random.seed() makes them reproducible; the draws differ from R's).

    Returns:
    pandas.DataFrame: One row with formula, method, kendall_w,
    [kendall_w_lower, kendall_w_upper,] Q, df, n, p.

    Examples:
    >>> compute_friedman_test("uptake ~ conc | Plant", df_co2)
    """
    left, right = formula.split("~")
    y_var = left.strip()
    g_var, b_var = (part.strip() for part in right.split("|"))
    data = df.loc[df[[g_var, b_var]].notna().all(axis=1), [y_var, g_var, b_var]]
    groups, blocks = r_levels(data[g_var]), r_levels(data[b_var])
    if data.groupby([b_var, g_var]).size().max() > 1:
        raise ValueError("each block must have at most one observation per group")
    y = np.full((len(blocks), len(groups)), np.nan)
    block_index = {b: i for i, b in enumerate(blocks)}
    group_index = {g: j for j, g in enumerate(groups)}
    for value, g, b in zip(data[y_var], data[g_var], data[b_var]):
        y[block_index[b], group_index[g]] = value
    y = y[~np.isnan(y).any(axis=1)]
    n, k = y.shape
    q = _friedman_q(y)
    df_q = k - 1
    p = stats.chi2.sf(q, df_q)
    kendall_w = q / (n * (k - 1))
    result = {"formula": f"{y_var} ~ {g_var} | {b_var}", "method": "Friedman rank sum test", "kendall_w": kendall_w}
    if ci:
        boot = [_friedman_q(y[np.random.randint(0, n, size=n)]) / (n * (k - 1)) for _ in range(nboot)]
        probs = [(1 - conf_level) / 2, 1 - (1 - conf_level) / 2]
        result["kendall_w_lower"], result["kendall_w_upper"] = np.nanquantile(boot, probs)
    result.update({"Q": q, "df": df_q, "n": n, "p": p})
    return pd.DataFrame([result])
##########################################################################################
# POSTHOC
##########################################################################################
def compute_posthoc(y, x):
    """
    Tukey (Tukey-Kramer) and Games-Howell pairwise comparisons of the group
    means of y (port of rwf::compute_posthoc). p values come from the
    studentized range distribution (R's ptukey).

    Parameters:
    y (array-like): Outcome values.
    x (array-like): Group labels, same length as y.

    Returns:
    dict: {"input": {"x": x, "y": y}, "output": {"tukey": DataFrame,
    "games.howell": DataFrame}}; each table has columns t, df, p and one row
    per pair of levels, indexed "level1:level2".

    Examples:
    >>> compute_posthoc(df_blood_pressure["bp_before"], df_blood_pressure["agegrp"])
    """
    x_series, y_series = pd.Series(np.asarray(x, dtype=object)), pd.Series(np.asarray(y, dtype=float))
    complete = x_series.notna() & y_series.notna()
    pair_levels = r_levels(x_series[complete])
    levels = r_levels(x_series)
    in_group = [x_series == level for level in levels]
    n = np.array([float(mask.sum()) for mask in in_group])
    means = np.array([y_series[mask].mean(skipna=False) for mask in in_group])
    variances = np.array([y_series[mask].var(ddof=1, skipna=False) for mask in in_group])
    groups = len(n)
    df_error = n.sum() - groups
    error_variance = np.sum((n - 1) * variances) / df_error
    pairs = list(combinations(range(groups), 2))
    names = [f"{r_label(pair_levels[i])}:{r_label(pair_levels[j])}" for i, j in combinations(range(len(pair_levels)), 2)]
    tukey_t = np.array([abs(means[j] - means[i]) / np.sqrt(error_variance * (1 / n[i] + 1 / n[j])) for i, j in pairs])
    tukey_p = stats.studentized_range.sf(tukey_t * np.sqrt(2), groups, df_error)
    se_parts = variances / n
    gh_df = np.array([(se_parts[i] + se_parts[j]) ** 2 / (se_parts[i] ** 2 / (n[i] - 1) + se_parts[j] ** 2 / (n[j] - 1))
                      for i, j in pairs])
    gh_t = np.array([abs(means[j] - means[i]) / np.sqrt(se_parts[i] + se_parts[j]) for i, j in pairs])
    gh_p = np.array([stats.studentized_range.sf(t * np.sqrt(2), groups, d) for t, d in zip(gh_t, gh_df)])
    tukey = pd.DataFrame({"t": tukey_t, "df": df_error, "p": tukey_p}, index=names)
    games_howell = pd.DataFrame({"t": gh_t, "df": gh_df, "p": gh_p}, index=names)
    return {"input": {"x": x, "y": y}, "output": {"tukey": tukey, "games.howell": games_howell}}
##########################################################################################
# REPORT ONEWAY
##########################################################################################
_ONEWAY_INSTRUCTIONS = {
    "fisher": "Fisher assumes homoscedasticity (equal variances)",
    "welch": "Welch does not assume homoscedasticity (it allows unequal variances)",
    "kruskal": "Kruskal Wallis procedure does not assume normality but it is not an alternative for violations of heteroscedasticity",
    "tukey": "Posthoc Tukey (Tukey-Kramer): handles unequal sample sizes but assumes equal variances",
    "games_howell": "Posthoc Games Howell: good for unequal sample sizes and heteroscedasticity",
    "homogeneity_instruction": "significant tests show heteroscedasticity and suggest the use of Welch or alternative procedures. Levene test depends on normality: Non normal distributions may result in false significant results. Sample size may affect test results",
}


def report_oneway(df, dv, iv, file=None, w=10, h=10, base_size=10, note="", title="", type="ci",
                  plot_means=False, plot_diagnostics=False, pb=False):
    """
    One-way ANOVA report for every dv x iv combination (port of
    rwf::report_oneway): Fisher's and Welch's F tests, Kruskal-Wallis,
    Levene's (mean-centred) and Bartlett's tests, and Tukey and Games-Howell
    post hoc comparisons, with a Bonferroni adjustment for the number of
    combinations. Levels with a single observation are dropped first.

    Parameters:
    df (pandas.DataFrame): Input data.
    dv (str, int or list): Dependent variables, as names or 1-based positions.
    iv (str, int or list): Grouping variables, as names or 1-based positions.
    file (str, optional): Output filename without extension: writes an Excel
        workbook (Fisher, Welch, Kruskal, Homogeneity, Tukey, Games-Howell,
        Descriptives sheets) and, if requested, PDF plots.
    w, h (float, optional): PDF page size in inches. Default 10.
    base_size (float, optional): Base font size of the plots. Default 10.
    note, title (str, optional): Passed to plot_oneway.
    type (str, optional): Error bars of plot_oneway ("ci" or "se"). Default "ci".
    plot_means (bool, optional): Write plots of the group means. Default False.
    plot_diagnostics (bool, optional): Write diagnostic plots. Default False.
    pb (bool, optional): Print progress. Default False.

    Returns:
    dict: instructions, fisher, welch, kruskal_wallis, games_howell, tukey,
    homogeneity (Levene and Bartlett), as in rwf.

    Examples:
    >>> report_oneway(df=df_blood_pressure, dv=["bp_before", "bp_after"], iv=["sex", "agegrp"])
    """
    dvs, ivs = resolve_columns(df, dv), resolve_columns(df, iv)
    fisher, welch, kruskal, tukey, games_howell, levene_rows, bartlett_rows = [], [], [], [], [], [], []
    combinations_ = [(factor, variable) for variable in dvs for factor in ivs]
    for i, (factor, variable) in enumerate(combinations_, start=1):
        if pb:
            print(f"report_oneway {i}/{len(combinations_)}")
        temp = df[df[[factor, variable]].notna().all(axis=1)].copy()
        counts = temp[factor].value_counts()
        temp = temp[temp[factor].isin(counts.index[counts > 1])]
        if temp[factor].nunique() <= 1:
            continue
        formula = f"{variable} ~ {factor}"
        fisher.append(pd.concat([pd.DataFrame({"DV": [variable], "IV": [factor]}),
                                 compute_one_way_test(formula, temp, var_equal=True)], axis=1))
        welch.append(pd.concat([pd.DataFrame({"DV": [variable], "IV": [factor]}),
                                compute_one_way_test(formula, temp, var_equal=False)], axis=1))
        kruskal.append(pd.concat([pd.DataFrame({"IV": [factor], "DV": [variable]}),
                                  compute_kruskal_wallis_test(formula, temp)], axis=1))
        samples = [temp.loc[temp[factor] == level, variable].astype(float) for level in r_levels(temp[factor])]
        levene_statistic, levene_p = stats.levene(*samples, center="mean")
        bartlett_statistic, bartlett_p = stats.bartlett(*samples)
        levene_rows.append({"Test": "Levene", "DV": variable, "IV": factor, "Statistic": levene_statistic,
                            "df_1": len(samples) - 1, "df_2": len(temp) - len(samples), "p": levene_p})
        bartlett_rows.append({"Test": "Bartlett", "DV": variable, "IV": factor, "Statistic": bartlett_statistic,
                              "df_1": float(len(samples) - 1), "p": bartlett_p})
        post_hoc = compute_posthoc(temp[variable], temp[factor])["output"]
        for table, label, key, rows in [("tukey", "Tukey", "Method", tukey),
                                        ("games.howell", "Games Howell", "method", games_howell)]:
            out = post_hoc[table]
            rows.append(pd.DataFrame({key: label, "IV": factor, "DV": variable, "LEVEL": out.index,
                                      "t": out["t"].to_numpy(), "df": out["df"].to_numpy(), "p": out["p"].to_numpy()}))
    adjustment = compute_adjustment(0.05, len(combinations_))["bonferroni"]

    def finish(blocks, significant=True):
        table = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        if len(table):
            table["bonferroni_p"] = adjustment
            if significant:
                table["significant"] = np.where(table["p"] < adjustment, "TRUE", "FALSE")
        return table

    homogeneity = pd.concat([finish([pd.DataFrame(levene_rows)], False), finish([pd.DataFrame(bartlett_rows)], False)],
                            ignore_index=True)
    homogeneity = homogeneity[["Test", "DV", "IV", "Statistic", "df_1", "df_2", "p", "bonferroni_p"]]
    homogeneity["significant"] = np.where(homogeneity["p"] < adjustment, "TRUE", "FALSE")
    result = {
        "instructions": dict(_ONEWAY_INSTRUCTIONS),
        "fisher": finish(fisher), "welch": finish(welch), "kruskal_wallis": finish(kruskal),
        "games_howell": finish(games_howell), "tukey": finish(tukey), "homogeneity": homogeneity,
    }

    descriptives = compute_descriptives(df=df, dv=dvs, iv=ivs, file=None)
    if plot_diagnostics:
        diagnostics = plot_oneway_diagnostics(df, dvs, ivs, base_size=base_size)
        report_pdf(plotlist=[p for p in diagnostics.values() if p is not None],
                   file=file, title="diagnostics", w=w, h=h, print_plot=False)
    if plot_means:
        oneway_means = plot_oneway(df, dv=dvs, iv=ivs, base_size=base_size, note=note, title=title, type=type)
        report_pdf(plotlist=[p for p in oneway_means["plots"].values() if p is not None],
                   file=file, title="means", w=w, h=h, print_plot=False)

    if file is not None:
        comment_text = {
            "DV": "Dependent Variable", "IV": "Independent Variable",
            "bonferroni_p": "Bonferroni adjustment\nadjusted critical value of p for familywise error",
            "significant": "Significant test after familywise error (Bonferroni) adjustment",
        }
        comment_fisher_welch = {**comment_text,
            "formula": "Model specification",
            "ss_effect": "Sum of Squares\nfor Effect", "ss_error": "Sum of Squares\nfor Error",
            "ms_effect": "Mean Sum of Squares\nfor Effect", "ms_error": "Mean Sum of Squares\nfor Error",
            "df_effect": "Degrees of Freedom\nfor Effect", "df_error": "Degrees of Freedom\nfor Error",
            "etasq": "Effect size\neta squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
            "partial.etasq": "Effect size\npartial eta squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
            "omegasq": "Effect size\nomega squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
            "partial.omegasq": "Effect size\npartial omega squared",
            "cohens.f": "Effect size\nCohen's f\n0.10 ~ small\n0.25 ~ medium\n0.40 ~ large",
            "statistic": "F"}
        comment_kruskal = {**comment_text, "formula": "Model specification",
                           "etasq": "Effect size\neta squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
                           "df": "Degrees of Freedom"}
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            for table, sheet, comment in [("fisher", "Fisher", comment_fisher_welch),
                                          ("welch", "Welch", comment_fisher_welch),
                                          ("kruskal_wallis", "Kruskal", comment_kruskal),
                                          ("homogeneity", "Homogeneity", comment_text),
                                          ("tukey", "Tukey", comment_text),
                                          ("games_howell", "Games-Howell", comment_text)]:
                excel_critical_value(result[table], writer, sheetname=sheet, comments=comment, critical={"p": "<0.05"})
            excel_critical_value(descriptives, writer, sheetname="Descriptives")
    return result
##########################################################################################
# FACTORIAL ANOVA
##########################################################################################
def _r_terms(factors):
    """
    Terms of R's a*b*c*... expansion in R's order: by degree, then by the binary code of the
    factors involved, first factor the lowest bit (a, b, c, a:b, a:c, b:c, a:d, b:d, c:d, ...).
    """
    subsets = [[f for i, f in enumerate(factors) if code >> i & 1] for code in range(1, 2 ** len(factors))]
    return sorted(subsets, key=len)


def _sum_contrast(k):
    """R's contr.sum(k): k x (k - 1)."""
    return np.vstack([np.eye(k - 1), -np.ones((1, k - 1))]) if k > 1 else np.zeros((1, 0))


def _orthonormal_contrast(k):
    """Orthonormal basis of the contrasts among k levels (k x (k - 1))."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(k), _sum_contrast(k)]))
    return q[:, 1:]


def _term_columns(codes, levels, term):
    """Sum-contrast model matrix columns of one term (row-wise products of the factors' contrasts)."""
    columns = np.ones((len(next(iter(codes.values()))), 1))
    for factor in term:
        contrast = _sum_contrast(len(levels[factor]))[codes[factor]]
        columns = np.einsum("ni,nj->nij", columns, contrast).reshape(len(columns), -1)
    return columns


def _effect_name(between_term, within_term):
    names = list(between_term) + list(within_term)
    return ":".join(names) if names else "(Intercept)"


def _mauchly(error, df, n_cells):
    """
    car's Mauchly test of sphericity on an error SSCP matrix of orthonormal contrasts. As in car,
    one term of w2 uses the total number of within cells (n_cells) instead of the term's p.
    """
    p = error.shape[0]
    log_w = np.log(np.linalg.det(error)) - p * np.log(np.trace(error) / p)
    rho = 1 - (2 * p ** 2 + p + 2) / (6 * p * df)
    w2 = (p + 2) * (p - 1) * (p - 2) * (2 * p ** 3 + 6 * p ** 2 + 3 * n_cells + 2) / (288 * (df * p * rho) ** 2)
    z = -df * rho * log_w
    f = p * (p + 1) / 2 - 1
    pr1, pr2 = stats.chi2.sf(z, f), stats.chi2.sf(z, f + 4)
    return float(np.exp(log_w)), float(pr1 + w2 * (pr2 - pr1))


def _sphericity_epsilons(error, df):
    """Greenhouse-Geisser and Huynh-Feldt epsilons, as car computes them."""
    p = error.shape[0]
    eigenvalues = np.linalg.eigvalsh(error)
    eigenvalues = eigenvalues[eigenvalues > 0]
    gg = (np.sum(eigenvalues) / p) ** 2 / (np.sum(eigenvalues ** 2) / p)
    hf = ((df + 1) * p * gg - 2) / (p * (df - p * gg))
    return float(gg), float(hf)


class _Design:
    """Subjects x within-cells layout of one dependent variable, with factor codings."""

    def __init__(self, data, dv, wid, within, between):
        self.wid, self.within, self.between = wid, within, between
        self.levels = {f: r_levels(data[f]) for f in within + between}
        self.subjects = r_levels(data[wid])
        subject_index = {s: i for i, s in enumerate(self.subjects)}
        # cells with the first within factor varying fastest
        self.within_cells = [tuple(reversed(c)) for c in product(*[self.levels[f] for f in reversed(within)])]
        cell_index = {c: j for j, c in enumerate(self.within_cells)}
        y = np.full((len(self.subjects), max(len(self.within_cells), 1)), np.nan)
        between_codes = {f: np.full(len(self.subjects), -1) for f in between}
        for _, row in data.iterrows():
            i = subject_index[row[wid]]
            j = cell_index[tuple(row[f] for f in within)] if within else 0
            y[i, j] = row[dv]
            for f in between:
                code = self.levels[f].index(row[f])
                if between_codes[f][i] not in (-1, code):
                    raise ValueError(f"between-subjects factor {f} varies within {wid} {row[wid]}")
                between_codes[f][i] = code
        if np.isnan(y).any():
            raise ValueError(f"every {wid} needs a value of {dv} in every within-subjects cell")
        self.y = y
        self.between_codes = between_codes
        self.between_terms = [[]] + _r_terms(between)
        blocks = [np.ones((len(self.subjects), 1))] + [
            _term_columns(between_codes, self.levels, t) for t in self.between_terms[1:]]
        self.x = np.column_stack(blocks)
        self.x_blocks = []
        start = 0
        for block in blocks:
            self.x_blocks.append(list(range(start, start + block.shape[1])))
            start += block.shape[1]
        self.within_terms = [[]] + _r_terms(within)

    def within_contrast(self, term):
        """Orthonormal contrast matrix (within cells x p) of a within-subjects term."""
        matrix = np.ones((1, 1))
        for factor in reversed(self.within):
            k = len(self.levels[factor])
            m = _orthonormal_contrast(k) if factor in term else np.ones((k, 1)) / np.sqrt(k)
            matrix = np.kron(matrix, m)
        return matrix


def _mlm_univariate_tests(design, white_adjust):
    """Type III univariate tests of every between x within term (car::Anova on an mlm)."""
    x, n = design.x, design.x.shape[0]
    xtx_inv = np.linalg.pinv(x.T @ x)
    df_error = n - np.linalg.matrix_rank(x)
    rows, sphericity = [], []
    for within_term in design.within_terms:
        z = design.y @ design.within_contrast(within_term) if design.within else design.y
        coef = xtx_inv @ x.T @ z
        residuals = z - x @ coef
        error = residuals.T @ residuals
        p = z.shape[1]
        if white_adjust and not design.within:
            leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
            meat = x.T @ (x * (residuals[:, 0] ** 2 / (1 - leverage) ** 2)[:, None])
            hc3 = xtx_inv @ meat @ xtx_inv
        for between_term, columns in zip(design.between_terms, design.x_blocks):
            q = len(columns)
            estimate = coef[columns]
            if white_adjust and not design.within:
                wald = float(estimate[:, 0] @ np.linalg.solve(hc3[np.ix_(columns, columns)], estimate[:, 0]))
                f_value = wald / q
                rows.append({"Effect": _effect_name(between_term, within_term), "DFn": q, "DFd": df_error,
                             "F": f_value, "p": stats.f.sf(f_value, q, df_error)})
                continue
            hypothesis = estimate.T @ np.linalg.solve(xtx_inv[np.ix_(columns, columns)], estimate)
            ssn, ssd = float(np.trace(hypothesis)), float(np.trace(error))
            dfn, dfd = q * p, df_error * p
            f_value = (ssn / dfn) / (ssd / dfd)
            rows.append({"Effect": _effect_name(between_term, within_term), "DFn": dfn, "DFd": dfd,
                         "SSn": ssn, "SSd": ssd, "F": f_value, "p": stats.f.sf(f_value, dfn, dfd)})
            if p > 1:
                w, p_w = _mauchly(error, df_error, design.y.shape[1])
                gg, hf = _sphericity_epsilons(error, df_error)
                hf_used = min(1.0, hf)
                sphericity.append({"Effect": rows[-1]["Effect"], "W": w, "p_W": p_w, "GGe": gg,
                                   "p[GG]": stats.f.sf(f_value, dfn * gg, dfd * gg), "HFe": hf,
                                   "p[HF]": stats.f.sf(f_value, dfn * hf_used, dfd * hf_used)})
    order = ["(Intercept)"] + [":".join(t) for t in _r_terms(design.between + design.within)]
    anova = pd.DataFrame(rows)
    anova = anova.iloc[np.argsort([order.index(e) for e in anova["Effect"]], kind="stable")].reset_index(drop=True)
    sphericity = pd.DataFrame(sphericity)
    if len(sphericity):
        sphericity = sphericity.iloc[np.argsort([order.index(e) for e in sphericity["Effect"]], kind="stable")]
        sphericity = sphericity.reset_index(drop=True)
    return anova, sphericity


def _levene_between(design):
    """ez's Levene test (car::leveneTest, median-centred): one-way ANOVA of |y - cell median|."""
    y = design.y[:, 0]
    cells = pd.Series(list(zip(*[design.between_codes[f] for f in design.between])))
    medians = pd.Series(y).groupby(cells).transform("median")
    absolute = pd.Series(np.abs(y - medians.to_numpy()))
    groups = absolute.groupby(cells)
    grand = absolute.mean()
    ssn = float(sum(len(g) * (g.mean() - grand) ** 2 for _, g in groups))
    ssd = float(sum(((g - g.mean()) ** 2).sum() for _, g in groups))
    dfn, dfd = groups.ngroups - 1, len(y) - groups.ngroups
    f_value = (ssn / dfn) / (ssd / dfd)
    return pd.DataFrame([{"DFn": dfn, "DFd": dfd, "SSn": ssn, "SSd": ssd, "F": f_value,
                          "p": stats.f.sf(f_value, dfn, dfd)}])


class _Strata:
    """Error strata of aov(y ~ between * within + Error(wid / (within))) on long data."""

    def __init__(self, long, dv, wid, within, between, levels):
        self.long, self.dv, self.wid, self.within, self.between = long, dv, wid, within, between
        self.levels = levels
        self.y = long[dv].to_numpy(dtype=float)
        n_subjects = long[wid].nunique()
        if within:
            self.names = [wid] + [":".join([wid] + t) for t in _r_terms(within)]
            self.within_sets = [[]] + _r_terms(within)
            self.ranks = [n_subjects - 1] + [n_subjects * int(np.prod([len(levels[f]) - 1 for f in t]))
                                             for t in _r_terms(within)]
        else:
            self.names, self.within_sets, self.ranks = [None], [None], [len(long) - 1]

    def _average(self, v, keys):
        if not keys:
            return np.full(len(v), v.mean())
        cache = self.__dict__.setdefault("_codes", {})
        key = tuple(keys)
        if key not in cache:
            arrays = [self.long[k].to_numpy() for k in keys]
            codes = (pd.MultiIndex.from_arrays(arrays).factorize()[0] if len(keys) > 1
                     else pd.factorize(arrays[0])[0])
            cache[key] = (codes, np.bincount(codes))
        codes, counts = cache[key]
        return (np.bincount(codes, weights=v) / counts)[codes]

    def project(self, v, stratum):
        """Projection of an observation vector onto one error stratum."""
        within_set = self.within_sets[stratum]
        if within_set is None:
            return v - v.mean()
        result = np.zeros_like(v, dtype=float)
        for size in range(len(within_set) + 1):
            for subset in combinations(within_set, size):
                sign = (-1) ** (len(within_set) - size)
                result += sign * self._average(v, [self.wid] + list(subset))
        if not within_set:
            result -= v.mean()
        return result


def _aov_table(strata):
    """Sequential (Type I) sums of squares per error stratum, as summary(aov) gives them."""
    long = strata.long
    factors = strata.between + strata.within
    codes = {f: np.array([strata.levels[f].index(v) for v in long[f]]) for f in factors}
    terms = _r_terms(factors)
    columns = {":".join(t): _term_columns(codes, strata.levels, t) for t in terms}
    rows, residual_ms = [], []
    for s, name in enumerate(strata.names):
        y_s = strata.project(strata.y, s)
        basis = np.zeros((len(y_s), 0))
        term_rows = []
        for term, x in columns.items():
            x_s = np.column_stack([strata.project(x[:, j], s) for j in range(x.shape[1])])
            if basis.shape[1]:
                x_s = x_s - basis @ (basis.T @ x_s)
            u, singular, _ = np.linalg.svd(x_s, full_matrices=False)
            new = u[:, singular > 1e-8 * max(1.0, singular.max(initial=0))]
            if new.shape[1] == 0:
                continue
            ss = float(np.sum((new.T @ y_s) ** 2))
            term_rows.append({"group": name, "term": term, "sumsq": ss, "df": new.shape[1]})
            basis = np.column_stack([basis, new])
        df_res = strata.ranks[s] - basis.shape[1]
        ss_res = float(y_s @ y_s - sum(r["sumsq"] for r in term_rows))
        ms_res = ss_res / df_res if df_res > 0 else np.nan
        for row in term_rows:
            row["meansq"] = row["sumsq"] / row["df"]
            row["statistic"] = row["meansq"] / ms_res if df_res > 0 else np.nan
            row["p.value"] = stats.f.sf(row["statistic"], row["df"], df_res) if df_res > 0 else np.nan
        rows.extend(term_rows)
        if df_res > 0:
            rows.append({"group": name, "term": "Residuals", "sumsq": ss_res, "df": df_res, "meansq": ms_res,
                         "statistic": np.nan, "p.value": np.nan})
        residual_ms.append((ms_res, df_res))
    return pd.DataFrame(rows), residual_ms


def _anova_stats(table, digits=3):
    """sjstats::anova_stats on a summary(aov) table (every row but the last is treated as a term)."""
    ss, df, ms = table["sumsq"].to_numpy(), table["df"].to_numpy(dtype=float), table["meansq"].to_numpy()
    n_terms = len(table) - 1
    ms_resid, ss_resid, ss_total, n = ms[-1], ss[-1], ss.sum(), df.sum() + 1
    t = slice(0, n_terms)
    etasq = ss[t] / ss_total
    partial_etasq = ss[t] / (ss[t] + ss_resid)
    omegasq = (ss[t] - df[t] * ms_resid) / (ss_total + ms_resid)
    partial_omegasq = (df[t] * (ms[t] - ms_resid)) / (df[t] * ms[t] + (n - df[t]) * ms_resid)
    epsilonsq = (ss[t] - df[t] * ms_resid) / ss_total
    cohens_f = np.sqrt(partial_etasq / (1 - partial_etasq))
    effect = pd.DataFrame({"etasq": etasq, "partial.etasq": partial_etasq, "omegasq": omegasq,
                           "partial.omegasq": partial_omegasq, "epsilonsq": epsilonsq, "cohens.f": cohens_f})
    effect = pd.concat([effect, pd.DataFrame([{c: np.nan for c in effect.columns}])], ignore_index=True)
    out = pd.concat([effect, table.reset_index(drop=True)], axis=1)
    df2 = df[-1]
    if (cohens_f ** 2 < 0).any() or (df[t] < 1).any() or df2 < 1:
        power = [np.nan] * len(out)
    else:
        lam = cohens_f ** 2 * (df[t] + df2 + 1)
        power = list(stats.ncf.sf(stats.f.isf(0.05, df[t], df2), df[t], df2, lam)) + [np.nan]
    out["power"] = power
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].apply(lambda c: c.map(lambda v: r_round(v, digits)))
    return out


def _pairwise_means(strata, aov_ms, factors_subset, family_factors):
    """emmeans pairwise comparisons (Tukey adjusted) of the cells of factors_subset."""
    long = strata.long
    model_factors = strata.between + strata.within
    full_levels = [strata.levels[f] for f in model_factors]
    full_cells = [tuple(reversed(c)) for c in product(*reversed(full_levels))]
    full_index = {c: i for i, c in enumerate(full_cells)}
    obs_cell = np.array([full_index[c] for c in zip(*[long[f].to_numpy() for f in model_factors])])
    counts = np.bincount(obs_cell, minlength=len(full_cells)).astype(float)
    cell_means = np.bincount(obs_cell, weights=strata.y, minlength=len(full_cells)) / counts
    positions = [model_factors.index(f) for f in factors_subset]
    level_lists = [strata.levels[f] for f in factors_subset]
    cells = [tuple(reversed(c)) for c in product(*reversed(level_lists))]  # first factor fastest
    labels = [" ".join(r_label(v) for v in c) for c in cells]
    membership = np.array([[tuple(fc[p] for p in positions) == c for fc in full_cells] for c in cells], dtype=float)
    weights = membership / membership.sum(axis=1, keepdims=True)  # equal weights over the other factors
    means = weights @ cell_means
    strata_used = [(s, ms, df_s) for s, (ms, df_s) in enumerate(aov_ms) if df_s > 0]
    rows = []
    for i, j in combinations(range(len(cells)), 2):
        a = ((weights[i] - weights[j]) / counts)[obs_cell]
        parts = []
        for s, ms, df_s in strata_used:
            projected = strata.project(a, s)
            parts.append((ms * float(projected @ projected), df_s))
        variance = sum(v for v, _ in parts)
        df = variance ** 2 / sum(v ** 2 / d for v, d in parts if v > 1e-14 * variance)
        estimate = means[i] - means[j]
        se = np.sqrt(variance)
        t_ratio = estimate / se
        rows.append({"contrast": f"{labels[i]} - {labels[j]}", "estimate": estimate, "SE": se, "df": df,
                     "t.ratio": t_ratio,
                     "p.value": float(stats.studentized_range.sf(abs(t_ratio) * np.sqrt(2), len(cells), df))})
    return pd.DataFrame(rows)


_FACTORIAL_COMMENT = {
    "DFn": "Degrees of Freedom\nfor numerator", "DFd": "Degrees of Freedom\nfor denominator",
    "SSn": "Sum of Squares\nfor numerator", "SSd": "Sum of Squares\nfor denominator",
    "dv": "Dependent Variable", "df": "Degrees of Freedom", "sumsq": "Sum of squares",
    "meansq": "Mean sum of squares", "statistic": "F value", "p.value": "p for F value",
    "etasq": "Effect size\neta squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
    "partial.etasq": "Effect size\npartial eta squared",
    "omegasq": "Effect size\nomega squared\n0.01 ~ small\n0.06 ~ medium\n0.14 ~ large",
    "partial.omegasq": "Effect size\npartial omega squared", "epsilonsq": "Effect size\nepsilon squared",
    "cohens.f": "Effect size\nCohen's f\n0.10 ~ small\n0.25 ~ medium\n0.40 ~ large", "power": "power",
    "DFn[L]": "Levene test\nDegrees of Freedom\nfor numerator", "DFd[L]": "Levene test\nDegrees of Freedom\nfor denominator",
    "SSn[L]": "Levene test\nSum of squares\nfor numerator", "SSd[L]": "Levene test\nSum of squares\nfor denominator",
    "F[L]": "Levene test\nF", "p[L]": "Levene test\np\nif significant, the assumption of homoscedasticity is violated",
    "W[M]": "Mauchly's Test\nW", "p[M]": "Mauchly's Test\np\nIf significant, the assumption of sphericity is violated",
    "GGe": "Greenhouse-Geisser\nepsilon", "p[GG]": "Greenhouse-Geisser\np adjusted for violated sphericity",
    "HFe": "Huynh-Feldt\n epsilon", "p[HF]": "Huynh-Feldt\np adjusted for violated sphericity",
}


def report_factorial_anova(df, dv, wid, within=None, within_full=None, between=None, within_covariates=None,
                           between_covariates=None, observed=None, diff=None, reverse_diff=False, type=3,
                           white_adjust=True, detailed=True, return_aov=True, file=None, post_hoc_test=True):
    """
    Factorial ANOVA for between-subjects, within-subjects (repeated measures)
    and mixed designs (port of rwf::report_factorial_anova, which wraps
    ez::ezANOVA, sjstats::anova_stats and emmeans), with Levene's test,
    Mauchly's test and Greenhouse-Geisser / Huynh-Feldt corrections, effect
    sizes and Tukey-adjusted pairwise comparisons of every main effect and
    interaction.

    The data are first averaged per participant and condition. The omnibus
    tests are Type III with sum-to-zero contrasts. With white_adjust=True
    (default) a between-subjects-only design uses HC3 heteroscedasticity
    corrected F tests, and, as in ez, the sums of squares and generalized
    eta-squared (ges) are then not reported. Effect sizes use sequential
    sums of squares per error stratum, as sjstats does. Within-subjects
    designs must be complete: every participant in every within cell.

    Not ported: within_covariates, between_covariates and diff (ez's
    covariate and difference-score paths) raise NotImplementedError, and
    type must be 3. return_aov is accepted for compatibility; the strata
    tables are returned instead of an aov object.

    Parameters:
    df (pandas.DataFrame): Long data, one row per observation.
    dv (str or list): Dependent variables, each analysed separately.
    wid (str): Participant id column.
    within, within_full, between (list, optional): Factor columns.
    observed (list, optional): Factors that are observed rather than
        manipulated; they change ges (only reported when white_adjust=False).
    type (int, optional): Sums of squares type; only 3. Defaults to 3.
    white_adjust (bool, optional): HC3 tests for between-only designs and,
        as in ez, no ges column. R's white.adjust. Defaults to True.
    detailed (bool, optional): Accepted for compatibility (always detailed).
    file (str, optional): Excel filename without extension.
    post_hoc_test (bool, optional): Write the post hoc sheet. Defaults to True.

    Returns:
    dict: omnibus, omnibus_effect_size, post_hoc (DataFrames, as in rwf) and
    object (per dv: the ANOVA, sphericity or Levene tables and the strata table).

    Examples:
    >>> report_factorial_anova(df=df, wid="id", dv=["DV1"], within=["W1", "W2"], between=["B1"])
    """
    within, within_full, between = list(within or []), list(within_full or []), list(between or [])
    if within_covariates or between_covariates or diff:
        raise NotImplementedError("within_covariates, between_covariates and diff are not ported")
    if type != 3:
        raise NotImplementedError("only type = 3 sums of squares are ported")
    if observed is not None and white_adjust:
        observed = None
    dvs = [dv] if isinstance(dv, str) else list(dv)
    design_columns = list(dict.fromkeys([wid] + within + within_full + between))
    data = df.copy()
    for column in design_columns:
        data[column] = data[column].map(lambda v: r_label(v) if pd.notna(v) else v)
    numeric = [c for c in data.columns if c not in design_columns and pd.api.types.is_numeric_dtype(data[c])
               and not pd.api.types.is_bool_dtype(data[c])]
    testdata = data.groupby(design_columns, sort=True)[numeric].mean().reset_index()
    factor_columns = [c for c in testdata.columns if c in set(within + within_full + between)]
    descriptives = compute_aggregate(df=testdata, iv=factor_columns, file=None)

    omnibus, effect_sizes, post_hoc, objects = [], [], [], {}
    for dependent in dvs:
        long = testdata.groupby([wid] + between + within, sort=True)[dependent].mean().reset_index()
        design = _Design(long, dependent, wid, within, between)
        anova, sphericity = _mlm_univariate_tests(design, white_adjust)
        tables = {"ANOVA": anova}
        if not white_adjust:
            unique_ssd = anova["SSd"].drop_duplicates().sum()
            if observed:
                obs = anova["Effect"].map(lambda e: any(o in e for o in observed))
                obs_ssn1, obs_ssn2 = (anova["SSn"] * obs).sum(), anova["SSn"] * obs
            else:
                obs_ssn1, obs_ssn2 = 0.0, 0.0
            anova["ges"] = anova["SSn"] / (anova["SSn"] + unique_ssd + obs_ssn1 - obs_ssn2)
        if not within:
            levene = _levene_between(design)
            levene.columns = [f"{c}[L]" for c in levene.columns]
            tables["Levene's Test for Homogeneity of Variance"] = levene
            merged = pd.concat([anova, pd.concat([levene] * len(anova), ignore_index=True)], axis=1)
        elif len(sphericity):
            mauchly = sphericity[["Effect", "W", "p_W"]].rename(columns={"W": "W[M]", "p_W": "p[M]"})
            corrections = sphericity[["Effect", "GGe", "p[GG]", "HFe", "p[HF]"]]
            tables["Mauchly's Test for Sphericity"] = mauchly
            tables["Sphericity Corrections"] = corrections
            matched = anova[anova["Effect"].isin(mauchly["Effect"])]
            unmatched = anova[~anova["Effect"].isin(mauchly["Effect"])]
            merged = pd.concat([matched.merge(mauchly, on="Effect").merge(corrections, on="Effect"), unmatched],
                               ignore_index=True)
        else:
            merged = anova
        omnibus.append(pd.concat([pd.DataFrame({"dv": [dependent] * len(merged)}), merged.reset_index(drop=True)],
                                 axis=1))

        long_obs = long.copy()
        strata = _Strata(long_obs, dependent, wid, within, between, design.levels)
        aov, residual_ms = _aov_table(strata)
        if not within:
            aov = aov.drop(columns="group")
        stats_table = _anova_stats(aov)
        effect_sizes.append(pd.concat([pd.DataFrame({"dv": [dependent] * len(stats_table)}), stats_table], axis=1))
        tables["effect_size"] = effect_sizes[-1]
        tables["aov"] = aov

        factors = list(dict.fromkeys(within + within_full + between))
        for size in range(1, len(factors) + 1):
            for subset in combinations(factors, size):
                pairs = _pairwise_means(strata, residual_ms, list(subset), factors)
                post_hoc.append(pd.concat([pd.DataFrame({"dv": [dependent] * len(pairs)}), pairs], axis=1))
        objects[dependent] = tables

    omnibus = pd.concat(omnibus, ignore_index=True)
    omnibus_effect_size = pd.concat(effect_sizes, ignore_index=True)
    post_hoc = pd.concat(post_hoc, ignore_index=True) if post_hoc else pd.DataFrame()
    result = {"omnibus": omnibus, "omnibus_effect_size": omnibus_effect_size, "post_hoc": post_hoc, "object": objects}

    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            if "p[GG]" in omnibus.columns:
                excel_critical_value(omnibus, writer, sheetname="ANOVA within", comments=_FACTORIAL_COMMENT,
                                     critical={"p": "<0.05", "p[M]": "<0.05", "p[GG]": "<0.05", "p[HF]": "<0.05"})
            if "p[L]" in omnibus.columns:
                excel_critical_value(omnibus, writer, sheetname="ANOVA between", comments=_FACTORIAL_COMMENT,
                                     critical={"p": "<0.05", "p[L]": "<0.05"})
            if "p[GG]" not in omnibus.columns and "p[L]" not in omnibus.columns:
                excel_critical_value(omnibus, writer, sheetname="ANOVA", comments=_FACTORIAL_COMMENT,
                                     critical={"p": "<0.05"})
            excel_critical_value(omnibus_effect_size, writer, sheetname="effect size", comments=_FACTORIAL_COMMENT,
                                 critical={"p.value": "<0.05"})
            if len(post_hoc) and post_hoc_test:
                excel_critical_value(post_hoc, writer, sheetname="post hoc", critical={"p.value": "<0.05"})
            excel_critical_value(descriptives, writer, sheetname="descriptives")
            call = (f"report_factorial_anova(dv={dvs},wid={wid},within={within},within_full={within_full},"
                    f"between={between},type={type},white_adjust={white_adjust})")
            excel_critical_value(pd.DataFrame({"call": [call]}), writer, sheetname="call")
    return result
##########################################################################################
# MANOVA RESULT
##########################################################################################
def _manova_term_name(name):
    """R's term label for a patsy term name: Intercept -> (Intercept), C(N, Sum):P -> N:P."""
    if name == "Intercept":
        return "(Intercept)"
    parts = []
    for part in name.split(":"):
        match = re.match(r"^C\(\s*([^,\)]+)", part)
        parts.append(match.group(1).strip() if match else part)
    return ":".join(parts)


def _multivariate_test(test, eigenvalues, p, q, df_res):
    """Statistic and approximate F of stats:::summary.manova for one term."""
    eig = np.asarray(eigenvalues)
    if test == "Pillai":
        statistic = np.sum(eig / (1 + eig))
        s, n, m = min(p, q), 0.5 * (df_res - p - 1), 0.5 * (abs(p - q) - 1)
        tmp1, tmp2 = 2 * m + s + 1, 2 * n + s + 1
        return statistic, (tmp2 / tmp1 * statistic) / (s - statistic), s * tmp1, s * tmp2
    if test == "Wilks":
        statistic = np.prod(1 / (1 + eig))
        tmp1 = df_res - 0.5 * (p - q + 1)
        tmp2 = (p * q - 2) / 4
        tmp3 = p ** 2 + q ** 2 - 5
        tmp3 = np.sqrt(((p * q) ** 2 - 4) / tmp3) if tmp3 > 0 else 1
        return (statistic, ((statistic ** (-1 / tmp3) - 1) * (tmp1 * tmp3 - 2 * tmp2)) / p / q,
                p * q, tmp1 * tmp3 - 2 * tmp2)
    if test == "Hotelling-Lawley":
        statistic = np.sum(eig)
        m, n, s = 0.5 * (abs(p - q) - 1), 0.5 * (df_res - p - 1), min(p, q)
        tmp1, tmp2 = 2 * m + s + 1, 2 * (s * n + 1)
        return statistic, (tmp2 * statistic) / s / s / tmp1, s * tmp1, tmp2
    statistic = np.max(eig)
    tmp1 = max(p, q)
    tmp2 = df_res - tmp1 + q
    return statistic, tmp2 * statistic / tmp1, tmp1, tmp2


def _manova_eigenvalues(hypothesis, error):
    """Eigenvalues of E^-1 H (real, non-negative)."""
    values = np.linalg.eigvals(np.linalg.solve(error, hypothesis)).real
    return np.sort(np.clip(values, 0, None))[::-1]


def report_manova(model, file=None):
    """
    MANOVA report (port of rwf::report_manova): Pillai's trace, Wilks' lambda,
    the Hotelling-Lawley trace and Roy's largest root with approximate F tests
    for every term, sequentially as summary(manova) gives them (intercept
    included), and the Type III MANOVA table (Pillai) of car::Anova. Both are
    printed and can be written to an Excel workbook.

    Type III tests need sum-to-zero contrasts, so fit the model with them,
    e.g. MANOVA.from_formula("y1 + y2 ~ C(a, Sum) * C(b, Sum)", data).

    Parameters:
    model (statsmodels.multivariate.manova.MANOVA): A model made with
        MANOVA.from_formula (R: a manova fit).
    file (str, optional): Excel filename without extension (sheets
        "critical" and "call").

    Returns:
    dict: multivariate (Group, Df, Statistic, approx F, num Df, den Df,
    Pr(>F), type), type_three (Df, test stat, approx F, num Df, den Df,
    Pr(>F), one row per term) and call, as in rwf.

    Examples:
    >>> from statsmodels.multivariate.manova import MANOVA
    >>> model = MANOVA.from_formula("bp_before + bp_after ~ C(sex, Sum) * C(agegrp, Sum)", data=df_blood_pressure)
    >>> report_manova(model)
    """
    x = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    slices = model.data.model_spec.term_name_slices
    n, p = y.shape
    rank = np.linalg.matrix_rank(x)
    df_res = n - rank
    xtx_inv = np.linalg.pinv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    residuals = y - x @ coef
    error = residuals.T @ residuals

    # sequential (Type I) hypothesis matrices, in model order
    sequential, basis = [], np.zeros((n, 0))
    for name, columns in slices.items():
        block = x[:, columns]
        if basis.shape[1]:
            block = block - basis @ (basis.T @ block)
        u, singular, _ = np.linalg.svd(block, full_matrices=False)
        new = u[:, singular > 1e-8 * max(1.0, singular.max(initial=0))]
        effects = new.T @ y
        sequential.append((_manova_term_name(name), new.shape[1], effects.T @ effects))
        basis = np.column_stack([basis, new])

    blocks = []
    for test in ["Pillai", "Wilks", "Hotelling-Lawley", "Roy"]:
        rows = []
        for term, q, hypothesis in sequential:
            statistic, f_value, df1, df2 = _multivariate_test(test, _manova_eigenvalues(hypothesis, error),
                                                              p, q, df_res)
            rows.append({"Group": term, "Df": q, "Statistic": statistic, "approx F": f_value, "num Df": df1,
                         "den Df": df2, "Pr(>F)": stats.f.sf(f_value, df1, df2)})
        rows.append({"Group": "Residuals", "Df": df_res, "Statistic": np.nan, "approx F": np.nan,
                     "num Df": np.nan, "den Df": np.nan, "Pr(>F)": np.nan})
        block = pd.DataFrame(rows)
        block["type"] = test
        blocks.append(block)
    pwhr = pd.concat(blocks, ignore_index=True)

    three = []
    for name, columns in slices.items():
        columns = list(range(columns.start, columns.stop))
        estimate = coef[columns]
        hypothesis = estimate.T @ np.linalg.solve(xtx_inv[np.ix_(columns, columns)], estimate)
        statistic, f_value, df1, df2 = _multivariate_test("Pillai", _manova_eigenvalues(hypothesis, error),
                                                          p, len(columns), df_res)
        three.append({"term": _manova_term_name(name), "Df": len(columns), "test stat": statistic,
                      "approx F": f_value, "num Df": df1, "den Df": df2, "Pr(>F)": stats.f.sf(f_value, df1, df2)})
    type_three = pd.DataFrame(three).set_index("term")
    type_three.index.name = None

    output_separator("Pillai,Wilks,Hotelling-Lawley,Roy Statistics", output=pwhr)
    output_separator("type Three", output=type_three)
    call = pd.DataFrame({"call": [f"MANOVA.from_formula({getattr(model, 'formula', '')!r})"]})
    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(pwhr, writer, sheetname="critical", critical={"Pr(>F)": "<0.05"})
            excel_critical_value(call, writer, sheetname="call")
    return {"multivariate": pwhr, "type_three": type_three, "call": call}
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os

    df_blood_pressure = pd.read_csv("data/blood_pressure.csv") if os.path.exists("data/blood_pressure.csv") \
        else pd.read_csv("../data/blood_pressure.csv")
    df_co2 = pd.read_csv("data/co2.csv") if os.path.exists("data/co2.csv") else pd.read_csv("../data/co2.csv")

    print("=" * 80, "\ncompute_kruskal_wallis_test\n", "=" * 80, sep="")
    print(compute_kruskal_wallis_test(formula="bp_before~agegrp", df=df_blood_pressure))

    print("\n" + "=" * 80, "\ncompute_one_way_test\n", "=" * 80, sep="")
    print(compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=True))
    print(compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=False))

    print("\n" + "=" * 80, "\ncompute_games_howell\n", "=" * 80, sep="")
    result = compute_games_howell(y=df_blood_pressure["bp_before"], x=df_blood_pressure["agegrp"])
    print(result["descriptives"])
    print(result["posthoc"])

    print("\n" + "=" * 80, "\ncompute_aov_es\n", "=" * 80, sep="")
    one_way_between = smf.ols('uptake~Treatment', data=df_co2).fit()
    factorial_between = smf.ols('uptake~Treatment*Type', data=df_co2).fit()
    for ss in ["I", "II", "III"]:
        print(f"one_way_between, ss={ss}:")
        print(compute_aov_es(model=one_way_between, ss=ss))
    for ss in ["I", "II", "III"]:
        print(f"factorial_between, ss={ss}:")
        print(compute_aov_es(model=factorial_between, ss=ss))
