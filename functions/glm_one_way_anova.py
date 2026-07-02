# -*- coding: utf-8 -*-
"""
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
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from itertools import combinations
##########################################################################################
# KRUSKALL WALLIS TEST WITH EFFECT SIZE
##########################################################################################
def compute_kruskal_wallis_test(formula, df):
    """
    Kruskal-Wallis rank sum test with eta-squared/epsilon-squared effect sizes.

    Parameters:
    formula (str): "dv ~ group_column" style formula (only the group
        column name after "~" is actually used; parsed with patsy just
        to validate/extract the DV column).
    df (pandas.DataFrame): Data containing the DV and group columns.

    Returns:
    pandas.DataFrame: One row with formula, method, etasq, epsilonsq, H, df, p.

    Examples:
    >>> import pandas as pd
    >>> df_blood_pressure = pd.read_csv("data/blood_pressure.csv")
    >>> compute_kruskal_wallis_test(formula="bp_before~agegrp", df=df_blood_pressure)
    """
    y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    x = y.iloc[:, 0].values

    group_labels = df[formula.split('~')[1].strip()].astype('category')
    k = group_labels.nunique()
    n = len(x)

    r = stats.rankdata(x)
    ranks_by_group = pd.Series(r).groupby(group_labels).agg(['sum', 'count'])

    h = np.sum((ranks_by_group['sum'] ** 2) / ranks_by_group['count'])
    H = (12 * h / (n * (n + 1))) - 3 * (n + 1)

    # Correction for ties
    ties = pd.Series(x).value_counts()
    tie_correction = 1 - np.sum(ties**3 - ties) / (n**3 - n)
    H /= tie_correction

    df_k = k - 1
    p = stats.chi2.sf(H, df_k)

    # Effect sizes
    etasq = (H - k + 1) / (n - k)
    epsilonsq = H / ((n ** 2 - 1) / (n + 1))

    result = pd.DataFrame([{
        "formula": formula,
        "method": "Kruskal-Wallis rank sum test",
        "etasq": etasq,
        "epsilonsq": epsilonsq,
        "H": H,
        "df": df_k,
        "p": p
    }])

    return result
##########################################################################################
# ONE WAY TEST WITH SS AND MS
##########################################################################################
def compute_one_way_test(formula, df, var_equal=True):
    """
    One-way ANOVA F-test built from scratch, with sums/means of squares
    and effect sizes -- Fisher's classical version (var_equal=True,
    matching scipy.stats.f_oneway) or Welch's heteroscedasticity-robust
    version (var_equal=False).

    Parameters:
    formula (str): "dv ~ group_column" style formula.
    df (pandas.DataFrame): Data containing the DV and group columns.
    var_equal (bool, optional): If True (default), assumes equal group
        variances (classical F-test). If False, uses Welch's correction.

    Returns:
    pandas.DataFrame: One row with ss/ms effect+error, etasq,
    partial.etasq, omegasq, partial.omegasq, cohens.f, power, statistic,
    df_effect, df_error, p.

    Examples:
    >>> compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=True)
    >>> compute_one_way_test(formula="bp_before ~ agegrp", df=df_blood_pressure, var_equal=False)
    """
    y_var, g_var = formula.replace(" ", "").split("~")
    y = df[y_var]
    g = df[g_var].astype("category")

    k = g.nunique()
    n_i = df.groupby(g_var)[y_var].count()
    m_i = df.groupby(g_var)[y_var].mean()
    v_i = df.groupby(g_var)[y_var].var()
    w_i = n_i / v_i
    sum_w_i = w_i.sum()
    n = len(y)
    df_effect = k - 1

    if var_equal:
        df_error = n - k
        ss_effect = np.sum(n_i * (m_i - y.mean()) ** 2)
        ss_error = np.sum((n_i - 1) * v_i)
        ms_effect = ss_effect / df_effect
        ms_error = ss_error / df_error
        method = "Assuming homoscedasticity"
    else:
        tmp = np.sum(((1 - (w_i / sum_w_i)) ** 2) / (n_i - 1)) / (k ** 2 - 1)
        df_error = 1 / (3 * tmp)
        m = np.sum(w_i * m_i) / sum_w_i
        ms_effect = np.sum(w_i * (m_i - m) ** 2)
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
    partial_omegasq = (df_effect * (ms_effect - ms_error)) / (df_effect * ms_effect + (n - df_effect) * ms_error)
    cohens_f = np.sqrt(etasq / (1 - etasq))
    lambda_ = cohens_f * (df_effect + df_error + 1)
    power = stats.ncf.sf(stats.f.isf(0.05, df_effect, df_error), df_effect, df_error, lambda_)

    result = pd.DataFrame([{
        "formula": formula,
        "method": method,
        "ss_effect": ss_effect,
        "ss_error": ss_error,
        "ms_effect": ms_effect,
        "ms_error": ms_error,
        "etasq": etasq,
        "partial.etasq": partial_etasq,
        "omegasq": omegasq,
        "partial.omegasq": partial_omegasq,
        "cohens.f": cohens_f,
        "power": power,
        "statistic": statistic,
        "df_effect": df_effect,
        "df_error": df_error,
        "p": p
    }])

    return result
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
def compute_aov_es(model, ss="I"):
    """
    Eta/partial-eta/omega/partial-omega-squared and Cohen's f effect
    sizes for every effect term in a fitted OLS ANOVA model, from its
    Type I, II, or III sums-of-squares table.

    Parameters:
    model (statsmodels RegressionResultsWrapper): A fitted
        statsmodels.formula.api.ols(...) model.
    ss (str, optional): "I", "II", or "III" -- which ANOVA sum-of-squares
        table (statsmodels.stats.anova.anova_lm) to compute effect sizes
        from. Defaults to "I".

    Returns:
    pandas.DataFrame: One row per effect term (Residual/Intercept
    excluded from the effect-size columns, but still present as rows
    with the raw ANOVA table columns), with call, ss, comparisons,
    etasq, partial_etasq, omegasq, partial_omegasq, epsilonsq, cohens_f,
    plus the underlying Df/Sum Sq/Mean Sq/F value/Pr(>F) columns.

    Examples:
    >>> import pandas as pd
    >>> import statsmodels.formula.api as smf
    >>> df_co2 = pd.read_csv("data/co2.csv")
    >>> one_way_between = smf.ols("uptake~Treatment", data=df_co2).fit()
    >>> compute_aov_es(model=one_way_between, ss="I")
    >>> factorial_between = smf.ols("uptake~Treatment*Type", data=df_co2).fit()
    >>> compute_aov_es(model=factorial_between, ss="III")
    """
    # Get the model dataframe
    model_df = model.model.data.frame
    n_total = model_df.shape[0]

    # Select appropriate summary table
    if ss == "I":
        ss1 = anova_lm(model, typ=1)
        ss1 = ss1.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'mean_sq': 'Mean Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss1 = ss1[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        summary_aov = ss1.reset_index()
    elif ss == "II":
        ss2 = anova_lm(model, typ=2)
        ss2['Mean Sq'] = ss2['sum_sq'] / ss2['df']
        ss2 = ss2.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss2 = ss2[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        summary_aov = ss2.reset_index()
    elif ss == "III":
        ss3 = anova_lm(model, typ=3)
        ss3['Mean Sq'] = ss3['sum_sq'] / ss3['df']
        ss3 = ss3.rename(columns={'df': 'Df', 'sum_sq': 'Sum Sq', 'F': 'F value', 'PR(>F)': 'Pr(>F)'})
        ss3 = ss3[['Df', 'Sum Sq', 'Mean Sq', 'F value', 'Pr(>F)']]
        ss3 = ss3.reset_index()

        # Put every effect term first (in their original order), then Residual, then
        # Intercept last -- generalizes to any model instead of assuming a single
        # factor literally named "Treatment" (the previous hardcoded order silently
        # dropped every other effect term for factorial/multi-factor models).
        effect_rows = ss3[~ss3['index'].isin(['Intercept', 'Residual'])]
        residual_row_df = ss3[ss3['index'] == 'Residual']
        intercept_row_df = ss3[ss3['index'] == 'Intercept']
        summary_aov = pd.concat([effect_rows, residual_row_df, intercept_row_df], ignore_index=True)
    else:
        raise ValueError('ss must be one of "I", "II", "III"')

    residual_row = summary_aov[summary_aov["index"].str.contains("Residual")].index[0]
    ms_effect = summary_aov.loc[:residual_row - 1, 'Mean Sq'].values
    ms_error = summary_aov.loc[residual_row, 'Mean Sq']
    df_effect = summary_aov.loc[:residual_row - 1, 'Df'].values
    df_error = summary_aov.loc[residual_row, 'Df']
    ss_effect = summary_aov.loc[:residual_row - 1, 'Sum Sq'].values
    ss_error = summary_aov.loc[residual_row, 'Sum Sq']
    ss_total = np.full_like(ss_effect, np.sum(summary_aov.loc[:, 'Sum Sq']), dtype=np.float64)

    omega = np.abs((ss_effect - df_effect * ms_error) / (ss_total + ms_error))
    partial_omega = np.abs((df_effect * (ms_effect - ms_error)) / (ss_effect + (n_total - df_effect) * ms_error))
    eta = ss_effect / ss_total
    partial_eta = ss_effect / (ss_effect + ss_error)
    cohens_f = np.sqrt(partial_eta / (1 - partial_eta))
    epsilon = (ss_effect - df_effect * ms_error) / ss_total

    comparison_names = summary_aov.loc[:residual_row - 1, 'index'].str.strip()

    result = pd.DataFrame({
        'call': str(model.model.formula),
        'ss': ss,
        'comparisons': comparison_names,
        'etasq': eta,
        'partial_etasq': partial_eta,
        'omegasq': omega,
        'partial_omegasq': partial_omega,
        'epsilonsq': epsilon,
        'cohens_f': cohens_f
    })

    summary_aov = summary_aov.rename(columns={'index': 'comparisons'})
    summary_aov['call'] = str(model.model.formula)
    summary_aov['ss'] = ss

    merged = pd.merge(summary_aov, result, how='outer', on=['call', 'ss', 'comparisons'])
    return merged
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
