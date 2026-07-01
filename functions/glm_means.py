# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:20:04 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
import itertools
from scipy.stats import levene, bartlett
try:
    from . import functions_excel as fe
except ImportError:
    import functions_excel as fe
##########################################################################################
# 
##########################################################################################
def report_ttests(df,dv,iv,paired=False,alternative="two-sided",correction=False):
    """
    Run pairwise independent t-tests across factor levels and return a
    reporting table.

    For every dependent variable in ``dv`` and every independent variable in
    ``iv``, this tests every pairwise combination of that factor's levels
    (e.g. a 3-level factor yields 3 comparisons: 1v2, 1v3, 2v3). Each
    comparison row combines descriptive statistics (means, variances,
    standard deviations, pooled sd), homogeneity-of-variance checks (Levene,
    Bartlett), and the t-test itself (via :py:func:`pingouin.ttest`).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain every column named in ``dv`` and ``iv``.
    dv : str or list of str
        Dependent variable column name(s). A single string is treated as a
        one-element list. Each dv is tested against every iv independently.
    iv : list of str
        Independent (grouping) variable column name(s). For each factor, the
        t-test is run on every pairwise combination of its unique levels.
    paired : bool, default False
        Whether the two levels being compared are related (repeated
        measures) rather than independent groups. Forwarded to
        :py:func:`pingouin.ttest`.
    alternative : str, default "two-sided"
        Alternative hypothesis / tail of the test: "two-sided", "greater",
        or "less". Forwarded to :py:func:`pingouin.ttest`.
    correction : bool or "auto", default False
        Whether to apply Welch's correction for unequal variances. If
        "auto", Welch's t-test is used automatically when group sizes are
        unequal. Forwarded to :py:func:`pingouin.ttest`.

    Returns
    -------
    pandas.DataFrame
        One row per pairwise level comparison, per (dv, iv) combination.
        Columns:

        * ``Paired`` : the ``paired`` argument used for this comparison
        * ``DV`` : dependent variable name for this row
        * ``IV`` : independent variable (factor) name for this row
        * ``L1``, ``L2`` : the two factor levels being compared
        * ``Mean_L1``, ``Mean_L2`` : dv mean within each level
        * ``Var_L1``, ``Var_L2`` : dv variance within each level
        * ``sd_L1``, ``sd_L2`` : dv standard deviation within each level
        * ``pooled_sd`` : sqrt((sd_L1**2 + sd_L2**2) / 2)
        * ``Levene``, ``Levene_p`` : Levene's test statistic and p-value for
          equal variances between L1 and L2
        * ``Bartlett``, ``Bartlett_p`` : Bartlett's test statistic and
          p-value for equal variances between L1 and L2
        * ``T``, ``dof``, ``alternative``, ``p_val``, ``CI95``,
          ``cohen_d``, ``power``, ``BF10`` : as returned by
          :py:func:`pingouin.ttest`. Note ``alternative`` appears twice
          (once from this function's own descriptives, once echoed by
          pingouin) since pandas allows duplicate column names.

    Examples
    --------
    >>> report_ttests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    >>> report_ttests(df=df_blood_pressure, dv=["bp_before", "bp_after"], iv=["sex", "agegrp"])
    """
    
    if isinstance(dv, str):
        dv = [dv]

    result_iterative=pd.DataFrame()
    for dependent in dv:
        for factor in iv:
            levels=df[factor].value_counts().index.values
            combinations=list(itertools.combinations(levels,2))
            for levels in combinations:
                variable_1=df[df[factor]==levels[0]][dependent].astype(float)
                variable_2=df[df[factor]==levels[1]][dependent].astype(float)
                # df_test=pd.concat(variable_1,variable_2,axis=0).dropna()
                levene_result=levene(variable_1,variable_2)
                bartlett_result=bartlett(variable_1,variable_2)
                ttest=pd.DataFrame(pg.ttest(variable_1,
                                            variable_2,
                                            paired=paired,
                                            alternative=alternative,
                                            correction=correction,
                                            r=0.707))
                descriptives=pd.DataFrame({"Paired":[paired],
                                           "DV":[dependent],
                                           "IV":[factor],
                                           "L1":[levels[0]],
                                           "L2":[levels[1]],
                                           "Mean_L1":[variable_1.mean()],
                                           "Mean_L2":[variable_2.mean()],
                                           "Var_L1":[variable_1.var()],
                                           "Var_L2":[variable_2.var()],
                                           "sd_L1":[variable_1.std()],
                                           "sd_L2":[variable_2.std()],
                                           "pooled_sd":[np.sqrt(((variable_1.std()**2)+(variable_2.std()**2))/2)],
                                           "Levene":[levene_result[0]],
                                           "Levene_p":[levene_result[1]],
                                           "Bartlett":[bartlett_result[0]],
                                           "Bartlett_p":[bartlett_result[1]]},
                                            index=ttest.index)
                result=pd.concat([descriptives,ttest],axis=1)
                result_iterative=pd.concat([result_iterative,result],axis=0)
    return result_iterative

import pandas as pd
DATA_DIR = pathlib.Path("/home/dimitrios/GitHub/pwf") / "data"
df_blood_pressure=pd.read_csv(DATA_DIR / "blood_pressure.csv")
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_ttests(df=df_blood_pressure, dv="bp_before", iv=["sex","agegrp"]).round(2))
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_ttests(df=df_blood_pressure, dv=["bp_before","bp_after"], iv=["sex","agegrp"]).round(2))
df_long = pd.melt(
    df_blood_pressure,
    id_vars=['patient', 'sex', 'agegrp'],
    value_vars=['bp_before', 'bp_after'],
    var_name='time',
    value_name='bp'
)
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_ttests(df=df_long, dv='bp', iv=['time'], paired=True))
##########################################################################################
# WILCOXON / MANN-WHITNEY
##########################################################################################
def report_wtests(df,dv,iv,paired=False,alternative="two-sided"):
    """
    Run pairwise Mann-Whitney U or Wilcoxon signed-rank tests across
    factor levels and return a reporting table.

    Non-parametric analogue of :py:func:`report_ttests`: for every
    dependent variable in ``dv`` and every independent variable in ``iv``,
    this tests every pairwise combination of that factor's levels (e.g. a
    3-level factor yields 3 comparisons: 1v2, 1v3, 2v3). Each comparison row
    combines descriptive statistics (means, variances, standard deviations,
    pooled sd), homogeneity-of-variance checks (Levene, Bartlett), and the
    rank test itself: Mann-Whitney U (via :py:func:`pingouin.mwu`) when
    ``paired=False`` for independent groups, or the Wilcoxon signed-rank
    test (via :py:func:`pingouin.wilcoxon`) when ``paired=True`` for
    repeated measures / matched pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain every column named in ``dv`` and ``iv``.
    dv : str or list of str
        Dependent variable column name(s). A single string is treated as a
        one-element list. Each dv is tested against every iv independently.
    iv : list of str
        Independent (grouping) variable column name(s). For each factor, the
        test is run on every pairwise combination of its unique levels.
    paired : bool, default False
        If False (default), runs the independent-samples Mann-Whitney U
        test — each level's rows are treated as separate, unrelated
        subjects. If True, runs the Wilcoxon signed-rank test instead,
        which requires the two level-slices to be the same length and
        row-aligned (e.g. via a long-format reshape of repeated measures,
        the same pattern used with ``report_ttests(..., paired=True)``).
    alternative : str, default "two-sided"
        Alternative hypothesis / tail of the test: "two-sided", "greater",
        or "less". Forwarded to :py:func:`pingouin.mwu` /
        :py:func:`pingouin.wilcoxon`.

    Returns
    -------
    pandas.DataFrame
        One row per pairwise level comparison, per (dv, iv) combination.
        Columns:

        * ``DV`` : dependent variable name for this row
        * ``IV`` : independent variable (factor) name for this row
        * ``L1``, ``L2`` : the two factor levels being compared
        * ``Mean_L1``, ``Mean_L2`` : dv mean within each level
        * ``Var_L1``, ``Var_L2`` : dv variance within each level
        * ``sd_L1``, ``sd_L2`` : dv standard deviation within each level
        * ``pooled_sd`` : sqrt((sd_L1**2 + sd_L2**2) / 2)
        * ``Levene``, ``L_p`` : Levene's test statistic and p-value for
          equal variances between L1 and L2
        * ``Bartlett``, ``B_p`` : Bartlett's test statistic and p-value for
          equal variances between L1 and L2
        * ``U_val`` (paired=False) or ``W_val`` (paired=True),
          ``alternative``, ``p_val``, ``RBC``, ``CLES`` : as returned by
          :py:func:`pingouin.mwu` / :py:func:`pingouin.wilcoxon`. Note
          ``alternative`` appears twice (once from this function's own
          descriptives, once echoed by pingouin) since pandas allows
          duplicate column names.

    Examples
    --------
    >>> report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex", "agegrp"])
    >>> report_wtests(df=df_long, dv="bp", iv=["time"], paired=True)
    """


    if isinstance(dv, str):
        dv = [dv]

    result_iterative=pd.DataFrame()
    for dependent in dv:
        for factor in iv:
            levels=df[factor].value_counts().index.values
            combinations=list(itertools.combinations(levels,2))
            for levels in combinations:
                variable_1=df[df[factor]==levels[0]][dependent].astype(float)
                variable_2=df[df[factor]==levels[1]][dependent].astype(float)
                levene_result=levene(variable_1,variable_2)
                bartlett_result=bartlett(variable_1,variable_2)
                if paired:
                    wtest=pd.DataFrame(pg.wilcoxon(variable_1,
                                              variable_2,
                                              alternative=alternative))
                else:
                    wtest=pd.DataFrame(pg.mwu(variable_1,
                                              variable_2,
                                              alternative=alternative))
                descriptives=pd.DataFrame({"Paired":[paired],
                                           "alternative":[alternative],
                                           "DV":[dependent],
                                           "IV":[factor],
                                           "L1":[levels[0]],
                                           "L2":[levels[1]],
                                           "Mean_L1":[variable_1.mean()],
                                           "Mean_L2":[variable_2.mean()],
                                           "Var_L1":[variable_1.var()],
                                           "Var_L2":[variable_2.var()],
                                           "sd_L1":[variable_1.std()],
                                           "sd_L2":[variable_2.std()],
                                           "pooled_sd":[np.sqrt(((variable_1.std()**2)+(variable_2.std()**2))/2)],
                                           "Levene":[levene_result[0]],
                                           "L_p":[levene_result[1]],
                                           "Bartlett":[bartlett_result[0]],
                                           "B_p":[bartlett_result[1]]},
                                            index=wtest.index)
                result=pd.concat([descriptives,wtest],axis=1)
                result_iterative=pd.concat([result_iterative,result],axis=0)
    return result_iterative

import pandas as pd
DATA_DIR = pathlib.Path("/home/dimitrios/GitHub/pwf") / "data"
df_blood_pressure=pd.read_csv(DATA_DIR / "blood_pressure.csv")
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_wtests(df=df_blood_pressure, dv="bp_before", iv=["sex","agegrp"]).round(2))
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_wtests(df=df_blood_pressure, dv=["bp_before","bp_after"], iv=["sex","agegrp"]).round(2))
df_long = pd.melt(
    df_blood_pressure,
    id_vars=['patient', 'sex', 'agegrp'],
    value_vars=['bp_before', 'bp_after'],
    var_name='time',
    value_name='bp'
)
with pd.option_context('display.max_columns', None, 'display.width', 180):
    print(report_wtests(df=df_long, dv='bp', iv=['time'], paired=True))
##########################################################################################
# LEVENE
##########################################################################################
from scipy.stats import levene
import pandas as pd
import itertools

def report_levene_bartlett(df, dv, iv):
    """
    Run pairwise Levene and Bartlett homogeneity-of-variance tests across
    factor levels and return a reporting table.

    For every independent variable in ``iv``, this tests every pairwise
    combination of that factor's levels (e.g. a 3-level factor yields 3
    comparisons: 1v2, 1v3, 2v3), checking whether ``dv`` has equal variance
    between the two levels via Levene's test and Bartlett's test.

    This is a standalone version of the same Levene/Bartlett columns that
    :py:func:`report_ttests` and :py:func:`report_wtests` compute inline —
    use it when you only need the variance-homogeneity check, without
    running a t-test/Mann-Whitney U alongside it.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain ``dv`` and every column named in ``iv``.
    dv : str
        Dependent variable column name. Unlike :py:func:`report_ttests` /
        :py:func:`report_wtests`, only a single column name is supported
        (not a list).
    iv : list of str
        Independent (grouping) variable column name(s). For each factor,
        the tests are run on every pairwise combination of its unique
        levels.

    Returns
    -------
    pandas.DataFrame
        One row per pairwise level comparison, per factor in ``iv``.
        Columns:

        * ``IV`` : independent variable (factor) name for this row
        * ``L1``, ``L2`` : the two factor levels being compared
        * ``Levene``, ``Levene_p`` : Levene's test statistic and p-value
        * ``Bartlett``, ``Bartlett_p`` : Bartlett's test statistic and
          p-value

    Notes
    -----
    ``dv`` is passed to ``levene``/``bartlett`` as-is, without casting to
    float. If ``dv`` is an integer-dtype column, this can trigger a known
    bug in some scipy versions (``bartlett`` raising
    ``ValueError: cannot convert float NaN to integer``) — see the
    ``.astype(float)`` workaround applied in :py:func:`report_ttests` /
    :py:func:`report_wtests` for the fix, which isn't applied here.

    Examples
    --------
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

import pandas as pd
DATA_DIR = pathlib.Path("/home/dimitrios/GitHub/pwf") / "data"
df_blood_pressure=pd.read_csv(DATA_DIR / "blood_pressure.csv")
report_levene_bartlett(df=df_blood_pressure,dv="bp_before",iv=["sex","agegrp"])

