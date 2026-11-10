# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LINEAR_REGRESSION.R — plot_scatterplot and
report_regression. R's version has a larger set of regression-modeling
functions; this file covers the two ported so far and is meant to be
extended later.

Deviations from the R original, by design:
  - ggExtra::ggMarginal (scatter + marginal histograms) has no plotnine
    equivalent — plotnine doesn't support marginal plots natively. Uses
    seaborn's jointplot(kind="reg") instead, which natively combines a
    scatter plot, OLS regression line with CI band, and marginal
    histograms in one call. Returns a seaborn JointGrid per pair
    (with a real matplotlib Figure at .figure), not a plotnine ggplot
    like the rest of this project's plotting functions.
  - Dropped the generic `method`/`formula` parameters (R forwards these
    to geom_smooth, supporting arbitrary smoothers/formulas). This
    always fits a simple OLS y~x line and always shows the full
    r/slope/angle caption — which is exactly what R itself does when
    the default y~x formula is used anyway.
  - The future/future.apply parallel-plotting branch is not replicated,
    for the same reason as in glm_anova_plot.py and functions_plot.py:
    plot objects don't reliably serialize across process boundaries in
    Python. Runs sequentially, matching R's own sequential fallback path.

report_regression matches rwf (tests/test_glm_linear_regression.py):
  - It takes a statsmodels formula OLS fit (statsmodels.formula.api.ols),
    viewed as R's lm(): columns in R's term order (patsy puts categorical
    terms first, which would change the sequential ANOVA), R's coefficient
    names, Type I ANOVA and lm's effects from the Householder QR.
  - car::outlierTest, car::durbinWatsonTest (bootstrap p value), car::vif
    (type = "predictor") and the standardized coefficients are native.
  - The diagnostic plot (ggfortify::autoplot(model, which = 1:6)) reuses
    _lm_diagnostics_figure from glm_anova_plot.py.
  - R's write_txt(block) evaluates the output_separator() calls lazily;
    here their stdout is captured and passed to write_txt.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import contextlib
import inspect
import io
import os
import re

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from patsy import ModelDesc
from scipy.stats import t as _tdist

try:
    from .functions import comparison_combinations
    from .functions_strings import str_aes, output_separator
    from .functions_mathematical import rad2deg
    from .functions_environment import write_txt
    from .functions_plot import report_pdf
    from .glm_anova_plot import _lm_diagnostics_figure
except ImportError:
    from functions import comparison_combinations
    from functions_strings import str_aes, output_separator
    from functions_mathematical import rad2deg
    from functions_environment import write_txt
    from functions_plot import report_pdf
    from glm_anova_plot import _lm_diagnostics_figure
##########################################################################################
# SCATTERPLOT
##########################################################################################
def plot_scatterplot(df, base_size=10, coord_equal=False, all_orders=False, title="",
                      combinations=None, str_aes_labels=True):
    """
    Scatter plot with an OLS regression line, CI band, and marginal
    histograms for variable pairs in a data frame. Each plot's caption
    reports pairwise n, Pearson r, explained variance, the regression
    equation, and the regression angle.

    Parameters:
    df (pandas.DataFrame): Numeric variables. When df has exactly 2
        columns and combinations=None, only that one (x, y) pair is used.
    base_size (int, optional): Base font size / figure height driver. Defaults to 10.
    coord_equal (bool, optional): If True, both axes share the same
        scale and limits. Defaults to False.
    all_orders (bool, optional): If True, both (X, Y) and (Y, X)
        orderings are plotted for each pair. Ignored if `combinations`
        is given. Defaults to False.
    title (str, optional): Plot title applied to every panel. Defaults to "".
    combinations (pandas.DataFrame, optional): Two-column frame of
        (x, y) column-name pairs to plot. If None (default), all pairs
        are derived automatically via comparison_combinations.
    str_aes_labels (bool, optional): If True (default), axis labels are
        cleaned via str_aes().

    Returns:
    dict: {"x_y": seaborn.axisgrid.JointGrid or None}, one per pair.
    None when a pair has fewer than 2 complete-case observations.

    Examples:
    >>> plot_scatterplot(df=df_insurance[["age", "bmi", "charges"]], coord_equal=True)
    """
    if combinations is None:
        combinations = comparison_combinations(df, all_orders=all_orders)
        combinations.columns = ['x', 'y']
    combinations = combinations[combinations.iloc[:, 0] != combinations.iloc[:, 1]]

    plots = {}
    for _, row in combinations.iterrows():
        xcol, ycol = row.iloc[0], row.iloc[1]
        key = f"{xcol}_{ycol}"
        tempdata = df[[xcol, ycol]].dropna()
        if len(tempdata) < 2:
            plots[key] = None
            continue

        x = tempdata[xcol].to_numpy(dtype=float)
        y = tempdata[ycol].to_numpy(dtype=float)
        pearsonr = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        degrees = rad2deg(np.arctan(slope))
        if degrees > 180:
            degrees = 360 - degrees

        note = (f"Pairwise n = {len(tempdata)}\n"
                f"Pearson r = {round(pearsonr, 4)}\n"
                f"Explained Variance = {round(pearsonr ** 2, 4) * 100}%\n"
                f"y = {round(slope, 4)}x + {round(intercept, 4)}\n"
                f"Angle = {round(degrees, 2)}")

        xlabel = str_aes(xcol) if str_aes_labels else xcol
        ylabel = str_aes(ycol) if str_aes_labels else ycol

        g = sns.jointplot(data=tempdata, x=xcol, y=ycol, kind="reg", height=max(base_size / 1.2, 4),
                           joint_kws={'scatter_kws': {'alpha': 0.3}})
        g.ax_joint.set_xlabel(xlabel)
        g.ax_joint.set_ylabel(ylabel)
        g.figure.suptitle(title)
        g.ax_joint.annotate(note, xy=(0.02, -0.3), xycoords='axes fraction',
                             fontsize=max(base_size * 0.6, 6), va='top')

        if x.min() <= 0 <= x.max():
            g.ax_joint.axvline(0, alpha=.5, color='gray')
        if y.min() <= 0 <= y.max():
            g.ax_joint.axhline(0, alpha=.5, color='gray')

        if coord_equal:
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            g.ax_joint.set_xlim(lo, hi)
            g.ax_joint.set_ylim(lo, hi)
            g.ax_joint.set_aspect('equal')

        plots[key] = g

    return plots
##########################################################################################
# R lm() VIEW OF A STATSMODELS FORMULA MODEL (internal)
##########################################################################################
def _r_column_name(name):
    """R's coefficient name of a patsy column: Intercept -> (Intercept), smoker[T.yes] -> smokeryes."""
    if name == "Intercept":
        return "(Intercept)"
    return re.sub(r"\[(?:T\.)?([^\]]*)\]", r"\1", name)


def _design_info(model):
    """patsy DesignInfo of a statsmodels formula model (data.model_spec from statsmodels 0.15, design_info before)."""
    data = model.model.data
    return getattr(data, "model_spec", None) or data.design_info


def _r_terms(formula):
    """patsy terms of a formula in R's order: by degree, then by appearance in the formula."""
    return sorted(ModelDesc.from_formula(formula).rhs_termlist, key=lambda term: len(term.factors))


def _deparse_formula(formula):
    """R's deparse() spacing of a formula: spaces around ~ + - * / and after commas, none around : and ^."""
    text = re.sub(r"\s+", "", formula)
    out = []
    for k, char in enumerate(text):
        previous = text[k - 1] if k else ""
        if char in "~+*/" or (char == "-" and previous not in ("", "~", "+", "*", "/", "(", ",", "-")):
            out.append(f" {char} ")
        elif char == ",":
            out.append(", ")
        else:
            out.append(char)
    return "".join(out).strip()


def _r_deparse_call(function, arguments, cutoff=60):
    """
    Lines of R's deparse() of a call function(name = value, ...): after each argument's ", " and after
    each binary operator of a formula (" ~ ", " + ", ...), deparse ends the line once it is longer than
    width.cutoff (60) characters; continuation lines are indented by 4 spaces.
    """
    lines, line = [], f"{function}("

    def check():
        nonlocal line
        if len(line) > cutoff:
            lines.append(line)
            line = "    "

    for k, (name, value) in enumerate(arguments):
        line += f"{name} = "
        for piece in re.split(r"( [~+\-*/] )", value):
            line += piece
            if re.fullmatch(r" [~+\-*/] ", piece):
                check()
        if k < len(arguments) - 1:
            line += ", "
            check()
    lines.append(line + ")")
    return lines


def _data_name(frame):
    """Name of the variable holding the model's data in a calling frame (R's model$call$data), else "data"."""
    caller = inspect.currentframe().f_back.f_back  # skip this helper and report_regression
    try:
        while caller is not None:
            for name, value in {**caller.f_globals, **caller.f_locals}.items():
                if value is frame and not name.startswith("_"):
                    return name
            caller = caller.f_back
    finally:
        del caller
    return "data"


def _r_lm(model):
    """
    The fitted statsmodels formula OLS model as R's lm() would hold it: the
    model-matrix columns in R's term order with R's coefficient names (refit,
    so every statsmodels result comes out in R's order), lm's assign vector
    and its QR effects (Householder QR, as LINPACK's dqrdc2 used by lm).
    """
    design_info = _design_info(model)
    terms = _r_terms(model.model.formula)
    columns, assign, labels = [], [], []
    for k, term in enumerate(terms):
        piece = design_info.term_slices[term]
        columns.extend(range(piece.start, piece.stop))
        assign.extend([k] * (piece.stop - piece.start))
        labels.append(term.name())
    exog = np.asarray(model.model.exog)[:, columns]
    names = [_r_column_name(design_info.column_names[c]) for c in columns]
    rows = model.model.data.row_labels
    endog = pd.Series(np.asarray(model.model.endog), index=rows, name=model.model.endog_names)
    fit = sm.OLS(endog, pd.DataFrame(exog, index=rows, columns=names)).fit()
    q, _ = np.linalg.qr(exog, mode="complete")
    has_intercept = bool(terms) and len(terms[0].factors) == 0
    return {"fit": fit, "assign": np.array(assign), "labels": labels, "effects": q.T @ endog.to_numpy(),
            "has_intercept": has_intercept, "formula": model.model.formula}


def _anova_lm(lm):
    """R's anova(lm): sequential sums of squares from the QR effects of each term."""
    fit, assign = lm["fit"], lm["assign"]
    effects = lm["effects"][:len(assign)]
    rows = []
    start = 1 if lm["has_intercept"] else 0
    df_residual, ss_residual = fit.df_resid, float(np.sum(fit.resid ** 2))
    for k in range(start, len(lm["labels"])):
        df = int(np.sum(assign == k))
        ss = float(np.sum(effects[assign == k] ** 2))
        rows.append([df, ss, ss / df])
    table = pd.DataFrame(rows, columns=["Df", "Sum Sq", "Mean Sq"], index=lm["labels"][start:])
    table["F value"] = table["Mean Sq"] / (ss_residual / df_residual)
    table["Pr(>F)"] = scipy.stats.f.sf(table["F value"], table["Df"], df_residual)
    residuals = pd.DataFrame([[df_residual, ss_residual, ss_residual / df_residual, np.nan, np.nan]],
                             columns=table.columns, index=["Residuals"])
    return pd.concat([table, residuals])
##########################################################################################
# OUTLIER TEST (internal, mirrors car::outlierTest.lm)
##########################################################################################
def _outlier_test(model, cutoff=0.05, n_max=10):
    """Bonferroni-corrected outlier test on externally studentized residuals, as car::outlierTest."""
    influence = model.get_influence()
    rstudent = pd.Series(influence.resid_studentized_external, index=model.model.data.row_labels).dropna()
    n = len(rstudent)
    p = pd.Series(2 * _tdist.sf(np.abs(rstudent), model.df_resid - 1), index=rstudent.index)
    bp = n * p
    order = [i for i in bp.sort_values(kind="stable").index if bp[i] <= cutoff]
    if not order:
        which = rstudent.abs().idxmax()
        return pd.DataFrame({"rstudent": [rstudent[which]], "p": [p[which]], "bonf.p": [bp[which]],
                             "signif": [False], "cutoff": [cutoff]}, index=[which])
    order = order[:n_max]
    return pd.DataFrame({"rstudent": rstudent[order], "p": p[order], "bonf.p": bp[order], "signif": True,
                         "cutoff": cutoff})
##########################################################################################
# DURBIN WATSON TEST (internal, mirrors car::durbinWatsonTest.lm, method="resample")
##########################################################################################
def _durbin_watson_test(model, reps=1000, alternative="two.sided"):
    """
    Lag-1 Durbin-Watson test with car's default bootstrap p value (residuals
    resampled onto the fitted values, refit, DW recomputed). The p value is
    random, in R as here.
    """
    resid = np.asarray(model.resid)
    n = len(resid)
    den = np.sum(resid ** 2)
    dw = np.sum(np.diff(resid) ** 2) / den
    r = np.sum(resid[1:] * resid[:-1]) / den

    X = np.asarray(model.model.exog)
    mu = np.asarray(model.fittedvalues)
    Y = np.random.choice(resid, size=(n, reps), replace=True) + mu[:, None]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ beta
    DW = np.sum(np.diff(E, axis=0) ** 2, axis=0) / np.sum(E ** 2, axis=0)

    if alternative == "two.sided":
        p0 = np.mean(dw < DW)
        p = 2 * min(p0, 1 - p0)
    elif alternative == "positive":
        p = np.mean(dw > DW)
    else:
        p = np.mean(dw < DW)
    return pd.DataFrame({"dw.r": [r], "dw.dw": [dw], "dw.p": [p], "dw.alternative": [alternative]})
##########################################################################################
# GVIF PER PREDICTOR (internal, mirrors car::vif(model, type="predictor"))
##########################################################################################
def _vif_predictor(model):
    """
    Generalized VIF per predictor (Fox & Monette 1992), grouping every
    model-matrix column belonging to a term that involves the predictor
    (directly or via any interaction) into one block, predictors in R's
    order. Raises if the model has fewer than 2 terms, as car::vif. When
    every block has one column car returns plain VIFs: a DataFrame with
    the single column "vif" is returned then.
    """
    di = _design_info(model)
    term_names = [term.name() for term in _r_terms(model.model.formula) if term.factors]
    if len(term_names) < 2:
        raise ValueError("model contains fewer than 2 terms")
    term_vars = {t: t.split(":") for t in term_names}
    predictors = list(dict.fromkeys(v for t in term_names for v in term_vars[t]))

    X = np.asarray(model.model.exog)
    intercept_slice = di.term_name_slices.get("Intercept")
    cols_no_intercept = [i for i in range(X.shape[1])
                         if not (intercept_slice and intercept_slice.start <= i < intercept_slice.stop)]
    X_ni = X[:, cols_no_intercept]

    term_of_col = [None] * X.shape[1]
    for t in term_names:
        sl = di.term_name_slices[t]
        for i in range(sl.start, sl.stop):
            term_of_col[i] = t
    term_of_col = [term_of_col[i] for i in cols_no_intercept]

    R = np.corrcoef(X_ni, rowvar=False)
    detR = np.linalg.det(R)
    n_cols = X_ni.shape[1]

    rows = []
    for predictor in predictors:
        which_terms = [t for t in term_names if predictor in term_vars[t]]
        related = set()
        for t in which_terms:
            related.update(t.split(":"))
        unrelated = [p for p in predictors if p not in related]
        if unrelated:
            unrelated_terms = {t for t in term_names if any(u in term_vars[t] for u in unrelated)}
            columns = np.array([i for i in range(n_cols) if term_of_col[i] not in unrelated_terms])
            complement = np.array([i for i in range(n_cols) if i not in columns])
            gvif = np.linalg.det(R[np.ix_(columns, columns)]) * np.linalg.det(R[np.ix_(complement, complement)]) / detR
            other_predictors = ", ".join(unrelated)
        else:
            columns = np.arange(n_cols)
            gvif = 1.0
            other_predictors = "--"
        interacts_with = ", ".join(p for p in predictors if p in related - {predictor}) or "--"
        p = len(columns)
        rows.append({"predictor": predictor, "GVIF": gvif, "Df": p, "GVIF^(1/(2*Df))": gvif ** (1 / (2 * p)),
                     "Interacts With": interacts_with, "Other Predictors": other_predictors})

    result = pd.DataFrame(rows).set_index("predictor")
    if (result["Df"] == 1).all():
        return result[["GVIF"]].rename(columns={"GVIF": "vif"})
    return result
##########################################################################################
# REGRESSION
##########################################################################################
def report_regression(model, base_size=10, title="", file=None, w=10, h=10, plot_diagnostics=True):
    """
    Report of a linear regression, as rwf::report_regression: R squared,
    coefficients (standardized, estimates, t tests, confidence intervals),
    the sequential ANOVA, deviance, coefficient covariances, car's outlier
    and Durbin-Watson tests, VIFs, case diagnostics and the call, plus the
    6-panel diagnostic plot; with file, a text log (file.txt), an Excel
    workbook (file.xlsx) and the plot PDF.

    The statsmodels model is viewed as R's lm(): coefficients, ANOVA rows,
    covariances and diagnostics follow R's term order (terms by degree,
    then formula order; patsy puts categorical terms first, which changes
    the sequential ANOVA) and R's coefficient names ((Intercept),
    smokeryes, ...). As in rwf the coefficient table is merged by row
    name, so the intercept, which has no standardized coefficient, comes
    last; vif is filled only when every predictor has one degree of
    freedom (rwf's tolerance = 1 / car::vif fails on car's GVIF table and
    is caught); the effects are lm's QR effects.

    Differences from rwf: the Durbin-Watson p value is a bootstrap (as
    car's) drawn with numpy's random generator, so it differs from R's
    draw; the call names the data by the variable holding it in a calling
    frame ("data" when none is found).

    Parameters:
    model (statsmodels RegressionResultsWrapper): OLS fitted with statsmodels.formula.api.ols(...).fit().
    base_size (int, optional): Base font size of the diagnostic plot. Defaults to 10.
    title (str, optional): Title of the diagnostic plot. Defaults to "".
    file (str, optional): Output filename without extension. Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10.
    plot_diagnostics (bool, optional): Draw the diagnostic plot. Defaults to True.

    Returns:
    dict: r, coeficients, anova, deviance, variance_covariance,
    outlier_test, durbin_watson, vif, call and diagnostics (DataFrames;
    rwf's names, including "coeficients").

    Examples:
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols("charges ~ age * bmi + smoker + region", data=df_insurance).fit()
    >>> result = report_regression(model, plot_diagnostics=False)
    >>> result["coeficients"]
    """
    instruction_coefficients = [
        "Unstandardized coefficients (b's) indicate the change in the outcome resulting from a unit change in the predictor",
        "Standardized coefficients (for more than one predictors), indicate the change in outcome as a result of a unit change by a standard deviation of the predictor",
        "t-test checks if coefficients are significantly different from 0. Coefficients of 0 indicate no predictor effects",
        "Significance value for t-test",
    ]
    instruction_anova = [
        "ANOVA tests for differences between the baseline model (model with no coefficient) and the predictive model (model with coefficient). A significant F shows that the predictor(s) significantly changes model predictability",
        "Significance value for ANOVA",
        "Null hypothesis: no variance explained by the predictor",
    ]
    instruction_durbin = [
        "Test the assumption of independent errors.\nTest values may vary between 0 and 4.\nValues above 3 and bellow 1 are problematic.\nValues of 2 are ideal indicating uncorrelated residuals.\n                        "
        "\nA value greater than 2 indicates a negative correlation between adjacent residuals.\nA value less than 2 indicates a positive correlation between adjacent residuals.",
        "Autocorrelation",
        "Durbin-Watson Statistic",
        "Significance value for Durbin-Watson Statistic",
    ]
    instruction_vif = [
        "Variance Inflation Factor VIF indicates whether a predictor has strong linear relationship with other predictors. If the largest VIF is greater than 10 it is problematic Myers(1990).",
        "Tolerance=1/VIF Tolerance bellow .1 indicates serious problem. Tolerance bellow .2 indicates potential problem Field & Myers(2012).",
        "Mean VIF if average VIF is greater than 1 the regression may be biased.",
    ]
    lm = _r_lm(model)
    fit = lm["fit"]
    formula = _deparse_formula(lm["formula"])

    if plot_diagnostics:
        caption = f"{formula}\nobservations={int(fit.nobs)}"
        fig = _lm_diagnostics_figure(fit, base_size=base_size, caption=f"{title}\n{caption}" if title else caption)
        report_pdf(fig, file=file, title=title, w=w, h=h, print_plot=(file is None))

    outlier_test = _outlier_test(fit)
    dw = _durbin_watson_test(fit)

    vif = pd.DataFrame()
    try:
        table = _vif_predictor(model)
        if list(table.columns) == ["vif"]:
            values = table["vif"]
            vif = pd.DataFrame({"vif": values, "tolerance": 1 / values, "mean_vif": values.mean()})
    except Exception:
        pass

    influence = fit.get_influence()
    names = list(fit.params.index)
    rows = fit.model.data.row_labels
    diagnostics = pd.DataFrame({
        "simple_residuals": fit.resid.to_numpy(),
        "standard_residuals": influence.resid_studentized_internal,
        "student_residuals": influence.resid_studentized_external,
        "fitted": fit.fittedvalues.to_numpy(),
        "cooks_distance": influence.cooks_distance[0],
        "dffits": influence.dffits[0],
        "hatvalues": influence.hat_matrix_diag,
        "covariance_ratio": influence.cov_ratio,
    }, index=rows)
    dfbeta = pd.DataFrame(influence.dfbeta, index=rows, columns=[f"dfbeta.{name}" for name in names])
    diagnostics = pd.concat([diagnostics, dfbeta], axis=1)
    diagnostics["effects"] = lm["effects"]

    # standardized coefficients: b * sd(x) / sd(y), with the sd of the coefficient's own model-matrix column
    exog = fit.model.exog
    y_sd = np.std(fit.model.endog, ddof=1)
    standardized = pd.Series({name: fit.params[name] * np.std(exog[:, k], ddof=1) / y_sd
                              for k, name in enumerate(names) if name != "(Intercept)"}, dtype=float)
    summary_table = pd.DataFrame({"Estimate": fit.params, "Std. Error": fit.bse, "t value": fit.tvalues,
                                  "Pr(>|t|)": fit.pvalues})
    ci = fit.conf_int()
    ci.columns = ["2.5 %", "97.5 %"]
    summary_table = summary_table.join(ci)
    order = list(standardized.index) + [name for name in names if name not in standardized.index]
    model_coefficients = pd.concat([standardized.rename("standardized"), summary_table], axis=1).loc[order]
    model_coefficients = model_coefficients.rename_axis("Row.names").reset_index()

    result = {
        "r": pd.DataFrame({"r_squared": [fit.rsquared], "adjusted_r_squared": [fit.rsquared_adj]}),
        "coeficients": model_coefficients,
        "anova": _anova_lm(lm),
        "deviance": pd.DataFrame({"deviance": [fit.ssr]}),
        "variance_covariance": fit.cov_params(),
        "outlier_test": outlier_test,
        "durbin_watson": dw,
        "vif": vif,
        "call": pd.DataFrame({"call": [f"lm({formula},data={_data_name(model.model.data.frame)})"]}),
        "diagnostics": diagnostics,
    }

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        output_separator("Summary", output=fit.summary())
        output_separator("Coefficients", output=result["coeficients"], instruction=instruction_coefficients)
        output_separator("ANOVA", output=result["anova"], instruction=instruction_anova)
        output_separator("Deviance", output=result["deviance"])
        output_separator("Outliers", output=result["outlier_test"])
        output_separator("Durbin Watson", output=result["durbin_watson"], instruction=instruction_durbin)
        if len(vif) > 0:
            output_separator("MULTICOLINEARITY", output=result["vif"], instruction=instruction_vif)
        output_separator("CALL", output=result["call"])
    write_txt(log_buffer.getvalue(), file=file)

    if file is not None:
        try:
            from .functions_excel import excel_critical_value
        except ImportError:
            from functions_excel import excel_critical_value
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(result["r"], writer, sheet="r")
            excel_critical_value(
                result["coeficients"], writer, sheet="Coefficients",
                comment={"Estimate": instruction_coefficients[0], "standardized": instruction_coefficients[1],
                          "t value": instruction_coefficients[2], "Pr(>|t|)": instruction_coefficients[3],
                          "Std. Error": "Standard Error of Estimates",
                          "2.5 %": "Confidence Intervals", "97.5 %": "Confidence Intervals"},
                critical={"Pr(>|t|)": "<0.05"})
            excel_critical_value(
                result["anova"].reset_index(names="term"), writer, sheet="ANOVA",
                comment={"F value": instruction_anova[0], "Pr(>F)": instruction_anova[1]},
                critical={"Pr(>F)": "<0.05"})
            excel_critical_value(result["variance_covariance"].reset_index(names="term"), writer,
                                 sheet="Variance Covariance")
            excel_critical_value(result["outlier_test"].reset_index(names="observation"), writer, sheet="Outliers")
            excel_critical_value(
                result["durbin_watson"], writer, sheet="Durbin Watson",
                comment={"dw.r": instruction_durbin[1], "dw.dw": instruction_durbin[2], "dw.p": instruction_durbin[3]})
            if len(vif) > 0:
                excel_critical_value(result["vif"].reset_index(names="predictor"), writer, sheet="VIF",
                                     comment={"vif": instruction_vif[0], "tolerance": instruction_vif[1],
                                               "mean_vif": instruction_vif[2]})
            excel_critical_value(
                result["diagnostics"], writer, sheet="Diagnostics",
                comment={"standard_residuals": "Problematic values for standardized residuals > +-1.96",
                          "student_residuals": "Studentized residuals indicate the the ability of the model to "
                                               "predict that case. They follow a t distribution",
                          "cooks_distance": "Cook's distance indicates leverage. Problematic values for cook's "
                                            "distance > 1 Cook and Weisberg (1982).",
                          "hatvalues": "Hat values indicate leverage. Problematic values are 2 or 3 times the "
                                       "average (k+1/n), Hoaglin and Welsch (1978) recommend investigating cases "
                                       "with values greater than twice the average (2(k+1)/n), Stevens (2002) "
                                       "recommend investigating cases with values greater than three times the "
                                       "average (3(k+1)/n)",
                          "dffits": "DFFits indicate the difference between the adjusted predicted value and the "
                                    "original predicted value. Adjusted predicted value for a case refers to the "
                                    "predicted value of that case, when that case is excluded from model fit."},
                critical={"standard_residuals": ["<-1.96", ">1.96"], "cooks_distance": ">1"})
            excel_critical_value(result["call"], writer, sheet="Call")
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os

    df_insurance = pd.read_csv("data/insurance.csv") if os.path.exists("data/insurance.csv") \
        else pd.read_csv("../data/insurance.csv")

    print("=" * 80, "\nplot_scatterplot\n", "=" * 80, sep="")
    plots = plot_scatterplot(df_insurance[["age", "bmi", "charges"]], coord_equal=False)
    print("keys:", list(plots.keys()))
    plots["age_charges"].figure.savefig("plot_scatterplot_age_charges.png")
    print("saved plot_scatterplot_age_charges.png")

    print("\n" + "=" * 80, "\nreport_regression\n", "=" * 80, sep="")
    import statsmodels.formula.api as smf
    model = smf.ols("charges ~ age * bmi + smoker + region", data=df_insurance).fit()
    res = report_regression(model=model, plot_diagnostics=False)
    for key in ["r", "coeficients", "anova", "durbin_watson", "vif", "call"]:
        print(f"\n{key}:\n", res[key])
