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

report_regression deviations, by design:
  - report_regression takes a fitted statsmodels OLS result (from
    statsmodels.formula.api.ols), the Python analogue of R's lm().
  - ggfortify::autoplot(model, which=1:6)'s 6-panel diagnostic plot reuses
    _lm_diagnostics_figure from glm_anova_plot.py (already built for the
    same purpose there) rather than duplicating it.
  - car::outlierTest, car::durbinWatsonTest, and QuantPsyc::lm.beta are
    reimplemented directly from their documented/source algorithms
    (verified numerically against R for the R docstring's own examples):
      * outlier_test: externally studentized residuals, two-tailed
        t-test p-value, Bonferroni-corrected (p * n).
      * durbin_watson_test: car's default method="resample" bootstrap
        (1000 resamples of residuals, refit, recompute DW, two-sided
        p-value from the simulated null distribution) — a randomized
        test in both R and here, so exact p-values won't reproduce
        run-to-run in either language, only converge to the same value.
      * lm_beta: beta_j = b_j * sd(x_j) / sd(y), using the fitted
        design-matrix columns for sd(x_j). QuantPsyc::lm.beta itself
        computes sd() from the raw model-frame columns, which is only
        equivalent to this for continuous predictors — like the R
        docstring's own examples, this hasn't been tested against
        multi-level categorical predictors, where QuantPsyc's own
        approach is documented to be fragile too (sd() of a factor).
  - car::vif(model, type="predictor")'s GVIF-per-predictor algorithm
    (Fox & Monette 1992) is reimplemented in _vif_predictor via patsy's
    design_info (term-to-column mapping) rather than the naive per-
    coefficient VIF, since for any model with interaction terms the
    naive version gives wildly inflated, misleading values that the
    R function specifically exists to avoid. Verified to match R exactly
    (GVIF, Df, and the derived column) for an additive model, a partial-
    interaction model, and the R docstring's fully-saturated 4-way
    interaction example.
  - model$effects (R's QR-decomposition-based orthogonal projection of y,
    an internal-ish diagnostic from lm's own fitting algorithm) is
    recomputed here via numpy's QR decomposition of the design matrix
    instead. This is the same *concept* (y projected onto an orthonormal
    basis of the column space) but won't numerically match R's values:
    a QR decomposition's Q is only unique up to sign/rotation in
    degenerate subspaces, and R's LAPACK routine and numpy's may pick
    different (equally valid) bases.
  - R's write_txt(block_of_output_separator_calls, file=file) relies on
    substitute()-based lazy evaluation of an unevaluated code block
    inside sink(); Python's write_txt (functions_environment.py) takes
    an already-rendered value instead. Reproduced here by capturing the
    combined stdout of the sequential output_separator() calls via
    contextlib.redirect_stdout into a string, then passing that string
    to write_txt() as one call.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import io
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as _tdist
import statsmodels.api as sm

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
# OUTLIER TEST (internal, mirrors car::outlierTest.lm)
##########################################################################################
def _outlier_test(model, cutoff=0.05, n_max=10):
    """Bonferroni-corrected outlier test on externally studentized residuals."""
    influence = model.get_influence()
    rstudent = pd.Series(influence.resid_studentized_external, index=model.model.data.row_labels)
    rstudent = rstudent.dropna()
    df = model.df_resid - 1
    n = len(rstudent)

    p = pd.Series(2 * _tdist.sf(np.abs(rstudent), df), index=rstudent.index)
    bp = n * p
    ord_idx = [i for i in bp.sort_values().index if bp[i] <= cutoff]

    if not ord_idx:
        which = rstudent.abs().idxmax()
        return pd.DataFrame({"rstudent": [rstudent[which]], "p": [p[which]], "bonf.p": [bp[which]],
                              "signif": [False], "cutoff": [cutoff]}, index=[which])
    ord_idx = ord_idx[:n_max]
    return pd.DataFrame({"rstudent": rstudent[ord_idx], "p": p[ord_idx], "bonf.p": bp[ord_idx],
                          "signif": True, "cutoff": cutoff})
##########################################################################################
# DURBIN WATSON TEST (internal, mirrors car::durbinWatsonTest.lm, method="resample")
##########################################################################################
def _durbin_watson_test(model, reps=1000, alternative="two-sided"):
    """Bootstrap (resample-residuals) Durbin-Watson autocorrelation test, lag 1."""
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
    den_sim = np.sum(E ** 2, axis=0)
    DW = np.sum(np.diff(E, axis=0) ** 2, axis=0) / den_sim

    if alternative == "two-sided":
        p0 = np.mean(dw < DW)
        p = 2 * min(p0, 1 - p0)
    elif alternative == "positive":
        p = np.mean(dw > DW)
    else:
        p = np.mean(dw < DW)
    return pd.DataFrame({"r": [r], "dw": [dw], "p": [p], "alternative": [alternative]})
##########################################################################################
# STANDARDIZED COEFFICIENTS (internal, mirrors QuantPsyc::lm.beta)
##########################################################################################
def _lm_beta(model):
    """Standardized regression coefficients: beta_j = b_j * sd(x_j) / sd(y)."""
    y = np.asarray(model.model.endog)
    X = np.asarray(model.model.exog)
    exog_names = model.model.exog_names
    sy = np.std(y, ddof=1)
    betas = {}
    for i, name in enumerate(exog_names):
        if name == "Intercept":
            continue
        sx = np.std(X[:, i], ddof=1)
        betas[name] = model.params[name] * sx / sy
    return pd.Series(betas, name="standardized")
##########################################################################################
# GVIF PER PREDICTOR (internal, mirrors car::vif(model, type="predictor"))
##########################################################################################
def _vif_predictor(model):
    """
    Generalized VIF per predictor (Fox & Monette 1992), grouping every
    model-matrix column belonging to a term that involves the predictor
    (directly or via any interaction) into one block. Raises if the
    model has fewer than 2 terms, matching car::vif's own behavior.
    """
    di = model.model.data.design_info
    term_names = [t for t in di.term_names if t != "Intercept"]
    if len(term_names) < 2:
        raise ValueError("model contains fewer than 2 terms")
    term_vars = {t: t.split(":") for t in term_names}
    predictors = sorted({v for vs in term_vars.values() for v in vs})

    X = np.asarray(model.model.exog)
    intercept_slice = di.term_name_slices.get("Intercept")
    cols_no_intercept = [i for i in range(X.shape[1])
                         if not (intercept_slice and intercept_slice.start <= i < intercept_slice.stop)]
    X_ni = X[:, cols_no_intercept]

    term_of_col = []
    for t in term_names:
        sl = di.term_name_slices[t]
        term_of_col.extend([t] * (sl.stop - sl.start))

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
        interacts_with = ", ".join(sorted(related - {predictor})) if related - {predictor} else "--"
        p = len(columns)
        rows.append({"predictor": predictor, "GVIF": gvif, "Df": p, "GVIF^(1/(2*Df))": gvif ** (1 / (2 * p)),
                     "Interacts With": interacts_with, "Other Predictors": other_predictors})

    result = pd.DataFrame(rows).set_index("predictor")
    if (result["Df"] == 1).all():
        return result[["GVIF"]].rename(columns={"GVIF": "vif"})
    return result
##########################################################################################
# QR EFFECTS (internal, conceptual analogue of R's lm()$effects -- see module docstring)
##########################################################################################
def _qr_effects(model):
    """y projected onto an orthonormal basis of the design matrix's column space."""
    X = np.asarray(model.model.exog)
    y = np.asarray(model.model.endog)
    Q, _ = np.linalg.qr(X, mode="complete")
    return Q.T @ y
##########################################################################################
# REGRESSION
##########################################################################################
def report_regression(model, base_size=10, title="", file=None, w=10, h=10, plot_diagnostics=True):
    """
    Full report for a fitted OLS regression: R^2, coefficients (raw +
    standardized + CI), sequential (Type I) ANOVA, deviance, coefficient
    covariance matrix, outlier test, Durbin-Watson test, VIF/GVIF (when
    the model has 2+ terms), per-observation diagnostics, and a 6-panel
    diagnostic plot — optionally written to a text log and an Excel
    workbook.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper):
        A fitted OLS model (e.g. from statsmodels.formula.api.ols(...).fit()).
    base_size (int, optional): Base font size for the diagnostic plot. Defaults to 10.
    title (str, optional): Title applied to the diagnostic plot. Defaults to "".
    file (str, optional): Output filename (without extension). When
        given, writes "<file>.log" (via write_txt) and "<file>.xlsx".
        Defaults to None.
    w, h (float, optional): PDF page size in inches for the diagnostic
        plot export. Defaults to 10, 10.
    plot_diagnostics (bool, optional): If True (default), build and
        export the 6-panel diagnostic plot.

    Returns:
    dict: r, coeficients, anova, deviance, variance_covariance,
    outlier_test, durbin_watson, vif, call, diagnostics.

    Examples:
    >>> import statsmodels.formula.api as smf
    >>> import statsmodels.api as sm
    >>> mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data
    >>> regressionmodel = smf.ols("mpg ~ qsec", data=mtcars).fit()
    >>> multipleregressionmodel = smf.ols("mpg ~ qsec*hp*wt*drat", data=mtcars).fit()
    >>> res = report_regression(model=regressionmodel, plot_diagnostics=True)
    >>> res = report_regression(model=multipleregressionmodel)
    >>> res = report_regression(model=regressionmodel, file="regression")
    >>> res = report_regression(model=multipleregressionmodel, file="regression", plot_diagnostics=True)
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
        "Test the assumption of independent errors.\nTest values may vary between 0 and 4.\nValues above 3 and bellow 1 are problematic.\nValues of 2 are ideal indicating uncorrelated residuals."
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

    if plot_diagnostics:
        caption = f"{model.model.formula}\nobservations={int(model.nobs)}"
        fig = _lm_diagnostics_figure(model, base_size=base_size, caption=(title or caption))
        report_pdf(fig, file=file, title=title, w=w, h=h, print_plot=(file is None))

    outlier_test = _outlier_test(model)
    dw = _durbin_watson_test(model)

    vif = pd.DataFrame()
    try:
        vif = _vif_predictor(model)
    except Exception:
        pass

    influence = model.get_influence()
    exog_names = model.model.exog_names
    dfbeta = pd.DataFrame(influence.dfbeta, index=model.model.data.row_labels,
                           columns=[f"dfbeta.{n}" for n in exog_names])
    diagnostics = pd.DataFrame({
        "simple_residuals": model.resid,
        "standard_residuals": influence.resid_studentized_internal,
        "student_residuals": influence.resid_studentized_external,
        "fitted": model.fittedvalues,
        "cooks_distance": influence.cooks_distance[0],
        "dffits": influence.dffits[0],
        "hatvalues": influence.hat_matrix_diag,
        "covariance_ratio": influence.cov_ratio,
    }, index=model.model.data.row_labels)
    diagnostics = pd.concat([diagnostics.reset_index(drop=True), dfbeta.reset_index(drop=True)], axis=1)
    diagnostics["effects"] = pd.Series(_qr_effects(model))

    coef_table = pd.DataFrame({
        "Estimate": model.params, "Std. Error": model.bse, "t value": model.tvalues, "Pr(>|t|)": model.pvalues,
    })
    ci = model.conf_int()
    ci.columns = ["2.5 %", "97.5 %"]
    model_coefficients = coef_table.join(ci)
    standardized = _lm_beta(model)
    model_coefficients = model_coefficients.join(standardized.rename("standardized"))

    formula_str = getattr(model.model, "formula", "")
    call = pd.DataFrame({"call": [f"lm({formula_str.replace(' ', '')})"]})

    result = {
        "r": pd.DataFrame({"r_squared": [model.rsquared], "adjusted_r_squared": [model.rsquared_adj]}),
        "coeficients": model_coefficients,
        "anova": sm.stats.anova_lm(model, typ=1),
        "deviance": pd.DataFrame({"deviance": [model.ssr]}),
        "variance_covariance": pd.DataFrame(model.cov_params()),
        "outlier_test": outlier_test,
        "durbin_watson": dw,
        "vif": vif,
        "call": call,
        "diagnostics": diagnostics,
    }

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        output_separator("Summary", output=model.summary())
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
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_critical_value(result["r"], writer, sheetname="r")
        excel_critical_value(
            result["coeficients"].reset_index(names="term"), writer, sheetname="Coefficients",
            comments={"Estimate": instruction_coefficients[0], "standardized": instruction_coefficients[1],
                      "t value": instruction_coefficients[2], "Pr(>|t|)": instruction_coefficients[3],
                      "Std. Error": "Standard Error of Estimates",
                      "2.5 %": "Confidence Intervals", "97.5 %": "Confidence Intervals"},
            critical={"Pr(>|t|)": "<0.05"})
        excel_critical_value(
            result["anova"].reset_index(names="term"), writer, sheetname="ANOVA",
            comments={"F": instruction_anova[0], "PR(>F)": instruction_anova[1]},
            critical={"PR(>F)": "<0.05"})
        excel_critical_value(result["variance_covariance"].reset_index(names="term"), writer,
                              sheetname="Variance Covariance")
        excel_critical_value(result["outlier_test"].reset_index(names="observation"), writer, sheetname="Outliers")
        excel_critical_value(
            result["durbin_watson"], writer, sheetname="Durbin Watson",
            comments={"r": instruction_durbin[1], "dw": instruction_durbin[2], "p": instruction_durbin[3]})
        if len(vif) > 0:
            excel_critical_value(result["vif"].reset_index(names="predictor"), writer, sheetname="VIF",
                                  comments={c: instruction_vif[i] for i, c in
                                            enumerate(vif.columns[:len(instruction_vif)])})
        excel_critical_value(
            result["diagnostics"], writer, sheetname="Diagnostics",
            comments={"standard_residuals": "Problematic values for standardized residuals > +-1.96",
                      "student_residuals": "Studentized residuals indicate the ability of the model to predict "
                                            "that case. They follow a t distribution",
                      "cooks_distance": "Cook's distance indicates leverage. Problematic values for cook's "
                                        "distance > 1 Cook and Weisberg (1982).",
                      "hatvalues": "Hat values indicate leverage. Problematic values are 2 or 3 times the average "
                                   "(k+1/n). Hoaglin and Welsch (1978) recommend investigating cases with values "
                                   "greater than twice the average (2(k+1)/n), Stevens (2002) recommend "
                                   "investigating cases with values greater than three times the average (3(k+1)/n)",
                      "dffits": "DFFits indicate the difference between the adjusted predicted value and the "
                                "original predicted value. Adjusted predicted value for a case refers to the "
                                "predicted value of that case, when that case is excluded from model fit."},
            critical={"standard_residuals": [">1.96", "<-1.96"], "cooks_distance": ">1"})
        excel_critical_value(result["call"], writer, sheetname="Call")
        writer._save()
        writer.close()

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

    print("\n" + "=" * 80, "\nreport_regression (simple)\n", "=" * 80, sep="")
    import statsmodels.formula.api as smf
    mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data
    regressionmodel = smf.ols("mpg ~ qsec", data=mtcars).fit()
    res = report_regression(model=regressionmodel, plot_diagnostics=False)
    print("r:\n", res["r"])
    print("\ncoeficients:\n", res["coeficients"])
    print("\nanova:\n", res["anova"])
    print("\ndeviance:\n", res["deviance"])
    print("\noutlier_test:\n", res["outlier_test"])
    print("\ndurbin_watson:\n", res["durbin_watson"])
    print("\nvif (empty expected, <2 terms):\n", res["vif"])
    print("\ncall:\n", res["call"])
    print("\ndiagnostics head:\n", res["diagnostics"].head())

    print("\n" + "=" * 80, "\nreport_regression (multiple, with interactions)\n", "=" * 80, sep="")
    multipleregressionmodel = smf.ols("mpg ~ qsec*hp*wt*drat", data=mtcars).fit()
    res2 = report_regression(model=multipleregressionmodel, plot_diagnostics=True, file="regression_test")
    print("vif:\n", res2["vif"])
    print("\ncoeficients:\n", res2["coeficients"])
    print("\nExcel/log written:", os.path.exists("regression_test.xlsx"), os.path.exists("regression_test.log"))
