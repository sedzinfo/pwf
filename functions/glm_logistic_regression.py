# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LOGISTIC_REGRESSION.R.

Deviations from the R original, by design:
  - report_logistic takes a fitted statsmodels GLM result (from
    statsmodels.formula.api.glm(...).fit()), the Python analogue of R's
    glm(). Only the GLM's own attributes/methods are used, so any
    statsmodels family works, matching R's family-agnostic design.
  - rstandard/rstudent are reimplemented from stats:::rstandard.glm /
    stats:::rstudent.glm's actual source (verified to match R exactly for
    a binomial GLM): pearson_resid / sqrt(dispersion * (1-hat)). For
    families where R estimates dispersion (gaussian, quasi*), R's
    rstudent additionally uses a per-observation leave-one-out dispersion
    estimate; that jackknife step isn't replicated here (rstudent equals
    rstandard for every family in this port), which only affects the
    dispersion-estimated families, not the fixed-dispersion ones
    (binomial, poisson) that this function is primarily used for.
  - dfbeta is GLMInfluence.d_params (statsmodels' unscaled per-observation
    parameter change), verified to match R's dfbeta(model) exactly.
    dffits is GLMInfluence.d_fittedvalues_scaled, verified to match R's
    dffits(model) exactly (car::dffits.glm's formula is not the simple
    OLS dffits formula and was checked directly rather than assumed).
  - car::vif(model)'s classical (non-"predictor"-grouped) GVIF-per-term
    algorithm is reimplemented in _vif_glm using cov2cor(vcov(model)),
    since R's own vif.lm dispatch always falls back to this classical
    version for any GLM regardless of the `type` argument (car:::vif.lm
    checks `inherits(mod,"glm")` and always uses NextMethod() ->
    vif.default in that case — the "predictor"-grouping extension is
    LM-only). Verified to match R exactly (GVIF, Df) for a 2-term
    additive model.
  - autoplot(model) (no `which=` argument, so ggfortify's default 4-panel
    subset) is instead always rendered as the same 6-panel diagnostic
    layout used in glm_linear_regression.report_regression, via a GLM-
    specific counterpart (_glm_diagnostics_figure, using Pearson/deviance
    residuals and GLMInfluence in place of OLSInfluence) rather than a
    4-panel subset — one consistent diagnostic figure across both report
    functions rather than reproducing R's exact panel-count default.
  - compute_descriptives (R rwf::EXPLORE_DESCRIPTIVES.R, not yet ported
    to this project) is stood in for by a minimal local _compute_descriptives,
    covering only the "describe every column of an already-numeric
    data frame" case actually used here (R's own call site omits the
    `dv` argument entirely — argument is missing but never forced except
    as an index, so `names(df)[dv]` behaves like `names(df)[]`, i.e. "all
    columns", a real R quirk, not a mistake in the R source). Reimplements
    psych::describe()'s statistics directly from its documented formulas
    (type=3 skew/kurtosis, verified against psych's own source code).
  - model$effects (R's QR-projection-of-y diagnostic) is approximated the
    same way as in glm_linear_regression.report_regression: a numpy QR
    decomposition of the (unweighted) design matrix, not R's IRLS-
    weighted-least-squares QR internals — same caveat about non-unique
    orthonormal bases applies.
  - Reference-level/column-order differences between patsy's default
    (alphabetical) categorical encoding and R's factor level order are
    expected wherever a model has multi-level categorical predictors
    (e.g. "education" here); dfbeta/coefficient columns are labeled from
    whatever order the fitted Python model actually used, which is
    internally consistent but not guaranteed to list categories in the
    same order R would.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from plotnine import ggplot, aes, geom_smooth, geom_count, theme_bw, labs

try:
    from .functions_strings import call_to_string, output_separator
    from .functions_environment import write_txt
    from .functions_plot import report_pdf
    from .functions_train_test_full import confusion_matrix_percent, plot_separability, result_confusion_performance
except ImportError:
    from functions_strings import call_to_string, output_separator
    from functions_environment import write_txt
    from functions_plot import report_pdf
    from functions_train_test_full import confusion_matrix_percent, plot_separability, result_confusion_performance
##########################################################################################
# LOGISTIC FUNCTION
##########################################################################################
def compute_y_logistic(intercept, coefficient, x):
    """
    Logistic function value(s) for a given intercept/coefficient/x.

    Parameters:
    intercept (float): Intercept term.
    coefficient (float): Slope/coefficient term.
    x (float or array-like): Predictor value(s).

    Returns:
    float or numpy.ndarray

    Examples:
    >>> compute_y_logistic(0, 1, range(-10, 11))
    >>> compute_y_logistic(0, 1, 1)
    """
    x = np.asarray(x, dtype=float) if not np.isscalar(x) else x
    y = 1 / (1 + np.exp(-(intercept + coefficient * x)))
    return y
##########################################################################################
# PLOT MODEL
##########################################################################################
def plot_logistic_model(df, outcome="outcome", title="", base_size=10):
    """
    Scatter (geom_count) + fitted binomial-GLM smooth curve for every
    predictor column in `df` against a shared binary outcome.

    Parameters:
    df (pandas.DataFrame): Must include `outcome` plus one or more
        predictor columns.
    outcome (str, optional): Name of the binary outcome column. Defaults to "outcome".
    title (str, optional): Unused, kept for R-signature parity (R's
        version also builds its own fixed title/caption and never uses
        this parameter).
    base_size (int, optional): Base font size. Defaults to 10.

    Returns:
    plotnine.ggplot

    Examples:
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> df = pd.DataFrame({
    ...     "outcome": [1]*10 + [0]*10,
    ...     "pd1": [1]*11 + [0]*9,
    ...     "pd2": [1]*9 + [0]*11,
    ...     "pc1": np.concatenate([np.random.normal(5, 1, 10), np.random.normal(10, 1, 10)]),
    ...     "pc2": np.concatenate([np.random.normal(5, 1, 10), np.random.normal(20, 1, 10)]),
    ... })
    >>> plot_logistic_model(df=df, base_size=15)
    """
    temp = df.melt(id_vars=outcome, var_name="Predictor", value_name="Value")
    return (ggplot(temp, aes(x="Value", y=outcome, color="Predictor"))
            + labs(x="Observed value", y="Outcome", title="Logistic function",
                   caption=f"Observations:{len(df)}")
            + geom_smooth(method="glm", method_args={"family": sm.families.Binomial()}, se=False, alpha=0.1)
            + geom_count(alpha=.5)
            + theme_bw(base_size=base_size))
##########################################################################################
# COMPARE MODELS
##########################################################################################
def output_compare_model_logistic(model1, model2):
    """
    Likelihood-ratio-style comparison of two nested GLMs via their
    deviances (a manual analogue of R's anova(model1, model2, test="Chisq")).

    Parameters:
    model1, model2 (statsmodels GLMResultsWrapper): Two fitted GLMs
        (typically nested — model2 with more predictors than model1).

    Returns:
    pandas.DataFrame: One row with columns "X^2", "df", "p".

    Examples:
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> infert = sm.datasets.get_rdataset("infert", "datasets").data
    >>> modelcategoricalpredictor = smf.glm("case ~ education", data=infert, family=sm.families.Binomial()).fit()
    >>> modelcontinuouspredictor = smf.glm("case ~ age", data=infert, family=sm.families.Binomial()).fit()
    >>> modeltwopredictors = smf.glm("case ~ education * age", data=infert, family=sm.families.Binomial()).fit()
    >>> output_compare_model_logistic(model1=modelcategoricalpredictor, model2=modeltwopredictors)
    >>> output_compare_model_logistic(model1=modelcontinuouspredictor, model2=modeltwopredictors)
    """
    x2 = model1.deviance - model2.deviance
    x2df = model1.df_resid - model2.df_resid
    x2p = 1 - scipy.stats.chi2.cdf(x2, x2df)
    return pd.DataFrame({"X^2": [x2], "df": [x2df], "p": [x2p]})
##########################################################################################
# GVIF PER TERM (internal, mirrors car::vif.default / car::vif.lm's glm fallback)
##########################################################################################
def _vif_glm(model):
    """Classical GVIF per model term, via cov2cor(vcov(model)) -- see module docstring."""
    di = model.model.data.design_info
    term_names = [t for t in di.term_names if t != "Intercept"]
    if len(term_names) < 2:
        raise ValueError("model contains fewer than 2 terms")

    V = model.cov_params()
    non_intercept = [n for n in V.index if n != "Intercept"]
    V = V.loc[non_intercept, non_intercept].to_numpy()
    sd = np.sqrt(np.diag(V))
    R = V / np.outer(sd, sd)
    detR = np.linalg.det(R)

    term_of_col = []
    for t in term_names:
        sl = di.term_name_slices[t]
        term_of_col.extend([t] * (sl.stop - sl.start))

    rows = []
    for t in term_names:
        idx = [i for i, tc in enumerate(term_of_col) if tc == t]
        comp = [i for i in range(len(term_of_col)) if i not in idx]
        gvif = np.linalg.det(R[np.ix_(idx, idx)]) * np.linalg.det(R[np.ix_(comp, comp)]) / detR
        rows.append({"term": t, "GVIF": gvif, "Df": len(idx), "GVIF^(1/(2*Df))": gvif ** (1 / (2 * len(idx)))})

    result = pd.DataFrame(rows).set_index("term")
    if (result["Df"] == 1).all():
        return result[["GVIF"]].rename(columns={"GVIF": "VIF"})
    return result
##########################################################################################
# DESCRIPTIVES (internal, minimal stand-in for the not-yet-ported compute_descriptives)
##########################################################################################
def _compute_descriptives(df):
    """
    Per-column descriptive statistics, matching psych::describe()'s
    output columns for the plain "describe every column" case -- see
    module docstring for scope/why this isn't the full compute_descriptives.
    """
    from scipy.stats import trim_mean
    rows = []
    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)
        median = np.median(x)
        q10, q25, q50, q75, q90 = np.quantile(x, [.1, .25, .5, .75, .9])
        with np.errstate(invalid="ignore", divide="ignore"):
            skew = np.sum((x - mean) ** 3) / (n * sd ** 3)
            kurtosis = np.sum((x - mean) ** 4) / (n * sd ** 4) - 3
        rows.append({
            "variable": col, "n": n, "mean": mean, "sd": sd, "median": median,
            "trimmed": trim_mean(x, 0.1), "mad": np.median(np.abs(x - median)) * 1.4826,
            "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
            "skew": skew, "kurtosis": kurtosis,
            "se": sd / np.sqrt(n), "IQR": q75 - q25,
            "Q0.1": q10, "Q0.25": q25, "Q0.5": q50, "Q0.75": q75, "Q0.9": q90,
        })
    return pd.DataFrame(rows).set_index("variable")
##########################################################################################
# GLM DIAGNOSTICS FIGURE (internal, GLM counterpart of glm_anova_plot._lm_diagnostics_figure)
##########################################################################################
def _glm_diagnostics_figure(model, base_size, caption):
    """6-panel GLM diagnostic plot, using Pearson residuals and GLMInfluence."""
    fitted = model.fittedvalues
    influence = model.get_influence()
    standardized_resid = np.asarray(influence.resid_studentized)
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    sqrt_abs_std_resid = np.sqrt(np.abs(standardized_resid))
    resid_pearson = np.asarray(model.resid_pearson)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    axes[0, 0].scatter(fitted, resid_pearson, alpha=0.6)
    axes[0, 0].axhline(0, linestyle="--", color="gray")
    axes[0, 0].set(xlabel="Fitted values", ylabel="Pearson residuals", title="Residuals vs Fitted")

    (osm, osr), (slope, intercept, _) = scipy.stats.probplot(standardized_resid, dist="norm")
    axes[0, 1].scatter(osm, osr, alpha=0.6)
    axes[0, 1].plot(osm, intercept + slope * osm, "r--")
    axes[0, 1].set(xlabel="Theoretical Quantiles", ylabel="Standardized residuals", title="Normal Q-Q")

    axes[1, 0].scatter(fitted, sqrt_abs_std_resid, alpha=0.6)
    axes[1, 0].set(xlabel="Fitted values", ylabel="sqrt(|Standardized residuals|)", title="Scale-Location")

    axes[1, 1].vlines(range(len(cooks_d)), 0, cooks_d)
    axes[1, 1].set(xlabel="Obs. number", ylabel="Cook's distance", title="Cook's Distance")

    axes[2, 0].scatter(leverage, standardized_resid, alpha=0.6)
    axes[2, 0].axhline(0, linestyle="--", color="gray")
    axes[2, 0].set(xlabel="Leverage", ylabel="Standardized residuals", title="Residuals vs Leverage")

    axes[2, 1].scatter(leverage, cooks_d, alpha=0.6)
    axes[2, 1].set(xlabel="Leverage", ylabel="Cook's distance", title="Cook's dist vs Leverage")

    fig.suptitle(caption, fontsize=base_size)
    fig.tight_layout()
    return fig
##########################################################################################
# LOGISTIC
##########################################################################################
def report_logistic(model, validation_data=None, file=None, title="", w=10, h=10, base_size=10, fast=False):
    """
    Full report for a fitted GLM (any family): pseudo/deviance R^2, chi-
    square model test, coefficients, GVIF (2+ terms), per-observation
    scores/residual diagnostics + their descriptives, a 6-panel
    diagnostic plot, and (for binary outcomes) confusion matrix
    performance and separability plots — optionally written to a text
    log and an Excel workbook.

    Parameters:
    model (statsmodels.genmod.generalized_linear_model.GLMResultsWrapper):
        A fitted GLM (e.g. from statsmodels.formula.api.glm(...).fit()).
    validation_data (pandas.DataFrame, optional): If given, predictions/
        observed values for the confusion-matrix/separability plots come
        from this data instead of the training data. Defaults to None.
    file (str, optional): Output filename (without extension). Defaults to None.
    title (str, optional): Title applied to plots. Defaults to "".
    w, h (float, optional): PDF page size in inches. Defaults to 10, 10.
    base_size (int, optional): Base font size for plots. Defaults to 10.
    fast (bool, optional): If True, skip writing the (large) per-
        observation scores table to the Excel export. Defaults to False.

    Returns:
    dict: model_summary, result_R2_logistic, result_X2_logistic,
    coefficients, logistic_output, scores, score_descriptives, VIF, model_call.

    Examples:
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> infert = sm.datasets.get_rdataset("infert", "datasets").data
    >>> modelcategoricalpredictor0 = smf.glm("case ~ education", data=infert, family=sm.families.Binomial()).fit()
    >>> modeltwopredictors0 = smf.glm("case ~ education + stratum", data=infert, family=sm.families.Binomial()).fit()
    >>> report_logistic(model=modelcategoricalpredictor0)
    >>> report_logistic(model=modeltwopredictors0, file="logistic_two_predictors", validation_data=infert)
    """
    score_notes = [
        "Problematic values for standardized residuals > +-1.96",
        "Problematic values for dfbeta >= 1",
        "Problematic values for Hat values (leverage) 2 or 3 times the average (k+1/n) where k=number of "
        "predictors n=number of participants",
    ]

    null_deviance = model.null_deviance
    model_chisquare = null_deviance - model.deviance
    r2 = model_chisquare / null_deviance
    model_df = model.df_model
    model_probability = 1 - scipy.stats.chi2.cdf(model_chisquare, model_df)
    chisquare = pd.DataFrame({
        "Chi.Squared": [model_chisquare], "df": [model_df], "p": [model_probability], "R2": [r2],
        "Note": ["Significance indicates that the model is better than chance at predicting the outcome"],
    })

    n_obs = len(model.fittedvalues)
    r_l = 1 - (model.deviance / null_deviance)
    r_cs = 1 - np.exp((model.deviance - null_deviance) / n_obs)
    r_n = r_cs / (1 - np.exp(-(null_deviance / n_obs)))
    pseudor2 = pd.DataFrame({
        "Test": ["Hosmer and Lemeshow R^2", "Cox and Snell R^2", "Nagelkerke R^2"],
        "Statistic": [r_l, r_cs, r_n],
        "Notes": ["", "Cox and Snell R^2 never reaches maximum of 1", ""],
    })

    model_call = pd.DataFrame({"call": [call_to_string(model)]})

    vif = pd.DataFrame()
    exog_names_no_intercept = [n for n in model.model.exog_names if n != "Intercept"]
    if len(exog_names_no_intercept) > 1:
        try:
            vif = _vif_glm(model)
        except Exception:
            pass

    influence = model.get_influence()
    hat = influence.hat_matrix_diag
    scale = model.scale
    pearson_resid = np.asarray(model.resid_pearson)
    rstandard = pearson_resid / (np.sqrt(scale) * np.sqrt(1 - hat))
    rstudent = rstandard

    di = model.model.data.design_info
    base_vars = sorted({v for t in di.term_names if t != "Intercept" for v in t.split(":")})
    outcome_name = model.model.endog_names
    variable_df = model.model.data.frame[[outcome_name] + base_vars].reset_index(drop=True)
    variable_df.columns = [f"variable.{c}" for c in variable_df.columns]

    dfbeta = pd.DataFrame(influence.d_params, columns=[f"dfbeta.{n}" for n in model.model.exog_names])
    scores = pd.concat([
        variable_df,
        pd.DataFrame({
            "weights": np.asarray(model.model.family.weights(model.fittedvalues)),
            "prior_weights": np.ones(n_obs),
            "linear_predictors": np.asarray(model.model.family.link(model.fittedvalues)),
            "fitted": np.asarray(model.fittedvalues),
            "residuals": np.asarray(model.resid_response),
            "standardized_residuals": rstandard,
            "student_residuals": rstudent,
        }),
        dfbeta,
        pd.DataFrame({"dffits": influence.d_fittedvalues_scaled, "hatvalues": hat}),
    ], axis=1)

    logistic_output = pd.DataFrame({
        "AIC": [model.aic], "df.residual": [model.df_resid], "df.null": [model.df_resid + model.df_model],
        "deviance": [model.deviance], "null.deviance": [null_deviance], "converged": [model.converged],
        "method": [model.method if hasattr(model, "method") else "IRLS"],
    })

    if validation_data is None:
        predicted = model.fittedvalues.to_numpy(dtype=float)
        observed = model.model.endog.astype(float)
    else:
        predicted = model.predict(validation_data).to_numpy(dtype=float)
        observed = validation_data[outcome_name].to_numpy(dtype=float)

    caption = f"{getattr(model.model, 'formula', '')}\nobservations={n_obs}"
    diagnostics_fig = _glm_diagnostics_figure(model, base_size=base_size, caption=(title or caption))
    confusion_performance = result_confusion_performance(observed=observed, predicted=predicted,
                                                           base_size=base_size, title=title, step=.01)
    separability = plot_separability(observed=observed, predicted=predicted, base_size=base_size, title=title)
    report_pdf(diagnostics_fig, confusion_performance["plot_performance"], separability,
               file=file, title=title, w=w, h=h, print_plot=(file is None))

    cut = np.nanmean(confusion_performance["cut"]) if np.ndim(confusion_performance["cut"]) else confusion_performance["cut"]
    cmatrix = confusion_matrix_percent(observed=observed, predicted=(predicted > cut).astype(int))

    score_descriptives = _compute_descriptives(scores.select_dtypes("number"))

    result = {
        "model_summary": model.summary(),
        "result_R2_logistic": pseudor2,
        "result_X2_logistic": chisquare,
        "coefficients": pd.DataFrame({
            "Estimate": model.params, "Std. Error": model.bse, "z value": model.tvalues, "Pr(>|z|)": model.pvalues,
        }),
        "logistic_output": logistic_output,
        "scores": scores,
        "score_descriptives": score_descriptives,
        "VIF": vif,
        "model_call": model_call,
    }

    import io
    import contextlib
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        output_separator("Model Summary", output=model.summary())
        output_separator("Summary", output=result["logistic_output"])
        output_separator("Confusion Matrix", output=cmatrix)
        output_separator("Confusion Matrix Performance", output=confusion_performance["cut_performance"])
        output_separator("Confusion Matrix Best Cut", output=confusion_performance["cut"])
        output_separator("Pseudo R^2", output=result["result_R2_logistic"])
        output_separator("X^2", output=result["result_X2_logistic"])
        output_separator("Variance Inflation Factor", output=result["VIF"])
        output_separator("Outlier Descriptives", output=result["score_descriptives"], instruction=score_notes)
        output_separator("Call", output=result["model_call"])
    write_txt(log_buffer.getvalue(), file=file)

    if file is not None:
        try:
            from .functions_excel import excel_critical_value, excel_confusion_matrix
        except ImportError:
            from functions_excel import excel_critical_value, excel_confusion_matrix
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_critical_value(result["logistic_output"], writer, sheetname="summary")
        excel_critical_value(result["coefficients"].reset_index(names="term"), writer, sheetname="coefficients",
                              critical={"Pr(>|z|)": "<0.05"})
        excel_confusion_matrix(cmatrix, writer)
        excel_critical_value(confusion_performance["cut_performance"], writer, sheetname="confusion matrix performance")
        excel_critical_value(result["result_X2_logistic"], writer, sheetname="X^2")
        excel_critical_value(result["result_R2_logistic"], writer, sheetname="pseudo R^2")
        if not fast:
            excel_critical_value(result["scores"], writer, sheetname="scores residuals")
        excel_critical_value(result["score_descriptives"].reset_index(names="variable"), writer,
                              sheetname="score residual descriptives")
        if len(vif) > 0:
            excel_critical_value(vif.reset_index(names="term"), writer, sheetname="VIF")
        excel_critical_value(result["model_call"], writer, sheetname="Call")
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
    import statsmodels.formula.api as smf

    infert = sm.datasets.get_rdataset("infert", "datasets").data

    print("=" * 80, "\ncompute_y_logistic\n", "=" * 80, sep="")
    x = list(range(-10, 11))
    print(compute_y_logistic(0, 1, x))
    print(compute_y_logistic(0, 1, 1))

    print("\n" + "=" * 80, "\nplot_logistic_model\n", "=" * 80, sep="")
    np.random.seed(0)
    df_logistic = pd.DataFrame({
        "outcome": [1] * 10 + [0] * 10,
        "pd1": [1] * 11 + [0] * 9,
        "pd2": [1] * 9 + [0] * 11,
        "pc1": np.concatenate([np.random.normal(5, 1, 10), np.random.normal(10, 1, 10)]),
        "pc2": np.concatenate([np.random.normal(5, 1, 10), np.random.normal(20, 1, 10)]),
    })
    p = plot_logistic_model(df=df_logistic, base_size=15)
    p.save("plot_logistic_model.png", verbose=False)
    print("saved plot_logistic_model.png")

    print("\n" + "=" * 80, "\noutput_compare_model_logistic\n", "=" * 80, sep="")
    modelcategoricalpredictor = smf.glm("case ~ education", data=infert, family=sm.families.Binomial()).fit()
    modelcontinuouspredictor = smf.glm("case ~ age", data=infert, family=sm.families.Binomial()).fit()
    modeltwopredictors = smf.glm("case ~ education * age", data=infert, family=sm.families.Binomial()).fit()
    print(output_compare_model_logistic(model1=modelcategoricalpredictor, model2=modeltwopredictors))
    print(output_compare_model_logistic(model1=modelcontinuouspredictor, model2=modeltwopredictors))
    print(output_compare_model_logistic(model1=modelcontinuouspredictor, model2=modelcategoricalpredictor))

    print("\n" + "=" * 80, "\nreport_logistic (single predictor, binomial)\n", "=" * 80, sep="")
    modelcategoricalpredictor0 = smf.glm("case ~ education", data=infert, family=sm.families.Binomial()).fit()
    res = report_logistic(model=modelcategoricalpredictor0)
    print("result_R2_logistic:\n", res["result_R2_logistic"])
    print("\nresult_X2_logistic:\n", res["result_X2_logistic"])
    print("\ncoefficients:\n", res["coefficients"])
    print("\nlogistic_output:\n", res["logistic_output"])
    print("\nVIF (empty expected, 1 term):\n", res["VIF"])
    print("\nscore_descriptives:\n", res["score_descriptives"])

    print("\n" + "=" * 80, "\nreport_logistic (two predictors, with VIF, Excel export)\n", "=" * 80, sep="")
    modeltwopredictors0 = smf.glm("case ~ education + stratum", data=infert, family=sm.families.Binomial()).fit()
    res2 = report_logistic(model=modeltwopredictors0, file="logistic_two_predictors", validation_data=infert)
    print("VIF:\n", res2["VIF"])
    print("\nscores head:\n", res2["scores"].head())
    print("\nExcel/log written:", os.path.exists("logistic_two_predictors.xlsx"),
          os.path.exists("logistic_two_predictors.log"))

    print("\n" + "=" * 80, "\nreport_logistic (gaussian family)\n", "=" * 80, sep="")
    modelcategoricalpredictor1 = smf.glm("case ~ education", data=infert, family=sm.families.Gaussian()).fit()
    res3 = report_logistic(model=modelcategoricalpredictor1)
    print("logistic_output:\n", res3["logistic_output"])
