# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LOGISTIC_REGRESSION.R.

report_logistic matches rwf (tests/test_glm_logistic_regression.py, on
binomial, poisson and gaussian models): the statsmodels formula GLM is
refitted as R's glm.fit (R's starting values, IRLS convergence rule, term
order and coefficient names), because R keeps the working weights, QR,
effects and therefore hat values, dfbeta, dffits and studentized
residuals from the last IRLS iteration, not from the converged fit.
car::vif, psych-style descriptives of the scores, rwf's confusion
results and R's deparsed call are reproduced too.

Differences from rwf: rwf's report_logistic returns nothing, this returns
its result list; the diagnostic plot is a 6-panel figure (ggfortify's
autoplot draws 4); plot_logistic_model fits its curves with statsmodels.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import contextlib
import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import statsmodels.api as sm
from patsy import ModelDesc
from plotnine import ggplot, aes, geom_smooth, geom_count, theme_bw, labs

try:
    from .explore_descriptives import compute_descriptives
    from .functions_strings import output_separator
    from .functions_environment import write_txt
    from .functions_plot import report_pdf
    from .functions_train_test import confusion_matrix_percent, plot_separability, result_confusion_performance
    from .glm_linear_regression import (_data_name, _deparse_formula, _design_info, _r_column_name,
                                        _r_deparse_call, _r_terms)
except ImportError:
    from explore_descriptives import compute_descriptives
    from functions_strings import output_separator
    from functions_environment import write_txt
    from functions_plot import report_pdf
    from functions_train_test import confusion_matrix_percent, plot_separability, result_confusion_performance
    from glm_linear_regression import (_data_name, _deparse_formula, _design_info, _r_column_name,
                                       _r_deparse_call, _r_terms)
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
    >>> infert = pd.read_csv("data/infert.csv")
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
# R glm() VIEW OF A STATSMODELS FORMULA GLM (internal)
##########################################################################################
_EPS = np.finfo(float).eps


def _r_family(model):
    """R's name of a statsmodels GLM family and its deviance contributions, starting values and AIC."""
    family = model.model.family
    name = {"Binomial": "binomial", "Poisson": "poisson", "Gaussian": "gaussian", "Gamma": "Gamma",
            "InverseGaussian": "inverse.gaussian"}.get(type(family).__name__)
    if name is None:
        raise ValueError(f"family {type(family).__name__} is not supported")

    def ylogy(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log(a / b), 0.0)

    def dev_resids(y, mu, w):
        if name == "binomial":
            return 2 * w * (ylogy(y, mu) + ylogy(1 - y, 1 - mu))
        if name == "poisson":
            return 2 * w * (ylogy(y, mu) - (y - mu))
        if name == "gaussian":
            return w * (y - mu) ** 2
        if name == "Gamma":
            return -2 * w * (np.log(np.where(y == 0, 1, y / mu)) - (y - mu) / mu)
        return w * (y - mu) ** 2 / (y * mu ** 2)

    def mustart(y, w):
        if name == "binomial":
            return (w * y + 0.5) / (w + 1)
        if name == "poisson":
            return y + 0.1
        return y.astype(float)

    def aic(y, mu, w, dev):
        n = len(y)
        if name == "binomial":
            return -2 * np.sum(w * scipy.stats.binom.logpmf(np.round(y), 1, mu))
        if name == "poisson":
            return -2 * np.sum(scipy.stats.poisson.logpmf(y, mu) * w)
        if name == "gaussian":
            return n * (np.log(2 * np.pi * dev / n) + 1) + 2
        if name == "Gamma":
            dispersion = dev / n
            return -2 * np.sum(scipy.stats.gamma.logpdf(y, 1 / dispersion, scale=mu * dispersion) * w) + 2
        dispersion = dev / np.sum(w)
        return np.sum(w) * (np.log(dispersion * 2 * np.pi) + 1) + 3 * np.sum(np.log(y) * w) + 2

    return {"name": name, "family": family, "dev_resids": dev_resids, "mustart": mustart, "aic": aic,
            "estimates_dispersion": name not in ("binomial", "poisson")}


def _r_glm(model, epsilon=1e-8, maxit=25):
    """
    The statsmodels formula GLM refitted as R's glm.fit: model-matrix
    columns in R's term order with R's names, R's starting values and IRLS
    with its convergence rule, keeping (as R) the working weights, QR and
    effects of the last iteration's weighted least squares, then the
    lm.influence quantities R's rstandard, rstudent, dfbeta, dffits and
    hatvalues use.
    """
    design_info = _design_info(model)
    terms = _r_terms(model.model.formula)
    columns, assign, labels = [], [], []
    for k, term in enumerate(terms):
        piece = design_info.term_slices[term]
        columns.extend(range(piece.start, piece.stop))
        assign.extend([k] * (piece.stop - piece.start))
        labels.append(term.name())
    X = np.asarray(model.model.exog, dtype=float)[:, columns]
    names = [_r_column_name(design_info.column_names[c]) for c in columns]
    y = np.asarray(model.model.endog, dtype=float)
    n, p = X.shape
    prior = np.ones(n)
    family = _r_family(model)
    link = family["family"].link
    has_intercept = bool(terms) and len(terms[0].factors) == 0

    mu = family["mustart"](y, prior)
    eta = link(mu)
    mu = link.inverse(eta)
    devold = np.sum(family["dev_resids"](y, mu, prior))
    converged = False
    for iteration in range(1, maxit + 1):
        mu_eta = link.inverse_deriv(eta)
        z = eta + (y - mu) / mu_eta
        w = np.sqrt(prior * mu_eta ** 2 / family["family"].variance(mu))
        q, r = np.linalg.qr(X * w[:, None], mode="complete")
        qty = q.T @ (z * w)
        coefficients = scipy.linalg.solve_triangular(r[:p], qty[:p])
        eta = X @ coefficients
        mu = link.inverse(eta)
        dev = np.sum(family["dev_resids"](y, mu, prior))
        if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
            converged = True
            break
        devold = dev
    working_residuals = (y - mu) / link.inverse_deriv(eta)
    weights = w ** 2
    rank = int(np.linalg.matrix_rank(r[:p]))
    wtdmu = np.sum(prior * y) / np.sum(prior) if has_intercept else link.inverse(np.zeros(n))
    null_deviance = np.sum(family["dev_resids"](y, np.broadcast_to(wtdmu, y.shape), prior))
    df_residual = n - rank
    df_null = n - int(has_intercept)
    if family["estimates_dispersion"]:
        dispersion = np.sum(weights * working_residuals ** 2) / df_residual
    else:
        dispersion = 1.0
    r_inverse = scipy.linalg.solve_triangular(r[:p], np.eye(p))
    unscaled = r_inverse @ r_inverse.T
    se = np.sqrt(np.diag(unscaled) * dispersion)
    statistic = coefficients / se
    if family["estimates_dispersion"]:
        p_values = 2 * scipy.stats.t.sf(np.abs(statistic), df_residual)
        labels_test = ["t value", "Pr(>|t|)"]
    else:
        p_values = 2 * scipy.stats.norm.sf(np.abs(statistic))
        labels_test = ["z value", "Pr(>|z|)"]

    # lm.influence on the last weighted least squares
    e = working_residuals * np.sqrt(weights)
    e = np.where(np.abs(e) < 100 * _EPS * np.median(np.abs(e)), 0.0, e)
    q1 = q[:, :rank]
    hat = np.sum(q1 ** 2, axis=1)
    hat = np.where(hat >= 1 - 10 * _EPS, 1.0, hat)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt((np.sum(e ** 2) - e ** 2 / (1 - hat)) / (n - p - 1))
        dfbeta = (q1 @ r_inverse.T) * np.where(hat == 1, 0, e / (1 - hat))[:, None]
        pearson = (y - mu) * np.sqrt(prior) / np.sqrt(family["family"].variance(mu))
        rstandard = pearson / (np.sqrt(dispersion) * np.sqrt(1 - hat))
        if family["estimates_dispersion"]:
            rstudent = pearson / (sigma * np.sqrt(1 - hat))
            dffits = e * np.sqrt(hat) / (sigma * (1 - hat))
        else:
            rstudent = rstandard
            dffits = e * np.sqrt(hat) / (np.sqrt(dispersion) * (1 - hat))
    return {
        "names": names, "labels": labels, "assign": np.array(assign), "has_intercept": has_intercept,
        "family": family["name"], "coefficients": coefficients, "se": se, "statistic": statistic,
        "p_values": p_values, "labels_test": labels_test, "unscaled": unscaled, "dispersion": dispersion,
        "fitted": mu, "linear_predictors": eta, "residuals": working_residuals, "weights": weights,
        "prior_weights": prior, "effects": qty, "rank": rank, "iter": iteration, "converged": converged,
        "deviance": dev, "null_deviance": null_deviance, "df_residual": df_residual, "df_null": df_null,
        "aic": family["aic"](y, mu, prior, dev) + 2 * rank, "epsilon": epsilon, "maxit": maxit,
        "hat": hat, "dfbeta": dfbeta, "dffits": dffits, "rstandard": rstandard, "rstudent": rstudent,
    }


def _vif_glm(fit):
    """car::vif of a glm (vif.default): GVIF per term from cov2cor of the coefficient covariances."""
    keep = fit["assign"] > 0 if fit["has_intercept"] else np.ones(len(fit["assign"]), bool)
    terms = [label for k, label in enumerate(fit["labels"]) if k > 0 or not fit["has_intercept"]]
    if len(terms) < 2:
        raise ValueError("model contains fewer than 2 terms")
    v = fit["unscaled"][np.ix_(keep, keep)]
    sd = np.sqrt(np.diag(v))
    correlations = v / np.outer(sd, sd)
    det_all = np.linalg.det(correlations)
    assign = fit["assign"][keep]
    rows = []
    for k, term in enumerate(terms, start=1 if fit["has_intercept"] else 0):
        inside = assign == k
        gvif = (np.linalg.det(correlations[np.ix_(inside, inside)])
                * np.linalg.det(correlations[np.ix_(~inside, ~inside)]) / det_all)
        rows.append({"term": term, "GVIF": gvif, "Df": int(inside.sum())})
    table = pd.DataFrame(rows).set_index("term")
    if (table["Df"] == 1).all():
        return table[["GVIF"]].rename(columns={"GVIF": "VIF"})
    table["GVIF^(1/(2*Df))"] = table["GVIF"] ** (1 / (2 * table["Df"]))
    return table.rename(columns=lambda c: f"VIF.{c}")


def _model_frame(model):
    """R's model$model: the response and the variables of the formula, in formula order."""
    desc = ModelDesc.from_formula(model.model.formula)
    variables = []
    for term in list(desc.lhs_termlist) + list(desc.rhs_termlist):
        for factor in term.factors:
            name = factor.name()
            if name not in variables:
                variables.append(name)
    frame = model.model.data.frame.loc[model.model.data.row_labels]
    return pd.DataFrame({name: frame.eval(name) if name not in frame.columns else frame[name] for name in variables})
##########################################################################################
# LOGISTIC
##########################################################################################
def report_logistic(model, validation_data=None, file=None, title="", w=10, h=10, base_size=10, fast=False):
    """
    Report of a generalized linear model, as rwf::report_logistic: the
    pseudo R squares, the model chi-square, coefficients with odds ratios,
    the fit summary, case scores (working weights, effects, residuals,
    standardized and studentized residuals, dfbeta, dffits, hat values)
    with their descriptives, VIFs, the call, the diagnostic, confusion
    performance and separability plots, and the confusion matrix at the
    best cut point; with file, a text log and an Excel workbook.

    The statsmodels GLM is refitted as R's glm.fit (R's starting values,
    IRLS convergence rule and term order, R's coefficient names), so the
    working weights, effects, iteration count and influence measures,
    which R takes from the last IRLS iteration, match R. VIF is computed
    (car::vif) when the model frame has more than two columns. As in rwf,
    predictions for validation_data are on the link scale (R's predict()
    default), not probabilities.

    Differences from rwf: rwf returns nothing; this returns the result
    list (and the confusion results). The diagnostic plot is the 6-panel
    figure of glm_logistic_regression, not ggfortify's 4 panels.

    Parameters:
    model (statsmodels GLMResultsWrapper): GLM fitted with statsmodels.formula.api.glm(...).fit()
        (binomial, poisson, gaussian, Gamma or inverse Gaussian family).
    validation_data (pandas.DataFrame, optional): Data for the confusion and separability results. Defaults to None.
    file (str, optional): Output filename without extension. Defaults to None.
    title (str, optional): Title of the plots. Defaults to "".
    w, h (float, optional): PDF page size in inches. Defaults to 10.
    base_size (int, optional): Base font size of the plots. Defaults to 10.
    fast (bool, optional): Leave the case scores out of the Excel file. Defaults to False.

    Returns:
    dict: model_summary, result_R2_logistic, result_X2_logistic,
    coefficients, logistic_output, scores, score_descriptives, VIF and
    model_call (rwf's result), plus cmatrix and confusion_performance.

    Examples:
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> model = smf.glm("case ~ age + parity + spontaneous", data=infert, family=sm.families.Binomial()).fit()
    >>> result = report_logistic(model)
    """
    score_notes = [
        "Problematic values for standardized residuals > +- 1.96",
        "Problematic values for dfbeta >= 1",
        "Problematic values for Hat values (leverage) 2 or 3 times the average (k+1/n) where k=number of "
        "predictors n=number of participants",
    ]
    fit = _r_glm(model)
    names = fit["names"]
    n_obs = len(fit["fitted"])

    model_chisquare = fit["null_deviance"] - fit["deviance"]
    model_df = fit["df_null"] - fit["df_residual"]
    chisquare = pd.DataFrame({
        "Chi.Squared": [model_chisquare], "df": [model_df], "p": [1 - scipy.stats.chi2.cdf(model_chisquare, model_df)],
        "R2": [model_chisquare / fit["null_deviance"]],
        "Note": ["Significance indicates that the model is better than chance at predicting the outcome"],
    })
    r_l = 1 - fit["deviance"] / fit["null_deviance"]
    r_cs = 1 - np.exp((fit["deviance"] - fit["null_deviance"]) / n_obs)
    r_n = r_cs / (1 - np.exp(-(fit["null_deviance"] / n_obs)))
    pseudor2 = pd.DataFrame({
        "Test": ["Hosmer and Lemeshow R^2", "Cox and Snell R^2", "Nagelkerke R^2"],
        "Statistic": [r_l, r_cs, r_n],
        "Notes": ["", "Cox and Snell R^2 never reaches maximum of 1", ""],
    })
    # rwf's call_to_string: R's deparse() lines of the call joined by ", ", spaces removed
    call_lines = _r_deparse_call("glm", [("formula", _deparse_formula(model.model.formula)),
                                         ("family", fit["family"]),
                                         ("data", _data_name(model.model.data.frame))])
    model_call = pd.DataFrame({"call": [", ".join(call_lines).replace(" ", "")]})

    frame = _model_frame(model)
    vif = pd.DataFrame()
    if frame.shape[1] > 2:
        try:
            vif = _vif_glm(fit)
        except Exception as error:
            print(error)

    variables = frame.reset_index(drop=True)
    variables.columns = [f"variable.{c}" for c in variables.columns]
    scores = pd.concat([variables, pd.DataFrame({
        "weights": fit["weights"], "prior_weights": fit["prior_weights"],
        "linear_predictors": fit["linear_predictors"], "effects": fit["effects"], "fitted": fit["fitted"],
        "residuals": fit["residuals"], "standardized_residuals": fit["rstandard"],
        "student_residuals": fit["rstudent"]}),
        pd.DataFrame(fit["dfbeta"], columns=[f"dfbeta.{name}" for name in names]),
        pd.DataFrame({"dffits": fit["dffits"], "hatvalues": fit["hat"]})], axis=1)
    scores.index = model.model.data.row_labels

    coefficients = pd.DataFrame({"Estimate": fit["coefficients"], "Std. Error": fit["se"],
                                 fit["labels_test"][0]: fit["statistic"], fit["labels_test"][1]: fit["p_values"],
                                 "Odds Ratio": np.exp(fit["coefficients"])}, index=names)
    logistic_output = pd.DataFrame({
        "AIC": [fit["aic"]], "df.residual": [fit["df_residual"]], "df.null": [fit["df_null"]],
        "deviance": [fit["deviance"]], "null.deviance": [fit["null_deviance"]], "epsilon": [fit["epsilon"]],
        "tol": [min(1e-07, fit["epsilon"] / 1000)], "qr.rank": [fit["rank"]], "rank": [fit["rank"]],
        "maxit": [fit["maxit"]], "iteration": [fit["iter"]], "method": ["glm.fit"], "converged": [fit["converged"]],
        "boundary": [False], "trace": [False],
    })
    numeric_scores = scores[[c for c in scores.columns if pd.api.types.is_numeric_dtype(scores[c])
                             and not pd.api.types.is_bool_dtype(scores[c])]]
    result = {
        "model_summary": model.summary(),
        "result_R2_logistic": pseudor2,
        "result_X2_logistic": chisquare,
        "coefficients": coefficients,
        "logistic_output": logistic_output,
        "scores": scores,
        "score_descriptives": compute_descriptives(numeric_scores, dv=None),
        "VIF": vif,
        "model_call": model_call,
    }

    if validation_data is None:
        predicted = fit["fitted"]
        observed = frame.iloc[:, 0].to_numpy(dtype=float)
    else:
        predicted = np.asarray(model.predict(validation_data, which="linear"), dtype=float)
        observed = validation_data[frame.columns[0]].to_numpy(dtype=float)
    caption = f"{_deparse_formula(model.model.formula)}\nobservations={n_obs}"
    diagnostics_fig = _glm_diagnostics_figure(model, base_size=base_size, caption=f"{title}\n{caption}" if title else caption)
    confusion_performance = result_confusion_performance(observed=observed, predicted=predicted,
                                                         base_size=base_size, title=title, step=.01)
    separability = plot_separability(observed=observed, predicted=predicted, base_size=base_size, title=title)
    report_pdf(diagnostics_fig, confusion_performance["plot_performance"], separability,
               file=file, title=title, w=w, h=h, print_plot=(file is None))
    cut = np.nanmean(confusion_performance["cut"])
    cmatrix = confusion_matrix_percent(observed=observed, predicted=(predicted > cut).astype(int))

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
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(result["logistic_output"], writer, sheet="summary")
            excel_critical_value(result["coefficients"].reset_index(names="term"), writer, sheet="coefficients",
                                 critical={fit["labels_test"][1]: "<0.05"})
            excel_confusion_matrix(cmatrix, writer)
            excel_critical_value(confusion_performance["cut_performance"], writer,
                                 sheet="confusion matrix performance")
            excel_critical_value(result["result_X2_logistic"], writer, sheet="X^2")
            excel_critical_value(result["result_R2_logistic"], writer, sheet="pseudo R^2")
            if not fast:
                excel_critical_value(result["scores"], writer, sheet="scores residuals")
            excel_critical_value(result["score_descriptives"], writer, sheet="score residual descriptives")
            if len(vif) > 0:
                excel_critical_value(vif.reset_index(names="term"), writer, sheet="VIF")
            excel_critical_value(result["model_call"], writer, sheet="Call")
    return {**result, "cmatrix": cmatrix, "confusion_performance": confusion_performance}
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os
    import statsmodels.formula.api as smf

    infert = pd.read_csv("data/infert.csv") if os.path.exists("data/infert.csv") else pd.read_csv("../data/infert.csv")

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
    modeltwopredictors0 = smf.glm("case ~ education + age", data=infert, family=sm.families.Binomial()).fit()
    res2 = report_logistic(model=modeltwopredictors0, file="logistic_two_predictors", validation_data=infert)
    print("VIF:\n", res2["VIF"])
    print("\nscores head:\n", res2["scores"].head())
    print("\nExcel/log written:", os.path.exists("logistic_two_predictors.xlsx"),
          os.path.exists("logistic_two_predictors.log"))

    print("\n" + "=" * 80, "\nreport_logistic (gaussian family)\n", "=" * 80, sep="")
    modelcategoricalpredictor1 = smf.glm("case ~ education", data=infert, family=sm.families.Gaussian()).fit()
    res3 = report_logistic(model=modelcategoricalpredictor1)
    print("logistic_output:\n", res3["logistic_output"])
