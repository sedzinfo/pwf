# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LDA.R, with MASS::lda (method "moment") and its
predict method ported natively, so report_lda takes the fitted model
alone, as in rwf.

lda() and report_lda() match MASS and rwf (tests/test_glm_lda.py): the
formula is expanded as R's model.matrix (treatment-coded factors, R's
term order and column names, intercept dropped), the within-group
covariance is sphered by singular value decomposition and the
discriminant axes come from the singular value decomposition of the
weighted group means, as MASS::lda.default. The signs of the
discriminant axes follow LAPACK's SVD, as in R.

Kept from rwf: the confusion matrix is built over the observed classes
and the predicted classes' factor codes too (R's c(observed, factor)
turns the predicted factor into its codes), so it has empty rows and
columns for those codes, as rwf's.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import re

import numpy as np
import pandas as pd
import patsy

try:
    from ._rcompat import r_levels, r_label, r_make_names
    from .functions_train_test import _confusion_counts, _percent_table
    from .glm_linear_regression import _data_name, _deparse_formula, _r_column_name, _r_deparse_call
except ImportError:
    from _rcompat import r_levels, r_label, r_make_names
    from functions_train_test import _confusion_counts, _percent_table
    from glm_linear_regression import _data_name, _deparse_formula, _r_column_name, _r_deparse_call
##########################################################################################
# LDA (port of MASS::lda, method = "moment")
##########################################################################################
class LDA:
    """
    A fitted linear discriminant analysis, with MASS::lda's components:
    prior, counts, means, scaling, lev, svd, N, call and terms (the
    formula), plus x (the model matrix) and grouping.
    """

    def predict(self, newdata=None, prior=None, dimen=None):
        """
        MASS::predict.lda (method "plug-in"): classes, posterior probabilities and discriminant scores.

        Parameters:
        newdata (pandas.DataFrame, optional): Data with the predictors; the training data when None.
        prior (array-like, optional): Prior probabilities. Defaults to the model's.
        dimen (int, optional): Number of discriminants used. Defaults to all.

        Returns:
        dict: class (pandas.Categorical), posterior (DataFrame) and x (DataFrame of scores).
        """
        x = self.x if newdata is None else _model_matrix(self.terms, newdata)[1].to_numpy(dtype=float)
        prior = self.prior.to_numpy() if prior is None else np.asarray(prior, dtype=float)
        means = prior @ self.means.to_numpy()
        scaling = self.scaling.to_numpy()
        scores = (x - means) @ scaling
        dm = (self.means.to_numpy() - means) @ scaling
        dimen = len(self.svd) if dimen is None else min(dimen, len(self.svd))
        dm = dm[:, :dimen]
        dist = 0.5 * np.sum(dm ** 2, axis=1) - np.log(prior) - scores[:, :dimen] @ dm.T
        dist = np.exp(-(dist - dist.min(axis=1, keepdims=True)))
        posterior = dist / dist.sum(axis=1, keepdims=True)
        classes = pd.Categorical([self.lev[k] for k in posterior.argmax(axis=1)], categories=self.lev)
        names = list(self.prior.index)
        return {"class": classes, "posterior": pd.DataFrame(posterior, columns=names),
                "x": pd.DataFrame(scores[:, :dimen], columns=list(self.scaling.columns[:dimen]))}


def _expand_dot(formula, data):
    """R's "y ~ ." : every other column of data."""
    lhs, rhs = formula.split("~", 1)
    if rhs.strip() == ".":
        response = lhs.strip()
        rhs = " + ".join(c for c in data.columns if c != response)
    return f"{lhs.strip()} ~ {rhs.strip()}"


def _patsy_formula(formula, data):
    """Quote column names for patsy (R names may hold dots and spaces)."""
    names = sorted(data.columns, key=len, reverse=True)
    pattern = "|".join(re.escape(n) for n in names)
    return re.sub(rf"(?<![\w.\"])({pattern})(?![\w.\"])", lambda m: f'Q("{m.group(1)}")', formula)


def _model_matrix(formula, data):
    """Response and model matrix (intercept dropped, R's term order and names) of a formula, as R's model.frame."""
    quoted = _patsy_formula(formula, data)
    y, x = patsy.dmatrices(quoted, data, return_type="dataframe", NA_action="drop")
    info = x.design_info
    terms = sorted(patsy.ModelDesc.from_formula(quoted).rhs_termlist, key=lambda term: len(term.factors))
    columns = []
    for term in terms:
        if term.factors:
            piece = info.term_slices[term]
            columns.extend(info.column_names[piece])
    x = x[columns]
    x.columns = [_r_column_name(re.sub(r'Q\("([^"]*)"\)', r"\1", c)) for c in columns]
    response = re.sub(r'Q\("([^"]*)"\)', r"\1", y.design_info.term_names[0]) if y.shape[1] == 1 else None
    return response, x


def lda(formula, data, prior=None, tol=1e-4):
    """
    Linear discriminant analysis, as MASS::lda(formula, data) with
    method = "moment".

    Parameters:
    formula (str): Grouping ~ predictors, e.g. "Species ~ ." or "case ~ age + parity".
    data (pandas.DataFrame): Data; rows with missing values in the formula's variables are dropped.
    prior (array-like, optional): Prior probabilities of the groups. Defaults to the group proportions.
    tol (float, optional): Tolerance for constant variables and rank. Defaults to 1e-4.

    Returns:
    LDA: prior, counts, means, scaling (coefficients of the linear discriminants), lev, svd
    (ratios of between- to within-group standard deviations), N, call and terms; .predict().

    Examples:
    >>> model = lda("Species ~ .", data=iris)
    >>> model.scaling
    """
    expanded = _expand_dot(formula, data)
    response = expanded.split("~", 1)[0].strip()
    used = data.dropna(subset=[c for c in data.columns if re.search(rf"(?<![\w.]){re.escape(c)}(?![\w.])",
                                                                    expanded)])
    _, x_frame = _model_matrix(expanded, used)
    x = x_frame.to_numpy(dtype=float)
    grouping = used[response].to_numpy()
    lev = [r_label(v) for v in r_levels(pd.Series(grouping))]
    labels = np.array([r_label(v) for v in grouping])
    n, p = x.shape
    counts = np.array([np.sum(labels == level) for level in lev])
    proportions = counts / n
    prior = proportions if prior is None else np.asarray(prior, dtype=float)
    ng = len(lev)
    group_means = np.array([x[labels == level].mean(axis=0) for level in lev])
    centered = x - group_means[[lev.index(v) for v in labels]]
    f1 = np.sqrt(np.diag(np.cov(centered, rowvar=False).reshape(p, p)))
    if np.any(f1 < tol):
        constant = " ".join(str(k + 1) for k in np.flatnonzero(f1 < tol))
        raise ValueError(f"variables {constant} appear to be constant within groups")
    scaling = np.diag(1 / f1)
    _, d, vt = np.linalg.svd(np.sqrt(1 / (n - ng)) * centered @ scaling, full_matrices=False)
    rank = int(np.sum(d > tol))
    if rank == 0:
        raise ValueError("rank = 0: variables are numerically constant")
    scaling = scaling @ vt.T[:, :rank] @ np.diag(1 / d[:rank])
    xbar = prior @ group_means
    between = np.sqrt(n * prior / (ng - 1))[:, None] * (group_means - xbar) @ scaling
    _, d, vt = np.linalg.svd(between, full_matrices=False)
    rank = int(np.sum(d > tol * d[0]))
    if rank == 0:
        raise ValueError("group means are numerically identical")
    scaling = scaling @ vt.T[:, :rank]
    ld = [f"LD{k}" for k in range(1, rank + 1)]

    model = LDA()
    model.prior = pd.Series(prior, index=lev)
    model.counts = pd.Series(counts, index=lev)
    model.means = pd.DataFrame(group_means, index=lev, columns=x_frame.columns)
    model.scaling = pd.DataFrame(scaling, index=x_frame.columns, columns=ld)
    model.lev = lev
    model.svd = d[:rank]
    model.N = n
    model.terms = expanded
    model.x = x
    model.grouping = grouping
    model.call = _r_deparse_call("lda", [("formula", _deparse_formula(formula)), ("data", _data_name(data))])
    return model
##########################################################################################
# REPORT LDA
##########################################################################################
def report_lda(model, file=None):
    """
    Report of a linear discriminant analysis, as rwf::report_lda: priors,
    counts and group means, the discriminant coefficients, the number of
    observations with the singular values, the confusion matrix of the
    predicted classes and the call; with file, an Excel workbook.

    Parameters:
    model (LDA): A model fitted with lda().
    file (str, optional): Excel filename without extension. Defaults to None.

    Returns:
    dict: prior_counts, means, coeficients (rwf's spelling), terms,
    model_description, cmatrix (rwf's text confusion table) and call.

    Examples:
    >>> result = report_lda(lda("Species ~ .", data=iris))
    >>> result = report_lda(lda("case ~ .", data=infert), file="lda")
    """
    prior_counts = pd.DataFrame({"prior": model.prior, "counts": model.counts})
    means = model.means.copy()
    means.columns = r_make_names([f"mean.{c}" for c in means.columns])
    prior_counts = pd.concat([prior_counts, means], axis=1)
    model_description = pd.DataFrame({"Observations": model.N, "SDV": model.svd})
    call = pd.DataFrame({"call": [", ".join(model.call).replace(" ", "")]})
    predicted = model.predict()["class"]
    codes = {str(code + 1) for code in predicted.codes}
    counts = _confusion_counts(model.grouping, np.asarray(predicted), extra_levels=codes)
    cmatrix = _percent_table(counts)
    result = {"prior_counts": prior_counts, "means": model.means, "coeficients": model.scaling,
              "terms": model.terms, "model_description": model_description, "cmatrix": cmatrix, "call": call}
    if file is not None:
        try:
            from .functions_excel import excel_critical_value, excel_confusion_matrix
        except ImportError:
            from functions_excel import excel_critical_value, excel_confusion_matrix
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(model.scaling.reset_index(names="variable"), writer, sheet="Coefficients")
            excel_confusion_matrix(cmatrix, writer)
            excel_critical_value(prior_counts.reset_index(names="class"), writer, sheet="Priors and Counts")
            excel_critical_value(model.means.reset_index(names="class"), writer, sheet="Means")
            excel_critical_value(model_description, writer, sheet="Descriptives")
            excel_critical_value(call, writer, sheet="Call")
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    from pwf.datasets import load_dataset
    iris = load_dataset("iris")
    infert = load_dataset("infert")
    result = report_lda(lda("Species ~ .", data=iris))
    for key in ["prior_counts", "coeficients", "model_description", "cmatrix", "call"]:
        print(f"\n{key}:\n", result[key])
    print(report_lda(lda("case ~ .", data=infert))["coeficients"])
