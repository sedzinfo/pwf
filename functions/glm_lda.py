# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_LDA.R — a report/export wrapper around a fitted
Linear Discriminant Analysis model.

Deviations from the R original, by design:
  - MASS::lda() carries the original formula/data/call inside the fitted
    model object (R's formula-interface reflection), so R's report_lda
    only needs `model` itself to recover predictions, the observed
    outcome, and a call string. scikit-learn's LinearDiscriminantAnalysis
    has none of that — it's fit on plain X/y arrays with no memory of
    where they came from. report_lda() here therefore takes `X` and `y`
    explicitly (the same values used to fit `model`), matching this
    project's established pattern for report_* functions wrapping
    scikit-learn/statsmodels objects (e.g. glm_hlr.report_hlr,
    functions_train_test_full.result_confusion_performance) rather than
    R's model-formula magic.
  - model$scaling (R's discriminant coefficients) is normalized so each
    linear discriminant has unit within-group variance. scikit-learn's
    `scalings_` uses a different internal normalization (from its SVD/
    eigen solver), so the coefficient values won't numerically match R's
    even though they describe the same discriminant directions — the
    ratios *between* variables within a given LD column are meaningful,
    but the absolute scale and sign of each column is solver-dependent
    in both R and Python, and isn't expected to match across them.
  - model$svd (the ratio of between- to within-group standard deviations
    per discriminant axis, on its own unnormalized scale) has no exposed
    scikit-learn equivalent for the default svd solver — only
    `explained_variance_ratio_` (each axis's share of the total
    between-class variance, normalized to sum to 1) is available.
    model_description's "SDV" column uses that instead; it's a related
    but numerically different diagnostic (relative rather than absolute),
    not a like-for-like replacement.
  - There is no `call` reflection in Python; report_lda's "call" field is
    a short descriptive string built from the model/X/y actually passed
    in, not a deparsed function call.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd

try:
    from .functions_train_test_full import confusion_matrix_percent
except ImportError:
    from functions_train_test_full import confusion_matrix_percent
##########################################################################################
# LDA
##########################################################################################
def report_lda(model, X, y, file=None, w=10, h=10, base_size=10, title=""):
    """
    Report for a fitted sklearn LinearDiscriminantAnalysis model: prior/
    class-count/means table, discriminant coefficients ("scalings"),
    between/within-variance diagnostic per axis, a confusion matrix, and
    (optionally) an Excel export.

    Parameters:
    model (sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
        A fitted LDA model.
    X (pandas.DataFrame): Predictors used to fit `model`.
    y (array-like): True outcome labels used to fit `model`.
    file (str, optional): Output filename (without extension) for an
        Excel report. If None (default), no file is written.
    w, h (float, optional): Unused placeholders, kept for R-signature
        parity (R's version accepts a PDF page size that this report
        never actually uses — no plot is produced by report_lda itself,
        matching the R original, which also never uses w/h/base_size/title).
    base_size, title: Unused, see above.

    Returns:
    dict: prior_counts, means, coeficients, model_description, cmatrix, call.

    Examples:
    >>> import pandas as pd
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> df = pd.read_csv("data/iris.csv")
    >>> X = df.drop(columns=["Species"])
    >>> y = df["Species"]
    >>> model = LinearDiscriminantAnalysis().fit(X, y)
    >>> result = report_lda(model=model, X=X, y=y)
    >>> result = report_lda(model=model, X=X, y=y, file="lda")
    """
    X = pd.DataFrame(X)
    y = pd.Series(y).reset_index(drop=True)
    classes = model.classes_
    n_components = model.scalings_.shape[1]
    ld_names = [f"LD{i + 1}" for i in range(n_components)]

    counts = y.value_counts().reindex(classes).fillna(0).astype(int)
    prior_counts = pd.DataFrame({"prior": model.priors_, "counts": counts.to_numpy()}, index=classes)
    means_df = pd.DataFrame(model.means_, index=classes, columns=X.columns).add_prefix("mean.")
    prior_counts = pd.concat([prior_counts, means_df], axis=1)

    coeficients = pd.DataFrame(model.scalings_, index=X.columns, columns=ld_names)

    model_description = pd.DataFrame({"Observations": [len(X)] * n_components,
                                       "SDV": model.explained_variance_ratio_[:n_components]}, index=ld_names)

    predicted = model.predict(X)
    cmatrix = confusion_matrix_percent(observed=y.to_numpy(), predicted=predicted)

    call = pd.DataFrame({"call": [f"LinearDiscriminantAnalysis(solver={model.solver!r})."
                                   f"fit(X=[{','.join(X.columns.astype(str))}],y={y.name or 'y'})"]})

    result = {
        "prior_counts": prior_counts,
        "means": pd.DataFrame(model.means_, index=classes, columns=X.columns),
        "coeficients": coeficients,
        "model_description": model_description,
        "cmatrix": cmatrix,
        "call": call,
    }

    if file is not None:
        try:
            from .functions_excel import excel_critical_value, excel_confusion_matrix
        except ImportError:
            from functions_excel import excel_critical_value, excel_confusion_matrix
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_critical_value(coeficients.reset_index(names="variable"), writer, sheetname="Coefficients")
        excel_confusion_matrix(cmatrix, writer)
        excel_critical_value(prior_counts.reset_index(names="class"), writer, sheetname="Priors and Counts")
        excel_critical_value(result["means"].reset_index(names="class"), writer, sheetname="Means")
        excel_critical_value(model_description.reset_index(names="LD"), writer, sheetname="Descriptives")
        excel_critical_value(call, writer, sheetname="Call")
        writer._save()
        writer.close()

    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import statsmodels.api as sm

    print("=" * 80, "\nreport_lda (infert, binary outcome)\n", "=" * 80, sep="")
    infert = sm.datasets.get_rdataset("infert", "datasets").data
    infert = pd.get_dummies(infert, columns=["education"], drop_first=True)
    X1 = infert.drop(columns=["case"])
    y1 = infert["case"]
    model1 = LinearDiscriminantAnalysis().fit(X1, y1)
    result1 = report_lda(model=model1, X=X1, y=y1)
    print("prior_counts:\n", result1["prior_counts"])
    print("\ncoeficients:\n", result1["coeficients"])
    print("\nmodel_description:\n", result1["model_description"])
    print("\ncmatrix:\n", result1["cmatrix"])
    print("\ncall:\n", result1["call"])

    print("\n" + "=" * 80, "\nreport_lda (iris, 3-class outcome, Excel export)\n", "=" * 80, sep="")
    iris = sm.datasets.get_rdataset("iris", "datasets").data
    X2 = iris.drop(columns=["Species"])
    y2 = iris["Species"]
    model2 = LinearDiscriminantAnalysis().fit(X2, y2)
    result2 = report_lda(model=model2, X=X2, y=y2, file="lda_iris")
    print("prior_counts:\n", result2["prior_counts"])
    print("\ncoeficients:\n", result2["coeficients"])
    print("\nmodel_description:\n", result2["model_description"])
    print("\ncmatrix:\n", result2["cmatrix"])
    print("\nExcel written:", os.path.exists("lda_iris.xlsx"))
