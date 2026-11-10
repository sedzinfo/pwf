# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_TRAIN_TEST.R: ROC, confusion-matrix and
separability plots, confusion matrices with row/column proportions, the
cut-point search for probabilistic predictions, k-fold and
train/test/validation splits, and min-max scaling with dummy coding.

Matches rwf (tests/test_functions_train_test.py):
  - confusion() orders the classes as gtools::mixedsort of their R labels
    and labels rows "predicted" and columns "observed".
  - confusion_matrix_percent() returns rwf's text table: counts, the
    "sum" row/column and the "p" row (recall) / column (precision), every
    value rounded to 2 decimals and formatted with 2 decimals, as R's
    format(round(x, 2), nsmall = 2).
  - result_confusion_performance() steps the cut point as R's
    seq(min, max, by = step); Mean_proportion averages the column and row
    proportions (not Overall); cut holds every best cut point.
  - plot_roc() returns the two pROC curves (control level first) with
    pROC's automatic direction and AUC in the caption.
  - proportion_accurate() kappas are irr::kappa2's (unweighted, equal,
    squared weights).
  - recode_scale_dummy() dummy codes as psych::dummy.code (columns ordered
    by frequency) with rwf's column names.

Differences from rwf: plotnine cannot put an x axis on top, so the
confusion plots keep it below; k_fold/k_sample draw with numpy's random
generator (R's draws cannot be reproduced) and take the model formula as
a string ("y ~ x1 + x2"); recode_scale_dummy does not reproduce rwf's
error when several categorical columns have the same number of levels.
functions_train_test_full is a deprecated alias of this module.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import re

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_abline, geom_tile, geom_text, geom_density, geom_vline,
    scale_fill_gradient, labs, theme, theme_bw, element_text, element_blank,
    scale_fill_identity, scale_color_identity, scale_x_continuous, scale_y_continuous,
    scale_x_reverse, scale_linetype_manual, coord_fixed, theme_minimal,
)
try:
    from ._rcompat import r_color_ramp as _r_color_ramp, r_label as _r_label, r_mixedsort_key, r_number, r_round
except ImportError:
    from _rcompat import r_color_ramp as _r_color_ramp, r_label as _r_label, r_mixedsort_key, r_number, r_round
##########################################################################################
# R HELPERS
##########################################################################################
_mixedsort_key = r_mixedsort_key


def _r_seq(start, stop, by):
    """R's seq(start, stop, by = by)."""
    n = int(np.floor((stop - start) / by + 1e-10))
    return start + np.arange(n + 1) * by


def _r_linetypes(n):
    """ggplot2's linetype palette (scales::linetype_pal): solid, 22, 42, 44, 13, 1343, 73, 2262, ... as dash tuples."""
    patterns = ["solid", (0, (2, 2)), (0, (4, 2)), (0, (4, 4)), (0, (1, 3)), (0, (1, 3, 4, 3)), (0, (7, 3)),
                (0, (2, 2, 6, 2)), (0, (1, 2, 1, 2, 7, 2)), (0, (1, 1, 6, 1)), (0, (3, 1, 1, 1)),
                (0, (5, 1, 1, 1)), (0, (1, 2, 1, 1, 1, 1))]
    return [patterns[k % len(patterns)] for k in range(n)]


def _format_nsmall2(values):
    """R's format(round(x, 2), nsmall = 2) of one numeric column: 2 decimals, right-aligned to a common width."""
    texts = [f"{r_round(v, 2):.2f}" for v in values]
    width = max(len(t) for t in texts)
    return [t.rjust(width) for t in texts]
##########################################################################################
# PLOT ROC
##########################################################################################
def _roc(observed, predicted, control, case):
    """pROC::roc(observed, predicted, levels = c(control, case)) with direction = "auto"."""
    controls = predicted[observed == control]
    cases = predicted[observed == case]
    direction = ">" if np.median(controls) > np.median(cases) else "<"
    sign = 1 if direction == "<" else -1
    a, b = sign * cases, sign * controls
    auc = (np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))
    thresholds = np.unique(np.concatenate([a, b]))
    cuts = np.concatenate([[-np.inf], (thresholds[:-1] + thresholds[1:]) / 2, [np.inf]])
    sensitivity = np.array([np.mean(a > t) for t in cuts])
    specificity = np.array([np.mean(b <= t) for t in cuts])
    return {"auc": auc, "direction": direction, "levels": [control, case],
            "curve": pd.DataFrame({"specificity": specificity, "sensitivity": sensitivity})}


def plot_roc(observed, predicted, base_size=10, title=""):
    """
    ROC curves of a binary outcome, as rwf::plot_roc: one pROC curve per
    choice of control level (the larger level first, then the smaller),
    each with pROC's automatic direction (cases higher, "<", unless the
    controls' median is larger, ">"), sensitivity against reversed
    specificity, the chance diagonal, and the AUC in the caption.

    Parameters:
    observed (array-like): Observed classes (two values).
    predicted (array-like): Predicted scores or probabilities.
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Added after "ROC" in the title. Defaults to "".

    Returns:
    dict: Two plotnine plots keyed by their levels as R's toString ("control, case").

    Examples:
    >>> plots = plot_roc(observed=[0, 0, 1, 1], predicted=[0.1, 0.4, 0.35, 0.8])
    """
    observed = np.array([_r_label(v) for v in observed])
    predicted = np.asarray(predicted, dtype=float)
    levels = sorted(set(observed), key=_mixedsort_key)
    plots = {}
    for control, case in [(levels[1], levels[0]), (levels[0], levels[1])]:
        roc = _roc(observed, predicted, control, case)
        caption = (f"Observations:{len(observed)}\nAUC:{r_number(r_round(roc['auc'] * 100, 2))}%"
                   f"\nControl Level:{control}\nDirection:{roc['direction']}")
        plots[f"{control}, {case}"] = (
            ggplot(roc["curve"], aes(x="specificity", y="sensitivity"))
            + geom_line(alpha=0.5)
            + geom_abline(intercept=1, slope=-1)
            + scale_x_reverse()
            + labs(title=f"ROC {title}", caption=caption)
            + theme_bw(base_size=base_size)
            + coord_fixed())
    return plots
##########################################################################################
# PLOT CONFUSION
##########################################################################################
def plot_confusion(observed, predicted, base_size=10, title=""):
    """
    Confusion matrix heatmap, as rwf::plot_confusion: counts of observed
    (x) by predicted (y) classes, with the accuracy, the accuracy counting
    adjacent classes, and the unweighted, linear and squared kappas in the
    caption.

    Parameters:
    observed (array-like): Observed classes.
    predicted (array-like): Predicted classes.
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Added after "Confusion Matrix" in the title. Defaults to "".

    Returns:
    plotnine.ggplot: The plot.

    Examples:
    >>> plot_confusion(observed=[1, 2, 3, 1, 2, 3, 1, 2, 3], predicted=[1, 2, 3, 1, 1, 3, 1, 2, 2])
    """
    cmatrix = confusion(observed=observed, predicted=predicted)
    pa = proportion_accurate(observed=observed, predicted=predicted)
    observations = int(cmatrix.to_numpy().sum())
    melted = cmatrix.stack().rename("value").reset_index()
    melted["observed"] = pd.Categorical(melted["observed"], categories=list(cmatrix.columns)[::-1])
    melted["predicted"] = pd.Categorical(melted["predicted"], categories=list(cmatrix.index))

    def rounded(name):
        return r_number(r_round(pa[name].iloc[0], 2))

    caption = (f"Observations:{observations}\nAccuracy:{rounded('cm_diagonal')}"
               f"\nAccuracy with off diagonals:{rounded('cm_off_diagonal')}"
               f"\nKappa unweighted:{rounded('kappa_unweighted')}\nKappa linear:{rounded('kappa_linear')}"
               f"\nKappa squared:{rounded('kappa_squared')}")
    return (ggplot(melted, aes(x="observed", y="predicted", fill="value"))
            + geom_tile(color="white")
            + geom_text(aes(label="value"), color="black", size=base_size / 2)
            + theme_bw(base_size=base_size)
            + theme(axis_ticks_major_x=element_blank(), axis_ticks_major_y=element_blank(),
                    panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
                    panel_border=element_blank(), panel_background=element_blank(), legend_position="none")
            + scale_fill_gradient(low="#fafaff", high="#132B43")
            + labs(title=f"Confusion Matrix {title}", caption=caption)
            + coord_fixed())
##########################################################################################
# PLOT SEPARABILITY
##########################################################################################
def plot_separability(observed, predicted, base_size=10, title=""):
    """
    Density of the predicted values per observed class, as
    rwf::plot_separability.

    Parameters:
    observed (array-like): Observed classes.
    predicted (array-like): Predicted probabilities or scores.
    base_size (int, optional): Base font size. Defaults to 10.
    title (str, optional): Added after the title. Defaults to "".

    Returns:
    plotnine.ggplot: The plot.

    Examples:
    >>> plot_separability(observed=[0, 0, 1, 1, 1], predicted=[0.1, 0.3, 0.6, 0.8, 0.9])
    """
    observed = [_r_label(v) for v in observed]
    df = pd.DataFrame({"observed": pd.Categorical(observed, categories=sorted(set(observed), key=_mixedsort_key)),
                       "predicted": np.asarray(predicted, dtype=float)})
    return (ggplot(df, aes(x="predicted", color="observed"))
            + geom_density(size=1)
            + labs(title=f"Predicted proportion vs Observed category {title}", color="observed",
                   caption=f"Observations:{len(df)}")
            + theme_bw(base_size=base_size))
##########################################################################################
# CONFUSION PERFORMANCE
##########################################################################################
def result_confusion_performance(observed, predicted, step=.1, base_size=10, title=""):
    """
    Cut-point search for probabilistic predictions, as
    rwf::result_confusion_performance: for every cut point from the
    smallest to the largest prediction in steps of step (R's seq), the
    predictions above it are classed 1 and confusion_matrix_percent gives
    the overall accuracy, the precision of each predicted class
    (Collumn_Observed.k) and the recall of each observed class
    (Row_Predicted.k), all rounded to 2 decimals as rwf's table. The best
    cut points are those with the largest mean of the precisions and
    recalls (Mean_proportion).

    Parameters:
    observed (array-like): Observed classes (0/1).
    predicted (array-like): Predicted probabilities or scores.
    step (float, optional): Distance between cut points. Defaults to 0.1.
    base_size (int, optional): Base font size of the plot. Defaults to 10.
    title (str, optional): Added after the plot title. Defaults to "".

    Returns:
    dict: plot_performance (plotnine plot), cut_performance (DataFrame,
    one row per cut point), cut (array of the best cut points) and
    confusion_matrix (confusion_matrix_percent at their mean).

    Examples:
    >>> result_confusion_performance(observed=[0, 0, 1, 1, 0, 1], predicted=[0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted, dtype=float)
    rows = []
    for cut in _r_seq(np.nanmin(predicted), np.nanmax(predicted), step):
        table = confusion_matrix_percent(observed=observed, predicted=np.where(predicted > cut, 1, 0))
        values = table.apply(pd.to_numeric).to_numpy()
        precision = values[:-2, -1]
        recall = values[-1, :-2]
        row = {"cut_point": cut, "Overall": values[-1, -1]}
        for name, part in [("Collumn_Observed", precision), ("Row_Predicted", recall)]:
            for k, value in enumerate(part):
                row[name if len(part) == 1 else f"{name}.{k + 1}"] = value
        rows.append(row)
    df_cut_performance = pd.DataFrame(rows)
    cp = df_cut_performance.melt(id_vars="cut_point")
    df_cut_performance["Mean_proportion"] = df_cut_performance.iloc[:, 2:].mean(axis=1, skipna=True)
    best = df_cut_performance["Mean_proportion"] == df_cut_performance["Mean_proportion"].max()
    mcp = df_cut_performance.loc[best, "cut_point"].to_numpy()
    cmatrixp = confusion_matrix_percent(observed=observed, predicted=np.where(predicted > np.nanmean(mcp), 1, 0))
    cut_text = ", ".join(r_number(r_round(c, 4)) for c in mcp)
    plot_performance = (
        ggplot(cp, aes(x="cut_point", y="value", color="variable", linetype="variable"))
        + geom_line(size=1)
        + geom_vline(xintercept=list(mcp), size=1)
        + scale_linetype_manual(values=_r_linetypes(cp["variable"].nunique()))
        + labs(title=f"Confusion Matrix Performance {title}", x="Cut Point", y="Proportion Correct",
               caption=f"Observations:{len(df_cut_performance)}\nCut point:{cut_text}", color="", linetype="")
        + theme_bw(base_size=base_size))
    return {"plot_performance": plot_performance, "cut_performance": df_cut_performance, "cut": mcp,
            "confusion_matrix": cmatrixp}
##########################################################################################
# K FOLD AND TRAIN TEST VALIDATION SAMPLES
##########################################################################################
def _formula_variables(model_formula):
    """Outcome and predictor variable names of an "y ~ x1 + x2" formula (R's all.vars of each side)."""
    lhs, rhs = model_formula.split("~", 1)

    def names(side):
        found = re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", side)
        return list(dict.fromkeys(n for n in found if n not in {"I", "C", "log", "exp", "sqrt"}))

    return names(lhs), names(rhs)


def _dmatrix(frame, predictors, outcome):
    import xgboost
    def data_matrix(column):
        """R's data.matrix: text becomes its (sorted) factor codes."""
        if pd.api.types.is_numeric_dtype(column):
            return column
        return pd.Series(pd.Categorical(column).codes + 1.0, index=column.index).where(column.notna())

    return xgboost.DMatrix(frame[predictors].apply(data_matrix).to_numpy(dtype=float),
                           label=frame[outcome].to_numpy(dtype=float))


def _r_cut_labels(n, k):
    """R's cut(1:n, breaks = k, labels = FALSE): k equal-width bins of 1..n, numbered 1..k."""
    values = np.arange(1, n + 1)
    low, high = values.min(), values.max()
    width = high - low
    edges = np.linspace(low, high, k + 1)
    edges[0] -= width / 1000
    edges[-1] += width / 1000
    return np.searchsorted(edges, values, side="left").clip(1, k)


def k_fold(df, model_formula, k=10):
    """
    k-fold cross-validation sets, as rwf::k_fold: rows are put in k
    folds at random (R's sample of cut(1:n, k)) and each fold is once the
    test set and the rest the training set, with xgboost DMatrix objects
    for each (characters as their factor codes, R's data.matrix).

    Parameters:
    df (pandas.DataFrame): Data.
    model_formula (str): Formula naming the outcome and predictors, e.g. "y ~ x1 + x2".
    k (int, optional): Number of folds. Defaults to 10.

    Returns:
    dict: f (index, train, test, x_test, y_test per fold "f1".."fk"),
    index (fold of each row, 1..k), model_formula, variables,
    predictors, outcome and xgb (train, test, evals, watchlist, ytrain,
    ytest per fold).

    Examples:
    >>> folds = k_fold(df_insurance, "charges ~ age + bmi", k=5)
    """
    outcome, predictors = _formula_variables(model_formula)
    outcome = outcome[0]
    variables = [outcome] + predictors
    df = df[variables]
    index = np.random.permutation(_r_cut_labels(len(df), k))
    rows = np.arange(len(df))
    fold = {"index": {}, "train": {}, "test": {}, "x_test": {}, "y_test": {}}
    xgb = {}
    for i in range(1, k + 1):
        name = f"f{i}"
        test_rows = rows[index == i]
        train, test = df.iloc[np.setdiff1d(rows, test_rows)], df.iloc[test_rows]
        fold["index"][name] = test_rows + 1
        fold["train"][name], fold["test"][name] = train, test
        fold["x_test"][name], fold["y_test"][name] = test[predictors], test[outcome]
        xgb_train, xgb_test = _dmatrix(train, predictors, outcome), _dmatrix(test, predictors, outcome)
        evals = {"train": xgb_train, "test": xgb_test}
        xgb[name] = {"train": xgb_train, "test": xgb_test, "evals": evals, "watchlist": evals,
                     "ytrain": train[outcome], "ytest": test[outcome]}
        print("Fold Cases:", i, "Train:", len(train), "Test:", len(test), "Total:", len(train) + len(test),
              "Unique Train:", train.index.nunique(), "Unique Test:", test.index.nunique())
    return {"f": fold, "index": index, "model_formula": model_formula, "variables": variables,
            "predictors": predictors, "outcome": outcome, "xgb": xgb}


def k_sample(df, model_formula, k=1):
    """
    Train, test and validation sets, as rwf::k_sample: within each of k
    random folds (all rows when k = 1) the shuffled rows are halved into
    train and the rest, and the rest halved into test and validation,
    with xgboost DMatrix objects for each.

    As in rwf, xgb's yvalidation holds the test outcomes.

    Parameters:
    df (pandas.DataFrame): Data.
    model_formula (str): Formula naming the outcome and predictors, e.g. "y ~ x1 + x2".
    k (int, optional): Number of folds. Defaults to 1.

    Returns:
    dict: f (index/train/test/validation sets per fold "fold1".."foldk"),
    index, model_formula, variables, predictors, outcome and xgb.

    Examples:
    >>> samples = k_sample(df_insurance, "charges ~ age + bmi")
    """
    def halves(values):
        """R's split(v, ceiling(seq_along(v) / (length(v) / 2)))."""
        values = np.asarray(values)
        groups = np.ceil(np.arange(1, len(values) + 1) / (len(values) / 2))
        return values[groups == 1], values[groups == 2]

    outcome, predictors = _formula_variables(model_formula)
    outcome = outcome[0]
    variables = [outcome] + predictors
    df = df[variables]
    index = np.random.permutation(_r_cut_labels(len(df), k)) if k > 1 else np.ones(len(df), dtype=int)
    rows = np.arange(len(df))
    fold = {"index": {"train": {}, "test": {}, "validation": {}}, "train": {}, "test": {}, "validation": {},
            "x_test": {}, "y_test": {}, "x_validation": {}, "y_validation": {}}
    xgb = {}
    for i in range(1, k + 1):
        name = f"fold{i}"
        train_rows, rest = halves(np.random.permutation(rows[index == i]))
        test_rows, validation_rows = halves(rest)
        for part, part_rows in [("train", train_rows), ("test", test_rows), ("validation", validation_rows)]:
            fold["index"][part][name] = part_rows + 1
            fold[part][name] = df.iloc[part_rows]
        train, test, validation = fold["train"][name], fold["test"][name], fold["validation"][name]
        fold["x_test"][name], fold["y_test"][name] = test[predictors], test[outcome]
        fold["x_validation"][name], fold["y_validation"][name] = validation[predictors], validation[outcome]
        xgb_train, xgb_test = _dmatrix(train, predictors, outcome), _dmatrix(test, predictors, outcome)
        evals = {"train": xgb_train, "test": xgb_test}
        xgb[name] = {"train": xgb_train, "test": xgb_test, "validation": _dmatrix(validation, predictors, outcome),
                     "evals": evals, "watchlist": evals, "ytrain": train[outcome], "ytest": test[outcome],
                     "yvalidation": test[outcome]}
        print("Fold Cases:", i, "Train:", len(train), "Test:", len(test), "Validation:", len(test),
              "Total:", len(train) + len(test) + len(validation), "Unique Train:", train.index.nunique(),
              "Unique Test:", test.index.nunique(), "Unique Validation:", validation.index.nunique())
    return {"f": fold, "index": index, "model_formula": model_formula, "variables": variables,
            "predictors": predictors, "outcome": outcome, "xgb": xgb}
##########################################################################################
# SCALE AND DUMMY CODE
##########################################################################################
def _dummy_code(values):
    """psych::dummy.code: one column per value, most frequent first (ties in sorted order), NA rows NA."""
    series = pd.Series(values)
    counts = series.dropna().value_counts()
    levels = sorted(counts.index, key=lambda v: (-counts[v], _mixedsort_key(_r_label(v))))
    result = pd.DataFrame({_r_label(level): (series == level).astype(float) for level in levels},
                          index=series.index)
    result[series.isna()] = np.nan
    return result


def recode_scale_dummy(df, categories=10):
    """
    Min-max scaled numeric columns with dummy-coded categorical columns,
    as rwf::recode_scale_dummy: text/categorical columns with fewer than
    categories values are dummy coded as psych::dummy.code (one 0/1
    column per value, most frequent first), named column.value when there
    are several categorical columns and value alone when there is one;
    numeric columns are scaled to [0, 1].

    Parameters:
    df (pandas.DataFrame): Data.
    categories (int, optional): Largest number of values (exclusive) for dummy coding. Defaults to 10.

    Returns:
    pandas.DataFrame: The dummy columns followed by the scaled numeric columns, with df's index.

    Examples:
    >>> recode_scale_dummy(pd.DataFrame({"x": [1, 2, 3, 4, 5], "group": ["B", "A", "A", "B", "C"]}))
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    keep = [c for c in categorical if df[c].nunique(dropna=True) < categories]
    parts = []
    if len(categorical) > 1:
        for column in keep:
            coded = _dummy_code(df[column])
            coded.columns = [f"{column}.{level}" for level in coded.columns]
            parts.append(coded)
    elif len(categorical) == 1 and keep:
        parts.append(_dummy_code(df[keep[0]]))
    values = df[numeric]
    scaled = (values - values.min()) / (values.max() - values.min())
    result = pd.concat(parts + [scaled], axis=1)
    result.index = df.index
    return result
##########################################################################################
# PROPORTION ACCURATE
##########################################################################################
def _kappa2(observed, predicted, weight):
    """
    irr::kappa2(data.frame(observed, predicted), weight, sort.levels = TRUE)$value. irr sorts the
    levels as text ("1", "10", "11", "2", ...), which sets the distances of the weighted kappas.
    """
    labels_observed = [_r_label(v) for v in observed]
    labels_predicted = [_r_label(v) for v in predicted]
    levels = sorted(set(labels_observed) | set(labels_predicted), key=lambda v: (v.casefold(), v.swapcase()))
    position = {level: k for k, level in enumerate(levels)}
    a = np.array([position[v] for v in labels_observed])
    b = np.array([position[v] for v in labels_predicted])
    n_levels = len(levels)
    table = np.zeros((n_levels, n_levels))
    np.add.at(table, (a, b), 1)
    table /= table.sum()
    distance = np.abs(np.subtract.outer(np.arange(n_levels), np.arange(n_levels)))
    if weight == "unweighted":
        weights = (distance == 0).astype(float)
    elif weight == "equal":
        weights = 1 - distance / (n_levels - 1)
    else:
        weights = 1 - distance ** 2 / (n_levels - 1) ** 2
    observed_agreement = np.sum(weights * table)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0))
    chance_agreement = np.sum(weights * expected)
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def proportion_accurate(observed, predicted):
    """
    Accuracy measures of a classification, as rwf::proportion_accurate:
    the proportion on the diagonal of the confusion matrix, the proportion
    on it or next to it (classes one apart), and irr::kappa2's unweighted,
    linear ("equal") and squared kappas.

    Parameters:
    observed (array-like): Observed classes.
    predicted (array-like): Predicted classes.

    Returns:
    pandas.DataFrame: cm_diagonal, cm_off_diagonal, kappa_unweighted, kappa_linear, kappa_squared.

    Examples:
    >>> proportion_accurate(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
    """
    cmatrix = confusion(observed=observed, predicted=predicted).to_numpy()
    n = len(cmatrix)
    total = cmatrix.sum()
    near = sum(cmatrix[i, j] for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n)
    return pd.DataFrame({"cm_diagonal": [np.trace(cmatrix) / total], "cm_off_diagonal": [near / total],
                         "kappa_unweighted": [_kappa2(observed, predicted, "unweighted")],
                         "kappa_linear": [_kappa2(observed, predicted, "equal")],
                         "kappa_squared": [_kappa2(observed, predicted, "squared")]})
##########################################################################################
# CONFUSION
##########################################################################################
def confusion(observed, predicted):
    """
    Confusion matrix, as rwf::confusion: counts of every predicted (rows)
    by observed (columns) class, over the classes of both, labelled and
    ordered as gtools::mixedsort of their R labels.

    Parameters:
    observed (array-like): Observed classes.
    predicted (array-like): Predicted classes.

    Returns:
    pandas.DataFrame: Counts; index "predicted", columns "observed".

    Examples:
    >>> confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
    """
    return _confusion_counts(observed, predicted)


def _confusion_counts(observed, predicted, extra_levels=()):
    """confusion() over the classes of both vectors plus extra_levels (labels R adds through c() coercion)."""
    observed = [_r_label(v) for v in observed]
    predicted = [_r_label(v) for v in predicted]
    levels = sorted(set(observed) | set(predicted) | set(extra_levels), key=_mixedsort_key)
    table = pd.crosstab(pd.Categorical(predicted, categories=levels), pd.Categorical(observed, categories=levels),
                        dropna=False)
    table.index = pd.Index(levels, name="predicted")
    table.columns = pd.Index(levels, name="observed")
    return table
##########################################################################################
# CONFUSION MATRIX PERCENT
##########################################################################################
def confusion_matrix_percent(observed, predicted):
    """
    Confusion matrix with totals and proportions, as
    rwf::confusion_matrix_percent: the counts of confusion() with a "sum"
    column and row, a "p" column with each predicted class's precision
    (diagonal / row sum) and a "p" row with each observed class's recall
    (diagonal / column sum), 1 for the sums and the overall accuracy in
    the corner (0 for 0/0). As rwf, every value is rounded to 2 decimals
    and returned as text with 2 decimals, right-aligned per column.

    Parameters:
    observed (array-like): Observed classes.
    predicted (array-like): Predicted classes.

    Returns:
    pandas.DataFrame: The text table.

    Examples:
    >>> confusion_matrix_percent(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
    """
    return _percent_table(confusion(observed=observed, predicted=predicted))


def _percent_table(counts):
    """confusion_matrix_percent's text table from a table of counts."""
    values = counts.to_numpy(dtype=float)
    overall = np.trace(values) / values.sum()
    values = np.column_stack([values, values.sum(axis=1)])
    values = np.vstack([values, values.sum(axis=0)])
    n = len(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.array([values[i, i] / values[i, -1] for i in range(n)])
        recall = np.array([values[i, i] / values[-1, i] for i in range(n)])
    values = np.vstack([np.column_stack([values, precision]), np.append(recall, overall)])
    values[np.isnan(values)] = 0
    names = list(counts.index) + ["sum", "p"]
    columns = list(counts.columns) + ["sum", "p"]
    table = pd.DataFrame(values, index=names, columns=columns)
    return table.apply(_format_nsmall2)
##########################################################################################
# PLOT CONFUSION EXTENDED
##########################################################################################
def plot_confusion_extended(observed, predicted, positive=None, base_size=10, title=""):
    """
    2x2 confusion matrix with its row, column and overall measures (port of
    rwf::plot_confusion_extended): counts shaded blue by size, precision /
    false discovery rate and NPV / false omission rate beside the predicted
    rows, sensitivity / miss rate and specificity / fall-out under the
    observed columns, and accuracy, prevalence and the proportion
    incorrectly classified in the corner.

    plotnine cannot put the x axis on top, so the "Observed" labels sit
    below the grid (above it in rwf). The plot's data (p.data) holds the
    same cells, labels and colours as rwf's.

    Parameters:
    observed (array-like): True class labels, 2 distinct values.
    predicted (array-like): Predicted class labels, same 2 values.
    positive (optional): Label of the positive class. Defaults to the second
        sorted label, as in rwf::confusion().
    base_size (float, optional): Base font size. Defaults to 10.
    title (str, optional): Added after "Confusion Matrix" in the title.

    Returns:
    plotnine.ggplot: The plot.

    Examples:
    >>> plot_confusion_extended(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0])
    >>> plot_confusion_extended(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0], positive=0)
    """
    observed = np.array([_r_label(v) for v in observed])
    predicted = np.array([_r_label(v) for v in predicted])
    lvls = sorted(set(observed) | set(predicted))
    if len(lvls) != 2:
        raise ValueError("plot_confusion_extended() only supports a binary (2-class) confusion matrix")
    positive = lvls[1] if positive is None else _r_label(positive)
    negative = next(level for level in lvls if level != positive)

    TP = int(np.sum((predicted == positive) & (observed == positive)))
    TN = int(np.sum((predicted == negative) & (observed == negative)))
    FP = int(np.sum((predicted == positive) & (observed == negative)))
    FN = int(np.sum((predicted == negative) & (observed == positive)))
    total = TP + TN + FP + FN

    def ratio(a, b):
        return a / b if b else np.nan

    accuracy, prevalence, picc = (TP + TN) / total, (TP + FN) / total, (FN + FP) / total
    sensitivity, miss_rate = ratio(TP, TP + FN), ratio(FN, TP + FN)
    specificity, fallout = ratio(TN, FP + TN), ratio(FP, FP + TN)
    precision, fdr = ratio(TP, TP + FP), ratio(FP, TP + FP)
    npv, for_rate = ratio(TN, FN + TN), ratio(FN, FN + TN)

    def pct(x):
        return "NaN%" if np.isnan(x) else f"{x * 100:.1f}%"

    seq_ramp = _r_color_ramp("#cde2fb", "#0d366b", 100)

    def fill_for(n):
        return seq_ramp[int(max(1, min(100, round(n / total * 99) + 1))) - 1]

    ink_primary, ink_secondary, margin_fill = "#0b0b0b", "#52514e", "#f0efec"
    cells = pd.DataFrame({
        "col": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "row": [3, 3, 3, 2, 2, 2, 1, 1, 1],
        "kind": ["main", "main", "margin", "main", "main", "margin", "margin", "margin", "corner"],
        "n": [TP, FP, np.nan, FN, TN, np.nan, np.nan, np.nan, np.nan],
        "label": [
            f"TP\n{TP}\n({pct(TP / total)})",
            f"FP\n{FP}\n({pct(FP / total)})",
            f"Precision\n{pct(precision)}\nFDR {pct(fdr)}",
            f"FN\n{FN}\n({pct(FN / total)})",
            f"TN\n{TN}\n({pct(TN / total)})",
            f"NPV\n{pct(npv)}\nFOR {pct(for_rate)}",
            f"Sensitivity\n{pct(sensitivity)}\nMiss Rate {pct(miss_rate)}",
            f"Specificity\n{pct(specificity)}\nFall-out {pct(fallout)}",
            f"Accuracy {pct(accuracy)}\nPrevalence {pct(prevalence)}\nIncorrect {pct(picc)}",
        ],
    })
    main = cells["kind"] == "main"
    cells["fill"] = [fill_for(n) if is_main else margin_fill for n, is_main in zip(cells["n"], main)]
    cells["text_color"] = np.where(main & cells["n"].notna() & (cells["n"] / total > 0.5), "#ffffff", ink_primary)

    col_labels = [f"Observed: {positive}", f"Observed: {negative}", ""]
    row_labels = ["", f"Predicted: {negative}", f"Predicted: {positive}"]
    return (ggplot(cells, aes(x="col", y="row"))
            + geom_tile(aes(fill="fill"), color="#fcfcfb", size=2, width=0.96, height=0.96)
            + geom_text(aes(label="label", color="text_color"), size=base_size * 0.95, fontweight="bold")
            + scale_fill_identity()
            + scale_color_identity()
            + scale_x_continuous(breaks=[1, 2, 3], labels=col_labels, expand=(0.02, 0.02))
            + scale_y_continuous(breaks=[1, 2, 3], labels=row_labels, expand=(0.02, 0.02))
            + coord_fixed()
            + labs(title=f"Confusion Matrix {title}", x="", y="", caption=f"Observations: {total}")
            + theme_minimal(base_size=base_size)
            + theme(axis_text=element_text(color=ink_secondary, weight="bold"),
                    axis_ticks=element_blank(),
                    panel_grid=element_blank(),
                    plot_title=element_text(color=ink_primary, weight="bold"),
                    plot_caption=element_text(color=ink_secondary)))
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    np.random.seed(0)

    print("=" * 80, "\nconfusion / confusion_matrix_percent\n", "=" * 80, sep="")
    print(confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11]))
    print(confusion_matrix_percent(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2]))

    print("\n" + "=" * 80, "\nproportion_accurate\n", "=" * 80, sep="")
    print(proportion_accurate(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11]))

    print("\n" + "=" * 80, "\nresult_confusion_performance\n", "=" * 80, sep="")
    observed_bin = np.random.binomial(1, 0.4, 200)
    predicted_prob = np.clip(observed_bin * 0.3 + np.random.uniform(0, 0.7, 200), 0, 1)
    performance = result_confusion_performance(observed=observed_bin, predicted=predicted_prob, step=0.05)
    print("best cut points:", performance["cut"])
    print(performance["confusion_matrix"])

    print("\n" + "=" * 80, "\nplots\n", "=" * 80, sep="")
    print(list(plot_roc(observed_bin, predicted_prob, title="Example")))
    plot_confusion(observed=[1, 2, 3, 1, 2, 3, 1, 2, 3], predicted=[1, 2, 3, 1, 1, 3, 1, 2, 2]).draw()
    plot_separability(observed=observed_bin, predicted=predicted_prob).draw()

    print("\n" + "=" * 80, "\nk_fold / k_sample / recode_scale_dummy\n", "=" * 80, sep="")
    df = pd.DataFrame({"y": np.random.normal(size=40), "x1": np.random.normal(size=40),
                       "group": np.random.choice(["a", "b", "c"], 40)})
    folds = k_fold(df, "y ~ x1 + group", k=4)
    samples = k_sample(df, "y ~ x1", k=1)
    print(recode_scale_dummy(df).head())
