# -*- coding: utf-8 -*-
"""
Reports for xgboost models (port of rwf ML_XGBOOST.R): a combined diagram
of all trees (plot_trees_xgboost) and a report of the model's structure,
training log, feature importance and validation performance
(report_xgboost).

Differences from rwf:
  - The model is a Python xgboost.Booster (from xgboost.train) or a
    scikit-learn style XGBModel, whose booster is used.
  - A Python Booster does not keep its evaluation log, call or the
    parameters it was trained with, as R's does. Pass them to
    report_xgboost with evals_result (the dict that xgboost.train fills),
    params and call; otherwise the parameters are read from the model's
    configuration and the log is empty.
  - plot_trees_xgboost draws the same merged tree diagram as
    xgb.plot.multi.trees into a self-contained HTML file with an SVG
    image, instead of a DiagrammeR widget (no Graphviz needed).
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import html
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_col, labs, theme_bw, coord_flip, position_dodge,
)
try:
    from ._rcompat import r_number
    from .functions_excel import excel_confusion_matrix, excel_critical_value
    from .functions_plot import report_pdf
    from .functions_strings import str_aes
    from .functions_train_test_full import result_confusion_performance
    from .glm_linear_regression import plot_scatterplot
except ImportError:
    from _rcompat import r_number
    from functions_excel import excel_confusion_matrix, excel_critical_value
    from functions_plot import report_pdf
    from functions_strings import str_aes
    from functions_train_test_full import result_confusion_performance
    from glm_linear_regression import plot_scatterplot
##########################################################################################
# HELPERS
##########################################################################################
def _booster(model):
    if isinstance(model, xgb.Booster):
        return model
    if hasattr(model, "get_booster"):
        return model.get_booster()
    raise TypeError("model must be an xgboost Booster from xgboost.train (or an XGBModel)")


def _r_format(values, digits=5):
    """R's format(x, digits = 5) of a numeric vector: common decimals, right-justified to a common width."""
    values = [float(v) for v in values]
    decimals = 0
    for v in values:
        if v == 0 or not np.isfinite(v):
            continue
        text = f"{v:.{digits - 1}e}"
        mantissa, exponent = text.split("e")
        significant = mantissa.replace("-", "").replace(".", "").rstrip("0") or "0"
        decimals = max(decimals, len(significant) - 1 - int(exponent))
    strings = [f"{v:.{decimals}f}" for v in values]
    width = max(len(s) for s in strings)
    return [s.rjust(width) for s in strings]


def _tree_table(booster):
    """xgb.model.dt.tree: one row per node (Tree, Node, ID, Feature, Yes, No, Gain, Cover)."""
    table = booster.trees_to_dataframe()
    return table[["Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Gain", "Cover"]]
##########################################################################################
# TREE PLOT
##########################################################################################
def _multi_trees(booster, features_keep=10):
    """Nodes and edges of xgb.plot.multi.trees: all trees merged by node position."""
    tree = _tree_table(booster).copy()
    position = {}
    for row in tree.itertuples():
        if row.Node == 0:
            position[row.ID] = "0"
    pending = [r for r in tree.itertuples() if r.Node == 0]
    while pending:
        children = []
        for row in pending:
            if isinstance(row.Yes, str):
                position[row.Yes] = position[row.ID] + "_0"
                position[row.No] = position[row.ID] + "_1"
                children.append(row.Yes)
                children.append(row.No)
        pending = [r for r in tree.itertuples() if r.ID in set(children)]
    tree["abs.node.position"] = tree["ID"].map(position)
    gains = tree.groupby(["abs.node.position", "Feature"], sort=False)["Gain"].sum().reset_index()
    nodes = []
    for node_position, group in gains.groupby("abs.node.position", sort=False):
        keep = group.head(features_keep)
        labels = [f"{f} ({g})" for f, g in zip(keep["Feature"], _r_format(keep["Gain"]))]
        nodes.append({"abs.node.position": node_position, "Text": "\n".join(labels)})
    nodes = pd.DataFrame(nodes)
    splits = tree[tree["Feature"] != "Leaf"]
    edges = pd.concat([
        pd.DataFrame({"From": splits["abs.node.position"], "To": splits["abs.node.position"] + "_0"}),
        pd.DataFrame({"From": splits["abs.node.position"], "To": splits["abs.node.position"] + "_1"}),
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)
    return nodes, edges


def _tree_svg(nodes, edges):
    """Left-to-right layered drawing of the merged tree, as SVG."""
    depth = {p: p.count("_") for p in nodes["abs.node.position"]}
    by_depth = {}
    for p in nodes["abs.node.position"]:
        by_depth.setdefault(depth[p], []).append(p)
    lines = {p: t.split("\n") for p, t in zip(nodes["abs.node.position"], nodes["Text"])}
    char_w, line_h, pad, gap_x, gap_y = 7.2, 15, 8, 60, 16
    widths = {p: max(len(line) for line in lines[p]) * char_w + 2 * pad for p in lines}
    heights = {p: len(lines[p]) * line_h + 2 * pad for p in lines}
    column_w = {d: max(widths[p] for p in ps) for d, ps in by_depth.items()}
    x0, coords = {}, {}
    x = 10
    for d in sorted(by_depth):
        x0[d] = x
        x += column_w[d] + gap_x
    column_h = {d: sum(heights[p] for p in ps) + gap_y * (len(ps) - 1) for d, ps in by_depth.items()}
    total_h = max(column_h.values()) + 20
    for d, ps in by_depth.items():
        y = (total_h - column_h[d]) / 2
        for p in ps:
            coords[p] = (x0[d], y)
            y += heights[p] + gap_y
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{x:.0f}" height="{total_h:.0f}" '
             'font-family="Helvetica, Arial, sans-serif" font-size="12">',
             '<defs><marker id="vee" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" '
             'orient="auto"><path d="M0,0 L10,5 L0,10 L3,5 z" fill="DimGray"/></marker></defs>']
    for start, end in zip(edges["From"], edges["To"]):
        if start not in coords or end not in coords:
            continue
        (sx, sy), (ex, ey) = coords[start], coords[end]
        parts.append(f'<line x1="{sx + widths[start]:.1f}" y1="{sy + heights[start] / 2:.1f}" x2="{ex:.1f}" '
                     f'y2="{ey + heights[end] / 2:.1f}" stroke="DimGray" marker-end="url(#vee)"/>')
    for p, (nx, ny) in coords.items():
        parts.append(f'<rect x="{nx:.1f}" y="{ny:.1f}" width="{widths[p]:.1f}" height="{heights[p]:.1f}" '
                     'fill="beige" stroke="DimGray"/>')
        for i, line in enumerate(lines[p]):
            parts.append(f'<text x="{nx + pad:.1f}" y="{ny + pad + (i + 0.8) * line_h:.1f}" '
                         f'xml:space="preserve">{html.escape(line)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def plot_trees_xgboost(model, train, file="xgboost"):
    """
    Draw all trees of an xgboost model merged into one diagram, keeping the
    10 most important features per position, and save it as <file>.html
    (port of rwf::plot_trees_xgboost, which uses xgb.plot.multi.trees).

    Parameters:
    model (xgboost.Booster or XGBModel): The model.
    train (pandas.DataFrame or array): Training data; its column names label
        the features when the model has none and the counts match.
    file (str, optional): Output filename without extension, relative to
        the working directory or absolute. Defaults to "xgboost".

    Returns:
    dict: file (path written), nodes (abs.node.position, Text) and edges
    (From, To) of the diagram.

    Examples:
    >>> plot_trees_xgboost(model=booster, train=x_train, file="classification_trees")
    """
    booster = _booster(model)
    columns = list(getattr(train, "columns", []))
    if not booster.feature_names and columns and booster.num_features() == len(columns):
        booster = booster.copy()
        booster.feature_names = [str(c) for c in columns]
    nodes, edges = _multi_trees(booster, features_keep=10)
    path = os.path.abspath(f"{file}.html")
    page = ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>xgboost trees</title></head>\n"
            f"<body style=\"margin:0\">\n{_tree_svg(nodes, edges)}\n</body></html>\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return {"file": path, "nodes": nodes, "edges": edges}
##########################################################################################
# XGBOOST REPORT
##########################################################################################
def _leaf_depths(booster):
    """xgb.plot.deepness(plot = FALSE): one row per leaf with ID, Tree, Depth, Cover, Weight."""
    tree = _tree_table(booster)
    parent = {}
    for row in tree[tree["Feature"] != "Leaf"].itertuples():
        parent[row.Yes] = row.ID
        parent[row.No] = row.ID
    rows = []
    for row in tree[tree["Feature"] == "Leaf"].itertuples():
        depth, node = 1, row.ID
        while node in parent:
            node, depth = parent[node], depth + 1
        rows.append({"ID": row.ID, "Tree": int(row.Tree), "Depth": depth, "Cover": row.Cover, "Weight": row.Gain})
    depths = pd.DataFrame(rows, columns=["ID", "Tree", "Depth", "Cover", "Weight"])
    return depths.sort_values(["Tree", "ID"], kind="mergesort").reset_index(drop=True)


def _importance(booster, feature_names):
    """xgb.importance for a tree model: Feature, Gain, Cover, Frequency (shares), by decreasing Gain."""
    tree = _tree_table(booster)
    splits = tree[tree["Feature"] != "Leaf"]
    if splits.empty:
        return pd.DataFrame(columns=["Feature", "Gain", "Cover", "Frequency"])
    grouped = splits.groupby("Feature", sort=False).agg(Gain=("Gain", "sum"), Cover=("Cover", "sum"),
                                                        Frequency=("Gain", "size"))
    grouped = grouped / grouped.sum()
    if feature_names and all(str(f).startswith("f") and str(f)[1:].isdigit() for f in grouped.index):
        grouped.index = [feature_names[int(str(f)[1:])] for f in grouped.index]
    grouped = grouped.rename_axis("Feature").reset_index()
    return grouped.sort_values("Gain", ascending=False, kind="mergesort").reset_index(drop=True)


def _config_parameters(booster):
    """Training parameters from the model's configuration (Python boosters do not keep the ones given)."""
    config = json.loads(booster.save_config())["learner"]
    params = {"objective": config["objective"]["name"]}
    gradient_booster = config.get("gradient_booster", {})
    for section in ("gbtree_train_param", "tree_train_param"):
        params.update(gradient_booster.get(section, {}))
    return params


def report_xgboost(model, validation_data=None, label=None, file="xgboost", w=10, h=10, base_size=10, title="",
                   fast=False, evals_result=None, params=None, call=None):
    """
    Report of an xgboost model (port of rwf::report_xgboost): structure of
    the trees, training log, feature importance and, with validation data,
    predictions, validation metrics and (for classification) cut-point
    performance. With file, the plots go to <file>_<title>.pdf and the
    tables to <file>.xlsx.

    Parameters:
    model (xgboost.Booster or XGBModel): The model.
    validation_data (pandas.DataFrame, optional): Data with the model's
        features and the label column.
    label (str, optional): Outcome column of validation_data.
    file (str, optional): Output filename without extension; None writes
        nothing. Defaults to "xgboost", as in rwf.
    w, h (float, optional): PDF page size in inches. Default 10.
    base_size (float, optional): Base font size. Default 10.
    title (str, optional): Added to plot titles and the PDF name.
    fast (bool, optional): Leave the evaluation log out of the workbook.
    evals_result (dict, optional): Python only: the dict xgboost.train(...,
        evals_result=...) filled, e.g. {"train": {"logloss": [...]}}; gives
        the evaluation log (R keeps it in the model).
    params (dict, optional): Python only: the parameters the model was
        trained with; defaults to those in the model's configuration.
    call (str, optional): Python only: text of the training call.

    Returns:
    dict: plots (regression, performance, depth, cover, weight, error,
    importance), result (importance, parameters, model_call,
    evaluation_log, model_summary, validation_metrics), observed,
    predicted, feature_names, as in rwf.

    Examples:
    >>> evals = {}
    >>> booster = xgboost.train({"objective": "binary:logistic"}, dtrain, 20,
    ...                         evals=[(dtrain, "train")], evals_result=evals)
    >>> report_xgboost(booster, validation_data=test, label="admit", evals_result=evals, file=None)
    """
    booster = _booster(model)
    plots, result = {}, {}
    observed = predicted = None
    objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
    is_regression = objective.startswith(("reg:", "count:", "survival:"))

    feature_names = list(booster.feature_names or [])
    if not feature_names:
        splits = _tree_table(booster)
        feature_names = list(pd.unique(splits.loc[splits["Feature"] != "Leaf", "Feature"]))
    if not feature_names and validation_data is not None:
        feature_names = [c for c in validation_data.columns if c != label]

    if validation_data is not None:
        if label is None or label not in validation_data.columns:
            raise ValueError("label must be provided and exist in validation_data")
        missing = [f for f in feature_names if f not in validation_data.columns]
        if missing:
            raise ValueError("validation_data is missing required features: " + ", ".join(missing))
        observed = validation_data[label].to_numpy()
        matrix = xgb.DMatrix(validation_data[feature_names].to_numpy(dtype=float),
                             feature_names=feature_names if booster.feature_names else None)
        predicted = booster.predict(matrix)
        plots["regression"] = plot_scatterplot(pd.DataFrame({"observed": observed, "predicted": predicted}))
        numeric = pd.api.types.is_numeric_dtype(validation_data[label]) and len(predicted) == len(observed)
        observed_float = observed.astype(float) if numeric else None
        if numeric and is_regression:
            error = predicted - observed_float
            keep = ~np.isnan(error)
            r = np.corrcoef(observed_float[keep], predicted[keep])[0, 1]
            result["validation_metrics"] = pd.DataFrame({
                "Metric": ["N", "RMSE", "MAE", "Mean error", "R squared"],
                "value": [keep.sum(), np.sqrt(np.mean(error[keep] ** 2)), np.mean(np.abs(error[keep])),
                          np.mean(error[keep]), r ** 2]})
        elif (numeric and np.isin(observed_float[~np.isnan(observed_float)], [0, 1]).all()
              and ((predicted >= 0) & (predicted <= 1)).all()):
            from sklearn.metrics import roc_auc_score
            keep = ~np.isnan(observed_float) & ~np.isnan(predicted)
            y = observed_float[keep]
            p = np.clip(predicted[keep], 1e-15, 1 - 1e-15)
            try:
                auc = roc_auc_score(y, p)
            except ValueError:
                auc = np.nan
            result["validation_metrics"] = pd.DataFrame({
                "Metric": ["N", "AUC", "Log loss", "Accuracy (cut-off 0.5)"],
                "value": [len(y), auc, -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                          np.mean((p >= 0.5) == y)]})
        if not is_regression:
            try:
                plots["performance"] = result_confusion_performance(observed=observed, predicted=predicted)
            except Exception:  # noqa: BLE001 - rwf also drops the cut-point plot when it cannot be made
                pass

    params = _config_parameters(booster) if params is None else dict(params)
    result["parameters"] = pd.DataFrame({"Hyperparameter": list(params), "value": [str(v) for v in params.values()]})
    result["model_call"] = pd.DataFrame({"Parameters": ["Call"], "value": [call.replace(" ", "") if call else np.nan]})

    evaluation_log = pd.DataFrame()
    if evals_result:
        columns = {f"{data}_{metric}": values for data, metrics in evals_result.items() for metric, values in metrics.items()}
        evaluation_log = pd.DataFrame({"iter": np.arange(1, len(next(iter(columns.values()))) + 1), **columns})
    result["evaluation_log"] = evaluation_log

    best_iteration = getattr(booster, "best_iteration", None) if "best_iteration" in booster.attributes() else None
    best_score = booster.attributes().get("best_score")
    result["model_summary"] = pd.DataFrame({
        "Statistic": ["Objective", "Boosting rounds", "Features", "Best iteration", "Best score"],
        "value": [objective, str(booster.num_boosted_rounds()), str(len(feature_names)),
                  str(best_iteration + 1) if best_iteration is not None else np.nan,
                  r_number(float(f"{float(best_score):.4g}")) if best_score is not None else np.nan]})

    depths = _leaf_depths(booster)
    if len(depths):
        for column in ["Depth", "Cover", "Weight"]:
            plots[column.lower()] = (ggplot(depths, aes(y=column, x="Tree")) + geom_point(alpha=0.1)
                                     + labs(x="Tree", y=column, title="") + theme_bw(base_size=base_size))
    if len(evaluation_log):
        error_df = evaluation_log.melt(id_vars="iter", var_name="Metric", value_name="value")
        error_df = error_df.rename(columns={"iter": "Iteration"})
        error_df["Metric"] = [str_aes(m) for m in error_df["Metric"]]
        plots["error"] = (ggplot(error_df, aes(y="value", x="Iteration", color="Metric"))
                          + geom_line(size=base_size / 15)
                          + labs(x="Iteration", y="Metric Value", title=f"Training Log: {title}")
                          + theme_bw(base_size=base_size))

    importance = _importance(booster, feature_names)
    result["importance"] = importance
    if len(importance):
        order = importance.sort_values("Gain", kind="mergesort")["Feature"].tolist()
        long = importance.melt(id_vars="Feature", value_vars=["Gain", "Cover", "Frequency"],
                               var_name="Metric", value_name="value").rename(columns={"Feature": "Factor"})
        long["Factor"] = pd.Categorical(long["Factor"], categories=order)
        plots["importance"] = (ggplot(long, aes(y="value", x="Factor", fill="Metric"))
                               + geom_col(position=position_dodge(), colour="white")
                               + labs(x="Predictor", y="Relative Importance", title=f"Importance {title}")
                               + theme_bw(base_size=base_size) + coord_flip())

    if file is not None and plots:
        figures = []
        for value in plots.values():
            if isinstance(value, dict):
                figures += [v for v in value.values() if hasattr(v, "draw") or hasattr(v, "savefig")]
            elif value is not None:
                figures.append(value)
        report_pdf(plotlist=figures, file=file, title=title or None, w=w, h=h, print_plot=False)
    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            performance = plots.get("performance")
            if isinstance(performance, dict) and performance.get("confusion_matrix") is not None:
                excel_confusion_matrix(performance["confusion_matrix"], writer)
            excel_critical_value(result["model_summary"], writer, sheetname="Model Summary")
            if "validation_metrics" in result:
                excel_critical_value(result["validation_metrics"], writer, sheetname="Validation Metrics")
            if len(importance):
                excel_critical_value(importance, writer, sheetname="Feature Importance")
            if len(result["parameters"]):
                excel_critical_value(result["parameters"], writer, sheetname="Hyperparameters")
            if not fast and len(evaluation_log):
                excel_critical_value(evaluation_log, writer, sheetname="Evaluation Log")
    return {"plots": plots, "result": result, "observed": observed, "predicted": predicted,
            "feature_names": feature_names}
