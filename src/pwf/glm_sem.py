# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_SEM.R: path diagrams (plot_cfa_gg, plot_cfa), a
full CFA/SEM report (report_cfa) and fit across sample sizes
(simulate_cfa_fit).

Models are fitted with pwf's own lavaan-compatible engine (pwf._sem, also
available here as cfa, sem and simulate_data), which reproduces lavaan's
estimates, standard errors, standardized solution, fit measures,
modification indices and factor scores for maximum likelihood, with
listwise deletion (default) or with missing = "ml" (full information
maximum likelihood, as lavaan's missing = "ml"). No R installation is
needed. With missing = "ml", lavaan's FIML-corrected ("robust") fit indices,
the GFI family and CRMR are not ported and come out missing.

Deviations from the R original, by design:
  - R's igraph-based node layouts (layout_in_circle, layout_as_tree,
    layout_with_fr) are replaced with networkx (circular_layout,
    graphviz's "dot" layout for "tree", spring_layout for "spring"); without
    Graphviz a simple 2-row layout (latents on top, their indicators below)
    is used for "tree".
  - ggplot2::geom_curve (latent covariance arcs) has no plotnine
    equivalent; a quadratic Bezier arc drawn with geom_path is used instead.
  - inspect(fit, "est"/"std") results are dicts of DataFrames (one per
    model matrix), and the text summary written to the log is pwf's own
    layout, not lavaan's summary() print.
  - simulate_cfa_fit runs its sample sizes one after another (rwf runs them
    in parallel with future.apply), and simulated data come from numpy, so
    they differ from R's for the same seed.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import contextlib
import io
import os

import numpy as np
import pandas as pd
import networkx as nx
from plotnine import (
    ggplot, aes, geom_segment, geom_rect, geom_text, geom_label, geom_path,
    coord_fixed, theme_void, theme, labs, arrow, element_text,
)

try:
    from ._sem import SEMFit, cfa, sem, simulate_data
    from .functions import remove_nc
    from .functions_generate_data import simulate_correlation_from_sample
    from .functions_environment import write_txt
    from .functions_excel import excel_critical_value, excel_generic_format, excel_matrix
    from .functions_strings import output_separator
    from .functions_plot import report_pdf
    from .glm_linear_regression import plot_scatterplot
except ImportError:
    from _sem import SEMFit, cfa, sem, simulate_data
    from functions import remove_nc
    from functions_generate_data import simulate_correlation_from_sample
    from functions_environment import write_txt
    from functions_excel import excel_critical_value, excel_generic_format, excel_matrix
    from functions_strings import output_separator
    from functions_plot import report_pdf
    from glm_linear_regression import plot_scatterplot

__all__ = ["cfa", "sem", "simulate_data", "plot_cfa_gg", "plot_cfa", "report_cfa", "simulate_cfa_fit"]
##########################################################################################
# GRAPH LAYOUT (internal)
##########################################################################################
def _simple_tree_layout(nodes, latent_vars, edges):
    """Fallback 2-row layout (latents on top, their own indicators below) when graphviz is unavailable."""
    latents = [n for n in nodes if n in latent_vars]
    observed = [n for n in nodes if n not in latent_vars]
    children_of = {lat: [] for lat in latents}
    for lhs, rhs in edges:
        if lhs in children_of:
            children_of[lhs].append(rhs)

    pos = {}
    for i, lat in enumerate(latents):
        pos[lat] = (i * 3.0, 1.0)
    for lat in latents:
        children = children_of.get(lat, [])
        n = max(len(children), 1)
        start = pos[lat][0] - (n - 1) / 2.0
        for j, child in enumerate(children):
            pos[child] = (start + j, 0.0)
    for k, n in enumerate([o for o in observed if o not in pos]):
        pos[n] = (len(latents) * 3 + k, 0.0)
    return pos


def _bezier_arc(x0, y0, x1, y1, curvature=0.35, n=30):
    """Quadratic Bezier arc between two points -- plotnine has no geom_curve."""
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist == 0:
        return np.full(n, x0), np.full(n, y0)
    nx_, ny_ = -dy / dist, dx / dist
    cx, cy = xm + nx_ * curvature * dist, ym + ny_ * curvature * dist
    t = np.linspace(0, 1, n)
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
    return xs, ys


def _check_model(model):
    if not isinstance(model, SEMFit):
        raise TypeError("model must be a fitted model from pwf cfa() or sem() (rwf: a lavaan object)")
##########################################################################################
# MODEL PLOT
##########################################################################################
def plot_cfa_gg(model, what="std", layout="tree", label_size=3.2, edge_label_size=2.6,
                color_latent="#4f8ef7", color_observed="#e8eaf0", **kwargs):
    """
    Path diagram of a fitted CFA/SEM model, drawn with plotnine (port of
    rwf::plot_cfa_gg).

    Parameters:
    model (SEMFit): A model fitted with cfa() or sem().
    what (str, optional): "std" (standardized), "est" (unstandardized),
        or "eq" (parameter labels). Defaults to "std".
    layout (str, optional): "tree", "circle", or "spring". Defaults to "tree".
    label_size (float, optional): Node label size. Defaults to 3.2.
    edge_label_size (float, optional): Path coefficient label size. Defaults to 2.6.
    color_latent (str, optional): Fill color for latent variable nodes.
    color_observed (str, optional): Fill color for observed variable nodes.
    **kwargs: Ignored (kept for API compatibility with plot_cfa).

    Returns:
    plotnine.ggplot

    Examples:
    >>> fit = cfa("LATENT1 =~ X1 + X2 + X3; LATENT2 =~ X4 + X5 + X6", df)
    >>> plot_cfa_gg(fit, what="std")
    >>> plot_cfa_gg(fit, what="std", layout="circle")
    """
    _check_model(model)
    if what not in ("std", "est", "eq"):
        raise ValueError('what must be one of "std", "est", "eq"')
    if layout not in ("tree", "circle", "spring"):
        raise ValueError('layout must be one of "tree", "circle", "spring"')

    pe = model.parameter_estimates()
    pe = pe[pe["op"] != "r2"]
    if "label" not in pe.columns:
        pe["label"] = ""
    edge_col = {"std": "std.all", "est": "est", "eq": "label"}[what]

    latent_vars = pe.loc[pe["op"] == "=~", "lhs"].unique().tolist()
    observed_vars = pe.loc[pe["op"] == "=~", "rhs"].unique().tolist()
    all_nodes = list(dict.fromkeys(latent_vars + observed_vars))
    node_df = pd.DataFrame({"name": all_nodes, "is_lat": [n in latent_vars for n in all_nodes]})

    edge_list = pe[pe["op"].isin(["=~", "~"])][["lhs", "rhs"]]
    edge_list = edge_list[edge_list["lhs"].isin(all_nodes) & edge_list["rhs"].isin(all_nodes)]
    edges = list(edge_list.itertuples(index=False, name=None))

    g = nx.DiGraph()
    g.add_nodes_from(all_nodes)
    g.add_edges_from(edges)

    if layout == "circle":
        pos = nx.circular_layout(g)
    elif layout == "spring":
        pos = nx.spring_layout(g, iterations=1000, seed=1)
    else:
        try:
            pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
        except Exception:
            pos = _simple_tree_layout(all_nodes, latent_vars, edges)

    xs = np.array([pos[n][0] for n in all_nodes], dtype=float)
    ys = np.array([pos[n][1] for n in all_nodes], dtype=float)

    def safe_scale(v):
        rng = v.max() - v.min()
        return np.full(len(v), 5.0) if rng == 0 else (v - v.min()) / rng * 10

    node_df["x"] = safe_scale(xs)
    node_df["y"] = safe_scale(ys)
    node_df["hw"] = np.where(node_df["is_lat"], 0.90, 0.70)
    node_df["hh"] = np.where(node_df["is_lat"], 0.55, 0.42)
    node_df["xmin"] = node_df["x"] - node_df["hw"]
    node_df["xmax"] = node_df["x"] + node_df["hw"]
    node_df["ymin"] = node_df["y"] - node_df["hh"]
    node_df["ymax"] = node_df["y"] + node_df["hh"]

    def _attach_coords(edf, xcol, ycol, side):
        merged = edf.merge(node_df[["name", "x", "y"]], left_on=side, right_on="name")
        return merged.rename(columns={"x": xcol, "y": ycol}).drop(columns="name")

    def _display(frame):
        if what == "eq":
            label_col = frame["label"] if "label" in frame.columns else pd.Series([""] * len(frame))
            est_fmt = frame["est"].map(lambda v: f"{v:.3f}")
            return np.where(label_col.isna() | (label_col == ""), est_fmt, label_col)
        return frame[edge_col].map(lambda v: f"{v:.3f}")

    loadings = pe[pe["op"] == "=~"].copy()
    edge_df = _attach_coords(loadings, "x_from", "y_from", "lhs")
    edge_df = _attach_coords(edge_df, "x_to", "y_to", "rhs")
    edge_df["display"] = _display(edge_df)
    edge_df["mx"] = (edge_df["x_from"] + edge_df["x_to"]) / 2
    edge_df["my"] = (edge_df["y_from"] + edge_df["y_to"]) / 2

    cov_df = pe[(pe["op"] == "~~") & (pe["lhs"] != pe["rhs"]) &
                pe["lhs"].isin(latent_vars) & pe["rhs"].isin(latent_vars)].copy()
    if len(cov_df) > 0:
        cov_df = _attach_coords(cov_df, "x_from", "y_from", "lhs")
        cov_df = _attach_coords(cov_df, "x_to", "y_to", "rhs")
        cov_df["display"] = _display(cov_df)
        cov_df["mx"] = (cov_df["x_from"] + cov_df["x_to"]) / 2
        cov_df["my"] = (cov_df["y_from"] + cov_df["y_to"]) / 2

    plot_title = {"std": "Standardised Estimates", "est": "Unstandardised Estimates",
                  "eq": "Parameters with Equality Constraints"}[what]

    p = (ggplot()
         + geom_segment(data=edge_df, mapping=aes(x="x_from", y="y_from", xend="x_to", yend="y_to"),
                        colour="#555a6b", size=0.55, arrow=arrow(length=0.02, type="closed", ends="last"))
         + geom_label(data=edge_df, mapping=aes(x="mx", y="my", label="display"),
                      size=edge_label_size, fill="white", label_size=0, colour="#333745")
         + geom_rect(data=node_df[~node_df["is_lat"]],
                     mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                     fill=color_observed, colour="#8890a8", size=0.4)
         + geom_rect(data=node_df[node_df["is_lat"]],
                     mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                     fill=color_latent, colour="#2a5cc7", size=0.5)
         + geom_text(data=node_df[~node_df["is_lat"]], mapping=aes(x="x", y="y", label="name"),
                     size=label_size, colour="#222533")
         + geom_text(data=node_df[node_df["is_lat"]], mapping=aes(x="x", y="y", label="name"),
                     size=label_size, colour="white", fontweight="bold")
         + coord_fixed()
         + theme_void(base_size=11)
         + theme(plot_title=element_text(ha="center", size=12))
         + labs(title=f"{plot_title} — {layout}"))

    if len(cov_df) > 0:
        arc_frames = []
        for idx, row in enumerate(cov_df.itertuples()):
            xs_arc, ys_arc = _bezier_arc(row.x_from, row.y_from, row.x_to, row.y_to)
            arc_frames.append(pd.DataFrame({"x": xs_arc, "y": ys_arc, "arc_id": idx}))
        arc_df = pd.concat(arc_frames, ignore_index=True)
        p = (p
             + geom_path(data=arc_df, mapping=aes(x="x", y="y", group="arc_id"),
                         colour="#e87c4f", size=0.5, linetype="dashed")
             + geom_label(data=cov_df, mapping=aes(x="mx", y="my", label="display"),
                          size=edge_label_size, fill="#fff5f0", label_size=0, colour="#993c1d"))
    return p
##########################################################################################
# MODEL PLOT (batch)
##########################################################################################
def plot_cfa(model, **kwargs):
    """
    Path diagrams of a CFA/SEM model in every layout x display combination
    (port of rwf::plot_cfa).

    Parameters:
    model (SEMFit): A model fitted with cfa() or sem().
    **kwargs: Passed to plot_cfa_gg().

    Returns:
    dict: {"<layout>_<mode>": plotnine.ggplot}, e.g. "circle_estimates",
    "tree_standard_estimates", "spring_parameters_wih_equality_constraints"
    (spelled as rwf spells it). Combinations that fail are skipped with a
    printed message.

    Examples:
    >>> plot_cfa(fit)
    """
    layouts = ["circle", "tree", "spring"]
    whats = ["est", "std", "eq"]
    what_key = {"est": "estimates", "std": "standard_estimates", "eq": "parameters_wih_equality_constraints"}
    plots = {}
    for lay in layouts:
        for wh in whats:
            key = f"{lay}_{what_key[wh]}"
            try:
                plots[key] = plot_cfa_gg(model, what=wh, layout=lay, **kwargs)
            except Exception as exc:  # noqa: BLE001 - rwf also skips diagrams it cannot draw
                print(f"Skipping {key}: {exc}")
    return plots
##########################################################################################
# MODEL
##########################################################################################
def _summary_text(model, fit_indices):
    """A short text summary of the model, in place of lavaan's summary() print."""
    fit = fit_indices["fit"]
    lines = [
        f"Estimator: ML    Observations: {model.n}    Free parameters: {len(model.free)}",
        f"Chi-square: {fit['chisq']:.3f}    df: {fit['df']:.0f}    p: {fit['pvalue']:.4f}",
        f"CFI: {fit['cfi']:.3f}    TLI: {fit['tli']:.3f}    RMSEA: {fit['rmsea']:.3f} "
        f"[{fit['rmsea.ci.lower']:.3f}, {fit['rmsea.ci.upper']:.3f}]    SRMR: {fit['srmr']:.3f}",
    ]
    return "\n".join(lines)


def report_cfa(model, file=None, w=10, h=10):
    """
    Full report of a fitted CFA/SEM model (port of rwf::report_cfa): R
    squared, fit indices, parameter estimates, modification indices, sample
    covariance, (un)standardized model matrices, factor scores and path
    diagrams, optionally written to <file>_diagram.pdf, <file>.log and
    <file>.xlsx.

    Parameters:
    model (SEMFit): A model fitted with cfa() or sem() (rwf: a lavaan object).
    file (str, optional): Output filename without extension.
    w, h (float, optional): PDF page size in inches. Defaults to 10, 10.

    Returns:
    dict: r_squared, fit_indices (lavaan's fitMeasures), parameters
    (parameterEstimates with standardized values and R squared),
    modification_indices, sample_covariance, unstandardized_estimates and
    standardized_estimates (dicts of model matrices), group (empty: one
    group), predict (data and factor scores) and call, as in rwf.

    Examples:
    >>> fit = cfa("LATENT =~ ITEM1 + ITEM2 + ITEM3 + ITEM4 + ITEM5", df)
    >>> result = report_cfa(fit)
    >>> result = report_cfa(fit, file="cfa")
    """
    _check_model(model)
    r_squared = pd.DataFrame({"r_squared": pd.Series(model.rsquare())})
    fit_indices = pd.DataFrame({"fit": pd.Series(model.fit_measures())})
    parameters = model.parameter_estimates()
    modification_indices = model.modification_indices()
    sample_covariance = pd.DataFrame(model.sample_cov, index=model.observed, columns=model.observed)
    unstandardized_estimates = model.inspect("est")
    standardized_estimates = model.inspect("std")
    group = pd.DataFrame()
    predict = pd.concat([model.data.reset_index(drop=True), model.predict().reset_index(drop=True)], axis=1)
    call_string = f"sem(model='{model.model_syntax}',std_lv={model.std_lv})".replace(" ", "")
    call = pd.DataFrame({"call": [call_string]})
    result = {
        "r_squared": r_squared, "fit_indices": fit_indices, "parameters": parameters,
        "modification_indices": modification_indices, "sample_covariance": sample_covariance,
        "unstandardized_estimates": unstandardized_estimates, "standardized_estimates": standardized_estimates,
        "group": group, "predict": predict, "call": call,
    }

    plots = plot_cfa(model)
    report_pdf(plotlist=list(plots.values()), file=file, title="diagram", w=w, h=h, print_plot=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for section, output in [
            ("SUMMARY", _summary_text(model, fit_indices)), ("R_SQUARED", r_squared),
            ("FIT INDICES", fit_indices), ("PARAMETERS", parameters),
            ("UNSTANDARDIZED PARAMETERS", "\n".join(f"{k}:\n{v}" for k, v in unstandardized_estimates.items())),
            ("STANDARDIZED PARAMETERS", "\n".join(f"{k}:\n{v}" for k, v in standardized_estimates.items())),
            ("SAMPLE COVARIANCE", sample_covariance), ("CALL", call_string),
        ]:
            output_separator(section, output=output)
    write_txt(buffer.getvalue().rstrip("\n"), file=file)

    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_critical_value(r_squared.reset_index(names="variable"), writer, sheet="R_Squared")
            excel_critical_value(fit_indices.reset_index(names="index"), writer, sheet="Fit_Indices")
            excel_critical_value(parameters, writer, sheet="Parameters")
            excel_critical_value(modification_indices, writer, sheet="Modification_Indices")
            excel_critical_value(group, writer, sheet="Groups")
            excel_matrix(sample_covariance, writer, sheet="Sample_Covariance")
            excel_matrix(predict, writer, sheet="Scores")
            excel_critical_value(call, writer, sheet="Call")
    return result
##########################################################################################
# SIMULATE CFA FROM COEFFICIENTS
##########################################################################################
def simulate_cfa_fit(model_sim=None, model=None, df=None, minnobs=50, maxnobs=1000, stepping=10,
                     file=None, w=10, h=10):
    """
    Fit a CFA at a range of sample sizes and collect its fit indices, for
    power and sample size planning (port of rwf::simulate_cfa_fit). Data
    come either from model_sim (lavaan syntax with fixed values, simulated
    with simulate_data) or from df (resampled from its correlations with
    simulate_correlation_from_sample).

    Parameters:
    model_sim (str, optional): Model with fixed values, e.g. "F =~ 1*x1 + 0.8*x2".
    model (str): Model with free parameters, fitted at every sample size.
    df (pandas.DataFrame, optional): Observed data to resample from.
    minnobs, maxnobs, stepping (int, optional): Sample sizes from minnobs to
        maxnobs in steps of stepping. Defaults to 50, 1000, 10.
    file (str, optional): Output filename (without extension) for a PDF and
        an Excel workbook.
    w, h (float, optional): PDF page size in inches. Defaults to 10, 10.

    Returns:
    list: [sim_results (DataFrame, one row per sample size: observations and
    every fit index), plots (dict of scatter plots, one per fit index
    against sample size)].

    Examples:
    >>> model_sim = "LATENT =~ 1*X1 + 0.5*X2 + 1.5*X3 + 1.5*X4 + X5"
    >>> model = "LATENT =~ X1 + X2 + X3 + X4 + X5"
    >>> result = simulate_cfa_fit(model_sim=model_sim, model=model, minnobs=50, maxnobs=300, stepping=100)
    """
    rows = []
    for nobs in range(minnobs, maxnobs + 1, stepping):
        if model_sim is not None:
            sim = simulate_data(model_sim, sample_nobs=nobs, orthogonal=True)
        else:
            sim = simulate_correlation_from_sample(df, nrows=nobs)
        row = {"observations": nobs}
        row.update(cfa(model, sim).fit_measures())
        rows.append(row)
    sim_results = pd.DataFrame(rows)
    plot_data = remove_nc(sim_results, remove_rows=True, aggressive=False, remove_cols=True,
                          remove_zero_variance=True)
    combinations = pd.DataFrame({"x": ["observations"] * len(plot_data.columns), "y": list(plot_data.columns)})
    combinations = combinations[combinations["y"] != "observations"]
    plots = plot_scatterplot(df=plot_data, combinations=combinations)

    if file is not None:
        filename = f"{file}.xlsx"
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            excel_generic_format(sim_results, writer, sheet="simulation")
        report_pdf(plotlist=[p for p in plots.values() if p is not None], w=w, h=h, file=file, print_plot=False)
    return [sim_results, plots]
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 80, "\nplot_cfa_gg / plot_cfa\n", "=" * 80, sep="")
    data = simulate_data("LATENT1 =~ 1*X1 + 0.8*X2 + 0.6*X3; LATENT2 =~ 1*X4 + 0.7*X5 + 0.9*X6",
                         sample_nobs=200, seed=12345)
    fit_multi = cfa("LATENT1 =~ X1 + X2 + X3; LATENT2 =~ X4 + X5 + X6", data)
    print(list(plot_cfa(fit_multi)))

    print("\n" + "=" * 80, "\nreport_cfa\n", "=" * 80, sep="")
    data2 = simulate_data("LATENT =~ 1*ITEM1 + 0.8*ITEM2 + 0.7*ITEM3 + 0.9*ITEM4 + 0.6*ITEM5", sample_nobs=200, seed=1)
    result = report_cfa(cfa("LATENT =~ ITEM1 + ITEM2 + ITEM3 + ITEM4 + ITEM5", data2))
    print(result["parameters"])

    print("\n" + "=" * 80, "\nsimulate_cfa_fit\n", "=" * 80, sep="")
    sim_result = simulate_cfa_fit(model_sim="LATENT =~ 1*X1 + 0.5*X2 + 1.5*X3 + 1.5*X4 + X5",
                                  model="LATENT =~ X1 + X2 + X3 + X4 + X5", minnobs=50, maxnobs=250, stepping=100)
    print(sim_result[0][["observations", "chisq", "cfi", "rmsea"]])
