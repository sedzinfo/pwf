# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_SEM.R.

Every function here operates on a fitted lavaan model, and several go
deep into lavaan's own introspection machinery (lavInspect, S4 slots like
model@Model@ngroups, model@SampleStats@cov, model@Data@X, parameterEstimates,
modificationIndices). There is no native Python SEM package with an
equivalent surface (semopy's API/output structure is entirely different),
so — consistent with mirt/lavaan/thurstonianIRT in glm_irt.py/glm_irt_t.py
and mixlm in glm_reliability.py — this whole file bridges to real R lavaan
via rpy2. Every function takes an rpy2 lavaan model object (built via rpy2
calls to lavaan::cfa()/sem(), typically through ro.r("fit <- lavaan::cfa(...)")).

Deviations from the R original, by design:
  - R's igraph-based node layouts (layout_in_circle, layout_as_tree,
    layout_with_fr) are replaced with networkx (circular_layout,
    graphviz's "dot" layout via nx.nx_pydot.graphviz_layout for "tree",
    spring_layout for "spring"). "tree" needs a system graphviz install
    (already present here); if unavailable, a simple hand-rolled 2-row
    layout (latents on top, their own indicators below) is used instead
    — good enough for a standard CFA model (a 2-level forest), not a
    faithful Reingold-Tilford tree layout for deeper hierarchical SEMs.
  - ggplot2::geom_curve (used for the latent-latent covariance arcs) has
    no plotnine equivalent (plotnine has no curved-line geom); replaced
    with a manually-computed quadratic Bezier arc rendered via geom_path
    — visually similar, not the identical curve-drawing algorithm.
  - Discovered and worked around an rpy2 stability issue: chained S4
    slot access (model.slots["Model"].slots["ngroups"]) on a lavaan
    model object reliably raises "ReferenceError: weakly-referenced
    object no longer exists" after any lavInspect()/parameterEstimates()
    call has run in the same session. Slot access here is instead done
    via a small helper that binds the model to a temporary name in R's
    global environment and evaluates the slot chain as an R expression
    string (e.g. "name@Model@ngroups") — fully server-side, and immune
    to whatever Python-side weak-reference caching causes the crash.
    A useful side effect: unlike glm_irt.py's mirt models (fit via
    direct Python-to-R function calls, which corrupts match.call()
    capture), a lavaan model fit via ro.r("fit <- lavaan::cfa(...)") — a
    plain R source string, not Python-side argument marshaling — has an
    intact model@call, verified directly; report_cfa's call string is
    therefore the real deparsed call, not a synthesized fallback.
  - lavInspect(model, "est"/"std", ...)'s per-model-matrix (lambda,
    theta, psi, ...) results are kept as a dict of pandas DataFrames
    (one per matrix name) rather than reproducing R's exact nested
    list-printing format — used for the same "just display/export it"
    purpose as R's own version, not for further computation either way.
  - simulate_cfa_fit's future/future.apply parallel-iteration branch is
    not replicated (same rationale as every other file in this project:
    plot/model objects don't reliably serialize across process
    boundaries) — this runs the same loop sequentially. It's genuinely
    slower for large sample-size sweeps than R's parallel version.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import networkx as nx
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from plotnine import (
    ggplot, aes, geom_segment, geom_rect, geom_text, geom_label, geom_path,
    coord_fixed, theme_void, theme, labs, arrow, element_text,
)

try:
    from .functions import remove_nc
    from .functions_generate_data import simulate_correlation_from_sample
    from .functions_environment import write_txt
    from .functions_strings import output_separator
    from .functions_plot import report_pdf
    from .glm_linear_regression import plot_scatterplot
except ImportError:
    from functions import remove_nc
    from functions_generate_data import simulate_correlation_from_sample
    from functions_environment import write_txt
    from functions_strings import output_separator
    from functions_plot import report_pdf
    from glm_linear_regression import plot_scatterplot

_base = importr("base")
_lavaan = importr("lavaan")
_MODEL_TMP_NAME = ".pwf_glm_sem_model"
##########################################################################################
# R <-> PANDAS / SLOT ACCESS HELPERS (internal)
##########################################################################################
def _is_r_null(x):
    return bool(_base.is_null(x)[0])


def _r_df_to_pandas(r_df):
    data = {name: list(r_df.rx2(name)) for name in r_df.names}
    return pd.DataFrame(data)


def _r_matrix_to_pandas(r_mat):
    df = pd.DataFrame(np.array(r_mat))
    colnames = _base.colnames(r_mat)
    rownames = _base.rownames(r_mat)
    if not _is_r_null(colnames):
        df.columns = list(colnames)
    if not _is_r_null(rownames):
        df.index = list(rownames)
    return df


def _r_slot(model, expr):
    """
    Evaluate an R slot-access expression (e.g. "@Model@ngroups") against
    `model`, entirely server-side. See module docstring for why chained
    Python-side .slots[...] access is avoided.
    """
    ro.globalenv[_MODEL_TMP_NAME] = model
    return ro.r(f"{_MODEL_TMP_NAME}{expr}")


def _r_call_string(model):
    """
    Deparsed model@call, as a plain Python string. Unlike _r_slot, this
    does the deparse()+paste() server-side in a single R expression:
    passing a fetched call/language object back into a *separate*
    Python-side rpy2 function call (e.g. base.deparse(_r_slot(model,
    "@call"))) reproduces the same match.call()-capture corruption bug
    documented in glm_irt.py, just triggered by deparse() itself rather
    than by how the model was originally fit -- verified directly: doing
    it in one string gives the real short call every time, doing it in
    two steps gives a multi-thousand-character dump of the entire model object.
    """
    ro.globalenv[_MODEL_TMP_NAME] = model
    return ro.r(f'paste(deparse({_MODEL_TMP_NAME}@call), collapse=" ")')[0]
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
##########################################################################################
# MODEL PLOT
##########################################################################################
def plot_cfa_gg(model, what="std", layout="tree", label_size=3.2, edge_label_size=2.6,
                 color_latent="#4f8ef7", color_observed="#e8eaf0", **kwargs):
    """
    Path diagram for a fitted lavaan CFA/SEM model, built directly with
    plotnine (no semPlot dependency).

    Parameters:
    model: An rpy2 fitted lavaan model object.
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
    >>> import rpy2.robjects as ro
    >>> ro.r('''
    ...     model <- "LATENT1=~X1+X2+X3\\nLATENT2=~X4+X5+X6"
    ...     df <- lavaan::simulateData(model=model, model.type="cfa",
    ...                                return.type="data.frame", sample.nobs=100)
    ...     fit <- lavaan::cfa(model, data=df)
    ... ''')
    >>> fit = ro.r["fit"]
    >>> plot_cfa_gg(fit, what="std")
    >>> plot_cfa_gg(fit, what="std", layout="circle")
    """
    if what not in ("std", "est", "eq"):
        raise ValueError('what must be one of "std", "est", "eq"')
    if layout not in ("tree", "circle", "spring"):
        raise ValueError('layout must be one of "tree", "circle", "spring"')

    pe = _r_df_to_pandas(ro.r["parameterEstimates"](model, standardized=True))
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
        merged = merged.rename(columns={"x": xcol, "y": ycol}).drop(columns="name")
        return merged

    loadings = pe[pe["op"] == "=~"].copy()
    edge_df = _attach_coords(loadings, "x_from", "y_from", "lhs")
    edge_df = _attach_coords(edge_df, "x_to", "y_to", "rhs")

    if what == "eq":
        label_col = edge_df["label"] if "label" in edge_df.columns else pd.Series([""] * len(edge_df))
        est_fmt = edge_df["est"].map(lambda v: f"{v:.3f}")
        edge_df["display"] = np.where(label_col.isna() | (label_col == ""), est_fmt, label_col)
    else:
        edge_df["display"] = edge_df[edge_col].map(lambda v: f"{v:.3f}")
    edge_df["mx"] = (edge_df["x_from"] + edge_df["x_to"]) / 2
    edge_df["my"] = (edge_df["y_from"] + edge_df["y_to"]) / 2

    cov_df = pe[(pe["op"] == "~~") & (pe["lhs"] != pe["rhs"]) &
                pe["lhs"].isin(latent_vars) & pe["rhs"].isin(latent_vars)].copy()
    if len(cov_df) > 0:
        cov_df = _attach_coords(cov_df, "x_from", "y_from", "lhs")
        cov_df = _attach_coords(cov_df, "x_to", "y_to", "rhs")
        if what == "eq":
            label_col = cov_df["label"] if "label" in cov_df.columns else pd.Series([""] * len(cov_df))
            est_fmt = cov_df["est"].map(lambda v: f"{v:.3f}")
            cov_df["display"] = np.where(label_col.isna() | (label_col == ""), est_fmt, label_col)
        else:
            cov_df["display"] = cov_df[edge_col].map(lambda v: f"{v:.3f}")
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
    Batch-plot a CFA/SEM model across all layout x display-mode combinations.

    Parameters:
    model: An rpy2 fitted lavaan model object.
    **kwargs: Forwarded to plot_cfa_gg().

    Returns:
    dict: {"<layout>_<mode>": plotnine.ggplot}, e.g. "circle_estimates",
    "tree_standard_estimates", "spring_parameters_wih_equality_constraints"
    (typo kept for parity with R's own key name). Combinations that raise
    an error are silently dropped, with a printed message.

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
            except Exception as exc:
                print(f"Skipping {key}: {exc}")
    return plots
##########################################################################################
# MODEL
##########################################################################################
def report_cfa(model, file=None, w=10, h=10):
    """
    Full report for a fitted lavaan CFA/SEM model: R^2, fit indices,
    (un)standardized estimates, parameter table, modification indices,
    sample covariance, factor scores, path diagrams, and (optionally) a
    text log and Excel export.

    Parameters:
    model: An rpy2 fitted lavaan model object.
    file (str, optional): Output filename (without extension). When
        given, writes "<file>_diagram.pdf", "<file>.log", "<file>.xlsx".
    w, h (float, optional): PDF page size in inches. Defaults to 10, 10.

    Returns:
    dict: r_squared, fit_indices, parameters, modification_indices,
    sample_covariance, unstandardized_estimates, standardized_estimates
    (both dicts of per-matrix DataFrames), group, predict, call.

    Examples:
    >>> import rpy2.robjects as ro
    >>> ro.r('''
    ...     model <- "LATENT=~ITEM1+ITEM2+ITEM3+ITEM4+ITEM5"
    ...     df <- lavaan::simulateData(model=model, model.type="cfa",
    ...                                return.type="data.frame", sample.nobs=100)
    ...     fit <- lavaan::cfa(model, data=df, missing="ML")
    ... ''')
    >>> fit = ro.r["fit"]
    >>> result = report_cfa(fit)
    >>> result = report_cfa(fit, file="cfa")
    """
    r_squared = pd.DataFrame({"r_squared": np.asarray(_lavaan.lavInspect(model, "rsquare"))},
                              index=list(_lavaan.lavInspect(model, "rsquare").names))
    fit_r = _lavaan.lavInspect(model, "fit")
    fit_indices = pd.DataFrame({"fit": np.asarray(fit_r)}, index=list(fit_r.names))

    def _named_list_to_dict(r_list):
        out = {}
        for name in r_list.names:
            element = r_list.rx2(name)
            out[name] = _r_matrix_to_pandas(element) if hasattr(element, "dim") and not _is_r_null(element.dim) \
                else pd.Series(np.asarray(element))
        return out

    unstandardized_estimates = _named_list_to_dict(_lavaan.lavInspect(model, "est"))
    standardized_estimates = _named_list_to_dict(_lavaan.lavInspect(model, "std"))

    ngroups = int(_r_slot(model, "@Model@ngroups")[0])
    group = pd.DataFrame()
    if ngroups > 1:
        group = pd.DataFrame({
            "GROUP_COLUMN": list(_lavaan.lavInspect(model, "group")),
            "GROUPS": list(_lavaan.lavInspect(model, "group.label")),
        })
        group["NGROUPS"] = ngroups
        group["OBSERVATIONS"] = list(_lavaan.lavInspect(model, "nobs"))
        group["ORIGINAL_OBSERVATIONS"] = list(_lavaan.lavInspect(model, "norig"))
        group["TOTAL"] = int(_lavaan.lavInspect(model, "ntotal")[0])

    pe_fn = ro.r["parameterEstimates"]
    parameters = _r_df_to_pandas(pe_fn(
        model, se=True, zstat=True, pvalue=True, ci=True, level=0.95, **{"boot.ci.type": "perc"},
        standardized=True, fmi=False,
        **{"remove.system.eq": True, "remove.eq": False, "remove.ineq": False, "remove.def": False},
        rsquare=True, **{"add.attributes": True}))

    mi_fn = ro.r["modificationIndices"]
    modification_indices = _r_df_to_pandas(mi_fn(
        model, standardized=True, **{"cov.std": True}, information="expected", power=True, delta=0.1, alpha=0.05,
        **{"high.power": 0.75, "sort.": True, "minimum.value": 0, "free.remove": False, "na.remove": True},
        op=ro.NULL))

    cov_list = _r_slot(model, "@SampleStats@cov")
    sample_covariance = _r_matrix_to_pandas(cov_list[0])

    pred = _lavaan.predict(model)
    x_list = _r_slot(model, "@Data@X")
    if len(_base.dim(pred)) if not _is_r_null(_base.dim(pred)) else 0:
        predict_df = pd.concat([pd.DataFrame(np.asarray(x_list[0])),
                                 pd.DataFrame(np.asarray(pred))], axis=1)
    else:
        predict_df = pd.concat([pd.DataFrame(np.asarray(x_list[0])),
                                 pd.DataFrame(np.asarray(pred[0]))], axis=1)

    call_str = _r_call_string(model).replace(" ", "")
    call = pd.DataFrame({"call": [call_str]})

    result = {
        "r_squared": r_squared, "fit_indices": fit_indices, "parameters": parameters,
        "modification_indices": modification_indices, "sample_covariance": sample_covariance,
        "unstandardized_estimates": unstandardized_estimates, "standardized_estimates": standardized_estimates,
        "group": group, "predict": predict_df, "call": call,
    }

    plots = plot_cfa(model)
    report_pdf(plotlist=list(plots.values()), file=file, title="diagram", w=w, h=h, print_plot=(file is None))

    summary_str = "".join(ro.r["capture.output"](ro.r["summary"](
        model, standardized=True, **{"fit.measures": True, "rsquare": True})))
    log_lines = []
    for section, output, instruction in [
        ("SUMMARY", summary_str, None), ("R_SQUARED", r_squared, None), ("FIT INDICES", fit_indices, None),
        ("PARAMETERS", parameters, None),
        ("UNSTANDARDIZED PARAMETERS", "\n".join(f"{k}:\n{v}" for k, v in unstandardized_estimates.items()), None),
        ("STANDARDIZED PARAMETERS", "\n".join(f"{k}:\n{v}" for k, v in standardized_estimates.items()), None),
        ("SAMPLE COVARIANCE", sample_covariance, None), ("CALL", call_str, None),
    ]:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            output_separator(section, output=output, instruction=instruction)
        log_lines.append(buf.getvalue())
    write_txt("".join(log_lines), file=file)

    if file is not None:
        try:
            from .functions_excel import excel_critical_value, excel_matrix
        except ImportError:
            from functions_excel import excel_critical_value, excel_matrix
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_critical_value(r_squared.reset_index(names="variable"), writer, sheetname="R_Squared")
        excel_critical_value(fit_indices.reset_index(names="index"), writer, sheetname="Fit_Indices")
        excel_critical_value(parameters, writer, sheetname="Parameters")
        excel_critical_value(modification_indices, writer, sheetname="Modification_Indices")
        if len(group) > 0:
            excel_critical_value(group, writer, sheetname="Groups")
        excel_matrix(sample_covariance, writer, sheetname="Sample_Covariance")
        excel_matrix(predict_df, writer, sheetname="Scores")
        excel_critical_value(call, writer, sheetname="Call")
        writer._save()
        writer.close()

    return result
##########################################################################################
# SIMULATE CFA FROM COEFFICIENTS
##########################################################################################
def simulate_cfa_fit(model_sim=None, model=None, df=None, minnobs=50, maxnobs=1000, stepping=10,
                      file=None, w=10, h=10):
    """
    Fit a CFA repeatedly across a range of sample sizes and collect fit
    indices for each — useful for power/sample-size planning in SEM. Two
    mutually exclusive workflows: `model_sim` (lavaan model string with
    fixed coefficients, used to simulate population data at each sample
    size) or `df` (observed data, resampled from its correlation
    structure at each sample size via simulate_correlation_from_sample).

    Parameters:
    model_sim (str, optional): lavaan model string with fixed
        coefficients, e.g. "F =~ 1*x1 + 0.8*x2".
    model (str): lavaan model string with free coefficients, fit at
        every sample size.
    df (pandas.DataFrame, optional): Observed data to resample from.
    minnobs, maxnobs, stepping (int, optional): Sample-size sweep range.
        Defaults to 50, 1000, 10.
    file (str, optional): Output filename (without extension) for a PDF
        and Excel export. Defaults to None.
    w, h (float, optional): PDF page size in inches. Defaults to 10, 10.

    Returns:
    list: [sim_results (pandas.DataFrame, one row per sample size), plots
    (dict of plotnine.ggplot, one per fit index vs. sample size)].

    Examples:
    >>> import rpy2.robjects as ro
    >>> model_sim = "LATENT =~ 1*X1 + 0.5*X2 + 1.5*X3 + 1.5*X4 + X5"
    >>> model = "LATENT =~ X1 + X2 + X3 + X4 + X5"
    >>> result = simulate_cfa_fit(model_sim=model_sim, model=model,
    ...                           minnobs=50, maxnobs=300, stepping=100)
    """
    sequence = range(minnobs, maxnobs + 1, stepping)
    rows = []
    for nobs in sequence:
        if model_sim is not None:
            sim_r = _lavaan.simulateData(model_sim, **{"model.type": "cfa"}, **{"return.type": "data.frame"},
                                          **{"sample.nobs": nobs}, orthogonal=True)
            sim = _r_df_to_pandas(sim_r)
        else:
            sim = simulate_correlation_from_sample(df, nrows=nobs)

        with localconverter(ro.default_converter + pandas2ri.converter):
            sim_r_conv = ro.conversion.py2rpy(sim)
        fit_r = _lavaan.cfa(model, data=sim_r_conv)
        fit_indices = _lavaan.lavInspect(fit_r, "fit")
        row = {"observations": nobs}
        row.update(dict(zip(list(fit_indices.names), np.asarray(fit_indices))))
        rows.append(row)

    sim_results = pd.DataFrame(rows)
    plot_data = remove_nc(sim_results, remove_rows=True, aggressive=False, remove_cols=True,
                           remove_zero_variance=True)
    combinations = pd.DataFrame({"x": ["observations"] * len(plot_data.columns), "y": list(plot_data.columns)})
    combinations = combinations[combinations["y"] != "observations"]
    plots = plot_scatterplot(df=plot_data, combinations=combinations)

    if file is not None:
        try:
            from .functions_excel import excel_generic_format
        except ImportError:
            from functions_excel import excel_generic_format
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_generic_format(sim_results, writer, sheetname="simulation")
        writer._save()
        writer.close()
    # plot_scatterplot returns seaborn JointGrid objects (see glm_linear_regression.py's
    # own documented deviation) -- report_pdf expects a plotnine ggplot or matplotlib
    # Figure, so the underlying Figure is extracted here for the export step only.
    figures = [p.figure for p in plots.values() if p is not None]
    report_pdf(plotlist=figures, w=w, h=h, file=file, print_plot=(file is None))

    return [sim_results, plots]
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import os

    print("=" * 80, "\nplot_cfa_gg / plot_cfa\n", "=" * 80, sep="")
    ro.r("""
        set.seed(12345)
        model <- "LATENT1=~X1+X2+X3\nLATENT2=~X4+X5+X6"
        df <- lavaan::simulateData(model=model, model.type="cfa",
                                    return.type="data.frame", sample.nobs=200)
        fit_multi <- lavaan::cfa(model, data=df)
    """)
    fit_multi = ro.r["fit_multi"]

    for lay in ["tree", "circle", "spring"]:
        p = plot_cfa_gg(fit_multi, what="std", layout=lay)
        p.save(f"plot_cfa_gg_{lay}.png", verbose=False, width=8, height=6)
        print(f"saved plot_cfa_gg_{lay}.png")

    all_plots = plot_cfa(fit_multi)
    print("plot_cfa keys:", list(all_plots.keys()))

    print("\n" + "=" * 80, "\nreport_cfa\n", "=" * 80, sep="")
    ro.r("""
        set.seed(12345)
        model2 <- "LATENT=~ITEM1+ITEM2+ITEM3+ITEM4+ITEM5"
        df2 <- lavaan::simulateData(model=model2, model.type="cfa",
                                     return.type="data.frame", sample.nobs=200)
        fit <- lavaan::cfa(model2, data=df2, missing="ML")
    """)
    fit = ro.r["fit"]
    result = report_cfa(fit, file="cfa_report")
    print("r_squared:\n", result["r_squared"])
    print("\nfit_indices head:\n", result["fit_indices"].head())
    print("\nparameters:\n", result["parameters"])
    print("\nmodification_indices head:\n", result["modification_indices"].head())
    print("\nsample_covariance:\n", result["sample_covariance"])
    print("\npredict head:\n", result["predict"].head())
    print("\ncall:", result["call"]["call"].iloc[0])
    print("\nExcel/log/pdf written:", os.path.exists("cfa_report.xlsx"), os.path.exists("cfa_report.log"),
          os.path.exists("cfa_report_diagram.pdf"))

    print("\n" + "=" * 80, "\nsimulate_cfa_fit (coefficient-based)\n", "=" * 80, sep="")
    model_sim = "LATENT =~ 1*X1 + 0.5*X2 + 1.5*X3 + 1.5*X4 + X5"
    model_free = "LATENT =~ X1 + X2 + X3 + X4 + X5"
    sim_result = simulate_cfa_fit(model_sim=model_sim, model=model_free, minnobs=50, maxnobs=250, stepping=100)
    print("sim_results:\n", sim_result[0])
    print("plot keys:", list(sim_result[1].keys()))
