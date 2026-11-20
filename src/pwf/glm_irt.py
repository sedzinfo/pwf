# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_IRT.R.

There is no native Python package with anything close to mirt's feature
set (multidimensional IRT, empirical-histogram latent trait estimation,
the M2 statistic, Q3 local-independence residuals, TLI/CFI/RMSEA/SRMR fit
indices, oblimin-rotated multidimensional coefficients). The closest pure
-Python packages (girth, py-irt) only estimate basic unidimensional item
parameters and provide none of the above. Rather than reimplement a large
and necessarily-incomplete slice of mirt from scratch, this module bridges
to the real R mirt package via rpy2 (both R and mirt are present in this
environment, and rpy2 was already part of this project's "full" extras) —
giving exact 1:1 output parity with R instead of an approximation.

fit_irt() is a *new* convenience wrapper, not part of the R source: R
users call mirt::mirt(...) inline, which has no equivalent Python call
syntax. It wraps the input DataFrame -> R data.frame conversion (see
below) and returns the raw rpy2 RS4 mirt model object, so `model` in
plot_irt_onefactor()/report_irt() is exactly the same kind of object R's
own docstring examples build.

Deviations/quirks, by design:
  - Every rpy2 call in this module deliberately avoids nesting R-to-R
    calls inside a `localconverter(pandas2ri)` context. Only the single
    Python-DataFrame -> R-data.frame conversion in fit_irt() needs that
    context; anywhere else, having it active makes rpy2 silently
    auto-convert R vectors/matrices to numpy arrays the moment they're
    extracted (e.g. via .rx2()), which then breaks the *next* R call that
    receives that numpy array as an argument (rpy2's default converter
    doesn't know how to convert numpy arrays back to R). All other R ->
    Python conversions here are done manually (_r_matrix_to_pandas /
    _r_df_to_pandas) precisely to sidestep this.
  - model_call (an informational field in report_irt's output) can't
    reliably reproduce R's `deparse(model@Call)`: rpy2 invokes R
    functions in a way that breaks match.call()/sys.call() capture inside
    the callee, so model@Call comes back as a multi-thousand-line dump of
    mirt's entire internal function body instead of the short call the
    user actually made. This is a general rpy2 limitation, not specific
    to mirt. _model_call_string() detects this (a suspiciously long
    deparse, or one containing "SingleGroupClass") and falls back to a
    short synthesized description (item count, factor count) instead.
  - plot_irt_onefactor preserves an R quirk verbatim: per-item testinfo
    is selected via `which.items=grep(item_name, all_item_names)`, i.e.
    substring matching, not exact matching. If one item's name is a
    substring of another's (e.g. items named "V1" and "V10"), that
    item's "per-item" information curve silently includes the other
    item's contribution too. Reproduced here with R's own grep() for
    parity, not fixed. Its series keep rwf's factor levels (information,
    expected_score, then the items in column order), which set the
    colours and the legend order.
  - report_irt's model_coefficients_oblimin replicates a specific,
    concrete R mechanism (rwf::flatten_list applied to
    coef(model, rotate="oblimin", simplify=TRUE)'s items/means/cov list)
    rather than a generic list-flattener: rows from the "items" matrix,
    "means" vector, and "cov" matrix are stacked with an ".id" label
    column and NaN-filled mismatched columns, then the first N row
    labels (N = item count) are overwritten with the item names — exactly
    matching R's `row.names(...)[1:N] <- row.names(model_coefficients)`
    step, including that any additional (means/cov) rows keep plain
    sequential-integer-as-string labels.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_point, theme, labs, facet_wrap

try:  # plotnine's theme_bw with ggplot2's left-aligned title
    from ._rcompat import theme_bw
except ImportError:
    from _rcompat import theme_bw

try:
    from .functions import remove_nc
    from .functions_matrix import matrix_triangle
except ImportError:
    from functions import remove_nc
    from functions_matrix import matrix_triangle

# These functions still run R's mirt through rpy2. The import is optional, so the module (and its
# documentation) loads without R; the functions say what is missing when they are called.
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    _base = importr("base")
    _mirt = importr("mirt")
    _R_ERROR = None
except Exception as _error:  # noqa: BLE001 - rpy2, R or mirt missing
    ro = importr = pandas2ri = localconverter = _base = _mirt = None
    _R_ERROR = _error


def _require_r():
    if _R_ERROR is not None:
        raise ImportError("this function needs R with the mirt package and the rpy2 Python package "
                          f"(pip install rpy2): {_R_ERROR}")
##########################################################################################
# R <-> PANDAS CONVERSION HELPERS (internal)
##########################################################################################
def _is_r_null(x):
    return bool(_base.is_null(x)[0])


def _r_matrix_to_pandas(r_mat):
    """R matrix -> pandas.DataFrame, preserving row/column names if present."""
    df = pd.DataFrame(np.array(r_mat))
    colnames = _base.colnames(r_mat)
    rownames = _base.rownames(r_mat)
    if not _is_r_null(colnames):
        df.columns = list(colnames)
    if not _is_r_null(rownames):
        df.index = list(rownames)
    return df


def _r_df_to_pandas(r_df):
    """R data.frame -> pandas.DataFrame, column by column (columns may be mixed type)."""
    data = {name: list(r_df.rx2(name)) for name in r_df.names}
    df = pd.DataFrame(data)
    rownames = _base.rownames(r_df)
    if not _is_r_null(rownames):
        df.index = list(rownames)
    return df


def _model_call_string(model):
    """
    Best-effort short description of the model's specification. See the
    module docstring: deparse(model@Call) is unusable via rpy2 (it dumps
    mirt's entire internal source instead of the user's actual call), so
    this falls back to a synthesized summary built from the model itself
    whenever that corruption is detected.
    """
    deparsed = "".join(_base.deparse(model.slots["Call"]))
    if len(deparsed) > 500 or "SingleGroupClass" in deparsed:
        nitems = int(_mirt.extract_mirt(model, "nitems")[0])
        nfact = int(_mirt.extract_mirt(model, "nfact")[0])
        itemtype = ",".join(sorted(set(_mirt.extract_mirt(model, "itemtype"))))
        return f"mirt(nitems={nitems},nfact={nfact},itemtype={itemtype}) [Call unavailable via rpy2]"
    return deparsed.replace(" ", "")
##########################################################################################
# FIT IRT MODEL (added convenience wrapper, not from R source)
##########################################################################################
def fit_irt(data, n_factors=1, itemtype=None, **kwargs):
    """
    Fit an IRT model via R's mirt::mirt(), returning the raw rpy2 model
    object for use by plot_irt_onefactor()/report_irt(). Not part of the
    R source — see module docstring for why this wrapper exists.

    Parameters:
    data (pandas.DataFrame): Item response data (one column per item).
    n_factors (int, optional): Number of latent factors. Defaults to 1.
    itemtype (str or sequence of str, optional): mirt itemtype(s), e.g.
        "graded", "Rasch", "2PL", "3PL". If None (default), mirt infers
        it from the data.
    **kwargs: Forwarded to mirt::mirt() verbatim (e.g. empiricalhist=True,
        calcNull=True, verbose=False).

    Returns:
    An rpy2 RS4 mirt model object.

    Examples:
    >>> import rpy2.robjects as ro
    >>> from rpy2.robjects.packages import importr
    >>> psych = importr("psych")
    >>> ro.r("set.seed(12345)")
    >>> sim = psych.sim_rasch(nvar=5, n=2000, low=-4, high=4, d=ro.NULL, a=1, mu=0, sd=1)
    >>> import numpy as np, pandas as pd
    >>> items = pd.DataFrame(np.array(sim.rx2("items")), columns=[f"V{i+1}" for i in range(5)])
    >>> model = fit_irt(items, n_factors=1, empiricalhist=True, calcNull=True, verbose=False)
    """
    _require_r()
    with localconverter(ro.default_converter + pandas2ri.converter) as converter:
        r_data = converter.py2rpy(pd.DataFrame(data))  # rpy2 3.6 deprecates ro.conversion.py2rpy
    call_kwargs = dict(kwargs)
    if itemtype is not None:
        call_kwargs["itemtype"] = itemtype
    return _mirt.mirt(r_data, n_factors, **call_kwargs)
##########################################################################################
# PLOT MODEL
##########################################################################################
def plot_irt_onefactor(model, theta=None, title="", base_size=10):
    """
    Item and total-test information/expected-score curves across theta.

    Parameters:
    model: An rpy2 mirt model object (from fit_irt() or built directly
        via rpy2 calls to mirt::mirt()).
    theta (array-like, optional): Theta grid. Defaults to
        numpy.arange(-6, 6.1, 0.1) (matching R's seq(-6,6,.1)).
    title (str, optional): Appended to the plot title. Defaults to "".
    base_size (int, optional): Base font size. Defaults to 10.

    Returns:
    plotnine.ggplot: Faceted (by "type": total / Item Information /
    Expected Score) line+point plot of value vs. theta, one series per
    item plus a "total" series.

    Examples:
    >>> model = fit_irt(df_items, n_factors=1, empiricalhist=True, calcNull=True, verbose=False)
    >>> plot_irt_onefactor(model=model, base_size=10, title="Normal Test")
    """
    _require_r()
    if theta is None:  # R's seq(-6, 6, .1): from + (0:n) * by, capped at to
        theta = np.minimum(-6 + np.arange(121) * 0.1, 6)
    theta = np.asarray(theta, dtype=float)
    theta_r = ro.FloatVector(theta)

    # rwf: names(data.frame(model@Data$data)), and each item's information over grep(name, names) (a regular
    # expression, so "V1" also selects V10, V11, ...)
    names_r = ro.r("function(model) names(data.frame(model@Data$data))")(model)
    item_names = list(names_r)
    n_items = len(item_names)
    all_items_r = ro.IntVector(range(1, n_items + 1))

    info_item = {"theta": theta}
    for name in item_names:
        which = _base.grep(name, names_r)
        info_item[name] = np.array(_mirt.testinfo(model, Theta=theta_r, **{"which.items": which}))
    info_item_df = pd.DataFrame(info_item)
    info_item_df.insert(1, "type", "Item Information")

    exp_mat = np.array(_mirt.expected_test(model, Theta=_base.matrix(theta_r), mins=True, individual=True,
                                            **{"which.items": all_items_r}))
    exp_mat = exp_mat.reshape((len(theta), n_items), order="F")
    expected_item_df = pd.DataFrame(exp_mat, columns=item_names)
    expected_item_df.insert(0, "theta", theta)
    expected_item_df.insert(1, "type", "Expected Score")

    info_total = np.array(_mirt.testinfo(model, Theta=theta_r, **{"which.items": all_items_r}))
    exp_total = np.array(_mirt.expected_test(model, Theta=_base.matrix(theta_r), mins=True, individual=False))
    df_total = pd.DataFrame({"theta": theta, "information": info_total, "expected_score": exp_total})

    df_total_melt = df_total.melt(id_vars="theta", var_name="variable", value_name="value")
    df_total_melt.insert(0, "type", "total")
    info_item_melt = info_item_df.melt(id_vars=["theta", "type"], var_name="variable", value_name="value")
    expected_item_melt = expected_item_df.melt(id_vars=["theta", "type"], var_name="variable", value_name="value")
    df_result = pd.concat([df_total_melt, info_item_melt, expected_item_melt], ignore_index=True, sort=False)
    # melt() makes variable a factor and plyr::rbind.fill joins the levels in order of appearance
    df_result["variable"] = pd.Categorical(df_result["variable"], categories=list(dict.fromkeys(df_result["variable"])))

    return (ggplot(df_result, aes(x="theta", y="value", group="variable", color="variable"))
            + geom_line()
            + geom_point()
            + theme_bw(base_size=base_size)
            + theme(legend_position="bottom")
            + labs(title=f"Total Score / Information {title}", y="", x=r"$\theta$")
            + facet_wrap("~type", scales="free"))
##########################################################################################
# FLATTEN OBLIMIN COEFFICIENT LIST (internal, mirrors rwf::flatten_list usage in report_irt)
##########################################################################################
def _flatten_coef_list(r_list, item_names):
    parts = []
    for name in r_list.names:
        element = r_list.rx2(name)
        d = _r_matrix_to_pandas(element) if hasattr(element, "dim") and not _is_r_null(element.dim) \
            else pd.DataFrame({"x": np.array(element)})
        d.insert(0, ".id", name)
        parts.append(d.reset_index(drop=True))
    result = pd.concat(parts, ignore_index=True, sort=False)
    n = len(item_names)
    result.index = list(item_names) + [str(i) for i in range(n + 1, len(result) + 1)]
    return result
##########################################################################################
# REPORT
##########################################################################################
def report_irt(model, m2=True, file=None):
    """
    IRT model report: item coefficients (unrotated and oblimin-rotated),
    Q3 local-independence residual matrix, expected residuals, item fit
    (S-X2), overall G2 fit statistics, and (optionally) the M2 statistic.

    Parameters:
    model: An rpy2 mirt model object (from fit_irt() or built directly
        via rpy2 calls to mirt::mirt()).
    m2 (bool, optional): If True (default), also compute M2 (Maydeu-
        Olivares & Joe, 2006) — can be slow/fail to converge for larger
        models, in which case m2_fit is None.
    file (str, optional): Output filename (without extension) for an
        Excel report. If None (default), no file is written.

    Returns:
    dict: model_coefficients, model_coefficients_oblimin, model_options,
    model_call, q3_matrix, exp_residuals, item_fit, g2_fit, m2_fit.

    Examples:
    >>> model = fit_irt(df_items, n_factors=1, empiricalhist=True, calcNull=True, verbose=False)
    >>> report_irt(model=model, file=None)
    """
    _require_r()
    comment = {
        "a1": "discrimination", "d": "difficulty", "g": "guessing", "u": "inattentiveness",
        "G2": "PARSCALE's G^2", "TLI": "Tucker Lewis Index TLI>0.95", "CFI": "Comparative Fit Index CFI>0.95",
        "RMSEA": "Root Mean Square Error of Approximation RMSEA<0.07", "df": "degrees of freedom",
        "AIC": "Akaike Information Criterion",
        "AICc": "small-sample-size adjusted Akaike Information Criterion",
        "BIC": "Bayesian Information Criterion", "SABIC": "sample-size adjusted Bayesian Information Criterion",
        "DIC": "Deviance Information Criterion",
        "SRMR(SRMSR)": "Standardized Root Mean Square Residual SRMR(SRMSR)<0.08",
    }

    coef_fn = ro.r["coef"]
    coef_none = coef_fn(model, CI=0.95, printSE=False, verbose=False, rotate="none",
                         **{"as.data.frame": False}, simplify=True, unique=False)
    model_coefficients = _r_matrix_to_pandas(coef_none.rx2("items"))
    item_names = model_coefficients.index.tolist()

    coef_oblimin = coef_fn(model, CI=0.95, printSE=False, verbose=False, rotate="oblimin",
                            **{"as.data.frame": False}, simplify=True, unique=False)
    model_coefficients_oblimin = _flatten_coef_list(coef_oblimin, item_names)

    q3_raw = ro.r["residuals"](model, digits=3, type="Q3", QMC=True)
    q3_df = _r_matrix_to_pandas(q3_raw)
    q3_arr = matrix_triangle(q3_df.to_numpy(), off_diagonal=np.nan, diagonal=np.nan, type="lower")
    q3_matrix = pd.DataFrame(q3_arr, index=q3_df.index, columns=q3_df.columns)
    q3_matrix["min"] = remove_nc(q3_matrix.min(axis=1, skipna=True), value=np.nan)
    q3_matrix["max"] = remove_nc(q3_matrix.max(axis=1, skipna=True), value=np.nan)
    # rwf's rbind(q3, min = apply(q3, 2, min, na.rm = TRUE), max = ...): both rows from the matrix before either
    # is added; R's min of no values is Inf and its max -Inf (the last item's column of the lower triangle)
    column_min = q3_matrix.apply(lambda column: column.min() if column.notna().any() else np.inf)
    column_max = q3_matrix.apply(lambda column: column.max() if column.notna().any() else -np.inf)
    q3_matrix.loc["min"] = column_min
    q3_matrix.loc["max"] = column_max

    m2_fit = None
    if m2:
        try:
            m2_r = ro.r["M2"](model, type="M2*", calcNull=True, **{"na.rm": True}, quadpts=ro.NULL,
                               theta_lim=ro.FloatVector([-6, 6]), CI=0.9, residmat=False, QMC=True)
            m2_fit = _r_df_to_pandas(m2_r)
        except Exception:
            m2_fit = None

    exp_residuals = _r_df_to_pandas(ro.r["residuals"](model, digits=3, type="exp", QMC=True))
    item_fit = _r_df_to_pandas(_mirt.itemfit(model, **{"na.rm": True}))

    fit_df_r = _base.data_frame(model.slots["Fit"], **{"check.names": False})
    g2_fit = remove_nc(_r_df_to_pandas(fit_df_r))

    opts_unlisted = ro.r["unlist"](model.slots["Options"])
    model_options = pd.DataFrame({"Options": list(opts_unlisted)}, index=list(opts_unlisted.names))

    # rwf passes model@Call to call_to_string(), which expects a model: both of its lookups give NULL, so
    # rwf's model_call is the text "NULL" (an rwf bug, kept for parity; _model_call_string has the details)
    model_call = "NULL"

    result = {
        "model_coefficients": model_coefficients,
        "model_coefficients_oblimin": model_coefficients_oblimin,
        "model_options": model_options,
        "model_call": model_call,
        "q3_matrix": q3_matrix,
        "exp_residuals": exp_residuals,
        "item_fit": item_fit,
        "g2_fit": g2_fit,
        "m2_fit": m2_fit,
    }

    if file is not None:
        try:
            from .functions_excel import excel_critical_value, excel_matrix
        except ImportError:
            from functions_excel import excel_critical_value, excel_matrix
        with pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter") as writer:  # saved once, when it closes
            # rwf's sheets: the item and option names are row names
            excel_critical_value(model_coefficients, writer, sheet="Coefficients", comment=comment)
            excel_critical_value(model_coefficients_oblimin, writer, sheet="Coefficients Oblimin", comment=comment)
            excel_critical_value(item_fit, writer, sheet="item fit", comment=comment)
            excel_critical_value(g2_fit, writer, sheet="G2", comment=comment)
            if isinstance(m2_fit, pd.DataFrame):
                excel_critical_value(m2_fit, writer, sheet="M2", comment=comment,
                                     title="M2 (Maydeu-Olivares & Joe, 2006) statistic\n when all data are dichotomous")
            excel_matrix(q3_matrix, writer, sheet="Q3", conditional_formatting=True,
                         title="Test for local independence:\nThere is no consensus about the critical values.\n"
                               "Critical values may range from absolute of .7 to .1")
            excel_critical_value(exp_residuals, writer, sheet="Residuals")
            excel_critical_value(model_options, writer, sheet="Model Options")

    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from rpy2.robjects.packages import importr as _importr
    _psych = _importr("psych")

    def _sim_rasch_df(nvar=5, n=2000, low=-4, high=4, a=1):
        ro.r("set.seed(12345)")
        sim = _psych.sim_rasch(nvar=nvar, n=n, low=low, high=high, d=ro.NULL, a=a, mu=0, sd=1)
        arr = np.array(sim.rx2("items"))
        return pd.DataFrame(arr, columns=[f"V{i + 1}" for i in range(nvar)])

    print("=" * 80, "\nfit_irt + plot_irt_onefactor\n", "=" * 80, sep="")
    df_normal = _sim_rasch_df(low=-4, high=4)
    model_normal = fit_irt(df_normal, n_factors=1, empiricalhist=True, calcNull=True, verbose=False)
    p = plot_irt_onefactor(model=model_normal, base_size=10, title="Normal Test")
    p.save("plot_irt_onefactor_normal.png", verbose=False, width=10, height=6)
    print("saved plot_irt_onefactor_normal.png")

    df_easy = _sim_rasch_df(low=-6, high=-4)
    model_easy = fit_irt(df_easy, n_factors=1, empiricalhist=True, calcNull=True, verbose=False)
    plot_irt_onefactor(model=model_easy, base_size=10, title="Easy Items").save(
        "plot_irt_onefactor_easy.png", verbose=False, width=10, height=6)
    print("saved plot_irt_onefactor_easy.png")

    ro.r("set.seed(12345)")
    sim_poly = _psych.sim_poly(nvar=5, n=2000, low=-4, high=4, a=1, c=0, z=1, d=ro.NULL,
                                mu=0, sd=1, cat=5, mod="logistic", theta=ro.NULL)
    df_poly = pd.DataFrame(np.array(sim_poly.rx2("items")), columns=[f"V{i + 1}" for i in range(5)])
    model_graded = fit_irt(df_poly, n_factors=1, itemtype="graded", verbose=False)
    plot_irt_onefactor(model=model_graded, base_size=10, title="Graded Response").save(
        "plot_irt_onefactor_graded.png", verbose=False, width=10, height=6)
    print("saved plot_irt_onefactor_graded.png")

    print("\n" + "=" * 80, "\nreport_irt\n", "=" * 80, sep="")
    result = report_irt(model=model_normal, file=None)
    print("keys:", list(result.keys()))
    print("model_coefficients:\n", result["model_coefficients"])
    print("\nmodel_coefficients_oblimin:\n", result["model_coefficients_oblimin"])
    print("\nq3_matrix:\n", result["q3_matrix"])
    print("\nitem_fit:\n", result["item_fit"])
    print("\ng2_fit:\n", result["g2_fit"])
    print("\nm2_fit:\n", result["m2_fit"])
    print("\nmodel_call:", result["model_call"])

    print("\n" + "=" * 80, "\nreport_irt (2-factor, Excel export)\n", "=" * 80, sep="")
    model_2f = fit_irt(df_normal, n_factors=2, empiricalhist=True, calcNull=True, verbose=False)
    result2 = report_irt(model=model_2f, file="irt_two_factor_report")
    print("model_coefficients_oblimin (2-factor):\n", result2["model_coefficients_oblimin"])
    import os
    print("Excel written:", os.path.exists("irt_two_factor_report.xlsx"))
