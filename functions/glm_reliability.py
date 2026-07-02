# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_RELIABILITY.R.

Deviations from the R original, by design:
  - report_alpha reimplements psych::alpha()'s statistics directly from
    its R source (not guessed from the docs): raw/standardized alpha,
    Guttman's G6 (via the squared-multiple-correlation identity
    smc = 1 - 1/diag(inv(R))), average/median/variance of inter-item r,
    signal-to-noise ratio, the Feldt-style alpha standard error ("ase"),
    item-total correlations (raw_r, std_r, r_cor, r_drop), and the
    alpha-if-item-removed table. Every one of these was verified to match
    real psych::alpha() output exactly (to full float precision) on a
    shared dataset generated in R and loaded into both languages.
    psych::responseFrequency()'s per-item response-category proportions
    are approximated (_response_frequency) rather than exactly replicated
    column-for-column, and psych's "Unidim"/"Goodfit" indices are not
    computed at all (both are internal/undocumented diagnostics returned
    by alpha.1() but never actually surfaced in R's own report_alpha output).
  - The bootstrap CI for alpha (n_iter > 1) resamples rows with
    replacement and recomputes the same statistics each iteration,
    matching psych::alpha()'s own bootstrap procedure — including its
    apparent quirk of labeling the bootstrap table's 6th column "ase"
    while actually storing the raw Feldt Q value there (unlike the
    point-estimate "ase", which is sqrt(Q/n)). This is a randomized
    procedure in both R and here, so exact bootstrap CI values won't
    reproduce run-to-run in either language, only converge similarly.
  - extract_components bridges to R's mixlm::Anova() via rpy2: Type III
    variance-component decomposition for crossed random-effects designs
    has no statsmodels equivalent (MixedLM handles nested/hierarchical
    structures, not this), the same category of gap as mirt/lavaan in
    glm_irt.py/glm_irt_t.py. It accepts an rpy2 mixlm model object built
    via R calls (see the __main__ block for a full worked example).
  - plot_mtmm's `round(value,2)` + "strip a leading zero" cell label
    (R: stringr::str_replace(round(value,2),"0\\.",".")) is reproduced by
    formatting to 2 decimals, stripping trailing zeros the way R's
    automatic numeric-to-character conversion would (so a perfect 1.00
    renders as "1", not "1.00"), then replacing the first "0." substring
    with "." — matching R's actual (slightly quirky) label text, not just
    its intent.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, facet_grid, labs, theme_bw, theme,
    element_rect, element_blank, element_line, element_text, scale_fill_brewer,
)
##########################################################################################
# ALPHA (raw Cronbach's alpha from a covariance matrix)
##########################################################################################
def compute_raw_alpha(df):
    """
    Cronbach's alpha computed directly from the (pairwise-complete)
    covariance matrix of a unidimensional set of items.

    Parameters:
    df (pandas.DataFrame): Numeric item columns of a single scale.

    Returns:
    float: Cronbach's alpha.

    Examples:
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(12345)
    >>> from functions_generate_data import generate_correlation_matrix
    >>> corr = np.full((6, 6), .5)
    >>> np.fill_diagonal(corr, 1)
    >>> df = np.round(generate_correlation_matrix(corr, nrows=1000), 0) + 5
    >>> compute_raw_alpha(df=pd.DataFrame(df))
    """
    cov = df.cov()
    k = df.shape[1]
    numerator = np.trace(cov.to_numpy())
    denominator = cov.to_numpy().sum()
    return k / (k - 1) * (1 - numerator / denominator)
##########################################################################################
# ALPHA DIAGNOSTICS
##########################################################################################
def compute_alpha_diagnostics(df):
    """
    Item-total correlations and alpha-if-item-removed for a
    unidimensional scale.

    Parameters:
    df (pandas.DataFrame): Numeric item columns of a single scale.

    Returns:
    pandas.DataFrame: One row per item — item, alpha.if.item.removed,
    item.total.correlation, item.total.correlation.r.drop.

    Examples:
    >>> compute_alpha_diagnostics(df=df)
    """
    row_sum_all = df.sum(axis=1, skipna=True)
    alpha_if_removed, item_total, item_total_rdrop = [], [], []
    for i, col in enumerate(df.columns):
        rest = df.drop(columns=[col])
        alpha_if_removed.append(compute_raw_alpha(rest))
        item_total.append(df[col].corr(row_sum_all))
        item_total_rdrop.append(df[col].corr(rest.sum(axis=1, skipna=True)))
    return pd.DataFrame({
        "item": df.columns, "alpha.if.item.removed": alpha_if_removed,
        "item.total.correlation": item_total, "item.total.correlation.r.drop": item_total_rdrop,
    }, index=df.columns)
##########################################################################################
# MEAN SD
##########################################################################################
def compute_mean_sd_alpha(df, divisor=None):
    """
    Mean and SD of scale scores.

    Parameters:
    df (pandas.DataFrame): Numeric item columns of a single scale.
    divisor (float, optional): If given, scale score = row sum / divisor.
        If None (default), scale score = row mean.

    Returns:
    pandas.DataFrame: One row with columns MEAN, SD (or Mean, SD when
    `divisor` is given, matching R's inconsistent capitalization between
    the two branches).

    Examples:
    >>> compute_mean_sd_alpha(df)
    >>> compute_mean_sd_alpha(df, divisor=100)
    """
    if divisor is None:
        scores = df.mean(axis=1, skipna=True)
        return pd.DataFrame({"MEAN": [scores.mean()], "SD": [scores.std(ddof=1)]})
    scores = df.sum(axis=1, skipna=True) / divisor
    return pd.DataFrame({"Mean": [scores.mean()], "SD": [scores.std(ddof=1)]})
##########################################################################################
# KEY TO CFA MODEL
##########################################################################################
def key_to_cfa_model(key):
    """
    Convert a {factor_name: [item names]} dict into a lavaan CFA model
    syntax string.

    Parameters:
    key (dict): Maps factor/trait names to lists of item column names.

    Returns:
    str: lavaan model specification, one factor definition per line.

    Examples:
    >>> key = {"f1": ["x1", "x2", "x3"], "f2": ["x4", "x5", "x6"]}
    >>> key_to_cfa_model(key)
    """
    parts = []
    for name, items in key.items():
        parts.extend([name, "=~", "+".join(items), "\n"])
    return " ".join(parts)
##########################################################################################
# PLOT MTMM
##########################################################################################
def _format_corr_label(v):
    """Round to 2dp, drop trailing zeros the way R's numeric->character conversion would, strip a leading '0.'."""
    if pd.isna(v):
        return ""
    s = f"{round(v, 2):.2f}".rstrip("0").rstrip(".")
    if s in ("", "-"):
        s = "0"
    return s.replace("0.", ".", 1)


def plot_mtmm(df, key, method, subject, title=""):
    """
    Campbell-Fiske multitrait-multimethod (MTMM) matrix, as a faceted
    tile plot: for each trait x method combination a scale score (row
    mean of items) is built, Cronbach's alpha is placed on the diagonal,
    and all scale-score pairs are correlated and classified into one of
    four MTMM relationship types.

    Parameters:
    df (pandas.DataFrame): Long-format data, one row per subject per method.
    key (dict): Maps trait names to lists of item column names.
    method (str): Column identifying the measurement method per row.
    subject (str): Column identifying the subject (for aligning scores
        across methods).
    title (str, optional): Appended to the plot title. Defaults to "".

    Returns:
    plotnine.ggplot

    Examples:
    >>> import rpy2.robjects as ro
    >>> from rpy2.robjects.packages import importr
    >>> lavaan = importr("lavaan")
    >>> # see glm_reliability.py's __main__ block for a full worked example
    >>> # building model_data via lavaan::simulateData through rpy2.
    """
    # R builds `trait` with one row per row of `df` (not per unique subject) and relies
    # on silent vector recycling to fill it from each (usually shorter) per-method subset
    # -- correct only because the R docstring's example data happens to repeat each
    # subject's block in lockstep with `subject`'s own cycle. pandas has no such
    # recycling, and it isn't needed: indexing `trait` by unique subject directly is
    # both robust to row ordering and produces the identical correlation matrix.
    trait = pd.DataFrame(index=pd.Index(df[subject].unique(), name="subject"))
    alpha_result = {}
    for z in df[method].unique():
        subset_rows = df[method] == z
        for i, items in key.items():
            scale_data = df.loc[subset_rows, items]
            scores = scale_data.mean(axis=1, skipna=True)
            scores.index = df.loc[subset_rows, subject].to_numpy()
            trait[f"{i}.{z}"] = scores.reindex(trait.index).to_numpy()
            alpha_result[f"{i}.{z}"] = compute_raw_alpha(scale_data)
    cors = trait.corr()
    corm = cors.reset_index(names="Var1").melt(id_vars="Var1", var_name="Var2", value_name="value")
    corm = corm[corm["Var1"] != corm["Var2"]]

    rel = pd.DataFrame({"Var1": list(alpha_result.keys()), "value": list(alpha_result.values())})
    rel["Var2"] = rel["Var1"]
    rel = rel[rel["Var1"].isin(cors.columns)][["Var1", "Var2", "value"]]
    corm = pd.concat([corm, rel], ignore_index=True)

    split1 = corm["Var1"].str.split(".", n=1, expand=True)
    corm["trait_x"], corm["method_x"] = split1[0], split1[1]
    split2 = corm["Var2"].str.split(".", n=1, expand=True)
    corm["trait_y"], corm["method_y"] = split2[0], split2[1]

    corm["var1_s"] = corm[["Var1", "Var2"]].min(axis=1)
    corm["var2_s"] = corm[["Var1", "Var2"]].max(axis=1)

    same_trait = corm["trait_x"] == corm["trait_y"]
    same_method = corm["method_x"] == corm["method_y"]
    corm["type"] = np.select(
        [same_trait & ~same_method, ~same_trait & same_method, ~same_trait & ~same_method, same_trait & same_method],
        ["monotrait-heteromethod (validity)", "heterotrait-monomethod",
         "heterotrait-heteromethod", "monotrait-monomethod (reliability)"],
        default="",
    )

    trait_levels = sorted(corm["trait_x"].unique())
    method_levels = sorted(corm["method_x"].unique())
    corm["trait_x"] = pd.Categorical(corm["trait_x"], categories=trait_levels)
    corm["method_x"] = pd.Categorical(corm["method_x"], categories=method_levels)
    corm["trait_y"] = pd.Categorical(corm["trait_y"], categories=list(reversed(trait_levels)))
    corm["method_y"] = pd.Categorical(corm["method_y"], categories=method_levels)

    corm = corm.sort_values(["method_x", "trait_x"])
    corm = corm.drop_duplicates(subset=["var1_s", "var2_s"])
    corm["label"] = corm["value"].apply(_format_corr_label)

    return (ggplot(corm)
            + geom_tile(aes(x="trait_x", y="trait_y", fill="type"), size=5)
            + geom_text(aes(x="trait_x", y="trait_y", label="label"))
            + facet_grid("method_y~method_x")
            + labs(x="", y="", title=f"Mulitrait Multimethod Matrix {title}")
            + theme_bw(base_size=10)
            + theme(panel_background=element_rect(colour=None), panel_grid_minor=element_blank(),
                    axis_line=element_line(), strip_background=element_blank(), panel_grid=element_blank(),
                    axis_text_x=element_text(angle=45, hjust=1), legend_title=element_blank())
            + scale_fill_brewer(type="seq", palette=1))
##########################################################################################
# SMC (internal, mirrors psych::smc)
##########################################################################################
def _smc(R):
    """Squared multiple correlation of each variable regressed on the rest of R."""
    return 1 - 1 / np.diag(np.linalg.inv(R))
##########################################################################################
# ALPHA STATS (internal, mirrors psych::alpha's alpha.1(C,R) helper)
##########################################################################################
def _alpha_stats(C, R):
    n = C.shape[0]
    trC = np.trace(C)
    sumC = C.sum()
    alpha_raw = (1 - trC / sumC) * (n / (n - 1))
    sumR = R.sum()
    alpha_std = (1 - n / sumR) * (n / (n - 1))
    smc_r = _smc(R)
    g6 = 1 - (n - smc_r.sum()) / sumR
    av_r = (sumR - n) / (n * (n - 1))
    lower = R[np.tril_indices(n, k=-1)]
    var_r = np.var(lower, ddof=1)
    med_r = np.median(lower)
    sn = n * av_r / (1 - av_r)
    q = (2 * n ** 2 / ((n - 1) ** 2 * (sumC ** 3))) * \
        (sumC * (np.trace(C @ C) + trC ** 2) - 2 * (trC * np.sum(C @ C)))
    return {"raw": alpha_raw, "std": alpha_std, "G6": g6, "av_r": av_r, "sn": sn, "Q": q,
            "var_r": var_r, "med_r": med_r}
##########################################################################################
# REVERSE CODE (internal, mirrors psych::reverse.code)
##########################################################################################
def _reverse_code(df, keys, mini=None, maxi=None):
    df = df.copy()
    for j, col in enumerate(df.columns):
        k = keys[j] if hasattr(keys, "__getitem__") else keys
        if k == -1:
            colmax = maxi if maxi is not None else df[col].max()
            colmin = mini if mini is not None else df[col].min()
            df[col] = (colmax + colmin) - df[col]
    return df
##########################################################################################
# RESPONSE FREQUENCY (internal, approximates psych::responseFrequency)
##########################################################################################
def _response_frequency(df, max_categories=10):
    """Proportion of each observed response value per item, for items with <= max_categories unique values."""
    rows = {}
    for col in df.columns:
        s = df[col]
        miss = s.isna().mean()
        valid = s.dropna()
        uniq = sorted(valid.unique())
        row = {"miss": miss}
        if len(uniq) <= max_categories:
            counts = valid.value_counts(normalize=True)
            for v in uniq:
                row[str(v)] = counts.get(v, 0.0)
        rows[col] = row
    return pd.DataFrame(rows).T
##########################################################################################
# PSYCH ALPHA (internal, mirrors the bulk of psych::alpha)
##########################################################################################
def _psych_alpha(df, cumulative=False, n_iter=1):
    n = df.shape[1]
    nsub = df.shape[0]
    C = df.cov(ddof=1).to_numpy()
    R = df.corr().to_numpy()
    stats = _alpha_stats(C, R)
    ase = np.sqrt(stats["Q"] / nsub)

    total_scores = df.sum(axis=1, skipna=True) if cumulative else df.mean(axis=1, skipna=True)
    total = pd.Series({
        "raw_alpha": stats["raw"], "std_alpha": stats["std"], "G6(smc)": stats["G6"],
        "average_r": stats["av_r"], "S/N": stats["sn"], "ase": ase,
        "mean": total_scores.mean(), "sd": total_scores.std(ddof=1), "median_r": stats["med_r"],
    })

    vt = R.sum()
    item_r = R.sum(axis=0) / np.sqrt(vt)
    rc = R.copy()
    np.fill_diagonal(rc, _smc(R))
    vtc = rc.sum()
    item_rc = rc.sum(axis=0) / np.sqrt(vtc)
    raw_r = np.array([df[col].corr(total_scores) for col in df.columns])

    r_drop = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        v_drop = C[np.ix_(idx, idx)].sum()
        c_drop = C[:, i].sum() - C[i, i]
        r_drop[i] = c_drop / np.sqrt(C[i, i] * v_drop)

    item_stats = pd.DataFrame({
        "n": df.notna().sum().to_numpy(), "raw_r": raw_r, "std_r": item_r, "r_cor": item_rc, "r_drop": r_drop,
        "mean": df.mean(skipna=True).to_numpy(), "sd": df.std(ddof=1, skipna=True).to_numpy(),
    }, index=df.columns)
    item_stats = pd.concat([item_stats, _response_frequency(df)], axis=1)

    drop_rows = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        s = _alpha_stats(C[np.ix_(idx, idx)], R[np.ix_(idx, idx)])
        ase_i = np.sqrt(s["Q"] / nsub) if nsub > n else np.nan
        drop_rows.append({"raw_alpha": s["raw"], "std.alpha": s["std"], "G6(smc)": s["G6"],
                           "average_r": s["av_r"], "S/N": s["sn"], "alpha se": ase_i,
                           "var.r": s["var_r"], "med.r": s["med_r"]})
    alpha_drop = pd.DataFrame(drop_rows, index=df.columns)

    boot = None
    boot_ci = None
    if n_iter > 1:
        boot_rows = []
        for _ in range(n_iter):
            xi = df.sample(n=nsub, replace=True)
            Ci = xi.cov(ddof=1).to_numpy()
            Ri = xi.corr().to_numpy()
            s = _alpha_stats(Ci, Ri)
            boot_rows.append({"raw_alpha": s["raw"], "std.alpha": s["std"], "G6(smc)": s["G6"],
                               "average_r": s["av_r"], "s/n": s["sn"], "ase": s["Q"],
                               "var.r": s["var_r"], "median.r": s["med_r"]})
        boot = pd.DataFrame(boot_rows)
        q = boot["raw_alpha"].quantile([.025, .5, .975])
        boot_ci = {"2.5%": q.iloc[0], "50%": q.iloc[1], "97.5%": q.iloc[2]}

    return {"nvar": n, "total": total, "item_stats": item_stats, "alpha_drop": alpha_drop,
            "boot": boot, "boot_ci": boot_ci}
##########################################################################################
# ALPHA OUTPUT
##########################################################################################
def report_alpha(df, key=None, questions=None, reverse=None, mini=None, maxi=None, file=None,
                  cumulative=False, n_iter=1):
    """
    Cronbach's alpha reliability report for one or more scales, with item-
    level diagnostics and optional Excel export. See module docstring for
    exactly which psych::alpha() statistics are reproduced.

    Parameters:
    df (pandas.DataFrame): All item columns.
    key (dict, optional): Maps scale name -> list of item column names. If
        None (default), all of df's columns form one scale, "dimension".
    questions (dict, optional): Same structure as `key`, question label
        strings appended to item names in the output. Defaults to None.
    reverse (dict, optional): Same structure as `key`, sign vectors (1 =
        keep, -1 = reverse) per scale. Defaults to None (no reversal).
    mini, maxi (float, optional): Min/max rating used for item reversal.
        If None (default), the empirical min/max is used.
    file (str, optional): Output filename (without extension) for an
        Excel report. Defaults to None.
    cumulative (bool, optional): If True, scale scores are row sums
        instead of row means. Defaults to False.
    n_iter (int, optional): Bootstrap iterations for alpha's CI. 1
        (default) skips the bootstrap.

    Returns:
    dict: result_total, result_boot, result_item_statistics, result_dropped.

    Examples:
    >>> import numpy as np, pandas as pd
    >>> from functions_generate_data import generate_correlation_matrix
    >>> np.random.seed(12345)
    >>> corr = np.full((6, 6), .5)
    >>> np.fill_diagonal(corr, 1)
    >>> df = pd.DataFrame(np.round(generate_correlation_matrix(corr, nrows=1000), 0) + 5,
    ...                   columns=["X1", "X2", "X3", "X4", "X5", "X6"])
    >>> key = {"f1": ["X1", "X2", "X3"], "f2": ["X4", "X5", "X6"]}
    >>> report_alpha(df=df, key=key, cumulative=True, n_iter=1)
    >>> report_alpha(df=df, key=key, reverse={"f1": [1, 1, 1], "f2": [1, 1, 1]}, n_iter=2)
    >>> report_alpha(df=df, key=key, n_iter=2, file="alpha")
    """
    if key is None:
        key = {"dimension": list(df.columns)}

    total_rows, boot_rows, item_rows, drop_rows = [], [], [], []
    for sc, items in key.items():
        if len(items) < 2:
            continue
        temp = df[items].copy()
        if reverse is not None and sc in reverse:
            temp = _reverse_code(temp, reverse[sc], mini, maxi)

        result = _psych_alpha(temp, cumulative=cumulative, n_iter=n_iter)
        try:
            eigenvalues = np.linalg.eigvalsh(temp.corr().to_numpy())
        except Exception:
            eigenvalues = np.array([np.nan])
        kaiser_criterion = int(np.sum(eigenvalues > 1))

        total_row = {"dimension": sc, "items": result["nvar"], "kaiser_criterion": kaiser_criterion}
        total_row.update(result["total"].to_dict())
        if result["boot_ci"] is not None:
            for k, v in result["boot_ci"].items():
                total_row[f"boot_ci_{k}"] = v
        total_rows.append(total_row)

        if result["boot"] is not None:
            for _, boot_row in result["boot"].iterrows():
                boot_rows.append({"dimension": sc, "items": result["nvar"],
                                   "kaiser_criterion": kaiser_criterion, **boot_row.to_dict()})

        question_labels = (list(result["item_stats"].index) if questions is None else
                            [f"{q} {questions[sc][i]}" for i, q in enumerate(result["item_stats"].index)])
        item_row = result["item_stats"].copy()
        item_row.insert(0, "raw_alpha", result["total"]["raw_alpha"])
        item_row.insert(0, "question", question_labels)
        item_row.insert(0, "dimension", sc)
        item_rows.append(item_row.reset_index(drop=True))

        drop_row = result["alpha_drop"].copy()
        drop_row.insert(0, "scale_alpha", result["total"]["raw_alpha"])
        drop_row.insert(0, "question", question_labels)
        drop_row.insert(0, "dimension", sc)
        drop_rows.append(drop_row.reset_index(drop=True))

    result_total = pd.DataFrame(total_rows)
    result_boot = pd.DataFrame(boot_rows) if boot_rows else pd.DataFrame()
    result_item_statistics = pd.concat(item_rows, ignore_index=True) if item_rows else pd.DataFrame()
    result_dropped = pd.concat(drop_rows, ignore_index=True) if drop_rows else pd.DataFrame()

    result_total.columns = [c.lower().replace(".", "_") for c in result_total.columns]
    if not result_boot.empty:
        result_boot.columns = [c.lower().replace(".", "_") for c in result_boot.columns]
    result_item_statistics.columns = [c.lower().replace(".", "_") for c in result_item_statistics.columns]
    result_dropped.columns = [c.lower().replace(".", "_") for c in result_dropped.columns]

    if not result_total.empty:
        result_total["alpha_criterion"] = np.select(
            [result_total["raw_alpha"] < .6, result_total["raw_alpha"] < .7, result_total["raw_alpha"] < .8,
             result_total["raw_alpha"] < .9, result_total["raw_alpha"] >= .9],
            ["Unacceptable", "Acceptable", "Good and Acceptable", "Good", "Excellent"], default="")

    if file is not None:
        try:
            from .functions_excel import excel_critical_value
        except ImportError:
            from functions_excel import excel_critical_value
        writer = pd.ExcelWriter(f"{file}.xlsx", engine="xlsxwriter")
        excel_critical_value(result_total, writer, sheetname="total statistics", critical={"raw_alpha": ">0.60"})
        if not result_boot.empty:
            excel_critical_value(result_boot, writer, sheetname="bootstrap", critical={"raw_alpha": ">0.60"})
        excel_critical_value(result_item_statistics, writer, sheetname="item statistics",
                              critical={"raw_alpha": ">0.60"})
        excel_critical_value(result_dropped, writer, sheetname="if dropped",
                              critical={"scale_alpha": ">0.60", "raw_alpha": ">0.60"})
        writer._save()
        writer.close()

    return {"result_total": result_total, "result_boot": result_boot,
            "result_item_statistics": result_item_statistics, "result_dropped": result_dropped}
##########################################################################################
# EXTRACT COMPONENTS (rpy2 bridge to R mixlm::Anova -- see module docstring)
##########################################################################################
def extract_components(model, title=""):
    """
    Extract variance components (Type III) from an R mixlm::lm() mixed
    model and their percentage contribution to total variance, plus a
    horizontal bar chart. Bridges to R via rpy2 — see module docstring.

    Parameters:
    model: An rpy2 R model object from mixlm::lm() (with random effects
        specified via r(), e.g. "response ~ r(time)*r(person)").
    title (str, optional): Plot title. Defaults to "".

    Returns:
    dict: {"components": pandas.DataFrame (component, VC, vc_percent),
    "plot": plotnine.ggplot}.

    Examples:
    >>> import rpy2.robjects as ro
    >>> from rpy2.robjects.packages import importr
    >>> mixlm = importr("mixlm")
    >>> # see glm_reliability.py's __main__ block for a full worked example
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    importr("mixlm")
    base = importr("base")

    res = ro.r["Anova"](model, type="III")
    anova_table = res.rx2("anova")
    var_comps = np.asarray(res.rx2("var.comps"))
    name_components = list(base.rownames(anova_table))

    vc_percent = np.abs(var_comps) / np.nansum(np.abs(var_comps)) * 100
    components = pd.DataFrame({"component": name_components, "VC": var_comps, "vc_percent": vc_percent})
    components["component"] = pd.Categorical(components["component"], categories=name_components)

    from plotnine import geom_bar, geom_line, geom_point, coord_flip
    plot = (ggplot(components, aes(x="component", y="vc_percent", group=1))
            + geom_bar(stat="identity")
            + geom_line(color="#404040")
            + geom_point(size=5, color="#404040")
            + labs(title=title, x="components", y="Explained variance (%)")
            + coord_flip()
            + theme_bw())
    return {"components": components, "plot": plot}
##########################################################################################
# SHROUT RELIABILITY
##########################################################################################
def compute_shrout(sperson, spersonitem, stime, spersontime, serror, m, k):
    """
    Shrout-Fleiss (1979) reliability coefficients from person x item x
    time variance components (see extract_components).

    Parameters:
    sperson (float): Variance component of the person main effect.
    spersonitem (float): Variance component of the person x item interaction.
    stime (float): Variance component of the time main effect.
    spersontime (float): Variance component of the person x time interaction.
    serror (float): Residual error variance component.
    m (int): Number of items averaged over.
    k (int): Number of time points averaged over.

    Returns:
    pandas.DataFrame: One row per coefficient (r1f, rkf, rkr, r1r, rc),
    with columns measure, result, description.

    Examples:
    >>> compute_shrout(sperson=1.2, spersonitem=0.3, stime=0.1,
    ...                spersontime=0.2, serror=0.4, m=3, k=3)
    """
    instruction = pd.DataFrame({
        "measure": ["r1f", "r1r", "rkf", "rkr", "rc"],
        "description": [
            "Reliability (between persons) of measures taken on the same fixed k time",
            "Reliability (between persons) of measures taken on the same random k time",
            "Reliability (between persons) of average measures taken over fixed m items and fixed k times",
            "Reliability (between persons) of different random time with same number of points k between periods",
            "Reliability (within persons) of change",
        ],
    })
    km = m * k
    r1f = (sperson + spersonitem / m) / (sperson + spersonitem / m + serror / m)
    r1r = (sperson + spersonitem / m) / (sperson + spersonitem / m + stime + spersontime + serror / m)
    rkf = (sperson + spersonitem / m) / (sperson + spersonitem / m + serror / km)
    rkr = (sperson + spersonitem / m) / (sperson + spersonitem / m + stime / k + spersontime / k + serror / km)
    rc = spersontime / (spersontime + (serror / m))
    result = pd.DataFrame({"measure": ["r1f", "rkf", "rkr", "r1r", "rc"], "result": [r1f, rkf, rkr, r1r, rc]})
    return result.merge(instruction, on="measure", how="outer")
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    try:
        from .functions_generate_data import generate_correlation_matrix
    except ImportError:
        from functions_generate_data import generate_correlation_matrix

    np.random.seed(12345)
    corr = np.full((6, 6), .5)
    np.fill_diagonal(corr, 1)
    df_scale = pd.DataFrame(np.round(generate_correlation_matrix(corr, nrows=1000), 0) + 5,
                             columns=[f"X{i + 1}" for i in range(6)])

    print("=" * 80, "\ncompute_raw_alpha / compute_alpha_diagnostics / compute_mean_sd_alpha\n", "=" * 80, sep="")
    print("raw alpha:", compute_raw_alpha(df_scale))
    print(compute_alpha_diagnostics(df_scale))
    print(compute_mean_sd_alpha(df_scale))
    print(compute_mean_sd_alpha(df_scale, divisor=100))

    print("\n" + "=" * 80, "\nkey_to_cfa_model\n", "=" * 80, sep="")
    key_cfa = {"f1": [f"x{i}" for i in range(1, 4)], "f2": [f"x{i}" for i in range(4, 7)]}
    print(repr(key_to_cfa_model(key_cfa)))

    print("\n" + "=" * 80, "\nreport_alpha (2 scales, with bootstrap CI)\n", "=" * 80, sep="")
    key = {"f1": ["X1", "X2", "X3"], "f2": ["X4", "X5", "X6"]}
    res = report_alpha(df=df_scale, key=key, n_iter=200)
    print("result_total:\n", res["result_total"])
    print("\nresult_boot head:\n", res["result_boot"].head())
    print("\nresult_item_statistics:\n", res["result_item_statistics"])
    print("\nresult_dropped:\n", res["result_dropped"])

    print("\n" + "=" * 80, "\nreport_alpha (reverse coding, Excel export)\n", "=" * 80, sep="")
    reverse = {"f1": [1, 1, 1], "f2": [1, 1, 1]}
    res2 = report_alpha(df=df_scale, key=key, reverse=reverse, n_iter=1, file="alpha_test")
    print("result_total:\n", res2["result_total"])
    import os
    print("Excel written:", os.path.exists("alpha_test.xlsx"))

    print("\n" + "=" * 80, "\nplot_mtmm (via rpy2 lavaan::simulateData)\n", "=" * 80, sep="")
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        lavaan = importr("lavaan")
        base = importr("base")

        ro.r("set.seed(12345)")
        population_model = ("t1=~x1+.9*x2+.9*x3\n"
                             "t2=~x4+.9*x5+.9*x6\n"
                             "t3=~x7+.9*x8+.9*x9")
        model_data_r = lavaan.simulateData(population_model, sample_nobs=1000)
        with_rows = ro.r["sample"](ro.IntVector(range(1, 1001)), 1000, replace=True)
        model_data_r = model_data_r.rx(with_rows, True)
        model_data = pd.DataFrame(np.asarray(base.as_matrix(model_data_r)),
                                   columns=list(base.colnames(model_data_r)))
        model_data = pd.concat([model_data, model_data, model_data], ignore_index=True)
        model_data["method"] = ["m1"] * 1000 + ["m2"] * 1000 + ["m3"] * 1000
        model_data["id"] = list(range(1, 1001)) * 3

        key_mtmm = {"t1": [f"x{i}" for i in range(1, 4)], "t2": [f"x{i}" for i in range(4, 7)],
                    "t3": [f"x{i}" for i in range(7, 10)]}
        p = plot_mtmm(df=model_data, key=key_mtmm, method="method", subject="id")
        p.save("plot_mtmm.png", verbose=False, width=10, height=8)
        print("saved plot_mtmm.png")
    except Exception as exc:
        print(f"Skipped plot_mtmm example (rpy2/lavaan not available or failed): {exc}")

    print("\n" + "=" * 80, "\nextract_components / compute_shrout (rpy2 bridge to R mixlm)\n", "=" * 80, sep="")
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        mixlm = importr("mixlm")
        base = importr("base")

        ro.r("""
        design <- expand.grid(time=1:3, item=1:3, person=1:10)
        design <- as.data.frame(lapply(design, as.factor))
        design$response <- as.numeric(as.character(design$time)) + as.numeric(as.character(design$item)) +
                            rnorm(90, 0, 0.1)
        model <- mixlm::lm(response ~ r(time)*r(person) + r(item)*r(person), data=design)
        """)
        model = ro.r["model"]
        result = extract_components(model)
        print(result["components"])
        result["plot"].save("plot_extract_components.png", verbose=False)
        print("saved plot_extract_components.png")

        vc = result["components"]
        print("\ncompute_shrout:")
        print(compute_shrout(sperson=vc.loc[1, "VC"], spersonitem=vc.loc[4, "VC"], stime=vc.loc[0, "VC"],
                              spersontime=vc.loc[3, "VC"], serror=vc.loc[5, "VC"], m=3, k=3))
    except Exception as exc:
        print(f"Skipped extract_components/compute_shrout example (rpy2/mixlm not available or failed): {exc}")
