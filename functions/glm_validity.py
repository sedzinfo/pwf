# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_VALIDITY.R: compute_efa_cfa, construct validity
of multi-item scales through exploratory and confirmatory factor analysis.

The EFAs use pwf's psych-compatible engine (pwf._efa, psych::fa with
fm = "ml" and rotate = "oblimin") and the CFAs pwf's lavaan-compatible
engine (pwf._sem with missing = "ml"), so no R installation is needed.
rwf's scree plots are switched off in rwf too, so plots_scree is empty.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import warnings

import numpy as np
import pandas as pd

try:
    from ._efa import fa, model_loadings
    from ._rcompat import r_make_names
    from ._sem import cfa
    from .functions_excel import excel_critical_value, excel_matrix
except ImportError:
    from _efa import fa, model_loadings
    from _rcompat import r_make_names
    from _sem import cfa
    from functions_excel import excel_critical_value, excel_matrix

_NAMES_ALL = ["n", "dimension", "variable", "F1", "S2F1", "S2F2", "P2F1", "P2F2", "S3F1", "S3F2", "S3F3", "P3F1",
              "P3F2", "P3F3", "index", "f1_v", "f2_v", "f3_v", "unstandardized", "standardized", "unstandardized_RI",
              "RI_u", "standardized_RI", "RI_s", "index_cfa", "fit_CFA", "fit_CFA_RI"]
_NAMES_NUMERIC = [c for c in _NAMES_ALL if c not in ("dimension", "variable", "index", "index_cfa")]
_FIT_NAMES = ["RMSEA", "RMSEA.CI.LOWER", "RMSEA.CI.UPPER", "RMSEA_CONF", "RMS", "CFI", "TLI"]
_COMMENT = {
    "F1": "Model 1: Factor Loadings for EFA One Factor Model",
    "S2F1": "Model 2: Factor Loadings\nStructure Matrix (item-factor correlations after rotation)\nfor EFA Two Factor Model 1st Factor",
    "S2F2": "Model 2: Factor Loadings\nStructure Matrix (item-factor correlations after rotation)\nfor EFA Two Factor Model 2nd Factor",
    "P2F1": "Model 2: Factor Loadings\nPattern Matrix (regression weights after rotation)\nfor EFA Two Factor Model 1st Factor",
    "P2F2": "Model 2: Factor Loadings\nPattern Matrix (regression weights after rotation)\nfor EFA Two Factor Model 2nd Factor",
    "S3F1": "Model 3: Factor Loadings\nStructure Matrix (item-factor correlations after rotation)\nfor EFA Three Factor Model 1st Factor",
    "S3F2": "Model 3: Factor Loadings\nStructure Matrix (item-factor correlations after rotation)\nfor EFA Three Factor Model 2nd Factor",
    "S3F3": "Model 3: Factor Loadings\nStructure Matrix (item-factor correlations after rotation)\nfor EFA Three Factor Model 3rd Factor",
    "P3F1": "Model 3: Factor Loadings\nPattern Matrix (regression weights after rotation)\nfor EFA Three Factor Model 1st Factor",
    "P3F2": "Model 3: Factor Loadings\nPattern Matrix (regression weights after rotation)\nfor EFA Three Factor Model 2nd Factor",
    "P3F3": "Model 3: Factor Loadings\nPattern Matrix (regression weights after rotation)\nfor EFA Three Factor Model 3rd Factor",
    "index": "Fit Index For EFA Models", "f1_v": "Model 1: Fit Index Value", "f2_v": "Model 2: Fit Index Value",
    "f3_v": "Model 3: Fit Index Value",
    "standardized": "Model 4: Standardized Factor Loadings for CFA 1 Factor Model",
    "unstandardized": "Model 4: Unstandardized Factor Loadings for CFA 1 Factor Model",
    "standardized_RI": "Model 5: Standardized Factor Loadings for CFA 1 Factor Random Intercept Model",
    "unstandardized_RI": "Model 5: Unstandardized Factor Loadings for CFA 1 Factor Random Intercept Model",
    "RI_u": "Model 5: Random Intercept Parameter Unstandardized",
    "RI_s": "Model 5: Random Intercept Parameter Standardized",
    "index_cfa": "Fit Index for CFA Models", "fit_CFA": "Model 4: Fit Index Value",
    "fit_CFA_RI": "Model 5: Fit Index Value",
}
##########################################################################################
# HELPERS
##########################################################################################
def _fit_efa(data, nfactors, dimension):
    """EFA with nfactors factors, or None when the model is not identified or fails."""
    p = data.shape[1]
    if nfactors >= p or ((p - nfactors) ** 2 - (p + nfactors)) / 2 < 0:
        return None
    try:
        return fa(data, nfactors=nfactors)
    except Exception as error:  # noqa: BLE001 - rwf reports the failure and goes on
        print(f"EFA of {dimension} with {nfactors} factor(s) failed: {error}")
        return None


def _efa_fit(model):
    """RMSEA, its limits and confidence level, RMS, CFI and TLI (missing when not finite)."""
    if model is None:
        return [np.nan] * 7
    values = [model.RMSEA["RMSEA"], model.RMSEA["lower"], model.RMSEA["upper"], model.RMSEA["confidence"],
              model.rms, model.CFI, model.TLI]
    return [float(v) if np.isfinite(v) else np.nan for v in values]


def _efa_loadings(model, matrix_type, labels, variables):
    if model is None:
        out = pd.DataFrame({"variable": variables})
        for label in labels:
            out[label] = np.nan
        return out
    out = model_loadings(model, cut=0.001, matrix_type=matrix_type, sort=True).drop(columns="Matrix")
    out.columns = ["variable"] + labels
    return out


def _bind_columns(frames):
    """rwf's c_bind for frames of different lengths: side by side, shorter ones padded with missing values."""
    length = max(len(f) for f in frames)
    return pd.concat([f.reset_index(drop=True).reindex(range(length)) for f in frames], axis=1)


def _lower_triangle_long(matrix):
    """reshape2::melt(matrix_triangle(m, diagonal = NA)): lower-triangle cells, first index fastest."""
    rows = []
    names = list(matrix.columns)
    for j, b in enumerate(names):
        for i, a in enumerate(names):
            if i > j:
                rows.append({"Var1": a, "Var2": b, "value": matrix.iloc[i, j]})
    return pd.DataFrame(rows, columns=["Var1", "Var2", "value"])


def _matrix_triangle(matrix):
    """rwf::matrix_triangle(m): the lower triangle with the diagonal, the rest missing."""
    values = matrix.to_numpy(dtype=float).copy()
    values[np.triu_indices_from(values, 1)] = np.nan
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def _cfa_tables(dimension, temp_df, n_rows):
    """Models 4 and 5 of rwf (one factor, and with a random intercept) for one dimension."""
    latent = r_make_names([dimension])[0]
    items = list(temp_df.columns)
    model = f"{latent} =~ " + " + ".join(items)
    model_ri = model + "\nRI =~ " + " + ".join(f"1*{v}" for v in items) + f"\nRI ~ 0*{latent}"
    fit = cfa(model, temp_df, missing="ml")
    fit_ri = cfa(model_ri, temp_df, missing="ml")
    if not (fit.converged and fit_ri.converged):
        raise RuntimeError("the CFA did not converge")
    score = pd.Series(np.nan, index=range(n_rows))
    score.iloc[np.flatnonzero(temp_df.notna().any(axis=1).to_numpy())] = fit.predict().iloc[:, 0].to_numpy()

    mi = fit.modification_indices(power=False, free_remove=True, minimum_value=10)
    if len(mi):
        modification = pd.concat([pd.DataFrame({"dimension": [dimension] * len(mi)}), mi], axis=1)
    else:
        modification = pd.DataFrame({"dimension": [dimension], **{c: [np.nan] for c in mi.columns}})

    est, std = fit.inspect("est"), fit.inspect("std")
    est_ri, std_ri = fit_ri.inspect("est"), fit_ri.inspect("std")
    coef = pd.DataFrame({"q": items, "lambda": est["lambda"].iloc[:, 0].to_numpy(),
                         "theta": np.diag(est["theta"]), "intercept": est["nu"]["intercept"].to_numpy()})
    coef_std = pd.DataFrame({"q": items, "lambda_std": std["lambda"].iloc[:, 0].to_numpy(),
                             "theta_std": np.diag(std["theta"]), "intercept_std": std["nu"]["intercept"].to_numpy()})
    coef_ri = pd.DataFrame({"q": items, "lambda": est_ri["lambda"].iloc[:, 0].to_numpy(),
                            "ri": est_ri["lambda"].iloc[:, 1].to_numpy(), "theta": np.diag(est_ri["theta"]),
                            "intercept": est_ri["nu"]["intercept"].to_numpy()})
    coef_std_ri = pd.DataFrame({"q": items, "lambda_std": std_ri["lambda"].iloc[:, 0].to_numpy(),
                                "ri": std_ri["lambda"].iloc[:, 1].to_numpy(), "theta_std": np.diag(std_ri["theta"]),
                                "intercept_std": std_ri["nu"]["intercept"].to_numpy()})
    # merge(by = "q") sorts by q, then the rows are ordered by the loading
    coefficients = coef_std.merge(coef, on="q").sort_values("q")
    coefficients = coefficients.sort_values("lambda", kind="stable")
    coefficients.insert(0, "dimension", dimension)
    coefficients_ri = coef_ri.merge(coef_std_ri, on="q", suffixes=(".x", ".y")).sort_values("q")
    coefficients_ri = coefficients_ri.sort_values("lambda", kind="stable")
    coefficients_ri.insert(0, "dimension", dimension)

    loadings = pd.DataFrame({"variable": items, "unstandardized": est["lambda"].iloc[:, 0].to_numpy(),
                             "standardized": std["lambda"].iloc[:, 0].to_numpy(),
                             "unstandardized_RI": est_ri["lambda"].iloc[:, 0].to_numpy(),
                             "RI_u": est_ri["lambda"].iloc[:, 1].to_numpy(),
                             "standardized_RI": std_ri["lambda"].iloc[:, 0].to_numpy(),
                             "RI_s": std_ri["lambda"].iloc[:, 1].to_numpy()})
    fit_measures, fit_measures_ri = fit.fit_measures(), fit_ri.fit_measures()
    order = ["rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr", "cfi", "tli"]
    fit_table = pd.DataFrame({"index_cfa": [o.upper() for o in order], "fit_CFA": [fit_measures[o] for o in order],
                              "fit_CFA_RI": [fit_measures_ri[o] for o in order]})
    return score, modification, coefficients, coefficients_ri.reset_index(drop=True), loadings, fit_table
##########################################################################################
# EFA AND CFA
##########################################################################################
def compute_efa_cfa(dimensions, df_response, filename="efa_cfa_models.xlsx", mini=1, maxi=5):
    """
    Exploratory and confirmatory factor analyses of every dimension of a
    questionnaire, written to an Excel workbook (port of rwf::compute_efa_cfa).

    For each dimension: EFAs with 1, 2 and 3 factors (maximum likelihood,
    oblimin; Models 1-3) and, with at least 4 items, a one-factor CFA (Model
    4) and a one-factor CFA with a random intercept (Model 5), both by full
    information maximum likelihood so missing responses are used.

    Parameters:
    dimensions (pandas.DataFrame): Columns dimension, subquestion_text (item
        column names in df_response) and key (1 for reverse-scored items,
        -1 for items scored as they are).
    df_response (pandas.DataFrame): Responses, one column per item.
    filename (str, optional): Excel file to write. Defaults to "efa_cfa_models.xlsx".
    mini, maxi (int, optional): Lowest and highest possible response, used to
        reverse-score items. Defaults to 1 and 5.

    Returns:
    dict: efa_cfa_models (one row per item: EFA loadings F1, S2F1-P3F3 and
    fit, CFA loadings and fit), cfa_scores (Model 4 factor scores, one column
    per dimension), plots_scree (empty, as in rwf) and cfa_coefficients
    (Model 4 loadings, residual variances and intercepts), as in rwf.

    Examples:
    >>> items = [f"E{i}" for i in range(1, 11)]
    >>> dimensions = pd.DataFrame({"dimension": "E", "subquestion_text": items,
    ...                            "key": [1 if i in ("E2", "E4") else -1 for i in items]})
    >>> result = compute_efa_cfa(dimensions, df_ocean[items], filename="efa_cfa.xlsx")
    """
    dimensions = pd.DataFrame(dimensions).copy()
    df_response = pd.DataFrame(df_response).copy()
    missing_columns = [c for c in ("dimension", "subquestion_text", "key") if c not in dimensions.columns]
    if missing_columns:
        raise ValueError("dimensions is missing the column(s): " + ", ".join(missing_columns))
    if not dimensions["key"].isin([-1, 1]).all():
        raise ValueError("dimensions$key must be 1 (reverse scored item) or -1 (item scored as it is)")
    items = list(dimensions["subquestion_text"])
    missing_items = [i for i in items if i not in df_response.columns]
    if missing_items:
        raise ValueError("these items of dimensions$subquestion_text are not columns of df_response: "
                         + ", ".join(missing_items))
    for item in items:
        if df_response[item].isna().all():
            df_response[item] = df_response[item].astype(float)
    if not all(pd.api.types.is_numeric_dtype(df_response[i]) for i in items):
        raise ValueError("the items in df_response must be numeric")
    no_variance = [i for i in items if df_response[i].dropna().nunique() < 2]
    if no_variance:
        warnings.warn("these items have no responses or the same response from everyone, so the EFA and CFA of "
                      "their dimension will fail: " + ", ".join(no_variance))
    responses = df_response[items].to_numpy(dtype=float)
    if np.any((responses < mini) | (responses > maxi)):
        warnings.warn(f"some responses are outside {mini}-{maxi} (for example a missing-data code such as 0); "
                      "recode them to NA first")
    dimensions["dimension_name"] = dimensions["dimension"].astype(str) + "_" + dimensions["subquestion_text"]
    dimensions = dimensions.sort_values("dimension", kind="stable")

    efa_cfa_models, residuals, modification, coefficients_all, coefficients_all_ri = [], [], [], [], []
    cfa_scores = pd.DataFrame(index=range(len(df_response)))
    for dimension in pd.unique(dimensions["dimension"]):
        print(f"Dimension: {dimension}")
        temp_df = df_response[list(dimensions.loc[dimensions["dimension"] == dimension, "subquestion_text"])]
        models = {k: _fit_efa(temp_df, k, dimension) for k in (1, 2, 3)}
        if models[1] is not None:
            long = _lower_triangle_long(models[1].residual)
            long = long[long["value"].abs() > 0.1]
            long = long.iloc[np.argsort(-long["value"].abs().to_numpy(), kind="stable")]
            if len(long):
                residuals.append(pd.DataFrame({"dimension": dimension, "Q1": long["Var1"].to_numpy(),
                                               "Q2": long["Var2"].to_numpy(), "Residual": long["value"].to_numpy()}))
        variables = list(temp_df.columns)
        loadings = [
            _efa_loadings(models[1], "pattern", ["F1"], variables),
            _efa_loadings(models[2], "structure", ["S2F1", "S2F2"], variables),
            _efa_loadings(models[2], "pattern", ["P2F1", "P2F2"], variables),
            _efa_loadings(models[3], "structure", ["S3F1", "S3F2", "S3F3"], variables),
            _efa_loadings(models[3], "pattern", ["P3F1", "P3F2", "P3F3"], variables),
        ]
        u_efa = loadings[0]
        for frame in loadings[1:]:
            u_efa = u_efa.merge(frame, on="variable", sort=False)
        fit_columns = pd.DataFrame({"index": _FIT_NAMES, "f1_v": _efa_fit(models[1]), "f2_v": _efa_fit(models[2]),
                                    "f3_v": _efa_fit(models[3])})
        u_efa = _bind_columns([u_efa, fit_columns])
        u_efa.insert(0, "n", range(1, len(u_efa) + 1))
        temp = u_efa

        if temp_df.shape[1] < 4:
            print(f"Skipping the CFA of {dimension}: the random intercept model needs at least 4 items")
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score, mi, coefficients, coefficients_ri, cfa_loadings, fit_table = _cfa_tables(
                        dimension, temp_df, len(df_response))
                merged = u_efa.merge(cfa_loadings, on="variable", how="left", sort=False).sort_values("n")
                temp = _bind_columns([merged, fit_table])
                cfa_scores[dimension] = score.to_numpy()
                modification.append(mi)
                coefficients_all.append(coefficients)
                coefficients_all_ri.append(coefficients_ri)
            except Exception as error:  # noqa: BLE001 - rwf skips the CFA of a dimension that fails
                print(f"Skipping the CFA of {dimension} due to an error: {error}")
        temp = temp.copy()
        temp["dimension"] = dimension
        efa_cfa_models.append(temp)

    models_table = pd.concat(efa_cfa_models, ignore_index=True)
    for column in _NAMES_ALL:
        if column not in models_table.columns:
            models_table[column] = np.nan
    for column in _NAMES_NUMERIC:
        models_table[column] = pd.to_numeric(models_table[column].replace("", np.nan), errors="coerce")
    models_table = models_table[_NAMES_ALL].reset_index(drop=True)
    modification_df = pd.concat(modification, ignore_index=True) if modification else pd.DataFrame()
    residuals_df = pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame()
    coefficients_df = pd.concat(coefficients_all, ignore_index=True) if coefficients_all else pd.DataFrame()
    coefficients_ri_df = pd.concat(coefficients_all_ri, ignore_index=True) if coefficients_all_ri else pd.DataFrame()

    keys = (dimensions["key"] * -1).to_numpy()
    reversed_items = df_response[list(dimensions["subquestion_text"])].copy()
    for item, key in zip(dimensions["subquestion_text"], keys):
        if key < 0:
            reversed_items[item] = (maxi + mini) - reversed_items[item]
    df_meanr = pd.DataFrame({d: reversed_items[list(dimensions.loc[dimensions["dimension"] == d, "subquestion_text"])]
                             .mean(axis=1) for d in pd.unique(dimensions["dimension"])})
    df_used = reversed_items.copy()
    df_used.columns = list(dimensions["dimension_name"])
    cor_used_items = pd.concat([df_used, df_meanr], axis=1).corr().loc[df_used.columns, df_meanr.columns]
    long = _lower_triangle_long(df_used.corr())
    long = long.dropna()
    long = long.iloc[np.argsort(-long["value"].abs().to_numpy(), kind="stable")]
    mean_score_correlation = _matrix_triangle(df_meanr.corr())

    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        critical = {c: [">0.32", "<-0.32"] for c in ["F1", "S2F1", "S2F2", "P2F1", "P2F2", "S3F1", "S3F2", "S3F3",
                                                     "P3F1", "P3F2", "P3F3", "standardized", "standardized_RI"]}
        excel_critical_value(models_table, writer, sheetname="critical", comments=_COMMENT, critical=critical)
        for frame, sheet in [(modification_df, "CFA_Modification_Indexes"), (residuals_df, "EFA1F_Residuals"),
                             (coefficients_df, "CFA_Coefficients"), (coefficients_ri_df, "CFA_RI_Coefficients")]:
            if len(frame):
                excel_critical_value(frame, writer, sheetname=sheet)
        excel_matrix(cor_used_items, writer, sheetname="used_items")
        if len(long):
            excel_matrix(long, writer, sheetname="used_items_item_correlation")
        excel_matrix(mean_score_correlation, writer, sheetname="used_items_mean_score")
        if cfa_scores.shape[1] > 0:
            excel_matrix(_matrix_triangle(cfa_scores.corr()), writer, sheetname="used_items_cfa_score")
    return {"efa_cfa_models": models_table, "cfa_scores": cfa_scores, "plots_scree": {},
            "cfa_coefficients": coefficients_df}
