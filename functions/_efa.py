# -*- coding: utf-8 -*-
"""
Maximum likelihood exploratory factor analysis that reproduces
psych::fa(x, nfactors, rotate = "oblimin", fm = "ml"): the factanal-style ML
extraction psych uses, GPArotation's oblique gradient projection rotation
(the Barzilai-Borwein version GPArotation uses by default), psych's sign
and order conventions, and psych's fit statistics (factor.stats: objective,
chi square, RMSEA with its confidence interval, TLI, CFI, rms). With missing
values the correlations are pairwise, as psych's default use = "pairwise".

Also fa_sort (psych::fa.sort) and model_loadings (rwf::model_loadings).
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


def _smc(r):
    """Squared multiple correlations, 1 - 1 / diag(R^-1)."""
    return 1 - 1 / np.diag(np.linalg.inv(r))


def _fa_objective(psi, s, nf):
    sc = np.diag(1 / np.sqrt(psi))
    values = np.linalg.eigvalsh(sc @ s @ sc)[::-1]
    e = values[nf:]
    return -(np.sum(np.log(e) - e) - nf + s.shape[0])


def _fa_gradient(psi, s, nf):
    sc = np.diag(1 / np.sqrt(psi))
    values, vectors = np.linalg.eigh(sc @ s @ sc)
    values, vectors = values[::-1], vectors[:, ::-1]
    load = vectors[:, :nf] * np.sqrt(np.maximum(values[:nf] - 1, 0))
    load = np.sqrt(psi)[:, None] * load
    g = load @ load.T + np.diag(psi) - s
    return np.diag(g) / psi ** 2


def _fa_loadings(psi, s, nf):
    sc = np.diag(1 / np.sqrt(psi))
    values, vectors = np.linalg.eigh(sc @ s @ sc)
    values, vectors = values[::-1], vectors[:, ::-1]
    load = vectors[:, :nf] * np.sqrt(np.maximum(values[:nf] - 1, 0))
    return np.sqrt(psi)[:, None] * load


def _minres_objective(psi, s, nf):
    """psych's fit.residuals for fm = "minres": squared lower-triangle residuals."""
    reduced = s.copy()
    np.fill_diagonal(reduced, 1 - psi)
    values, vectors = np.linalg.eigh(reduced)
    values, vectors = values[::-1], vectors[:, ::-1]
    values = np.where(values < np.finfo(float).eps, 100 * np.finfo(float).eps, values)
    load = vectors[:, :nf] * np.sqrt(values[:nf])
    residual = reduced - load @ load.T
    return np.sum(residual[np.tril_indices_from(residual, -1)] ** 2)


def _minres_gradient(psi, s, nf):
    """psych's FAgr.minres."""
    values, vectors = np.linalg.eigh(s - np.diag(psi))
    values, vectors = values[::-1], vectors[:, ::-1]
    load = vectors[:, :nf] * np.sqrt(np.maximum(values[:nf], 0))
    g = load @ load.T + np.diag(psi) - s
    return np.diag(g)


def _minres_loadings(psi, s, nf):
    """psych's FAout.wls."""
    reduced = s.copy()
    np.fill_diagonal(reduced, np.diag(s) - psi)
    values, vectors = np.linalg.eigh(reduced)
    values, vectors = values[::-1], vectors[:, ::-1]
    return vectors[:, :nf] * np.sqrt(np.maximum(values[:nf], 0))



def _oblimin(loadings):
    """GPArotation's vgQ.oblimin with gamma = 0 (quartimin)."""
    squared = loadings ** 2
    x = squared.sum(axis=1, keepdims=True) - squared
    return np.sum(squared * x) / 4, loadings * x


def gpf_oblique(a, eps=1e-5, maxit=2000, fwindow=10):
    """
    GPArotation::GPFoblq(A, method = "oblimin") with its default Barzilai-Borwein steps, from the
    identity: returns the rotated loadings and the factor correlations Phi.
    """
    k = a.shape[1]
    t = np.eye(k)
    alpha = 1.0
    t_prev = gp_prev = None
    t_inv = np.linalg.inv(t)
    loadings = a @ t_inv.T
    f, gq = _oblimin(loadings)
    g = -(loadings.T @ gq @ t_inv).T
    history = []
    for _ in range(maxit + 1):
        gp = g - t @ np.diag(np.sum(t * g, axis=0))
        s = np.sqrt(np.sum(gp ** 2))
        history.append(f)
        if s < eps:
            break
        if t_prev is not None:
            dt, dgp = t - t_prev, gp - gp_prev
            if np.sum(dgp ** 2) > 0:
                alpha = np.sum(dt ** 2) / abs(np.sum(dt * dgp))
                alpha = max(1e-10, min(alpha, 20))
        else:
            alpha = 2 * alpha
        target = max(history[-fwindow:])
        for _ in range(11):
            x = t - alpha * gp
            t_new = x / np.sqrt(np.sum(x ** 2, axis=0))
            t_new_inv = np.linalg.inv(t_new)
            loadings = a @ t_new_inv.T
            f_new, gq = _oblimin(loadings)
            if target - f_new > 0.5 * s ** 2 * alpha:
                break
            alpha /= 2
        t_prev, gp_prev = t, gp
        t, f = t_new, f_new
        g = -(loadings.T @ gq @ t_new_inv).T
    return loadings, t.T @ t


class FAResult:
    """Result of fa(): psych::fa's main elements, with Python attribute names."""


def fa(data, nfactors=1, rotate="oblimin", fm="ml", alpha=0.1):
    """
    Exploratory factor analysis, as psych::fa(data, nfactors, rotate = "oblimin",
    fm = "ml", residuals = TRUE).

    Parameters:
    data (pandas.DataFrame): Raw data (correlations are computed pairwise).
    nfactors (int, optional): Number of factors. Defaults to 1.
    rotate (str, optional): "oblimin" (default) or "none".
    fm (str, optional): Only "ml" (maximum likelihood) is supported.
    alpha (float, optional): 1 minus the RMSEA confidence level. Defaults to 0.1.

    Returns:
    FAResult: loadings (pattern), Structure, Phi (None for one factor or no
    rotation), residual, model, communality, uniquenesses, RMSEA (Series:
    RMSEA, lower, upper, confidence), rms, CFI, TLI, STATISTIC, dof,
    objective, null_chisq, null_dof, n_obs. Factors are ordered by the
    variance they explain, as in psych; with several factors their ML1, ML2,
    ... labels can be in a different order from psych's (the values and
    column order are the same).

    Examples:
    >>> model = fa(df_ocean[["E1", "E2", "E3", "E4", "E5"]], nfactors=1)
    >>> model.loadings
    """
    if fm not in ("ml", "minres"):
        raise NotImplementedError('fm must be "ml" or "minres"')
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    names = list(data.columns)
    r = data.corr().to_numpy()
    n = r.shape[0]
    # psych counts the pairs of every two variables (used for rms) and takes n.obs as the number of rows
    mask = data.notna().to_numpy().astype(float)
    np_obs = mask.T @ mask
    n_obs = len(data)

    smc = _smc(r)
    upper = max(np.max(smc), 1)
    start = np.full(nfactors, 0.5) if (np.sum(smc) == nfactors and nfactors > 1) else 1 - smc
    scale = 0.01  # R's optim(..., control = list(parscale = 0.01))
    if fm == "minres":
        objective, gradient, extract = _minres_objective, _minres_gradient, _minres_loadings
        start = 1 - smc
    else:
        objective, gradient, extract = _fa_objective, _fa_gradient, _fa_loadings
    result = optimize.minimize(lambda z: objective(z * scale, r, nfactors), start / scale,
                               jac=lambda z: gradient(z * scale, r, nfactors) * scale, method="L-BFGS-B",
                               bounds=[(0.005 / scale, upper / scale)] * n,
                               options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
    psi = result.x * scale
    loadings = extract(psi, r, nfactors)
    unrotated_model = loadings @ loadings.T
    reduced = r.copy()
    np.fill_diagonal(reduced, np.diag(unrotated_model))
    values = np.linalg.eigvalsh(reduced)[::-1]
    if nfactors > 1:
        signs = np.sign(loadings.sum(axis=0))
        signs[signs == 0] = 1
        loadings = loadings * signs
    elif loadings.sum() < 0:
        loadings = -loadings
    phi = None
    if nfactors > 1 and rotate == "oblimin":
        loadings, phi = gpf_oblique(loadings)
    elif rotate not in ("oblimin", "none"):
        raise NotImplementedError('rotate must be "oblimin" or "none"')
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1
    loadings = loadings * signs
    if phi is not None:
        phi = np.diag(signs) @ phi @ np.diag(signs)
    prefix = "ML" if fm == "ml" else "MR"
    columns = [f"{prefix}{i + 1}" for i in range(nfactors)]
    if nfactors > 1:
        variance = np.diag(loadings.T @ loadings) if phi is None else np.diag(phi @ loadings.T @ loadings)
        order = np.argsort(-variance, kind="stable")
        loadings, columns = loadings[:, order], [columns[i] for i in order]
        if phi is not None:
            phi = phi[np.ix_(order, order)]

    out = FAResult()
    out.loadings = pd.DataFrame(loadings, index=names, columns=columns)
    out.Phi = None if phi is None else pd.DataFrame(phi, index=columns, columns=columns)
    model = loadings @ (np.eye(nfactors) if phi is None else phi) @ loadings.T
    out.model = pd.DataFrame(model, index=names, columns=names)
    out.Structure = pd.DataFrame(loadings @ (np.eye(nfactors) if phi is None else phi), index=names, columns=columns)
    out.communality = pd.Series(np.diag(model), index=names)
    out.uniquenesses = pd.Series(np.diag(r - model), index=names)
    out.r = pd.DataFrame(r, index=names, columns=names)
    out.n_obs = n_obs
    out.np_obs = pd.DataFrame(np_obs, index=names, columns=names)
    out.factors = nfactors
    out.fm = fm
    out.values = values
    out.e_values = np.linalg.eigvalsh(r)[::-1]
    out.communalities = pd.Series(1 - psi, index=names)
    _factor_stats(out, r, model, n, nfactors, n_obs, np_obs, alpha)
    out.complexity = pd.Series((loadings ** 2).sum(axis=1) ** 2 / (loadings ** 4).sum(axis=1), index=names)

    # factor scores by psych::fa's default, scores = "regression" (Thurstone): W = R^-1 F Phi
    phi_matrix = np.eye(nfactors) if phi is None else phi
    weights = np.linalg.solve(r, loadings @ phi_matrix)
    out.weights = pd.DataFrame(weights, index=names, columns=columns)
    standardized = (data - data.mean()) / data.std(ddof=1)
    out.scores = pd.DataFrame(standardized.to_numpy() @ weights, columns=columns, index=data.index)

    variance = np.diag(phi_matrix @ loadings.T @ loadings) if nfactors > 1 else np.array([np.sum(loadings ** 2)])
    total = np.sum(out.communality + out.uniquenesses)
    rows = {"SS loadings": variance, "Proportion Var": variance / total}
    if nfactors > 1:
        rows.update({"Cumulative Var": np.cumsum(variance / total), "Proportion Explained": variance / variance.sum(),
                     "Cumulative Proportion": np.cumsum(variance / variance.sum())})
    out.Vaccounted = pd.DataFrame(rows, index=columns).T
    return out


def _factor_stats(out, r, model, n, nfactors, n_obs, np_obs, alpha):
    """psych:::factor.stats for an ML solution."""
    residual = r - model
    out.residual = pd.DataFrame(residual, index=out.loadings.index, columns=out.loadings.index)
    dof = n * (n - 1) / 2 - n * nfactors + nfactors * (nfactors - 1) / 2
    out.dof = dof
    off = residual.copy()
    np.fill_diagonal(off, 0)
    rstar_off = np.sum(off ** 2 * np_obs)
    harmonic = stats.hmean(np_obs.ravel())
    out.rms = np.sqrt(rstar_off / (harmonic * n * (n - 1)))
    chi = rstar_off / 2
    out.chi, out.nh = chi, harmonic
    if dof > 0:
        out.crms = np.sqrt(rstar_off / (2 * harmonic * dof))
        out.EBIC = chi - dof * np.log(harmonic)
        out.ESABIC = chi - dof * np.log((harmonic + 2) / 24)
    else:
        out.crms = out.EBIC = out.ESABIC = np.nan
    r2_weighted = r * r * np_obs
    out.fit = 1 - np.sum(residual * residual) / np.sum(r * r)
    out.fit_off = 1 - rstar_off / (np.sum(r2_weighted) - np.trace(r2_weighted))
    fitted = model.copy()
    np.fill_diagonal(fitted, np.diag(r))
    m_inv_r = np.linalg.solve(fitted, r)
    objective = np.trace(m_inv_r) - np.log(np.linalg.det(m_inv_r)) - n
    out.objective = objective
    chisq = max(objective * ((n_obs - 1) - (2 * n + 5) / 6 - (2 * nfactors) / 3), 0)
    out.STATISTIC = chisq
    out.PVAL = stats.chi2.sf(chisq, dof) if dof > 0 else np.nan
    f0 = np.trace(r) - np.log(np.linalg.det(r)) - n
    mm = objective / dof if dof else np.nan
    m0 = f0 * 2 / (n * (n - 1))
    nm = (n_obs - 1) - (2 * n + 5) / 6 - (2 * nfactors) / 3
    out.null_model = f0
    out.null_dof = n * (n - 1) / 2
    out.null_chisq = f0 * ((n_obs - 1) - (2 * n + 5) / 6)
    out.TLI = (m0 - mm) / (m0 - 1 / nm)
    out.CFI = (max(out.null_chisq - out.null_dof, 0) - max(chisq - dof, 0)) / max(out.null_chisq - out.null_dof, 0)
    rmsea = lower = upper = np.nan
    if dof > 0:
        rmsea = np.sqrt(max(chisq / (dof * n_obs) - 1 / (n_obs - 1), 0))
        tail = alpha / 2
        bound = max(n_obs, chisq) + 2 * n_obs

        def ncp_root(probability):
            return optimize.brentq(lambda x: stats.ncx2.cdf(chisq, dof, x) - probability if x > 0
                                   else stats.chi2.cdf(chisq, dof) - probability, 0, bound, xtol=1e-12)

        upper = np.sqrt(ncp_root(tail) / (n_obs - 1) / dof) if stats.chi2.cdf(chisq, dof) > tail else 0.0
        lower = np.sqrt(ncp_root(1 - tail) / (n_obs - 1) / dof) if stats.chi2.cdf(chisq, dof) > 1 - tail else 0.0
        if upper < rmsea:
            upper = np.nan
        if lower > rmsea:
            lower = np.nan
    out.RMSEA = pd.Series({"RMSEA": rmsea, "lower": lower, "upper": upper, "confidence": 1 - alpha})
    out.BIC = chisq - dof * np.log(n_obs) if dof > 0 else np.nan
    out.SABIC = chisq - dof * np.log((n_obs + 2) / 24) if dof > 0 else np.nan


def fa_sort(loadings):
    """psych::fa.sort: items grouped by their largest absolute loading, largest first within a factor."""
    values = np.asarray(loadings, dtype=float)
    cluster = np.argmax(np.abs(values), axis=1)
    order = np.argsort(cluster, kind="stable")
    sorted_clusters = cluster[order]
    result = []
    first = 0
    for block, factor in enumerate(sorted(set(cluster))):
        members = order[sorted_clusters == factor]
        column = block  # psych indexes the column by the block number
        within = np.argsort(-np.abs(values[members, column]), kind="stable")
        result.extend(members[within])
        first += len(members)
    return loadings.iloc[result] if isinstance(loadings, pd.DataFrame) else values[result]


def model_loadings(model, cut=None, matrix_type="pattern", sort=True):
    """
    Loadings rounded to 2 decimals, sorted like psych::fa.sort, with values below cut blanked
    (port of rwf::model_loadings for fa() results).

    Parameters:
    model (FAResult): Result of fa().
    cut (float, optional): Smallest absolute loading shown. By default the critical loading for
        the sample size, as rwf.
    matrix_type (str, optional): "pattern" (default), "structure" or "all".
    sort (bool, optional): Sort the items. Defaults to True.

    Returns:
    pandas.DataFrame: Columns Matrix, variable and one per factor (text, as in rwf).
    """
    if cut is None:
        samples = np.array([50, 60, 70, 85, 100, 120, 150, 200, 250, 350])
        critical = np.array([0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])
        cut = critical[np.argmin(np.abs(samples - model.n_obs))]

    def table(matrix, label):
        rounded = matrix.round(2)
        if sort:
            rounded = fa_sort(rounded)
        text = rounded.map(lambda v: "" if abs(v) < cut else _r_text(v))
        return pd.concat([pd.DataFrame({"Matrix": label, "variable": rounded.index}), text.reset_index(drop=True)],
                         axis=1)

    pattern, structure = table(model.loadings, "Pattern"), table(model.Structure, "Structure")
    if matrix_type == "pattern":
        return pattern
    if matrix_type == "structure":
        return structure
    return pd.concat([pattern, structure], ignore_index=True)


def _r_text(value):
    """as.character() of a number rounded to 2 decimals: 0.8 -> "0.8", -0.76 -> "-0.76"."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
