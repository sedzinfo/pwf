# -*- coding: utf-8 -*-
"""
Exploratory factor analysis that reproduces psych::fa (psych 2.6.9 with
GPArotation 2026.8.2): psych's extraction methods (fm = "minres", "uls",
"ols", "wls", "gls", "ml", "minchi", "old.min", "pa" and "alpha"; their
objectives, gradients, starts and bounds, optimized by a port of R's
optim(method = "L-BFGS-B") because psych's wls, gls, minchi and old.min
use the ML gradient and so stop where R's optimizer stops, not at an
optimum), its rotations (stats::varimax, psych's promax/Promax and equamax,
and GPArotation's gradient projection rotations: Varimax, quartimax,
bentlerT, geominT, oblimin, quartimin, simplimax, geominQ, bentlerQ, with
GPArotation's Barzilai-Borwein steps), psych's sign and order conventions,
and psych's fit statistics (factor.stats: objective, chi square, RMSEA with
its confidence interval, TLI, CFI, rms). With missing values the
correlations are pairwise, as psych's default use = "pairwise".

psych::fa rotates from 20 starts (n.rotations = 20: the identity and 19
random orthogonal matrices from R's random numbers) and keeps the one with
the most loadings in the hyperplane; pwf rotates from the identity only.
The two agree when the starts reach the same solution (the usual case),
up to the rotation's tolerance (eps = 1e-5) and the order of the factors.
varimax, promax and equamax have no random starts in psych.

Not ported: fm = "minrank" (psych's glb.algebraic needs the Rcsdp
semidefinite programming solver), the target, bifactor, biquartimin,
cluster, varimin and special rotations, and score methods other than
psych::fa's default "regression".

Also fa_sort (psych::fa.sort) and model_loadings (rwf::model_loadings).
"""
import sys
import warnings as _warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    from ._rcompat import r_round, r_uniroot
    from ._lbfgsb import r_optim_lbfgsb
except ImportError:
    from _rcompat import r_round, r_uniroot
    from _lbfgsb import r_optim_lbfgsb

_FM_ALIASES = {"mle": "ml", "MLE": "ml", "ML": "ml"}
_FM_METHODS = ("pa", "alpha", "minrank", "wls", "gls", "minres", "minchi", "uls", "ml", "ols", "old.min")
# psych's column names (its second switch); "ols" has none: rwf's data frames show X1, X2, ...
_FM_PREFIX = {"alpha": "alpha", "wls": "WLS", "pa": "PA", "gls": "GLS", "ml": "ML", "minres": "MR", "minrank": "MRFA",
              "uls": "ULS", "old.min": "oldmin", "minchi": "MC", "ols": "X"}
_ORTHOGONAL = ("varimax", "Varimax", "quartimax", "bentlerT", "geominT", "equamax")
_OBLIQUE = ("oblimin", "quartimin", "simplimax", "geominQ", "bentlerQ", "promax", "Promax")
_NOT_PORTED = ("targetT", "TargetT", "targetQ", "TargetQ", "bifactor", "varimin", "specialT", "specialQ", "cluster",
               "biquartimin")


def _r_message(text):
    """R's message(): a line on stderr."""
    print(text, file=sys.stderr)


def _r_eigen(matrix):
    """eigen(x, symmetric = TRUE): values in decreasing order with their vectors."""
    values, vectors = np.linalg.eigh(matrix)
    return values[::-1], vectors[:, ::-1]


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


def _fit_residuals(psi, s, nf, s_inv, fm, np_obs):
    """psych's fit.residuals (inside fac): the objective of uls, wls, gls, minchi and old.min."""
    s = s.copy()
    np.fill_diagonal(s, 1 - psi)
    values, vectors = _r_eigen(s)
    values = np.where(values < np.finfo(float).eps, 100 * np.finfo(float).eps, values)
    load = vectors[:, :nf] * np.sqrt(values[:nf])
    residual = s - load @ load.T
    if fm == "wls":
        weights = 1 / np.diag(s_inv)  # sd.inv %*% residual^2 %*% sd.inv with sd.inv diagonal
        return np.sum(weights[:, None] * residual ** 2 * weights[None, :])
    if fm == "gls":
        return np.sum((s_inv @ residual) ** 2)
    if fm == "uls":
        return np.sum(residual ** 2)
    if fm == "minchi":
        weighted = residual ** 2 * np_obs
        np.fill_diagonal(weighted, 0)
        return np.sum(weighted)
    return np.sum(residual[np.tril_indices_from(residual, -1)] ** 2)  # minres, old.min


def _minres2_gradient(psi, s, nf, fm, np_obs):
    """psych's FAgr.minres2: the ML gradient, which psych uses for wls, gls, minchi and old.min."""
    sc = np.diag(1 / np.sqrt(psi))
    values, vectors = _r_eigen(sc @ s @ sc)
    load = vectors[:, :nf] * np.sqrt(np.maximum(values[:nf] - 1, 0))
    load = np.sqrt(psi)[:, None] * load
    g = load @ load.T + np.diag(psi) - s
    if fm == "minchi":
        g = g * np_obs
    return np.diag(g) / psi ** 2


def _ols_objective(psi, s, nf):
    """psych's FA.OLS (optimized with optim's numerical gradient)."""
    values, vectors = _r_eigen(s - np.diag(psi))
    d = np.maximum(values[:nf], 0)
    load = vectors[:, :nf] * np.sqrt(d)
    model = load @ load.T
    np.fill_diagonal(model, np.diag(s))
    return np.sum((s - model) ** 2) / 2


def _fit(r, nf, fm, np_obs):
    """psych's fit() inside fac: the uniquenesses (res$par) and unrotated loadings of an optimized method."""
    smc = _smc(r)
    upper = max(np.max(smc), 1)
    s_inv = np.linalg.inv(r) if fm in ("wls", "gls") else None
    gradient = None
    if fm == "ml":
        start = np.full(nf, 0.5) if (np.sum(smc) == nf and nf > 1) else np.diag(r) - smc
        objective, gradient = (lambda p: _fa_objective(p, r, nf)), (lambda p: _fa_gradient(p, r, nf))
    elif fm == "ols":
        start = np.diag(r) - smc
        objective = lambda p: _ols_objective(p, r, nf)  # noqa: E731
    elif fm in ("minres", "uls"):
        start = np.diag(r) - smc
        objective = lambda p: _fit_residuals(p, r, nf, None, fm, np_obs)  # noqa: E731
        gradient = lambda p: _minres_gradient(p, r, nf)  # noqa: E731
    else:  # wls, gls, minchi, old.min
        start = smc
        objective = lambda p: _fit_residuals(p, r, nf, s_inv, fm, np_obs)  # noqa: E731
        gradient = lambda p: _minres2_gradient(p, r, nf, fm, np_obs)  # noqa: E731
    result = r_optim_lbfgsb(objective, start, gr=gradient, lower=0.005, upper=upper,
                            parscale=np.full(len(start), 0.01))
    psi = np.asarray(result["par"], dtype=float)
    loadings = _fa_loadings(psi, r, nf) if fm in ("ml", "minchi") else _minres_loadings(psi, r, nf)
    return psi, loadings


def _principal_axis(r, r_mat, nf, min_err, max_iter, warn):
    """psych's fm = "pa" iterations: loadings, eigen values and the summed communality."""
    comm = err = np.sum(np.diag(r_mat))
    iteration = 1
    comm1 = np.nan
    while err > min_err:
        values, vectors = _r_eigen(r_mat)
        with np.errstate(invalid="ignore"):
            loadings = vectors[:, :nf] * np.sqrt(values[:nf])
        new = np.diag(loadings @ loadings.T).copy()
        comm1 = np.sum(new)
        np.fill_diagonal(r_mat, new)
        err = abs(comm - comm1)
        if np.isnan(err):
            _warnings.warn("imaginary eigen value condition encountered in fa\n Try again with SMC=FALSE \n exiting fa")
            break
        comm = comm1
        iteration += 1
        if iteration > max_iter:
            if warn:
                _r_message("maximum iteration exceeded")
            err = 0
    return loadings, values, comm1


def _alpha_factoring(r, r_mat, nf, min_err, max_iter, warn):
    """psych's fm = "alpha" iterations: loadings, psych's 'values' (sqrt(H2) * eigen values) and sum(H2)."""
    err = np.sum(np.diag(r_mat))
    iteration = 1
    h2 = np.diag(r_mat).copy()
    while err > min_err:
        scale = np.sqrt(1 / np.diag(r_mat))  # cov2cor(), with an exact unit diagonal
        r_mat = scale[:, None] * r_mat * scale[None, :]
        np.fill_diagonal(r_mat, 1)
        values, vectors = _r_eigen(r_mat)
        if nf == 1:
            # psych's diag(sqrt(values[1])) of one number is an identity matrix of that size (R's diag()), so
            # one factor runs only when 1 <= sqrt(value) < 2, with unscaled loadings, and fails otherwise
            if int(np.floor(np.sqrt(values[0]))) != 1:
                raise ValueError("non-conformable arguments (psych's alpha factoring of one factor: "
                                 "diag(sqrt(eigens$values[1])) is not a 1 x 1 matrix)")
            loadings = vectors[:, :1].copy()
        else:
            loadings = vectors[:, :nf] * np.sqrt(values[:nf])
        new_h2 = h2 * np.diag(loadings @ loadings.T)
        err = np.sum(np.abs(h2 - new_h2))
        r_mat = r.copy()
        np.fill_diagonal(r_mat, new_h2)
        h2 = new_h2
        iteration += 1
        if iteration > max_iter:
            if warn:
                _r_message("maximum iteration exceeded")
            err = 0
    return np.sqrt(h2)[:, None] * loadings, np.sqrt(h2) * values, np.sum(h2)


# GPArotation's rotation criteria (vgQ.*): each returns the criterion f and its gradient Gq
def _vgq_oblimin(loadings, gam=0):
    squared = loadings ** 2
    x = squared.sum(axis=1, keepdims=True) - squared
    if gam != 0:
        x = x - (gam / loadings.shape[0]) * x.sum(axis=0)
    return np.sum(squared * x) / 4, loadings * x


def _vgq_quartimin(loadings):
    k = loadings.shape[1]
    squared = loadings ** 2
    x = squared @ (np.ones((k, k)) - np.eye(k))
    return np.sum(squared * x) / 4, loadings * x


def _vgq_quartimax(loadings):
    return -np.sum(loadings ** 4) / 4, -loadings ** 3


def _vgq_varimax(loadings):
    squared = loadings ** 2
    centred = squared - squared.mean(axis=0)
    return -np.sum(centred ** 2) / 4, -loadings * centred


def _vgq_geomin(loadings, delta=0.01):
    k = loadings.shape[1]
    squared = loadings ** 2 + delta
    product = np.exp(np.log(squared).mean(axis=1))
    return np.sum(product), (2 / k) * (loadings / squared) * product[:, None]


def _vgq_bentler(loadings):
    squared = loadings ** 2
    m = squared.T @ squared
    dm = np.diag(m)
    f = -(np.linalg.slogdet(m)[1] - np.sum(np.log(dm))) / 4
    return f, -loadings * (squared @ (np.linalg.inv(m) - np.diag(1 / dm)))


def _vgq_simplimax(loadings, k=None):
    squared = loadings ** 2
    if k is None:
        k = loadings.shape[0] * (loadings.shape[1] - 1)
    flat = squared.ravel(order="F")
    chosen = np.zeros(flat.size, dtype=bool)
    chosen[np.argsort(flat, kind="stable")[:k]] = True  # R's order(): column-major, ties by position
    chosen = chosen.reshape(squared.shape, order="F")
    return np.sum(squared[chosen]), 2 * chosen * loadings


def _vgq_cf(loadings, kappa=0):
    squared = loadings ** 2
    row_rest = squared.sum(axis=1, keepdims=True) - squared
    column_rest = squared.sum(axis=0, keepdims=True) - squared
    f = (1 - kappa) * np.sum(squared * row_rest) / 4 + kappa * np.sum(squared * column_rest) / 4
    return f, (1 - kappa) * loadings * row_rest + kappa * loadings * column_rest


_VGQ = {"oblimin": _vgq_oblimin, "quartimin": _vgq_quartimin, "quartimax": _vgq_quartimax,
        "varimax": _vgq_varimax, "geomin": _vgq_geomin, "bentler": _vgq_bentler, "simplimax": _vgq_simplimax,
        "cf": _vgq_cf}


def gpf_oblique(a, method="oblimin", eps=1e-5, maxit=2000, fwindow=10, **method_args):
    """
    GPArotation::GPFoblq(A, method = method) with its default Barzilai-Borwein steps, from the
    identity: returns the rotated loadings and the factor correlations Phi.
    """
    criterion = _VGQ[method]
    k = a.shape[1]
    t = np.eye(k)
    alpha = 1.0
    t_prev = gp_prev = None
    t_inv = np.linalg.inv(t)
    loadings = a @ t_inv.T
    f, gq = criterion(loadings, **method_args)
    g = -(loadings.T @ gq @ t_inv).T
    history = []
    converged = False
    for _ in range(maxit + 1):
        gp = g - t @ np.diag(np.sum(t * g, axis=0))
        s = np.sqrt(np.sum(gp ** 2))
        history.append(f)
        if s < eps:
            converged = True
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
            f_new, gq = criterion(loadings, **method_args)
            if target - f_new > 0.5 * s ** 2 * alpha:
                break
            alpha /= 2
        t_prev, gp_prev = t, gp
        t, f = t_new, f_new
        g = -(loadings.T @ gq @ t_new_inv).T
    if not converged:
        _warnings.warn(f"Convergence not obtained in GPFoblq. {maxit} iterations used.")
    return loadings, t.T @ t


def gpf_orthogonal(a, method="varimax", eps=1e-5, maxit=2000, fwindow=10, **method_args):
    """
    GPArotation::GPForth(A, method = method) with its default Barzilai-Borwein steps, from the
    identity: returns the rotated loadings and the rotation matrix Th.
    """
    criterion = _VGQ[method]
    t = np.eye(a.shape[1])
    alpha = 1.0
    t_prev = gp_prev = None
    loadings = a @ t
    f, gq = criterion(loadings, **method_args)
    g = a.T @ gq
    history = []
    converged = False
    for _ in range(maxit + 1):
        m = t.T @ g
        gp = g - t @ ((m + m.T) / 2)
        s = np.sqrt(np.sum(gp ** 2))
        history.append(f)
        if s < eps:
            converged = True
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
            u, _, vt = np.linalg.svd(t - alpha * gp)
            t_new = u @ vt
            loadings = a @ t_new
            f_new, gq = criterion(loadings, **method_args)
            if target - f_new > 0.5 * s ** 2 * alpha:
                break
            alpha /= 2
        t_prev, gp_prev = t, gp
        t, f = t_new, f_new
        g = a.T @ gq
    if not converged:
        _warnings.warn(f"Convergence not obtained in GPForth. {maxit} iterations used.")
    return loadings, t


def r_varimax(x, normalize=True, eps=1e-5):
    """stats::varimax: Kaiser's varimax (with row normalization by default); returns loadings and rotmat."""
    if x.shape[1] < 2:
        return x, None
    if normalize:
        sc = np.sqrt(np.sum(x ** 2, axis=1))
        x = x / sc[:, None]
    p, nc = x.shape
    tt = np.eye(nc)
    d = 0.0
    for _ in range(1000):
        z = x @ tt
        b = x.T @ (z ** 3 - z @ np.diag(np.sum(z ** 2, axis=0)) / p)
        u, singular, vt = np.linalg.svd(b)
        tt = u @ vt
        d_past, d = d, np.sum(singular)
        if d < d_past * (1 + eps):
            break
    z = x @ tt
    if normalize:
        z = z * sc[:, None]
    return z, tt


def _promax(x, m=4):
    """psych::Promax: GPArotation's Varimax followed by the promax power target; returns loadings and Phi."""
    rotated, th = gpf_orthogonal(x, "varimax")
    q = rotated * np.abs(rotated) ** (m - 1)
    u = np.linalg.lstsq(rotated, q, rcond=None)[0]  # lm.fit(x, Q)$coefficients
    try:
        d = np.diag(np.linalg.inv(u.T @ u))
    except np.linalg.LinAlgError:
        _warnings.warn("Factors are exactly uncorrelated and the model produces a singular matrix. An approximation "
                      "is used")
        values, vectors = _r_eigen(u.T @ u)
        values = np.where(values < np.finfo(float).eps, 100 * np.finfo(float).eps, values)
        uu = vectors @ np.diag(values) @ vectors.T
        np.fill_diagonal(uu, 1)
        d = np.diag(np.linalg.inv(uu))
    u = u @ np.diag(np.sqrt(d))
    z = rotated @ u
    ui = np.linalg.inv(th @ u)
    return z, ui @ ui.T


def _rotate(loadings, rotate):
    """psych's rotation of unrotated loadings (fac/faRotations from the identity): loadings and Phi."""
    if rotate == "varimax":
        return r_varimax(loadings)[0], None
    if rotate == "Varimax":
        return gpf_orthogonal(loadings, "varimax")[0], None
    if rotate in ("quartimax", "bentlerT"):
        return gpf_orthogonal(loadings, {"quartimax": "quartimax", "bentlerT": "bentler"}[rotate])[0], None
    if rotate == "geominT":
        return gpf_orthogonal(loadings, "geomin", delta=0.01)[0], None
    if rotate == "equamax":  # psych's equamax: GPArotation::cfT(kappa = k / (2 p), maxit = 1000)
        kappa = loadings.shape[1] / (2 * loadings.shape[0])
        return gpf_orthogonal(loadings, "cf", maxit=1000, kappa=kappa)[0], None
    if rotate == "Promax":
        return _promax(loadings)
    if rotate == "promax":  # psych's kaiser(loadings, rotate = "Promax"): rows normalized, then Promax
        h2 = np.sqrt(np.diag(loadings @ loadings.T))
        rotated, phi = _promax(loadings / h2[:, None])
        return rotated * h2[:, None], phi
    method = {"oblimin": "oblimin", "quartimin": "quartimin", "simplimax": "simplimax", "geominQ": "geomin",
              "bentlerQ": "bentler"}[rotate]
    return gpf_oblique(loadings, method)


class FAResult:
    """Result of fa(): psych::fa's main elements, with Python attribute names."""


def fa(data, nfactors=1, rotate="oblimin", fm="minres", alpha=0.1, oblique_scores=False, min_err=0.001, max_iter=50,
       warnings=True):
    """
    Exploratory factor analysis, as psych::fa(data, nfactors, rotate, fm,
    oblique.scores, min.err, max.iter, residuals = TRUE).

    Parameters:
    data (pandas.DataFrame): Raw data (correlations are computed pairwise).
    nfactors (int, optional): Number of factors. Defaults to 1.
    rotate (str, optional): "oblimin" (default), "none", "varimax", "Varimax",
        "quartimax", "bentlerT", "geominT", "equamax", "promax", "Promax",
        "quartimin", "simplimax", "geominQ" or "bentlerQ", as psych.
    fm (str, optional): "minres" (default, as psych), "ml", "uls", "ols",
        "wls", "gls", "minchi", "old.min", "pa" or "alpha". "mle", "MLE" and "ML" mean "ml"; an unknown method is
        minres, with psych's message.
    alpha (float, optional): 1 minus the RMSEA confidence level. Defaults to 0.1.
    oblique_scores (bool, optional): psych's oblique.scores: True bases the
        score weights on the pattern loadings alone (R^-1 F), False (psych's
        default) on the structure (R^-1 F Phi). Defaults to False.
    min_err (float, optional): Convergence criterion of "pa" and "alpha". Defaults to 0.001.
    max_iter (int, optional): Iteration limit of "pa" and "alpha". Defaults to 50.
    warnings (bool, optional): psych's warnings: its "maximum iteration
        exceeded" and SMC messages. Defaults to True.

    Returns:
    FAResult: loadings (pattern), Structure, Phi (None for one factor or an
    orthogonal rotation), residual, model, communality, uniquenesses, RMSEA
    (Series: RMSEA, lower, upper, confidence), rms, CFI, TLI, STATISTIC, dof,
    objective, null_chisq, null_dof, n_obs. Factors are ordered by the
    variance they explain, as in psych, and labelled by method (ML1, MR1,
    PA1, ...; X1, X2, ... for "ols", whose columns psych leaves unnamed, as
    rwf's tables show them) in that order. psych's labels of the same columns depend on R's random
    number state (after set.seed(1) they are pwf's; set.seed(2) gives ML1,
    ML3, ML2): the values and column order are the same. communalities is
    psych's: 1 minus the optimized uniquenesses, or for "pa" and "alpha" the
    sum of the final communality estimates (one number, as psych).

    Examples:
    >>> model = fa(df_ocean[["E1", "E2", "E3", "E4", "E5"]], nfactors=1)
    >>> model.loadings
    >>> fa(df_ocean[["E1", "E2", "E3", "N1", "N2", "N3"]], nfactors=2, fm="pa", rotate="varimax").loadings
    """
    fm = _FM_ALIASES.get(fm, fm)
    if fm == "minrank":
        raise NotImplementedError('fm = "minrank" is not ported (psych\'s glb.algebraic needs the Rcsdp solver)')
    if fm not in _FM_METHODS:
        _r_message("factor method not specified correctly, minimum residual (unweighted least squares  used")
        fm = "minres"
    if nfactors > 1 and rotate != "none":
        if rotate in _NOT_PORTED:
            raise NotImplementedError(f'rotate = "{rotate}" is not ported')
        if rotate not in _ORTHOGONAL + _OBLIQUE:
            # psych prints "Specified rotation not found, rotate='none' used" for each start, then fails
            raise ValueError(f"Specified rotation not found: {rotate!r} (psych::fa stops with \"object 'rot.mat' "
                             "not found\")")
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    names = list(data.columns)
    r = data.corr().to_numpy()
    n = r.shape[0]
    # psych counts the pairs of every two variables (used for rms) and takes n.obs as the number of rows
    mask = data.notna().to_numpy().astype(float)
    np_obs = mask.T @ mask
    n_obs = len(data)

    e_values = np.linalg.eigvalsh(r)[::-1]
    if fm in ("pa", "alpha"):
        r_mat = r.copy()
        if nfactors <= n:
            np.fill_diagonal(r_mat, _smc(r))
        elif warnings:
            _r_message("In fa, too many factors requested for this number of variables to use SMC for communality "
                       "estimates, 1s are used instead")
        iterate = _principal_axis if fm == "pa" else _alpha_factoring
        loadings, values, communalities = iterate(r, r_mat, nfactors, min_err, max_iter, warnings)
    else:
        psi, loadings = _fit(r, nfactors, fm, np_obs)
        communalities = pd.Series(1 - psi, index=names)
        reduced = r.copy()
        np.fill_diagonal(reduced, np.diag(loadings @ loadings.T))
        values = np.linalg.eigvalsh(reduced)[::-1]
    if nfactors > 1:
        signs = np.sign(loadings.sum(axis=0))
        signs[signs == 0] = 1
        loadings = loadings * signs
    elif loadings.sum() < 0:
        loadings = -loadings
    loadings = np.where(loadings == 0, 1e-15, loadings)
    unrotated_model = loadings @ loadings.T
    phi = None
    if nfactors > 1 and rotate != "none":
        loadings, phi = _rotate(loadings, rotate)
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1
    loadings = loadings * signs
    if phi is not None:
        phi = np.diag(signs) @ phi @ np.diag(signs)
    columns = [f"{_FM_PREFIX[fm]}{i + 1}" for i in range(nfactors)]
    if nfactors > 1:
        variance = np.diag(loadings.T @ loadings) if phi is None else np.diag(phi @ loadings.T @ loadings)
        order = np.argsort(-variance, kind="stable")
        loadings, columns = loadings[:, order], [columns[i] for i in order]
        if phi is not None:
            phi = phi[np.ix_(order, order)]
    if fm == "ols":  # unnamed in psych (its sign flip by %*% drops "MR1"): data.frame() numbers them in order
        columns = [f"X{i + 1}" for i in range(nfactors)]

    out = FAResult()
    out.loadings = pd.DataFrame(loadings, index=names, columns=columns)
    out.Phi = None if phi is None else pd.DataFrame(phi, index=columns, columns=columns)
    model = unrotated_model
    out.model = pd.DataFrame(model, index=names, columns=names)
    out.Structure = pd.DataFrame(loadings @ (np.eye(nfactors) if phi is None else phi), index=names, columns=columns)
    out.communality = pd.Series(np.diag(model), index=names)
    if np.max(out.communality) > 1:
        _warnings.warn("An ultra-Heywood case was detected.  Examine the results carefully")
    out.uniquenesses = pd.Series(np.diag(r - model), index=names)
    out.r = pd.DataFrame(r, index=names, columns=names)
    out.n_obs = n_obs
    out.np_obs = pd.DataFrame(np_obs, index=names, columns=names)
    out.factors = nfactors
    out.fm = fm
    out.rotation = rotate
    out.values = values
    out.e_values = e_values
    out.communalities = communalities
    # psych's factor.stats works from the rotated loadings and Phi
    _factor_stats(out, r, loadings @ (np.eye(nfactors) if phi is None else phi) @ loadings.T, n, nfactors, n_obs,
                  np_obs, alpha)
    out.complexity = pd.Series((loadings ** 2).sum(axis=1) ** 2 / (loadings ** 4).sum(axis=1), index=names)

    # factor scores by psych::fa's default, scores = "regression" (Thurstone): W = R^-1 F Phi, or
    # R^-1 F with oblique.scores = TRUE (psych then passes Phi = NULL to factor.scores)
    score_phi = np.eye(nfactors) if (phi is None or oblique_scores) else phi
    weights = np.linalg.solve(r, loadings @ score_phi)
    out.weights = pd.DataFrame(weights, index=names, columns=columns)
    standardized = (data - data.mean()) / data.std(ddof=1)
    out.scores = pd.DataFrame(standardized.to_numpy() @ weights, columns=columns, index=data.index)

    phi_matrix = np.eye(nfactors) if phi is None else phi
    variance = np.diag(phi_matrix @ loadings.T @ loadings) if nfactors > 1 else np.array([np.sum(loadings ** 2)])
    total = np.sum(out.communality + out.uniquenesses)
    rows = {"SS loadings": variance, "Proportion Var": variance / total}
    if nfactors > 1:
        rows.update({"Cumulative Var": np.cumsum(variance / total), "Proportion Explained": variance / variance.sum(),
                     "Cumulative Proportion": np.cumsum(variance / variance.sum())})
    out.Vaccounted = pd.DataFrame(rows, index=columns).T
    return out


def _factor_stats(out, r, model, n, nfactors, n_obs, np_obs, alpha):
    """psych:::factor.stats (fa.stats) of a solution of any method."""
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
            # uniroot(..., c(0, max)) with R's default tol; psych's try() turns a failure into NA
            try:
                return r_uniroot(lambda x: (stats.ncx2.cdf(chisq, dof, x) if x > 0 else stats.chi2.cdf(chisq, dof))
                                 - probability, 0, bound)
            except ValueError:
                return np.nan

        upper = np.sqrt(ncp_root(tail) / (n_obs - 1) / dof) if stats.chi2.cdf(chisq, dof) > tail else 0.0
        if np.isnan(upper) and rmsea <= 0:
            upper = 0.0
        lower = np.sqrt(ncp_root(1 - tail) / (n_obs - 1) / dof) if stats.chi2.cdf(chisq, dof) > 1 - tail else 0.0
        if not np.isnan(upper) and upper < rmsea:
            upper = np.nan
        if not np.isnan(lower) and lower > rmsea:
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
        rounded = matrix.apply(lambda column: column.map(lambda v: r_round(v, 2)))  # R's round()
        if sort:
            rounded = fa_sort(rounded)
        # as R's ifelse(): text when a loading is blanked, otherwise the numbers themselves
        if (rounded.abs() < cut).any().any():
            text = rounded.map(lambda v: "" if abs(v) < cut else _r_text(v))
        else:
            text = rounded
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
