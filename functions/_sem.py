# -*- coding: utf-8 -*-
"""
A small structural equation modelling engine that reproduces lavaan for the
models rwf fits: confirmatory factor analysis and structural models with
latent variables, estimated by maximum likelihood on complete data.

It reads lavaan model syntax (=~, ~, ~~, fixed values such as 0.8*x, NA*x to
free a parameter, ";" or new lines between statements, # comments) and adds
lavaan's automatic parameters: the first loading of each factor fixed to 1
(or the factor variances fixed to 1 with std_lv=True), residual variances of
every observed and latent variable, and covariances among exogenous latent
variables. Estimates minimise lavaan's ML discrepancy with the sample
covariance computed with divisor N; standard errors come from the expected
information, as lavaan's defaults.

Not supported (they raise NotImplementedError): observed exogenous
regressors, mean structures, missing-data estimation, multiple groups,
labels and constraints.
"""
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

_OPERATORS = ("=~", "~~", "~")


def _parse(model):
    """Statements of lavaan syntax as (lhs, op, rhs, fixed value or None, free modifier)."""
    statements = []
    for raw in re.split(r"[;\n]", model):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        op = next((o for o in _OPERATORS if o in line), None)
        if op is None:
            raise ValueError(f"cannot read the model statement {line!r}")
        lhs, rhs = (part.strip() for part in line.split(op, 1))
        for term in rhs.split("+"):
            term = term.strip()
            fixed, free = None, False
            if "*" in term:
                modifier, term = (part.strip() for part in term.split("*", 1))
                if modifier.upper() == "NA":
                    free = True
                else:
                    try:
                        fixed = float(modifier)
                    except ValueError as error:
                        raise NotImplementedError("parameter labels are not supported") from error
            for name in (lhs, term):
                if not re.match(r"^[A-Za-z.][A-Za-z0-9._]*$", name):
                    raise ValueError(f"invalid variable name {name!r}")
            statements.append((lhs, op, term, fixed, free))
    return statements


class Parameter:
    def __init__(self, lhs, op, rhs, free, value, user):
        self.lhs, self.op, self.rhs, self.free, self.value, self.user = lhs, op, rhs, free, value, user


class SEMFit:
    """A fitted model: parameter table, implied covariance, sample statistics and helpers."""

    def __init__(self, model, data, std_lv=False):
        statements = _parse(model)
        self.model_syntax = model
        self.std_lv = std_lv
        latent = list(dict.fromkeys(s[0] for s in statements if s[1] == "=~"))
        names = list(dict.fromkeys(n for s in statements for n in (s[0], s[2])))
        observed = [n for n in names if n not in latent]
        # lavaan orders observed variables by first appearance as indicators, then elsewhere
        indicators = [s[2] for s in statements if s[1] == "=~"]
        observed = list(dict.fromkeys(indicators + observed))
        regressions = [s for s in statements if s[1] == "~"]
        for lhs, _, rhs, _, _ in regressions:
            if lhs not in latent or rhs not in latent:
                raise NotImplementedError("regressions are supported among latent variables only")
        self.latent, self.observed = latent, observed
        endogenous = list(dict.fromkeys(s[0] for s in regressions))
        self.endogenous = endogenous

        table = []
        first_indicator = {}
        for lhs, op, rhs, fixed, free in statements:
            if op == "=~" and lhs not in first_indicator:
                first_indicator[lhs] = rhs
                if fixed is None and not free and not std_lv:
                    fixed = 1.0
            table.append(Parameter(lhs, op, rhs, fixed is None, 0.0 if fixed is None else fixed, True))
        existing = {(p.lhs, p.op, p.rhs) for p in table} | {(p.rhs, p.op, p.lhs) for p in table if p.op == "~~"}
        for name in observed:
            if (name, "~~", name) not in existing:
                table.append(Parameter(name, "~~", name, True, 0.0, False))
        for name in latent:
            if (name, "~~", name) not in existing:
                fixed_one = std_lv and name not in endogenous
                table.append(Parameter(name, "~~", name, not fixed_one, 1.0 if fixed_one else 0.0, False))
        exogenous = [n for n in latent if n not in endogenous]
        for i, a in enumerate(exogenous):
            for b in exogenous[i + 1:]:
                if (a, "~~", b) not in existing:
                    table.append(Parameter(a, "~~", b, True, 0.0, False))
        self.table = table

        missing = [v for v in observed if v not in data.columns]
        if missing:
            raise ValueError("data has no column " + ", ".join(missing))
        values = data[observed].apply(pd.to_numeric, errors="coerce").dropna()
        self.data = values
        self.n = len(values)
        x = values.to_numpy(dtype=float)
        centred = x - x.mean(axis=0)
        self.sample_cov = centred.T @ centred / self.n
        self.p, self.m = len(observed), len(latent)
        self._index = {"ov": {v: i for i, v in enumerate(observed)}, "lv": {v: i for i, v in enumerate(latent)}}
        self.free = [p for p in table if p.free]
        self._start()
        self._estimate()

    # ------------------------------------------------------------------ matrices
    def _matrices(self, theta=None):
        if theta is not None:
            for parameter, value in zip(self.free, theta):
                parameter.value = value
        ov, lv = self._index["ov"], self._index["lv"]
        lam = np.zeros((self.p, self.m))
        th = np.zeros((self.p, self.p))
        psi = np.zeros((self.m, self.m))
        beta = np.zeros((self.m, self.m))
        for prm in self.table:
            if prm.op == "=~":
                lam[ov[prm.rhs], lv[prm.lhs]] = prm.value
            elif prm.op == "~":
                beta[lv[prm.lhs], lv[prm.rhs]] = prm.value
            elif prm.lhs in ov:
                th[ov[prm.lhs], ov[prm.rhs]] = th[ov[prm.rhs], ov[prm.lhs]] = prm.value
            else:
                psi[lv[prm.lhs], lv[prm.rhs]] = psi[lv[prm.rhs], lv[prm.lhs]] = prm.value
        return lam, th, psi, beta

    def implied(self, theta=None):
        lam, th, psi, beta = self._matrices(theta)
        c = np.linalg.inv(np.eye(self.m) - beta)
        a = c @ psi @ c.T
        return lam @ a @ lam.T + th

    def _derivatives(self):
        """dSigma/dtheta for every free parameter (list of p x p matrices)."""
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        a = c @ psi @ c.T
        ov, lv = self._index["ov"], self._index["lv"]
        result = []
        for prm in self.free:
            d = np.zeros((self.p, self.p))
            if prm.op == "=~":
                i, j = ov[prm.rhs], lv[prm.lhs]
                e = np.zeros((self.p, self.m))
                e[i, j] = 1
                d = e @ a @ lam.T + lam @ a @ e.T
            elif prm.op == "~":
                i, j = lv[prm.lhs], lv[prm.rhs]
                e = np.zeros((self.m, self.m))
                e[i, j] = 1
                dc = c @ e @ c
                d = lam @ (dc @ psi @ c.T + c @ psi @ dc.T) @ lam.T
            elif prm.lhs in ov:
                i, j = ov[prm.lhs], ov[prm.rhs]
                d[i, j] = d[j, i] = 1
            else:
                i, j = lv[prm.lhs], lv[prm.rhs]
                e = np.zeros((self.m, self.m))
                e[i, j] = e[j, i] = 1
                d = lam @ c @ e @ c.T @ lam.T
            result.append(d)
        return result

    # ------------------------------------------------------------------ estimation
    def _start(self):
        """Simple starting values: residual variances half the observed, latent variances 0.05."""
        s = self.sample_cov
        ov = self._index["ov"]
        marker = {}
        for prm in self.table:
            if prm.op == "=~" and not prm.free:
                marker.setdefault(prm.lhs, prm.rhs)
        for prm in self.free:
            if prm.op == "=~":
                ref = marker.get(prm.lhs)
                if ref is not None and ref != prm.rhs:
                    i, j = ov[prm.rhs], ov[ref]
                    prm.value = s[i, j] / s[j, j] if s[j, j] > 0 else 1.0
                else:
                    prm.value = np.sqrt(np.mean(np.diag(s))) * 0.7
            elif prm.op == "~~" and prm.lhs == prm.rhs:
                prm.value = 0.5 * s[ov[prm.lhs], ov[prm.lhs]] if prm.lhs in ov else 0.05
            else:
                prm.value = 0.0

    def _discrepancy(self, theta):
        sigma = self.implied(theta)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            return np.inf
        return logdet + np.trace(self.sample_cov @ np.linalg.inv(sigma)) - self._logdet_s - self.p

    def _gradient(self, theta):
        sigma = self.implied(theta)
        inv = np.linalg.inv(sigma)
        w = inv @ (sigma - self.sample_cov) @ inv
        return np.array([np.sum(w * d) for d in self._derivatives()])

    def _estimate(self):
        self._logdet_s = np.linalg.slogdet(self.sample_cov)[1]
        theta0 = np.array([p.value for p in self.free])
        fit = optimize.minimize(self._discrepancy, theta0, jac=self._gradient, method="BFGS",
                                options={"gtol": 1e-10, "maxiter": 10000})
        if not np.all(np.isfinite(fit.x)):
            raise RuntimeError("the model did not converge")
        polish = optimize.minimize(self._discrepancy, fit.x, jac=self._gradient, method="L-BFGS-B",
                                   options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 10000})
        theta = polish.x if polish.fun <= fit.fun else fit.x
        self._matrices(theta)
        self.converged = bool(fit.success or polish.success)
        self.fmin = 0.5 * self._discrepancy(theta)
        self.sigma = self.implied()
        derivatives = self._derivatives()
        inv = np.linalg.inv(self.sigma)
        q = len(derivatives)
        information = np.empty((q, q))
        products = [inv @ d for d in derivatives]
        for k in range(q):
            for l in range(k, q):
                information[k, l] = information[l, k] = 0.5 * np.sum(products[k] * products[l].T)
        self.vcov = np.linalg.inv(self.n * information)
        for prm, se in zip(self.free, np.sqrt(np.diag(self.vcov))):
            prm.se = se

    # ------------------------------------------------------------------ results
    def latent_variances(self):
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        return np.diag(c @ psi @ c.T)

    def _standardized(self, prm):
        ov, lv = self._index["ov"], self._index["lv"]
        var_lv = self.latent_variances()
        var_ov = np.diag(self.sigma)

        def sd(name, total):
            if name in lv:
                return np.sqrt(var_lv[lv[name]])
            return np.sqrt(var_ov[ov[name]]) if total else 1.0

        value = prm.value
        if prm.op == "=~":
            std_lv = value * sd(prm.lhs, True)
            return std_lv, std_lv / sd(prm.rhs, True)
        if prm.op == "~":
            std = value * sd(prm.rhs, True) / sd(prm.lhs, True)
            return std, std
        if prm.lhs == prm.rhs:
            std_lv = value / (sd(prm.lhs, False) * sd(prm.rhs, False))
            return std_lv, value / (sd(prm.lhs, True) * sd(prm.rhs, True))
        # a (residual) covariance is standardized by the two residual variances, as lavaan does
        lam, th, psi, beta = self._matrices()

        def residual_sd(name):
            return np.sqrt(psi[lv[name], lv[name]] if name in lv else th[ov[name], ov[name]])

        std_all = value / (residual_sd(prm.lhs) * residual_sd(prm.rhs))
        std_lv = std_all if prm.lhs in lv else value
        return std_lv, std_all

    def parameter_estimates(self, level=0.95):
        """lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE) without std.nox."""
        z_crit = stats.norm.ppf(1 - (1 - level) / 2)
        rows = []
        for prm in self.table:
            se = getattr(prm, "se", 0.0) if prm.free else 0.0
            z = prm.value / se if prm.free else np.nan
            p_value = 2 * stats.norm.sf(abs(z)) if prm.free else np.nan
            std_lv, std_all = self._standardized(prm)
            rows.append({"lhs": prm.lhs, "op": prm.op, "rhs": prm.rhs, "est": prm.value, "se": se, "z": z,
                         "pvalue": p_value, "ci.lower": prm.value - z_crit * se, "ci.upper": prm.value + z_crit * se,
                         "std.lv": std_lv, "std.all": std_all})
        for name, value in self.rsquare().items():
            rows.append({"lhs": name, "op": "r2", "rhs": name, "est": value, "se": np.nan, "z": np.nan,
                         "pvalue": np.nan, "ci.lower": np.nan, "ci.upper": np.nan, "std.lv": np.nan,
                         "std.all": np.nan})
        return pd.DataFrame(rows)

    def fit_measures(self):
        """lavaan::fitMeasures(fit) for ML on complete data, in lavaan's order and names."""
        n, p = self.n, self.p
        s, sigma = self.sample_cov, self.sigma
        npar = len(self.free)
        moments = p * (p + 1) / 2
        df = moments - npar
        chisq = 2 * n * self.fmin
        baseline_f = np.sum(np.log(np.diag(s))) - self._logdet_s
        baseline_chisq, baseline_df = n * baseline_f, moments - p
        out = {"npar": npar, "fmin": self.fmin, "chisq": chisq, "df": df,
               "pvalue": stats.chi2.sf(chisq, df) if df > 0 else np.nan,
               "baseline.chisq": baseline_chisq, "baseline.df": baseline_df,
               "baseline.pvalue": stats.chi2.sf(baseline_chisq, baseline_df)}
        d, db = max(chisq - df, 0), max(baseline_chisq - baseline_df, 0)
        out["cfi"] = 1 - d / max(db, d) if max(db, d) > 0 else 1.0
        tli = ((baseline_chisq / baseline_df) - (chisq / df)) / ((baseline_chisq / baseline_df) - 1) if df > 0 else 1.0
        out["tli"] = out["nnfi"] = tli
        out["rfi"] = ((baseline_chisq / baseline_df) - (chisq / df)) / (baseline_chisq / baseline_df) if df > 0 else np.nan
        out["nfi"] = (baseline_chisq - chisq) / baseline_chisq
        out["pnfi"] = (df / baseline_df) * out["nfi"]
        out["ifi"] = (baseline_chisq - chisq) / (baseline_chisq - df)
        out["rni"] = 1 - (chisq - df) / (baseline_chisq - baseline_df)
        sign, logdet_sigma = np.linalg.slogdet(sigma)
        inv = np.linalg.inv(sigma)
        logl = -n / 2 * (p * np.log(2 * np.pi) + logdet_sigma + np.trace(s @ inv))
        unrestricted = -n / 2 * (p * np.log(2 * np.pi) + self._logdet_s + p)
        out["logl"], out["unrestricted.logl"] = logl, unrestricted
        out["aic"] = -2 * logl + 2 * npar
        out["bic"] = -2 * logl + npar * np.log(n)
        out["ntotal"] = n
        out["bic2"] = -2 * logl + npar * np.log((n + 2) / 24)
        out["rmsea"] = np.sqrt(max(chisq / n / df - 1 / n, 0)) if df > 0 else 0.0

        def ncp_at(probability):
            f = lambda lam: stats.ncx2.cdf(chisq, df, lam) - probability if lam > 0 else stats.chi2.cdf(chisq, df) - probability
            if f(0) < 0:
                return 0.0
            upper = max(1.0, chisq)
            while f(upper) > 0:
                upper *= 2
            return optimize.brentq(f, 0, upper, xtol=1e-12)

        out["rmsea.ci.lower"] = np.sqrt(ncp_at(0.95) / (n * df))
        out["rmsea.ci.upper"] = np.sqrt(ncp_at(0.05) / (n * df))
        out["rmsea.ci.level"] = 0.90
        out["rmsea.pvalue"] = stats.ncx2.sf(chisq, df, n * df * 0.05 ** 2)
        out["rmsea.close.h0"] = 0.05
        out["rmsea.notclose.pvalue"] = stats.ncx2.cdf(chisq, df, n * df * 0.08 ** 2)
        out["rmsea.notclose.h0"] = 0.08
        lower = np.tril_indices(p)
        residual = s - sigma
        out["rmr"] = out["rmr_nomean"] = np.sqrt(np.mean(residual[lower] ** 2))
        scale = np.sqrt(np.outer(np.diag(s), np.diag(s)))
        bentler = np.sqrt(np.mean((residual / scale)[lower] ** 2))
        out["srmr"] = out["srmr_bentler"] = out["srmr_bentler_nomean"] = bentler
        corr_s = s / scale
        corr_sigma = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
        strict_lower = np.tril_indices(p, -1)
        out["crmr"] = out["crmr_nomean"] = np.sqrt(np.mean((corr_s - corr_sigma)[strict_lower] ** 2))
        mplus = (corr_s - corr_sigma)
        mplus_diag = (np.diag(s) - np.diag(sigma)) / np.diag(s)
        mplus = mplus.copy()
        np.fill_diagonal(mplus, mplus_diag)
        out["srmr_mplus"] = out["srmr_mplus_nomean"] = np.sqrt(np.mean(mplus[lower] ** 2))
        # lavaan 0.7's gfi is the population estimate n p / (n p + 2 x2 - 2 df) with Browne's
        # reweighted least squares statistic x2 = N/2 tr((Sigma^-1 (S - Sigma))^2); the classic
        # (LISREL) GFI is gfi_lisrel
        np_ = n * p
        weighted = inv @ (s - sigma)
        x2_rls = n / 2 * np.trace(weighted @ weighted)

        def ncp_rls(probability):
            f = lambda lam: stats.ncx2.cdf(x2_rls, df, lam) - probability if lam > 0 else stats.chi2.cdf(x2_rls, df) - probability
            if f(0) < 0:
                return 0.0
            upper = max(1.0, x2_rls)
            while f(upper) > 0:
                upper *= 2
            return optimize.brentq(f, 0, upper, xtol=1e-12)

        out["gfi"] = min(np_ / (np_ + 2 * x2_rls - 2 * df), 1.0)
        out["gfi.ci.lower"] = min(np_ / (np_ + 2 * ncp_rls(0.05)), 1.0)
        out["gfi.ci.upper"] = min(np_ / (np_ + 2 * ncp_rls(0.95)), 1.0)
        out["gfi.ci.level"] = 0.90
        out["cn_05"] = stats.chi2.ppf(0.95, df) / (chisq / n) + 1 if chisq > 0 else np.nan
        out["cn_01"] = stats.chi2.ppf(0.99, df) / (chisq / n) + 1 if chisq > 0 else np.nan
        product = inv @ s
        gfi_lisrel = 1 - np.trace((product - np.eye(p)) @ (product - np.eye(p))) / np.trace(product @ product)
        out["gfi_lisrel"] = gfi_lisrel
        out["agfi_lisrel"] = 1 - (moments / df) * (1 - gfi_lisrel)
        out["pgfi"] = (df / moments) * gfi_lisrel
        out["mfi"] = np.exp(-0.5 * (chisq - df) / n)
        out["ecvi"] = chisq / n + 2 * npar / n
        return out

    def rsquare(self):
        """Explained variance of every observed variable and every endogenous latent variable."""
        lam, th, psi, beta = self._matrices()
        result = {}
        for name, i in self._index["ov"].items():
            result[name] = 1 - th[i, i] / self.sigma[i, i]
        var_lv = self.latent_variances()
        for name in self.endogenous:
            i = self._index["lv"][name]
            result[name] = 1 - psi[i, i] / var_lv[i]
        return result


def sem(model, data, std_lv=False):
    """
    Fit a structural equation model given in lavaan syntax, like lavaan::sem
    (and lavaan::cfa, which has the same defaults for these models).

    Parameters:
    model (str): lavaan model syntax.
    data (pandas.DataFrame): Observed variables; rows with a missing value
        in a model variable are dropped (listwise deletion).
    std_lv (bool, optional): Fix the latent variances to 1 instead of the
        first loadings. R's std.lv. Defaults to False.

    Returns:
    SEMFit: The fitted model.

    Examples:
    >>> fit = sem("E =~ E1 + E2 + E3; N =~ N1 + N2 + N3", df_ocean)
    >>> fit.parameter_estimates()
    """
    return SEMFit(model, data, std_lv=std_lv)


cfa = sem
