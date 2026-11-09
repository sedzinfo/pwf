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


def _missing_patterns(x):
    """Rows grouped by which variables are observed: (observed indices, n, mean, covariance with divisor n)."""
    observed_mask = ~np.isnan(x)
    keys, inverse = np.unique(observed_mask, axis=0, return_inverse=True)
    patterns = []
    for k, key in enumerate(keys):
        rows = x[inverse.ravel() == k][:, key]
        mean = rows.mean(axis=0)
        centred = rows - mean
        patterns.append((np.flatnonzero(key), len(rows), mean, centred.T @ centred / len(rows)))
    return patterns


def _em_saturated(x, patterns, tol=1e-12, max_iter=100000):
    """
    Maximum likelihood mean and covariance of incomplete normal data by EM: the unrestricted
    (H1) model lavaan compares a FIML model with.
    """
    p = x.shape[1]
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0))
    n = sum(n_k for _, n_k, _, _ in patterns)
    for _ in range(max_iter):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for observed, n_k, mean_k, cov_k in patterns:
            missing = np.setdiff1d(np.arange(p), observed)
            t1 = n_k * mean_k
            t2 = n_k * (cov_k + np.outer(mean_k, mean_k))
            if len(missing) == 0:
                sum_x += t1
                sum_xx += t2
                continue
            b = sigma[np.ix_(missing, observed)] @ np.linalg.inv(sigma[np.ix_(observed, observed)])
            a = mu[missing] - b @ mu[observed]
            conditional = sigma[np.ix_(missing, missing)] - b @ sigma[np.ix_(observed, missing)]
            bt1 = b @ t1
            sum_x[observed] += t1
            sum_x[missing] += n_k * a + bt1
            sum_xx[np.ix_(observed, observed)] += t2
            cross = np.outer(a, t1) + b @ t2
            sum_xx[np.ix_(missing, observed)] += cross
            sum_xx[np.ix_(observed, missing)] += cross.T
            sum_xx[np.ix_(missing, missing)] += (n_k * conditional + n_k * np.outer(a, a) + np.outer(a, bt1)
                                                 + np.outer(bt1, a) + b @ t2 @ b.T)
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)
        change = max(np.max(np.abs(new_mu - mu)), np.max(np.abs(new_sigma - sigma)))
        mu, sigma = new_mu, new_sigma
        if change < tol:
            break
    return mu, sigma


class Parameter:
    def __init__(self, lhs, op, rhs, free, value, user):
        self.lhs, self.op, self.rhs, self.free, self.value, self.user = lhs, op, rhs, free, value, user


class SEMFit:
    """A fitted model: parameter table, implied covariance, sample statistics and helpers."""

    def __init__(self, model, data, std_lv=False, missing="listwise", meanstructure=False):
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
        if missing not in ("listwise", "ml"):
            raise ValueError('missing must be "listwise" or "ml"')
        self.missing = missing
        self.meanstructure = meanstructure or missing == "ml"
        if self.meanstructure and missing != "ml":
            raise NotImplementedError("a mean structure is supported with missing='ml' only")
        if self.meanstructure:
            table += [Parameter(name, "~1", "", True, 0.0, False) for name in observed]
            table += [Parameter(name, "~1", "", False, 0.0, False) for name in latent]
        self.table = table

        absent = [v for v in observed if v not in data.columns]
        if absent:
            raise ValueError("data has no column " + ", ".join(absent))
        values = data[observed].apply(pd.to_numeric, errors="coerce")
        values = values.dropna() if missing == "listwise" else values.dropna(how="all")
        self.data = values
        self.n = len(values)
        self.p, self.m = len(observed), len(latent)
        self._index = {"ov": {v: i for i, v in enumerate(observed)}, "lv": {v: i for i, v in enumerate(latent)}}
        self.free = [p for p in table if p.free]
        x = values.to_numpy(dtype=float)
        if missing == "listwise":
            centred = x - x.mean(axis=0)
            self.sample_cov = centred.T @ centred / self.n
        else:
            self._patterns = _missing_patterns(x)
            self.sample_mean, self.sample_cov = _em_saturated(x, self._patterns)
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
            elif prm.op == "~1":
                continue
            elif prm.lhs in ov:
                th[ov[prm.lhs], ov[prm.rhs]] = th[ov[prm.rhs], ov[prm.lhs]] = prm.value
            else:
                psi[lv[prm.lhs], lv[prm.rhs]] = psi[lv[prm.rhs], lv[prm.lhs]] = prm.value
        return lam, th, psi, beta

    def implied_mean(self, theta=None):
        """Model-implied means: the observed intercepts (latent means are fixed at 0)."""
        if theta is not None:
            self._matrices(theta)
        mean = np.zeros(self.p)
        for prm in self.table:
            if prm.op == "~1" and prm.lhs in self._index["ov"]:
                mean[self._index["ov"][prm.lhs]] = prm.value
        return mean

    def implied(self, theta=None):
        lam, th, psi, beta = self._matrices(theta)
        c = np.linalg.inv(np.eye(self.m) - beta)
        a = c @ psi @ c.T
        return lam @ a @ lam.T + th

    def _derivatives(self, parameters=None):
        """dSigma/dtheta (p x p) for each (op, lhs, rhs) parameter; by default every free parameter."""
        if parameters is None:
            parameters = [(prm.op, prm.lhs, prm.rhs) for prm in self.free]
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        a = c @ psi @ c.T
        ov, lv = self._index["ov"], self._index["lv"]
        result = []
        for op, lhs, rhs in parameters:
            d = np.zeros((self.p, self.p))
            if op == "~1":
                pass  # intercepts move the means, not the covariances
            elif op == "=~":
                i, j = ov[rhs], lv[lhs]
                e = np.zeros((self.p, self.m))
                e[i, j] = 1
                d = e @ a @ lam.T + lam @ a @ e.T
            elif op == "~":
                i, j = lv[lhs], lv[rhs]
                e = np.zeros((self.m, self.m))
                e[i, j] = 1
                dc = c @ e @ c
                d = lam @ (dc @ psi @ c.T + c @ psi @ dc.T) @ lam.T
            elif lhs in ov:
                i, j = ov[lhs], ov[rhs]
                d[i, j] = d[j, i] = 1
            else:
                i, j = lv[lhs], lv[rhs]
                e = np.zeros((self.m, self.m))
                e[i, j] = e[j, i] = 1
                d = lam @ c @ e @ c.T @ lam.T
            result.append(d)
        return result

    def _mean_derivatives(self, parameters):
        """dmu/dtheta (length p) for each (op, lhs, rhs) parameter."""
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        result = []
        for op, lhs, rhs in parameters:
            d = np.zeros(self.p)
            if op == "~1" and lhs in self._index["ov"]:
                d[self._index["ov"][lhs]] = 1.0
            elif op == "~1":
                d = lam @ c[:, self._index["lv"][lhs]]
            result.append(d)
        return result

    def _expected_information(self, parameters):
        """
        Expected information per observation for the given parameters: the complete-data formula,
        or with missing data its sum over the patterns (means included), as lavaan computes it.
        """
        d_sigma = self._derivatives(parameters)
        if self.missing != "ml":
            return self._information(d_sigma)
        d_mu = self._mean_derivatives(parameters)
        q = len(parameters)
        information = np.zeros((q, q))
        for observed, n_k, _, _ in self._patterns:
            inv = np.linalg.inv(self.sigma[np.ix_(observed, observed)])
            products = [inv @ d[np.ix_(observed, observed)] for d in d_sigma]
            means = [inv @ m[observed] for m in d_mu]
            for k in range(q):
                for l in range(k, q):
                    value = 0.5 * np.sum(products[k] * products[l].T) + d_mu[k][observed] @ means[l]
                    information[k, l] += n_k * value
                    if l != k:
                        information[l, k] += n_k * value
        return information / self.n

    def _score(self, parameters):
        """Derivative of -logl / N (lavaan's fmin for complete data) for the given parameters."""
        d_sigma = self._derivatives(parameters)
        if self.missing != "ml":
            inv = np.linalg.inv(self.sigma)
            w = inv @ (self.sigma - self.sample_cov) @ inv
            return np.array([0.5 * np.sum(w * d) for d in d_sigma])
        d_mu = self._mean_derivatives(parameters)
        mu = self.implied_mean()
        grad_sigma, grad_mu = np.zeros((self.p, self.p)), np.zeros(self.p)
        for observed, n_k, mean_k, cov_k in self._patterns:
            inv = np.linalg.inv(self.sigma[np.ix_(observed, observed)])
            diff = mean_k - mu[observed]
            grad_sigma[np.ix_(observed, observed)] += 0.5 * n_k * (inv - inv @ (cov_k + np.outer(diff, diff)) @ inv)
            grad_mu[observed] -= n_k * inv @ diff
        return np.array([np.sum(grad_sigma * ds) + grad_mu @ dm for ds, dm in zip(d_sigma, d_mu)]) / self.n

    def _information(self, derivatives):
        """Expected information per observation, 0.5 tr(Sigma^-1 dS_k Sigma^-1 dS_l)."""
        inv = np.linalg.inv(self.sigma)
        products = [inv @ d for d in derivatives]
        q = len(products)
        information = np.empty((q, q))
        for k in range(q):
            for l in range(k, q):
                information[k, l] = information[l, k] = 0.5 * np.sum(products[k] * products[l].T)
        return information

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
            elif prm.op == "~1":
                prm.value = self.sample_mean[ov[prm.lhs]]
            else:
                prm.value = 0.0

    # ------------------------------------------------------------------ FIML
    def _fiml_logl(self, theta=None):
        """Casewise normal log-likelihood summed over the missing-data patterns."""
        sigma, mu = self.implied(theta), self.implied_mean()
        total = 0.0
        for observed, n_k, mean_k, cov_k in self._patterns:
            sub = sigma[np.ix_(observed, observed)]
            sign, logdet = np.linalg.slogdet(sub)
            if sign <= 0:
                return -np.inf
            diff = mean_k - mu[observed]
            inv = np.linalg.inv(sub)
            total -= 0.5 * n_k * (len(observed) * np.log(2 * np.pi) + logdet + np.sum(inv * cov_k)
                                  + diff @ inv @ diff)
        return total

    def _fiml_objective(self, theta):
        value = -self._fiml_logl(theta) / self.n
        return value if np.isfinite(value) else np.inf

    def _fiml_gradient(self, theta):
        sigma, mu = self.implied(theta), self.implied_mean()
        d_sigma = np.zeros((self.p, self.p))
        d_mu = np.zeros(self.p)
        for observed, n_k, mean_k, cov_k in self._patterns:
            inv = np.linalg.inv(sigma[np.ix_(observed, observed)])
            diff = mean_k - mu[observed]
            g = inv - inv @ (cov_k + np.outer(diff, diff)) @ inv
            d_sigma[np.ix_(observed, observed)] += 0.5 * n_k * g
            d_mu[observed] -= n_k * inv @ diff
        derivatives = self._derivatives()
        grad = np.array([np.sum(d_sigma * d) for d in derivatives])
        ov = self._index["ov"]
        for k, prm in enumerate(self.free):
            if prm.op == "~1":
                grad[k] = d_mu[ov[prm.lhs]]
        return grad / self.n

    def _fiml_hessian(self, theta):
        """Observed information: numerical derivative of the analytic gradient of -logl."""
        q = len(theta)
        hessian = np.empty((q, q))
        for j in range(q):
            step = 1e-5 * max(1.0, abs(theta[j]))
            up, down = theta.copy(), theta.copy()
            up[j] += step
            down[j] -= step
            hessian[:, j] = (self._fiml_gradient(up) - self._fiml_gradient(down)) / (2 * step)
        self._matrices(theta)
        return self.n * (hessian + hessian.T) / 2

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
        if self.missing == "ml":
            objective, gradient = self._fiml_objective, self._fiml_gradient
        else:
            objective, gradient = self._discrepancy, self._gradient
        theta0 = np.array([p.value for p in self.free])
        fit = optimize.minimize(objective, theta0, jac=gradient, method="BFGS",
                                options={"gtol": 1e-10, "maxiter": 10000})
        if not np.all(np.isfinite(fit.x)):
            raise RuntimeError("the model did not converge")
        polish = optimize.minimize(objective, fit.x, jac=gradient, method="L-BFGS-B",
                                   options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 10000})
        theta = polish.x if polish.fun <= fit.fun else fit.x
        self._matrices(theta)
        self.converged = bool(fit.success or polish.success)
        self.sigma = self.implied()
        if self.missing == "ml":
            self.logl = self._fiml_logl(theta)
            self.vcov = np.linalg.inv(self._fiml_hessian(theta))
        else:
            self.fmin = 0.5 * self._discrepancy(theta)
            self.vcov = np.linalg.inv(self.n * self._information(self._derivatives()))
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
        if prm.op == "~1":
            std_lv = value / sd(prm.lhs, False)
            return std_lv, value / sd(prm.lhs, True)
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
        moments = p * (p + 1) / 2 + (p if self.meanstructure else 0)
        df = moments - npar
        if self.missing == "ml":
            unrestricted = self._pattern_logl(self.sigma_h1(), self.sample_mean)
            chisq = 2 * (unrestricted - self.logl)
            self.fmin = chisq / (2 * n)
            baseline_chisq, baseline_df = 2 * (unrestricted - self._independence_logl()), moments - 2 * p
        else:
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
        if self.missing == "ml":
            logl = self.logl
        else:
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
        # with a mean structure the "rmr"/"srmr" family also averages the mean residuals
        mean_residual = self.sample_mean - self.implied_mean() if self.meanstructure else np.zeros(0)

        def root_mean_square(values, means):
            return np.sqrt(np.mean(np.concatenate([values, means]) ** 2))

        sd_s = np.sqrt(np.diag(s))
        scale = np.outer(sd_s, sd_s)
        out["rmr"] = root_mean_square(residual[lower], mean_residual)
        out["rmr_nomean"] = root_mean_square(residual[lower], np.zeros(0))
        bentler_means = mean_residual / sd_s if self.meanstructure else np.zeros(0)
        out["srmr"] = out["srmr_bentler"] = root_mean_square((residual / scale)[lower], bentler_means)
        out["srmr_bentler_nomean"] = root_mean_square((residual / scale)[lower], np.zeros(0))
        corr_s = s / scale
        sd_sigma = np.sqrt(np.diag(sigma))
        corr_sigma = sigma / np.outer(sd_sigma, sd_sigma)
        strict_lower = np.tril_indices(p, -1)
        crmr_means = (self.sample_mean / sd_s - self.implied_mean() / sd_sigma) if self.meanstructure else np.zeros(0)
        out["crmr"] = root_mean_square((corr_s - corr_sigma)[strict_lower], crmr_means)
        out["crmr_nomean"] = root_mean_square((corr_s - corr_sigma)[strict_lower], np.zeros(0))
        mplus = (corr_s - corr_sigma).copy()
        np.fill_diagonal(mplus, (np.diag(s) - np.diag(sigma)) / np.diag(s))
        out["srmr_mplus"] = root_mean_square(mplus[lower], crmr_means)
        out["srmr_mplus_nomean"] = root_mean_square(mplus[lower], np.zeros(0))
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
        if self.missing == "ml":
            # with missing data lavaan weights the moments for these differently; not ported yet,
            # so they are missing rather than different from lavaan
            for name in ["crmr", "gfi", "gfi.ci.lower", "gfi.ci.upper", "gfi_lisrel", "agfi_lisrel", "pgfi"]:
                out[name] = np.nan
            # lavaan adds FIML-corrected ("robust") versions; not ported yet, so they are missing
            robust = {"tli": ["cfi.robust", "tli.robust"], "rni": ["nnfi.robust", "rni.robust"],
                      "rmsea.notclose.h0": ["rmsea.robust", "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
                                            "rmsea.pvalue.robust", "rmsea.notclose.pvalue.robust"],
                      "gfi.ci.level": ["gfi.robust", "gfi.ci.lower.robust", "gfi.ci.upper.robust"]}
            ordered = {}
            for name, value in out.items():
                ordered[name] = value
                for extra in robust.get(name, []):
                    ordered[extra] = np.nan
            out = ordered
        return out

    def sigma_h1(self):
        """The unrestricted covariance: the sample covariance, or its EM estimate with missing data."""
        return self.sample_cov

    def _pattern_logl(self, sigma, mu):
        """FIML log-likelihood of given moments over the missing-data patterns."""
        total = 0.0
        for observed, n_k, mean_k, cov_k in self._patterns:
            sub = sigma[np.ix_(observed, observed)]
            inv = np.linalg.inv(sub)
            diff = mean_k - mu[observed]
            total -= 0.5 * n_k * (len(observed) * np.log(2 * np.pi) + np.linalg.slogdet(sub)[1]
                                  + np.sum(inv * cov_k) + diff @ inv @ diff)
        return total

    def _independence_logl(self):
        """Log-likelihood of the baseline model (free means and variances, no covariances) with FIML."""
        x = self.data.to_numpy(dtype=float)
        total = 0.0
        for column in x.T:
            values = column[~np.isnan(column)]
            variance = values.var()
            total -= 0.5 * len(values) * (np.log(2 * np.pi) + np.log(variance) + 1)
        return total

    def _candidates(self):
        """lavaan's full parameter list (lav_pt_full) for modification indices."""
        ov, lv = self.observed, self.latent
        rows = [("=~", f, v) for f in lv for v in ov]
        rows += [("~~", ov[j], ov[i]) for j in range(len(ov)) for i in range(j, len(ov))]
        rows += [("~~", lv[j], lv[i]) for j in range(len(lv)) for i in range(j, len(lv))]
        eqs_y = list(dict.fromkeys(p.lhs for p in self.table if p.op == "~"))
        eqs_x = list(dict.fromkeys(p.rhs for p in self.table if p.op == "~"))
        names = list(dict.fromkeys(eqs_y + eqs_x))
        rows += [("~", a, b) for a in names for b in names if a != b]
        if self.meanstructure:
            rows += [("~1", v, "") for v in ov + lv]
        return rows

    def modification_indices(self, delta=0.1, alpha=0.05, high_power=0.75, power=True, free_remove=False,
                             minimum_value=0):
        """
        lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE, information =
        "expected", sort. = TRUE, ...): score tests of every parameter lavaan considers, with
        expected parameter changes. power=True (rwf's report_cfa) adds delta, ncp, power and
        Saris' decision and keeps sepc.all only; power=False (lavaan's default) gives sepc.lv,
        sepc.all and sepc.nox. free_remove=True drops the model's own parameters (lavaan's
        default); rows with an index of minimum_value or less are dropped when it is above 0.
        """
        def key(op, lhs, rhs):
            return (op, *sorted((lhs, rhs))) if op == "~~" else (op, lhs, rhs)

        existing = {key(p.op, p.lhs, p.rhs) for p in self.table}
        model_rows = [(p.op, p.lhs, p.rhs) for p in self.free]
        # fixed parameters of the model (e.g. marker loadings) are candidates too; freeing one is
        # only testable while the model stays identified, otherwise its index is missing
        fixed_rows = [(p.op, p.lhs, p.rhs) for p in self.table if not p.free]
        extra_rows = fixed_rows + [r for r in self._candidates() if key(*r) not in existing]
        information = self._expected_information(model_rows + extra_rows)
        q = len(model_rows)
        i22, i12, i11 = information[:q, :q], information[q:, :q], information[q:, q:]
        i22_inv = np.linalg.inv(i22)
        v_diag = np.diag(i11) - np.einsum("ij,jk,ik->i", i12, i22_inv, i12)
        v_diag = np.where(v_diag < np.finfo(float).eps ** (1 / 3), np.nan, v_diag)
        score = self._score(model_rows + extra_rows)
        n = self.n
        mi = np.concatenate([n * score[:q] ** 2 / np.diag(i22), n * score[q:] ** 2 / v_diag])
        d = -n * score
        d = np.where(np.abs(d) < 1e-15, 1.0, d)
        epc = mi / d
        values = {key(p.op, p.lhs, p.rhs): p.value for p in self.table}
        rows = []
        for (op, lhs, rhs), m, e in zip(model_rows + extra_rows, mi, epc):
            size = e
            if op == "~~" and lhs == rhs:
                size = values.get(key(op, lhs, rhs), e)  # cov.std: a variance is standardized by itself
                if abs(size) < np.sqrt(np.finfo(float).eps):
                    size = np.nan
            sepc_lv, sepc_all = self._standardized(Parameter(lhs, op, rhs, True, abs(size), False))
            rows.append({"lhs": lhs, "op": op, "rhs": rhs, "mi": m, "epc": e, "sepc.lv": np.sign(e) * sepc_lv,
                         "sepc.all": np.sign(e) * sepc_all, "sepc.nox": np.sign(e) * sepc_all,
                         "_model": (op, lhs, rhs) in model_rows})
        table = pd.DataFrame(rows)
        if free_remove:
            table = table[~table["_model"]]
        table = table.drop(columns="_model")
        if not power:
            table = table.iloc[np.argsort(-table["mi"].to_numpy(), kind="stable")]
            if minimum_value > 0:
                table = table[table["mi"] > minimum_value]
            return table[table["mi"].notna()].reset_index(drop=True)
        table = table.drop(columns=["sepc.lv", "sepc.nox"])
        table["delta"] = delta
        table["ncp"] = (table["mi"] / (table["epc"] * table["epc"])) * delta ** 2
        table["power"] = 1 - stats.ncx2.cdf(stats.chi2.ppf(1 - alpha, 1), 1, table["ncp"])
        significant = table["mi"] > stats.chi2.ppf(1 - alpha, 1)
        high = table["power"] > high_power
        epc_high = table["epc"].abs() > delta
        decision = np.full(len(table), "", dtype=object)
        decision[~significant & ~high] = "(i)"
        decision[significant & ~high] = "**(m)**"
        decision[~significant & high] = "(nm)"
        decision[significant & high & ~epc_high] = "epc:nm"
        decision[significant & high & epc_high] = "*epc:m*"
        table["decision"] = decision
        table = table.iloc[np.argsort(-table["mi"].to_numpy(), kind="stable")]
        if minimum_value > 0:
            table = table[table["mi"] > minimum_value]
        return table[table["mi"].notna()].reset_index(drop=True)

    def inspect(self, what="est"):
        """
        lavaan::inspect(fit, "est") or (what="std") the completely standardized estimates, as a
        dict of model matrices: lambda, theta, psi and, with regressions, beta.
        """
        if what not in ("est", "std"):
            raise ValueError('what must be "est" or "std"')
        lam, th, psi, beta = (m.copy() for m in self._matrices())
        if what == "std":
            # the same standardization as parameter_estimates' std.all, parameter by parameter
            ov, lv = self._index["ov"], self._index["lv"]
            for prm in self.table:
                value = self._standardized(prm)[1]
                if prm.op == "=~":
                    lam[ov[prm.rhs], lv[prm.lhs]] = value
                elif prm.op == "~":
                    beta[lv[prm.lhs], lv[prm.rhs]] = value
                elif prm.lhs in ov:
                    th[ov[prm.lhs], ov[prm.rhs]] = th[ov[prm.rhs], ov[prm.lhs]] = value
                else:
                    psi[lv[prm.lhs], lv[prm.rhs]] = psi[lv[prm.rhs], lv[prm.lhs]] = value
        result = {
            "lambda": pd.DataFrame(lam, index=self.observed, columns=self.latent),
            "theta": pd.DataFrame(th, index=self.observed, columns=self.observed),
            "psi": pd.DataFrame(psi, index=self.latent, columns=self.latent),
        }
        if self.endogenous:
            result["beta"] = pd.DataFrame(beta, index=self.latent, columns=self.latent)
        if self.meanstructure:
            intercepts = {p.lhs: (self._standardized(p)[1] if what == "std" else p.value)
                          for p in self.table if p.op == "~1" and p.lhs in self._index["ov"]}
            result["nu"] = pd.DataFrame({"intercept": [intercepts[v] for v in self.observed]}, index=self.observed)
        return result

    def predict(self):
        """Factor scores by the regression method, as lavaan::lavPredict(fit)."""
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        a = c @ psi @ c.T
        x = self.data.to_numpy(dtype=float)
        if self.missing != "ml":
            centred = x - x.mean(axis=0)
            scores = centred @ np.linalg.solve(self.sigma, lam @ a)
        else:
            # each case is scored from the variables it has, with the model-implied means
            mu = self.implied_mean()
            scores = np.empty((len(x), self.m))
            for i, row in enumerate(x):
                observed = np.flatnonzero(~np.isnan(row))
                weights = np.linalg.solve(self.sigma[np.ix_(observed, observed)], lam[observed] @ a)
                scores[i] = (row[observed] - mu[observed]) @ weights
        return pd.DataFrame(scores, columns=self.latent, index=self.data.index)

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


def sem(model, data, std_lv=False, missing="listwise", meanstructure=False):
    """
    Fit a structural equation model given in lavaan syntax, like lavaan::sem
    (and lavaan::cfa, which has the same defaults for these models).

    Parameters:
    model (str): lavaan model syntax.
    data (pandas.DataFrame): Observed variables.
    std_lv (bool, optional): Fix the latent variances to 1 instead of the
        first loadings. R's std.lv. Defaults to False.
    missing (str, optional): "listwise" (default) drops rows with a missing
        value; "ml" uses every available value (full information maximum
        likelihood, with intercepts and standard errors from the observed
        information, as lavaan's missing = "ml"). Rows without any value are
        dropped.
    meanstructure (bool, optional): Estimate intercepts; implied by
        missing="ml" (the only case supported).

    Returns:
    SEMFit: The fitted model.

    Examples:
    >>> fit = sem("E =~ E1 + E2 + E3; N =~ N1 + N2 + N3", df_ocean)
    >>> fit.parameter_estimates()
    >>> sem("E =~ E1 + E2 + E3", df_with_missing_values, missing="ml")
    """
    return SEMFit(model, data, std_lv=std_lv, missing=missing, meanstructure=meanstructure)


cfa = sem


def simulate_data(model, sample_nobs=500, orthogonal=False, seed=None):
    """
    Normal data from a model with fixed values, as lavaan::simulateData(model, model.type =
    "cfa", return.type = "data.frame"): unspecified variances are 1, everything else not given
    is 0, so latent variables are uncorrelated unless covariances are written. Draws use numpy,
    so values differ from R's for the same seed.

    Parameters:
    model (str): lavaan syntax with fixed values, e.g. "F =~ 1*x1 + 0.8*x2 + 0.5*x3".
    sample_nobs (int, optional): Number of rows. R's sample.nobs. Defaults to 500.
    orthogonal (bool, optional): Accepted for compatibility; latent variables are already
        uncorrelated unless the model gives covariances.
    seed (int, optional): Seed of the random generator.

    Returns:
    pandas.DataFrame: One column per observed variable.

    Examples:
    >>> simulate_data("F =~ 1*x1 + 0.5*x2 + 1.5*x3", sample_nobs=200, seed=1)
    """
    statements = _parse(model)
    latent = list(dict.fromkeys(s[0] for s in statements if s[1] == "=~"))
    names = list(dict.fromkeys(n for s in statements for n in (s[0], s[2])))
    observed = list(dict.fromkeys([s[2] for s in statements if s[1] == "=~"] + [n for n in names if n not in latent]))
    ov, lv = {v: i for i, v in enumerate(observed)}, {v: i for i, v in enumerate(latent)}
    lam, th = np.zeros((len(observed), len(latent))), np.eye(len(observed))
    psi, beta = np.eye(len(latent)), np.zeros((len(latent), len(latent)))
    for lhs, op, rhs, fixed, _ in statements:
        value = 0.0 if fixed is None else fixed
        if op == "=~":
            lam[ov[rhs], lv[lhs]] = value
        elif op == "~":
            if lhs not in lv or rhs not in lv:
                raise NotImplementedError("regressions are supported among latent variables only")
            beta[lv[lhs], lv[rhs]] = value
        elif lhs in ov:
            th[ov[lhs], ov[rhs]] = th[ov[rhs], ov[lhs]] = value if fixed is not None else th[ov[lhs], ov[rhs]]
        else:
            psi[lv[lhs], lv[rhs]] = psi[lv[rhs], lv[lhs]] = value if fixed is not None else psi[lv[lhs], lv[rhs]]
    c = np.linalg.inv(np.eye(len(latent)) - beta)
    sigma = lam @ c @ psi @ c.T @ lam.T + th
    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(np.zeros(len(observed)), sigma, size=sample_nobs)
    return pd.DataFrame(data, columns=observed)
