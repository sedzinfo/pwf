# -*- coding: utf-8 -*-
"""
A small structural equation modelling engine that reproduces lavaan for the
models rwf fits: confirmatory factor analysis and structural models with
latent variables, estimated by maximum likelihood on complete data.

It reads lavaan model syntax (=~, ~, ~~, fixed values such as 0.8*x, NA*x to
free a parameter, start(1)*x, ";" or new lines between statements, #
comments) and adds lavaan's automatic parameters: the first loading of each
factor fixed to 1 (or the factor variances fixed to 1 with std_lv=True),
residual variances of every observed and latent variable, and covariances
among exogenous latent variables. Estimates minimise lavaan's ML discrepancy
with the sample covariance computed with divisor N; standard errors come
from the expected information, as lavaan's defaults.

Labels and constraints, as lavaan: parameters sharing a label (a*x1 + a*x2)
are equal; a label shared with a fixed parameter fixes them all; linear
equality constraints (a == b, a == 2*b + 0.1) are kept exactly; defined
parameters (ab := a*b) are reported with delta-method standard errors and
their standardized values. The estimates and standard errors are lavaan's
for the constrained model, npar counts the free parameters less the
constraints, and modification indices respect the constraints (as lavaan's,
which test the model's parameters, not the constraints themselves).

Not supported (they raise NotImplementedError): observed exogenous
regressors, mean structures without missing="ml", multiple groups,
inequality and non-linear constraints.
"""
import ast
import inspect
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    from ._ordinal import _NAME, _linear, _parse_term
    from ._rcompat import r_deparse_call
except ImportError:
    from _ordinal import _NAME, _linear, _parse_term
    from _rcompat import r_deparse_call

_OPERATORS = ("=~", "~~", "~")


def _parse(model):
    """
    lavaan model syntax as parameter statements and the user's constraints and definitions.

    Statements are dicts lhs, op, rhs, fixed, label, start and free; a parameter named twice ("start(1)*x +
    a*x") is one statement with all its modifiers. The constraints and definitions keep their order in the
    syntax, as lavaan's parameter table: {"op": "==", "lhs", "rhs" (lavaan's text, without spaces),
    "linear": ({label: coefficient}, constant)} meaning sum(coefficient * label) == constant, and
    {"op": ":=", "lhs": name, "rhs": expression}.
    """
    statements, extras, index = [], [], {}
    for raw in re.split(r"[;\n]", model):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if ":=" in line:
            name, expression = (part.strip() for part in line.split(":=", 1))
            if not re.fullmatch(_NAME, name):
                raise ValueError(f"invalid name of a defined parameter {name!r}")
            extras.append({"op": ":=", "lhs": name, "rhs": expression.replace(" ", "")})
            continue
        if "==" in line:
            left, right = line.split("==", 1)
            a, constant_a = _linear(left)  # NotImplementedError for a constraint that isn't linear
            b, constant_b = _linear(right)
            coefficients = dict(a)
            for label, value in b.items():
                coefficients[label] = coefficients.get(label, 0.0) - value
            extras.append({"op": "==", "lhs": left.replace(" ", ""), "rhs": right.replace(" ", ""),
                           "linear": (coefficients, constant_b - constant_a)})
            continue
        if re.search(r"[<>]", line):
            raise NotImplementedError(f"inequality constraints are not supported: {line!r}")
        op = next((o for o in _OPERATORS if o in line), None)
        if op is None:
            raise ValueError(f"cannot read the model statement {line!r}")
        lhs, rhs = (part.strip() for part in line.split(op, 1))
        if not re.fullmatch(_NAME, lhs):
            raise ValueError(f"invalid variable name {lhs!r}")
        for term in rhs.split("+"):
            name, fixed, label, start, free = _parse_term(term.strip())
            key = (lhs, op, name)
            if op == "~~" and key not in index and (name, op, lhs) in index:
                key = (name, op, lhs)
            if key in index:
                row = statements[index[key]]
                row["fixed"] = fixed if fixed is not None else row["fixed"]
                row["label"] = label or row["label"]
                row["start"] = start if start is not None else row["start"]
                row["free"] = row["free"] or free
            else:
                index[key] = len(statements)
                statements.append({"lhs": lhs, "op": op, "rhs": name, "fixed": fixed, "label": label or "",
                                   "start": start, "free": free})
    return statements, extras


_R_FUNCTIONS = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs}


def _evaluate(expression, values):
    """
    An R arithmetic expression of labels (lavaan's := definitions): numbers, names, + - * / ^ and
    parentheses, sqrt(), exp(), log() and abs(); values maps names to numbers (complex ones too).
    """
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(f"{expression!r} names {node.id!r}, which is not a label or a defined parameter")
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return walk(node.operand) if isinstance(node.op, ast.UAdd) else -walk(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = walk(node.left), walk(node.right)
            return {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                    ast.Div: lambda: left / right, ast.Pow: lambda: left ** right}[type(node.op)]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _R_FUNCTIONS \
                and len(node.args) == 1 and not node.keywords:
            return _R_FUNCTIONS[node.func.id](walk(node.args[0]))
        raise NotImplementedError(f"the definition {expression!r} is not supported")

    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot read the definition {expression!r}") from error
    return walk(tree)


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
    def __init__(self, lhs, op, rhs, free, value, user, label="", start=None):
        self.lhs, self.op, self.rhs, self.free, self.value, self.user = lhs, op, rhs, free, value, user
        self.label, self.start = label, start


class SEMFit:
    """A fitted model: parameter table, implied covariance, sample statistics and helpers."""

    def __init__(self, model, data, std_lv=False, missing="listwise", meanstructure=False):
        statements, extras = _parse(model)
        self.model_syntax = model
        self.std_lv = std_lv
        latent = list(dict.fromkeys(s["lhs"] for s in statements if s["op"] == "=~"))
        names = list(dict.fromkeys(n for s in statements for n in (s["lhs"], s["rhs"])))
        observed = [n for n in names if n not in latent]
        # lavaan orders observed variables by first appearance as indicators, then elsewhere
        indicators = [s["rhs"] for s in statements if s["op"] == "=~"]
        observed = list(dict.fromkeys(indicators + observed))
        regressions = [s for s in statements if s["op"] == "~"]
        for s in regressions:
            if s["lhs"] not in latent or s["rhs"] not in latent:
                raise NotImplementedError("regressions are supported among latent variables only")
        self.latent, self.observed = latent, observed
        endogenous = list(dict.fromkeys(s["lhs"] for s in regressions))
        self.endogenous = endogenous

        table = []
        first_indicator = {}
        for s in statements:
            fixed = s["fixed"]
            if s["op"] == "=~" and s["lhs"] not in first_indicator:
                first_indicator[s["lhs"]] = s["rhs"]
                if fixed is None and not s["free"] and not std_lv:
                    fixed = 1.0
            table.append(Parameter(s["lhs"], s["op"], s["rhs"], fixed is None, 0.0 if fixed is None else fixed, True,
                                   s["label"], s["start"]))
        # lavaan: a label shared with a fixed parameter fixes every parameter with that label at its value
        fixed_labels = {}
        for prm in table:
            if prm.label and not prm.free:
                fixed_labels.setdefault(prm.label, prm.value)
        for prm in table:
            if prm.free and prm.label in fixed_labels:
                prm.free, prm.value = False, fixed_labels[prm.label]
        labels = {prm.label for prm in table if prm.label}
        defined = [extra["lhs"] for extra in extras if extra["op"] == ":="]
        clash = [name for name in defined if name in labels or defined.count(name) > 1]
        if clash:
            raise ValueError(f"the defined parameter {clash[0]!r} is already a label or defined twice")
        self._extras = extras
        existing = {(p.lhs, p.op, p.rhs) for p in table} | {(p.rhs, p.op, p.lhs) for p in table if p.op == "~~"}
        for name in observed:
            if (name, "~~", name) not in existing:
                table.append(Parameter(name, "~~", name, True, 0.0, False))
        for name in latent:
            if (name, "~~", name) not in existing:
                # lavaan's std.lv fixes every latent (residual) variance to 1, endogenous ones included
                fixed_one = std_lv
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
        self._setup_constraints()
        x = values.to_numpy(dtype=float)
        if missing == "listwise":
            centred = x - x.mean(axis=0)
            self.sample_cov = centred.T @ centred / self.n
        else:
            self._patterns = _missing_patterns(x)
            self.sample_mean, self.sample_cov = _em_saturated(x, self._patterns)
        self._start()
        self._estimate()

    # ------------------------------------------------------------------ labels and constraints
    def _setup_constraints(self):
        """
        The free parameters as functions of the parameters the optimizer moves, values = offset + jac @ theta:
        parameters sharing a label share one value, and the linear equality constraints restrict those values
        to an affine subspace of which theta are the coordinates (lavaan's reparameterisation of linear
        equality constraints). Without labels and constraints theta are the free parameters themselves.
        """
        keys, slot_of = {}, []
        for k, prm in enumerate(self.free):
            key = prm.label or f"#{k}"
            keys.setdefault(key, len(keys))
            slot_of.append(keys[key])
        slots = np.zeros((len(self.free), len(keys)))
        slots[np.arange(len(self.free)), slot_of] = 1.0
        linear = [extra["linear"] for extra in self._extras if extra["op"] == "=="]
        if linear:
            a = np.zeros((len(linear), len(keys)))
            c = np.zeros(len(linear))
            for i, (coefficients, constant) in enumerate(linear):
                for label, value in coefficients.items():
                    if label not in keys:
                        raise ValueError(f"a constraint names {label!r}, which is not the label of a free parameter")
                    a[i, keys[label]] += value
                c[i] = constant
            _, singular, vt = np.linalg.svd(a)
            rank = int(np.sum(singular > 1e-10 * max(singular.max(), 1.0)))
            p0 = np.linalg.pinv(a) @ c
            if not np.allclose(a @ p0, c, rtol=0, atol=1e-10):
                raise ValueError("the equality constraints contradict each other")
            k = vt[rank:].T
        else:
            p0, k = np.zeros(len(keys)), np.eye(len(keys))
        self._slots, self._slot_p0, self._slot_k = slots, p0, k
        self._jac, self._offset = slots @ k, slots @ p0
        self._constrained = len(keys) < len(self.free) or bool(linear)
        self.npar = k.shape[1]  # lavaan's npar: the free parameters less the (independent) constraints
        # lavaan's hidden "==" rows of shared labels, one for every parameter after the first with a label
        self._system_eq = len(self.free) - len(keys)

    def _rows(self, theta):
        """The free parameters' values from the optimizer's parameters."""
        if not self._constrained:
            return np.asarray(theta, dtype=float)
        return self._offset + self._jac @ np.asarray(theta, dtype=float)

    def _theta(self, rows):
        """The optimizer's parameters of free parameters' values: the first value of each label, projected on the
        constraints."""
        rows = np.asarray(rows, dtype=float)
        if not self._constrained:
            return rows
        first = np.array([rows[np.flatnonzero(self._slots[:, j])[0]] for j in range(self._slots.shape[1])])
        return self._slot_k.T @ (first - self._slot_p0)

    def _constrained_inverse(self, information):
        """
        The inverse of an information matrix of the free parameters under the constraints, J (J' I J)^-1 J':
        what lavaan's augmented inverse (lav_model_information_augment_invert) gives; the plain inverse without
        constraints.
        """
        if not self._constrained:
            return np.linalg.inv(information)
        return self._jac @ np.linalg.inv(self._jac.T @ information @ self._jac) @ self._jac.T

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
            if prm.start is not None:  # start(1)*x
                prm.value = prm.start

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
            row_objective, row_gradient = self._fiml_objective, self._fiml_gradient
        else:
            row_objective, row_gradient = self._discrepancy, self._gradient
        if self._constrained:  # the optimizer moves theta; the free parameters follow (shared labels, constraints)
            def objective(theta):
                return row_objective(self._rows(theta))

            def gradient(theta):
                return self._jac.T @ row_gradient(self._rows(theta))
        else:
            objective, gradient = row_objective, row_gradient
        theta0 = self._theta(np.array([p.value for p in self.free]))
        fit = optimize.minimize(objective, theta0, jac=gradient, method="BFGS",
                                options={"gtol": 1e-10, "maxiter": 10000})
        if not np.all(np.isfinite(fit.x)):
            raise RuntimeError("the model did not converge")
        polish = optimize.minimize(objective, fit.x, jac=gradient, method="L-BFGS-B",
                                   options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 10000})
        theta = polish.x if polish.fun <= fit.fun else fit.x
        # At the minimum the objective can no longer be lowered in floating point, so BFGS may stop there with
        # "precision loss" and L-BFGS-B with "ABNORMAL", depending on rounding that differs between platforms.
        # The fit has also converged when the gradient is (near) zero by lavaan's check of the solution
        # (check.gradient, with every element within optim.dx.tol = 1e-3).
        dx = gradient(theta)
        rows = self._rows(theta)
        self._matrices(rows)
        self.converged = bool(fit.success or polish.success or np.all(np.abs(dx) <= 1e-3))
        self.sigma = self.implied()
        if self.missing == "ml":
            self.logl = self._fiml_logl(rows)
            self.vcov = self._constrained_inverse(self._fiml_hessian(rows))
        else:
            self.fmin = 0.5 * self._discrepancy(rows)
            self.vcov = self._constrained_inverse(self.n * self._information(self._derivatives()))
        variances = np.diag(self.vcov).copy()
        if self._constrained:  # a parameter the constraints pin down has variance 0, which rounding can make < 0
            variances[np.abs(variances) <= 1e-12 * max(1.0, np.max(np.abs(variances)))] = 0.0
        for prm, se in zip(self.free, np.sqrt(variances)):
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
            with np.errstate(invalid="ignore"):  # a negative residual variance (Heywood case): missing, as lavaan
                return np.sqrt(psi[lv[name], lv[name]] if name in lv else th[ov[name], ov[name]])

        std_all = value / (residual_sd(prm.lhs) * residual_sd(prm.rhs))
        std_lv = std_all if prm.lhs in lv else value
        return std_lv, std_all

    def parameter_estimates(self, level=0.95, remove_eq=True):
        """
        lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE) without std.nox: the parameters,
        the defined parameters (:=, with delta-method standard errors, and standardized as their expressions of
        the standardized parameters), with remove_eq=False (lavaan's remove.eq, as rwf's report_cfa) the
        user's equality constraints (==, est the difference of their sides), then the R squared rows. A label
        column follows rhs when the model has labels. The index is lavaan's row numbers: positions in its
        parameter table, whose hidden rows (the constraints of shared labels) are skipped.
        """
        z_crit = stats.norm.ppf(1 - (1 - level) / 2)
        standardized = [self._standardized(prm) for prm in self.table]
        rows, index = [], []
        for k, (prm, (std_lv, std_all)) in enumerate(zip(self.table, standardized), start=1):
            se = getattr(prm, "se", 0.0) if prm.free else 0.0
            z = prm.value / se if prm.free and se > 0 else np.nan
            p_value = 2 * stats.norm.sf(abs(z)) if prm.free and se > 0 else np.nan
            rows.append({"lhs": prm.lhs, "op": prm.op, "rhs": prm.rhs, "label": prm.label, "est": prm.value,
                         "se": se, "z": z, "pvalue": p_value, "ci.lower": prm.value - z_crit * se,
                         "ci.upper": prm.value + z_crit * se, "std.lv": std_lv, "std.all": std_all})
            index.append(k)
        for k, row in enumerate(self._extra_rows(z_crit, standardized), start=len(self.table) + 1):
            if row["op"] == ":=" or not remove_eq:
                rows.append(row)
                index.append(k)
        first_r2 = len(self.table) + len(self._extras) + self._system_eq + 1
        for k, (name, value) in enumerate(self.rsquare().items(), start=first_r2):
            rows.append({"lhs": name, "op": "r2", "rhs": name, "label": "", "est": value, "se": np.nan, "z": np.nan,
                         "pvalue": np.nan, "ci.lower": np.nan, "ci.upper": np.nan, "std.lv": np.nan,
                         "std.all": np.nan})
            index.append(k)
        table = pd.DataFrame(rows, index=index)
        if not any(prm.label for prm in self.table) and not any(e["op"] == ":=" for e in self._extras):
            table = table.drop(columns="label")  # lavaan has the column only when the model has labels
        return table

    def _label_values(self, values):
        """Each label's value (its first parameter's), given a value for every row of the parameter table."""
        out = {}
        for prm, value in zip(self.table, values):
            if prm.label and prm.label not in out:
                out[prm.label] = value
        return out

    def _definitions(self, values):
        """The defined parameters (:=) in their order, a later one may use an earlier one, from per-row values."""
        names = self._label_values(values)
        out = {}
        for extra in self._extras:
            if extra["op"] == ":=":
                out[extra["lhs"]] = names[extra["lhs"]] = _evaluate(extra["rhs"], names)
        return out

    def _definition_jacobian(self, names):
        """d(defined parameter)/d(free parameter): complex-step derivatives (lavaan's lav_func_jacobian_complex),
        central differences when a definition uses abs()."""
        free_rows = [k for k, prm in enumerate(self.table) if prm.free]
        base = np.array([prm.value for prm in self.table], dtype=float)
        jacobian = np.zeros((len(names), len(free_rows)))
        complex_step = not any("abs(" in e["rhs"] for e in self._extras if e["op"] == ":=")
        for j, k in enumerate(free_rows):
            if complex_step:
                values = base.astype(complex)
                values[k] += 1e-20j
                defined = self._definitions(values)
                jacobian[:, j] = [np.imag(defined[name]) / 1e-20 for name in names]
            else:
                step = 1e-6 * max(1.0, abs(base[k]))
                up, down = base.copy(), base.copy()
                up[k] += step
                down[k] -= step
                high, low = self._definitions(up), self._definitions(down)
                jacobian[:, j] = [(high[name] - low[name]) / (2 * step) for name in names]
        return jacobian

    def _extra_rows(self, z_crit, standardized):
        """parameterEstimates' rows of the user's definitions (:=) and equality constraints (==), in their order."""
        if not self._extras:
            return []
        values = np.array([prm.value for prm in self.table], dtype=float)
        std_lv = np.array([s[0] for s in standardized], dtype=float)
        std_all = np.array([s[1] for s in standardized], dtype=float)
        names = [e["lhs"] for e in self._extras if e["op"] == ":="]
        estimates = self._definitions(values)
        defined_lv, defined_all = self._definitions(std_lv), self._definitions(std_all)
        se = {}
        if names:
            jacobian = self._definition_jacobian(names)
            variances = np.diag(jacobian @ self.vcov @ jacobian.T)
            se = dict(zip(names, np.sqrt(np.clip(variances, 0, None))))
        labels, labels_lv, labels_all = (self._label_values(v) for v in (values, std_lv, std_all))
        rows = []
        for extra in self._extras:
            if extra["op"] == ":=":
                name = extra["lhs"]
                est, error = float(np.real(estimates[name])), se[name]
                z = est / error if error > 0 else np.nan
                rows.append({"lhs": name, "op": ":=", "rhs": extra["rhs"], "label": name, "est": est, "se": error,
                             "z": z, "pvalue": 2 * stats.norm.sf(abs(z)) if error > 0 else np.nan,
                             "ci.lower": est - z_crit * error, "ci.upper": est + z_crit * error,
                             "std.lv": float(np.real(defined_lv[name])), "std.all": float(np.real(defined_all[name]))})
            else:  # lavaan reports the constraint's value, lhs - rhs (0 when it holds), with a zero standard error
                coefficients, constant = extra["linear"]

                def difference(label_values):
                    return sum(c * label_values[label] for label, c in coefficients.items()) - constant

                est = difference(labels)
                rows.append({"lhs": extra["lhs"], "op": "==", "rhs": extra["rhs"], "label": "", "est": est,
                             "se": 0.0, "z": np.nan, "pvalue": np.nan, "ci.lower": est, "ci.upper": est,
                             "std.lv": difference(labels_lv), "std.all": difference(labels_all)})
        return rows

    def fit_measures(self):
        """lavaan::fitMeasures(fit) for ML on complete data, in lavaan's order and names."""
        n, p = self.n, self.p
        s, sigma = self.sample_cov, self.sigma
        npar = self.npar  # the free parameters less the equality constraints, as lavaan's fitMeasures
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
        # under the model's equality constraints, as lavaan's (which still doesn't test releasing them)
        i22_inv = self._constrained_inverse(i22)
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
        # lavaan's row names: positions in its parameter table (with its rows of definitions and constraints,
        # which are no candidates), then in the candidates it adds
        lavaan_order = [key(p.op, p.lhs, p.rhs) for p in self.table]
        lavaan_order += [None] * (len(self._extras) + self._system_eq)
        lavaan_order += [key(*r) for r in self._candidates() if key(*r) not in existing]
        position = {k: i + 1 for i, k in enumerate(lavaan_order)}
        table.index = [position.get(key(op, lhs, rhs)) for op, lhs, rhs in zip(table["op"], table["lhs"], table["rhs"])]
        if free_remove:
            table = table[~table["_model"]]
        table = table.drop(columns="_model")
        if not power:
            table = table.iloc[np.argsort(-table["mi"].to_numpy(), kind="stable")]
            if minimum_value > 0:
                table = table[table["mi"] > minimum_value]
            return table[table["mi"].notna()]
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
        return table[table["mi"].notna()]  # indexed by lavaan's row numbers, as rwf's sheet shows them

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
                if prm.op == "~1":
                    continue  # intercepts are reported separately, as nu
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
        """
        Explained variance of every observed variable and every endogenous latent variable, in the order of
        their (residual) variances in the parameter table, as lavaan's (so variances the model gives come first).
        """
        lam, th, psi, beta = self._matrices()
        var_lv = self.latent_variances()
        result = {}
        for prm in self.table:
            if prm.op != "~~" or prm.lhs != prm.rhs or prm.lhs in result:
                continue
            if prm.lhs in self._index["ov"]:
                i = self._index["ov"][prm.lhs]
                result[prm.lhs] = 1 - th[i, i] / self.sigma[i, i]
            elif prm.lhs in self.endogenous:
                i = self._index["lv"][prm.lhs]
                result[prm.lhs] = 1 - psi[i, i] / var_lv[i]
        # lavaan (lav_inspect_rsquare): above 1, after a negative residual variance, is missing
        return {name: (np.nan if value > 1 else value) for name, value in result.items()}


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
    fit = SEMFit(model, data, std_lv=std_lv, missing=missing, meanstructure=meanstructure)
    fit.r_call = _lavaan_call("sem", inspect.currentframe().f_back, model, data, std_lv, missing, meanstructure)
    return fit


def cfa(model, data, std_lv=False, missing="listwise", meanstructure=False):
    """
    lavaan::cfa: the same fit as sem() (the defaults are the same for these
    models); only the call text rwf reports says cfa.
    """
    fit = SEMFit(model, data, std_lv=std_lv, missing=missing, meanstructure=meanstructure)
    fit.r_call = _lavaan_call("cfa", inspect.currentframe().f_back, model, data, std_lv, missing, meanstructure)
    return fit


def _lavaan_call(kind, frame, model, data, std_lv, missing, meanstructure):
    """
    Lines of lavaan's fit@call as rwf's report_cfa writes them (deparse, spaces removed): the arguments
    given, then the ones lavaan::sem/cfa add; model and data by the names they have in the caller.
    """
    def name_of(value, default):
        caller = frame
        while caller is not None:
            found = [k for k, v in {**caller.f_globals, **caller.f_locals}.items()
                     if v is value and not k.startswith("_")]
            if found:
                return found[0]
            caller = caller.f_back
        return default

    arguments = [("model", name_of(model, "model")), ("data", name_of(data, "data"))]
    arguments += [("std.lv", "TRUE")] if std_lv else []
    arguments += [("missing", '"ml"')] if missing == "ml" else []
    arguments += [("meanstructure", "TRUE")] if meanstructure else []
    arguments += [("model.type", f'"{kind}"'), ("cmd", f'"{kind}"')]
    arguments += [(name, "TRUE") for name in ["int.ov.free"]] + [("int.lv.free", "FALSE")]
    arguments += [(name, "TRUE") for name in ["auto.fix.first", "auto.fix.single", "auto.var", "auto.cov.lv.x",
                                              "auto.cov.y", "auto.th", "auto.delta", "auto.efa"]]
    return [line.replace(" ", "") for line in r_deparse_call("lavaan::lavaan", arguments)]


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
    statements = [(s["lhs"], s["op"], s["rhs"], s["fixed"]) for s in _parse(model)[0]]
    latent = list(dict.fromkeys(s[0] for s in statements if s[1] == "=~"))
    names = list(dict.fromkeys(n for s in statements for n in (s[0], s[2])))
    observed = list(dict.fromkeys([s[2] for s in statements if s[1] == "=~"] + [n for n in names if n not in latent]))
    ov, lv = {v: i for i, v in enumerate(observed)}, {v: i for i, v in enumerate(latent)}
    lam, th = np.zeros((len(observed), len(latent))), np.eye(len(observed))
    psi, beta = np.eye(len(latent)), np.zeros((len(latent), len(latent)))
    for lhs, op, rhs, fixed in statements:
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
