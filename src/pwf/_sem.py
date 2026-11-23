# -*- coding: utf-8 -*-
"""
A small structural equation modelling engine that reproduces lavaan for the
models rwf fits: confirmatory factor analysis and structural models with
latent and observed variables, estimated by maximum likelihood.

It reads lavaan model syntax (=~, ~, ~~, intercepts y ~ 1, fixed values such
as 0.8*x or y ~ 0*1, NA*x to free a parameter, start(1)*x, ";" or new lines
between statements, # comments) and adds lavaan's automatic parameters: the first loading of each
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

Observed variables in regressions, as lavaan (conditional.x = FALSE): its
classes of variables (lav_pt_vnames: indicators, outcomes, predictors) and
their order; its automatic rows (auto.cov.lv.x, auto.cov.y: covariances of
outcomes that predict nothing); fixed.x (the default), which fixes the
predictors' variances, covariances and means at their sample values (divisor
N), leaves their moments out of the degrees of freedom and their marginal
likelihood out of logl, and with missing="ml" drops cases without a
predictor; and lavaan's LISREL representation (lav_lisrel), in which observed
variables in regressions are dummy latent variables, which inspect(), the
derivatives and lavPredict's factor scores follow. A predictor the syntax
gives a variance or covariance is random (lavaan's warning).

Mean structures, as lavaan: with meanstructure=True, missing="ml" or
intercepts in the syntax, every observed variable gets a free intercept and
every latent variable a mean fixed at 0 unless the syntax says otherwise
(F ~ 1 frees it); with complete data the ML discrepancy and the expected
information gain the mean term (ybar - mu)' Sigma^-1 (ybar - mu).

The fit measures follow lavaan's lav_fit_* functions, also at their edges
(df = 0): the GFI from Browne's residual-based statistic, the classic GFI
with lavaan's ML weights, the RMSEA and GFI intervals with R's uniroot.

Not supported (they raise NotImplementedError): higher-order factors,
multiple groups, inequality and non-linear constraints, conditional.x.
"""
import ast
import inspect
import re
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    from ._ordinal import _NAME, _linear, _parse_term
    from ._rcompat import r_deparse_call, r_uniroot
except ImportError:
    from _ordinal import _NAME, _linear, _parse_term
    from _rcompat import r_deparse_call, r_uniroot

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
            term, term_op = term.strip(), op
            if op == "~" and term.split("*")[-1].strip() == "1":
                # an intercept (y ~ 1, y ~ 0*1, F ~ a*1): lavaan's "~1" row, with the usual modifiers
                term_op, name, modifiers = "~1", "", term.rpartition("*")[0]
                _, fixed, label, start, free = _parse_term(f"{modifiers}*intercept" if modifiers else "intercept")
            else:
                name, fixed, label, start, free = _parse_term(term)
            key = (lhs, term_op, name)
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
                statements.append({"lhs": lhs, "op": term_op, "rhs": name, "fixed": fixed, "label": label or "",
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


def _ncp_interval(x2, df, n, level=0.90):
    """
    lavaan's noncentrality bounds for the RMSEA and GFI intervals (lav_fit_rmsea_ci, lav_fit_gfi_ci): uniroot
    of pchisq(x2, df, ncp) on [0, x2] and [0, max(n, 4 x2)], 0 when a bound does not exist (or df < 1).
    """
    upper_perc, lower_perc = 1 - (1 - level) / 2, (1 - level) / 2

    def cdf(lam):
        return stats.chi2.cdf(x2, df) if lam <= 0 else stats.ncx2.cdf(x2, df, lam)

    def root(probability, upper):
        try:
            return r_uniroot(lambda lam: cdf(lam) - probability, 0, upper)
        except ValueError:  # R's try(uniroot()) gives NA
            return np.nan

    lower = 0.0 if df < 1 or cdf(0) - upper_perc < 0 else root(upper_perc, x2)
    bound = max(n, x2 * 4)
    upper = 0.0 if df < 1 or cdf(bound) - lower_perc > 0 or cdf(0) - lower_perc < 0 else root(lower_perc, bound)
    return lower, upper


def _nfi(x2, df, x2_null, df_null):
    """lavaan's lav_fit_nfi."""
    if df > df_null or np.isclose(x2_null, 0):
        return np.nan
    return (x2_null - x2) / x2_null if df > 0 else 1.0


def _rfi(x2, df, x2_null, df_null):
    """lavaan's lav_fit_rfi."""
    if df > df_null:
        return np.nan
    if df > 0 and df_null > 0:
        t1, t2 = x2_null / df_null - x2 / df, x2_null / df_null
        if not (np.isfinite(t1) and np.isfinite(t2)):
            return np.nan
        return 1.0 if t1 < 0 or t2 < 0 else t1 / t2
    return 1.0


class Parameter:
    def __init__(self, lhs, op, rhs, free, value, user, label="", start=None):
        self.lhs, self.op, self.rhs, self.free, self.value, self.user = lhs, op, rhs, free, value, user
        self.label, self.start = label, start


class SEMFit:
    """A fitted model: parameter table, implied covariance, sample statistics and helpers."""

    def __init__(self, model, data, std_lv=False, missing="listwise", meanstructure=False, fixed_x=True):
        statements, extras = _parse(model)
        self.model_syntax = model
        self.std_lv = std_lv
        latent = list(dict.fromkeys(s["lhs"] for s in statements if s["op"] == "=~"))
        indicators = list(dict.fromkeys(s["rhs"] for s in statements if s["op"] == "=~"))
        if any(name in latent for name in indicators):
            raise NotImplementedError("higher-order factors (latent variables measured by latent variables) are not "
                                      "supported")
        regressions = [s for s in statements if s["op"] == "~"]
        eqs_y = list(dict.fromkeys(s["lhs"] for s in regressions))
        eqs_x = list(dict.fromkeys(s["rhs"] for s in regressions))
        # lavaan's variable classes (lav_pt_vnames): outcomes that are no indicators (ov.y), observed predictors
        # that are neither (ov.x), and the order of the observed variables: indicators, outcomes, predictors, then
        # variables named only in covariances (all left-hand sides, then all right-hand sides)
        ov_y = [v for v in eqs_y if v not in latent and v not in indicators]
        ov_x = [v for v in eqs_x if v not in latent and v not in indicators and v not in ov_y]
        covariances = [s for s in statements if s["op"] == "~~"]
        named = [s["lhs"] for s in covariances] + [s["rhs"] for s in covariances]
        named += [s["lhs"] for s in statements if s["op"] == "~1"]
        observed = list(dict.fromkeys(indicators + ov_y + ov_x + [v for v in named if v not in latent]))
        # a predictor the syntax gives a variance, covariance or intercept is random, not exogenous (lavaan warns)
        random_x = [v for v in ov_x if v in named]
        if random_x:
            warnings.warn("model syntax contains variance/covariance/intercept formulas involving (an) exogenous "
                          f"variable(s): [{' '.join(random_x)}]; these variables will now be treated as random "
                          "introducing additional free parameters. If you wish to treat those variables as fixed, "
                          "remove these formulas from the model syntax. Otherwise, consider adding the fixed_x = "
                          "FALSE option.")
        ov_x = [v for v in ov_x if v not in random_x]
        self.latent, self.observed = latent, observed
        self.ov_x, self.fixed_x = ov_x, fixed_x
        self.endogenous = [v for v in eqs_y if v in latent]
        lv_x = [v for v in latent if v not in eqs_y]  # lavaan's lv.x: latent variables that depend on nothing
        # lavaan's lv.y and ov.y: outcomes that predict nothing; their residuals covary (auto.cov.y)
        pure_y = [v for v in latent if v in eqs_y and v not in eqs_x]
        pure_y += [v for v in observed if v in eqs_y and v not in eqs_x and v not in indicators]

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

        def add(lhs, rhs, free=True, value=0.0, exo=False):
            if (lhs, "~~", rhs) not in existing:
                existing.update({(lhs, "~~", rhs), (rhs, "~~", lhs)})
                table.append(Parameter(lhs, "~~", rhs, free, value, False))
                table[-1].exo = exo

        # lavaan's automatic rows (lav_pt_flat), in its order
        for name in observed:
            if name not in ov_x:
                add(name, name)
        for name in latent:
            # lavaan's std.lv fixes every latent (residual) variance to 1, endogenous ones included
            add(name, name, free=not std_lv, value=1.0 if std_lv else 0.0)
        for i, a in enumerate(lv_x):  # auto.cov.lv.x
            for b in lv_x[i + 1:]:
                add(a, b)
        for i, a in enumerate(pure_y):  # auto.cov.y
            for b in pure_y[i + 1:]:
                add(a, b)
        for j, a in enumerate(ov_x):  # the predictors' (co)variances: fixed at the sample's with fixed.x
            for b in ov_x[j:]:
                add(a, b, free=not fixed_x, exo=fixed_x)
        if missing not in ("listwise", "ml"):
            raise ValueError('missing must be "listwise" or "ml"')
        self.missing = missing
        # lavaan adds a mean structure with missing = "ml" and when the syntax has intercepts
        self.meanstructure = meanstructure or missing == "ml" or any(prm.op == "~1" for prm in table)
        if self.meanstructure:
            for name in observed:  # free intercepts (int.ov.free), the fixed predictors' at their means
                if (name, "~1", "") not in existing:
                    table.append(Parameter(name, "~1", "", not (fixed_x and name in ov_x), 0.0, False))
                    table[-1].exo = fixed_x and name in ov_x
            for name in latent:  # latent means fixed at 0 (int.lv.free = FALSE) unless the syntax frees them
                if (name, "~1", "") not in existing:
                    table.append(Parameter(name, "~1", "", False, 0.0, False))
        for prm in table:
            prm.exo = getattr(prm, "exo", False)
        self.table = table
        self._setup_representation()

        absent = [v for v in observed if v not in data.columns]
        if absent:
            raise ValueError("data has no column " + ", ".join(absent))
        values = data[observed].apply(pd.to_numeric, errors="coerce")
        if missing == "listwise":
            values = values.dropna()
        else:
            if fixed_x and ov_x:  # lavaan drops the cases without a fixed predictor
                incomplete = values[ov_x].isna().any(axis=1)
                if incomplete.any():
                    warnings.warn(f"{int(incomplete.sum())} cases were deleted due to missing values in exogenous "
                                  "variable(s), while fixed_x = TRUE.")
                values = values[~incomplete]
            values = values.dropna(how="all")
        self.data = values
        self.n = len(values)
        self.p, self.m = len(observed), len(self.lv_names)
        self.free = [p for p in table if p.free]
        self._setup_constraints()
        x = values.to_numpy(dtype=float)
        if missing == "listwise":
            self.sample_mean = x.mean(axis=0)
            centred = x - self.sample_mean
            self.sample_cov = centred.T @ centred / self.n
        else:
            self._patterns = _missing_patterns(x)
            self.sample_mean, self.sample_cov = _em_saturated(x, self._patterns)
        ov = self._index["ov"]
        for prm in table:  # fixed.x: the predictors' sample (co)variances and means, divisor N
            if prm.exo and prm.op == "~~":
                prm.value = self.sample_cov[ov[prm.lhs], ov[prm.rhs]]
            elif prm.exo:
                prm.value = self.sample_mean[ov[prm.lhs]]
        self._start()
        self._estimate()

    def _setup_representation(self):
        """
        lavaan's LISREL representation (lav_lisrel, conditional.x = FALSE): observed variables in regressions,
        and those sharing a covariance with them or with a latent variable, become dummy latent variables (lambda
        1, no residual of their own); the latent variables are the model's followed by the dummies in the order of
        the observed variables. Each parameter's matrix and position: lambda, theta, psi, beta, nu or alpha.
        """
        latent, observed = self.latent, self.observed
        regressions = [p for p in self.table if p.op == "~"]
        names = list(dict.fromkeys([p.lhs for p in regressions] + [p.rhs for p in regressions]))
        dummy = {v for v in names if v not in latent}
        covariances = [p for p in self.table if p.op == "~~"]
        linked = [p for p in covariances if p.lhs in dummy or p.rhs in dummy]
        lhs_linked, rhs_linked = {p.lhs for p in linked}, {p.rhs for p in linked}
        linked += [p for p in covariances if p.lhs in lhs_linked or p.rhs in rhs_linked]
        dummy |= {v for p in linked for v in (p.lhs, p.rhs)}
        dummy |= {p.lhs for p in covariances if p.lhs in observed and p.rhs in latent}
        dummy |= {p.rhs for p in covariances if p.lhs in latent and p.rhs in observed}
        self.dummies = [v for v in observed if v in dummy]
        self.lv_names = latent + self.dummies
        self._index = {"ov": {v: i for i, v in enumerate(observed)}, "lv": {v: i for i, v in enumerate(self.lv_names)}}
        ov, lv = self._index["ov"], self._index["lv"]
        for prm in self.table:
            if prm.op == "=~":
                prm.where = ("beta", lv[prm.rhs], lv[prm.lhs]) if prm.rhs in lv else ("lambda", ov[prm.rhs],
                                                                                       lv[prm.lhs])
            elif prm.op == "~":
                prm.where = ("beta", lv[prm.lhs], lv[prm.rhs])
            elif prm.op == "~1":
                prm.where = ("alpha", lv[prm.lhs], 0) if prm.lhs in lv else ("nu", ov[prm.lhs], 0)
            elif prm.lhs in lv:
                prm.where = ("psi", lv[prm.lhs], lv[prm.rhs])
            else:
                prm.where = ("theta", ov[prm.lhs], ov[prm.rhs])

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
    def _where(self, op, lhs, rhs):
        """The matrix and position of a parameter (op, lhs, rhs), as lavaan's representation places it."""
        ov, lv = self._index["ov"], self._index["lv"]
        if op == "=~":
            return ("beta", lv[rhs], lv[lhs]) if rhs in lv else ("lambda", ov[rhs], lv[lhs])
        if op == "~":
            return "beta", lv[lhs], lv[rhs]
        if op == "~1":
            return ("alpha", lv[lhs], 0) if lhs in lv else ("nu", ov[lhs], 0)
        return ("psi", lv[lhs], lv[rhs]) if lhs in lv else ("theta", ov[lhs], ov[rhs])

    def _all_matrices(self, theta=None):
        """lavaan's model matrices: lambda (with 1 for each dummy), theta, psi, beta, nu and alpha."""
        if theta is not None:
            for parameter, value in zip(self.free, theta):
                parameter.value = value
        mats = {"lambda": np.zeros((self.p, self.m)), "theta": np.zeros((self.p, self.p)),
                "psi": np.zeros((self.m, self.m)), "beta": np.zeros((self.m, self.m)), "nu": np.zeros(self.p),
                "alpha": np.zeros(self.m)}
        for name in self.dummies:
            mats["lambda"][self._index["ov"][name], self._index["lv"][name]] = 1.0
        for prm in self.table:
            kind, i, j = prm.where
            if kind in ("nu", "alpha"):
                mats[kind][i] = prm.value
            else:
                mats[kind][i, j] = prm.value
                if kind in ("theta", "psi"):
                    mats[kind][j, i] = prm.value
        return mats

    def _matrices(self, theta=None):
        mats = self._all_matrices(theta)
        return mats["lambda"], mats["theta"], mats["psi"], mats["beta"]

    def implied_mean(self, theta=None):
        """Model-implied means, nu + Lambda (I - B)^-1 alpha (the model's latent means are fixed at 0)."""
        mats = self._all_matrices(theta)
        c = np.linalg.inv(np.eye(self.m) - mats["beta"])
        return mats["nu"] + mats["lambda"] @ c @ mats["alpha"]

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

        def effect(name):
            """How a unit of a variable's disturbance moves the observed variables: through (I - B)^-1 for a
            latent variable or dummy, only itself for any other observed variable."""
            if name in lv:
                return lam @ c[:, lv[name]]
            vector = np.zeros(self.p)
            vector[ov[name]] = 1.0
            return vector

        result = []
        for op, lhs, rhs in parameters:
            kind, i, j = self._where(op, lhs, rhs)
            d = np.zeros((self.p, self.p))
            if op == "~~":
                # a (residual) covariance of two disturbances; for a candidate between a dummy and another
                # observed variable this is lavaan's, which turns that variable into a dummy too
                first, second = effect(lhs), effect(rhs)
                d = np.outer(first, second)
                if lhs != rhs:
                    d = d + d.T
            elif kind == "lambda":
                e = np.zeros((self.p, self.m))
                e[i, j] = 1
                d = e @ a @ lam.T + lam @ a @ e.T
            elif kind == "beta":
                e = np.zeros((self.m, self.m))
                e[i, j] = 1
                dc = c @ e @ c
                d = lam @ (dc @ psi @ c.T + c @ psi @ dc.T) @ lam.T
            result.append(d)  # intercepts (nu, alpha) move the means, not the covariances
        return result

    def _mean_derivatives(self, parameters):
        """dmu/dtheta (length p) for each (op, lhs, rhs) parameter, mu = nu + Lambda (I - B)^-1 alpha."""
        mats = self._all_matrices()
        lam = mats["lambda"]
        c = np.linalg.inv(np.eye(self.m) - mats["beta"])
        eta = c @ mats["alpha"]
        result = []
        for op, lhs, rhs in parameters:
            kind, i, j = self._where(op, lhs, rhs)
            d = np.zeros(self.p)
            if kind == "nu":
                d[i] = 1.0
            elif kind == "alpha":
                d = lam @ c[:, i]
            elif kind == "lambda":
                d[i] = eta[j]
            elif kind == "beta":
                d = lam @ c[:, i] * eta[j]
            result.append(d)
        return result

    def _fixed_x(self):
        """The positions of the fixed predictors (fixed.x) among the observed variables; none without them."""
        return [self._index["ov"][v] for v in self.ov_x] if self.fixed_x else []

    def _without_x(self, d_sigma, d_mu=None):
        """
        The derivatives with the fixed predictors' own moments left out (fixed.x): lavaan's expected information
        zeroes the rows and columns of those moments in the unrestricted model's information, which is the
        same as zeroing the predictors' block of dSigma (and their entries of dmu).
        """
        x = self._fixed_x()
        if not x:
            return d_sigma, d_mu
        d_sigma = [d.copy() for d in d_sigma]
        for d in d_sigma:
            d[np.ix_(x, x)] = 0.0
        if d_mu is not None:
            d_mu = [d.copy() for d in d_mu]
            for d in d_mu:
                d[x] = 0.0
        return d_sigma, d_mu

    def _expected_information(self, parameters):
        """
        Expected information per observation for the given parameters: the complete-data formula,
        or with missing data its sum over the patterns (means included), as lavaan computes it; without the
        fixed predictors' own moments (fixed.x).
        """
        d_sigma = self._derivatives(parameters)
        if self.missing != "ml":
            return self._information(d_sigma, self._mean_derivatives(parameters) if self.meanstructure else None)
        d_sigma, d_mu = self._without_x(d_sigma, self._mean_derivatives(parameters))
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
            if not self.meanstructure:
                return np.array([0.5 * np.sum(w * d) for d in d_sigma])
            # the mean term (ybar - mu)' Sigma^-1 (ybar - mu) of the discrepancy
            g = inv @ (self.sample_mean - self.implied_mean())
            w = w - np.outer(g, g)
            return np.array([0.5 * np.sum(w * ds) - g @ dm
                             for ds, dm in zip(d_sigma, self._mean_derivatives(parameters))])
        d_mu = self._mean_derivatives(parameters)
        mu = self.implied_mean()
        grad_sigma, grad_mu = np.zeros((self.p, self.p)), np.zeros(self.p)
        for observed, n_k, mean_k, cov_k in self._patterns:
            inv = np.linalg.inv(self.sigma[np.ix_(observed, observed)])
            diff = mean_k - mu[observed]
            grad_sigma[np.ix_(observed, observed)] += 0.5 * n_k * (inv - inv @ (cov_k + np.outer(diff, diff)) @ inv)
            grad_mu[observed] -= n_k * inv @ diff
        return np.array([np.sum(grad_sigma * ds) + grad_mu @ dm for ds, dm in zip(d_sigma, d_mu)]) / self.n

    def _information(self, derivatives, mean_derivatives=None):
        """
        Expected information per observation, 0.5 tr(Sigma^-1 dS_k Sigma^-1 dS_l), plus dmu_k' Sigma^-1 dmu_l
        with a mean structure, without the fixed predictors' own moments (fixed.x).
        """
        derivatives, mean_derivatives = self._without_x(derivatives, mean_derivatives)
        inv = np.linalg.inv(self.sigma)
        products = [inv @ d for d in derivatives]
        q = len(products)
        information = np.empty((q, q))
        for k in range(q):
            for l in range(k, q):
                value = 0.5 * np.sum(products[k] * products[l].T)
                if mean_derivatives is not None:
                    value += mean_derivatives[k] @ inv @ mean_derivatives[l]
                information[k, l] = information[l, k] = value
        return information

    def _x_logl(self):
        """The fixed predictors' own log-likelihood at their sample moments (lavaan's logl leaves it out)."""
        x = self._fixed_x()
        if not x:
            return 0.0
        logdet = np.linalg.slogdet(self.sample_cov[np.ix_(x, x)])[1]
        return -self.n / 2 * (len(x) * np.log(2 * np.pi) + logdet + len(x))

    # ------------------------------------------------------------------ estimation
    def _start(self):
        """
        Simple starting values: residual variances half the observed (a random predictor's its whole variance),
        latent variances 0.05, covariances among random predictors the sample's.
        """
        s = self.sample_cov
        ov = self._index["ov"]
        predictors = {p.rhs for p in self.table if p.op == "~"} - {p.lhs for p in self.table if p.op == "~"}
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
            elif prm.op == "~~" and prm.lhs in ov and prm.rhs in ov:
                i, j = ov[prm.lhs], ov[prm.rhs]
                if prm.lhs in predictors and prm.rhs in predictors:
                    prm.value = s[i, j]  # a random predictor's (co)variance: the sample's
                else:
                    prm.value = 0.5 * s[i, i] if i == j else 0.0
            elif prm.op == "~~" and prm.lhs == prm.rhs:
                prm.value = 0.05
            elif prm.op == "~1":
                prm.value = self.sample_mean[ov[prm.lhs]] if prm.lhs in ov else 0.0
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
        parameters = [(prm.op, prm.lhs, prm.rhs) for prm in self.free]
        grad = np.array([np.sum(d_sigma * ds) + d_mu @ dm
                         for ds, dm in zip(self._derivatives(parameters), self._mean_derivatives(parameters))])
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
        """lavaan's ML discrepancy, with a mean structure + (ybar - mu)' Sigma^-1 (ybar - mu)."""
        sigma = self.implied(theta)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            return np.inf
        inv = np.linalg.inv(sigma)
        value = logdet + np.trace(self.sample_cov @ inv) - self._logdet_s - self.p
        if self.meanstructure:
            diff = self.sample_mean - self.implied_mean()
            value += diff @ inv @ diff
        return value

    def _gradient(self, theta):
        sigma = self.implied(theta)
        inv = np.linalg.inv(sigma)
        w = inv @ (sigma - self.sample_cov) @ inv
        if not self.meanstructure:
            return np.array([np.sum(w * d) for d in self._derivatives()])
        g = inv @ (self.sample_mean - self.implied_mean())
        w = w - np.outer(g, g)
        parameters = [(prm.op, prm.lhs, prm.rhs) for prm in self.free]
        return np.array([np.sum(w * ds) - 2 * g @ dm
                         for ds, dm in zip(self._derivatives(parameters), self._mean_derivatives(parameters))])

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
            parameters = [(prm.op, prm.lhs, prm.rhs) for prm in self.free]
            means = self._mean_derivatives(parameters) if self.meanstructure else None
            self.vcov = self._constrained_inverse(self.n * self._information(self._derivatives(parameters), means))
        variances = np.diag(self.vcov)  # exactly 0 for a parameter the constraints pin down
        with np.errstate(invalid="ignore"):
            ses = np.sqrt(variances)
        if np.any(variances < 0):
            warnings.warn("some standard errors could not be computed (missing): the information matrix is not "
                          "positive definite at the solution, so the model may not be identified")
        for prm, se in zip(self.free, ses):
            prm.se = se

    # ------------------------------------------------------------------ results
    def latent_variances(self):
        lam, th, psi, beta = self._matrices()
        c = np.linalg.inv(np.eye(self.m) - beta)
        return np.diag(c @ psi @ c.T)

    def _standardized(self, prm):
        """
        (std.lv, std.all, std.nox) of a parameter, as lavaan's lav_standardize_lv, _all and _all_nox (cov.std):
        std.lv scales by the latent variables' standard deviations only (dummies are observed variables),
        std.all by every variable's, std.nox by every variable's but the fixed predictors'. A (residual)
        covariance is scaled by the two residual standard deviations.
        """
        ov, lv = self._index["ov"], self._index["lv"]
        var_lv = self.latent_variances()
        var_ov = np.diag(self.sigma)
        lam, th, psi, beta = self._matrices()
        real = set(self.latent)
        fixed = set(self.ov_x) if self.fixed_x else set()

        def total(name):
            return var_lv[lv[name]] if name in real else var_ov[ov[name]]

        def residual(name):
            return psi[lv[name], lv[name]] if name in lv else th[ov[name], ov[name]]

        def scale(name, kind, variance):
            if (kind == "lv" and name not in real) or (kind == "nox" and name in fixed):
                return 1.0
            with np.errstate(invalid="ignore"):  # a negative residual variance (Heywood case): missing, as lavaan
                return np.sqrt(variance(name))

        value, lhs, rhs = prm.value, prm.lhs, prm.rhs
        out = []
        for kind in ("lv", "all", "nox"):
            if prm.op == "~1":
                out.append(value / scale(lhs, kind, total))
            elif prm.op == "=~":
                out.append(value * scale(lhs, kind, total) / scale(rhs, kind, total))
            elif prm.op == "~":
                out.append(value * scale(rhs, kind, total) / scale(lhs, kind, total))
            elif lhs == rhs:
                out.append(value / (scale(lhs, kind, total) * scale(rhs, kind, total)))
            else:
                out.append(value / (scale(lhs, kind, residual) * scale(rhs, kind, residual)))
        return tuple(out)

    def parameter_estimates(self, level=0.95, remove_eq=True):
        """
        lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE): the parameters, the defined
        parameters (:=, with delta-method standard errors, and standardized as their expressions of the
        standardized parameters), with remove_eq=False (lavaan's remove.eq, as rwf's report_cfa) the user's
        equality constraints (==, est the difference of their sides), then the R squared rows. A label column
        follows rhs when the model has labels; std.nox follows std.all when the model has fixed predictors
        (fixed.x), as lavaan's. The index is lavaan's row numbers: positions in its parameter table, whose hidden
        rows (the constraints of shared labels) are skipped.
        """
        z_crit = stats.norm.ppf(1 - (1 - level) / 2)
        standardized = [self._standardized(prm) for prm in self.table]
        rows, index = [], []
        for k, (prm, (std_lv, std_all, std_nox)) in enumerate(zip(self.table, standardized), start=1):
            se = getattr(prm, "se", 0.0) if prm.free else 0.0
            z = prm.value / se if prm.free and se > 0 else np.nan
            p_value = 2 * stats.norm.sf(abs(z)) if prm.free and se > 0 else np.nan
            rows.append({"lhs": prm.lhs, "op": prm.op, "rhs": prm.rhs, "label": prm.label, "est": prm.value,
                         "se": se, "z": z, "pvalue": p_value, "ci.lower": prm.value - z_crit * se,
                         "ci.upper": prm.value + z_crit * se, "std.lv": std_lv, "std.all": std_all,
                         "std.nox": std_nox})
            index.append(k)
        for k, row in enumerate(self._extra_rows(z_crit, standardized), start=len(self.table) + 1):
            if row["op"] == ":=" or not remove_eq:
                rows.append(row)
                index.append(k)
        first_r2 = len(self.table) + len(self._extras) + self._system_eq + 1
        for k, (name, value) in enumerate(self.rsquare().items(), start=first_r2):
            rows.append({"lhs": name, "op": "r2", "rhs": name, "label": "", "est": value, "se": np.nan, "z": np.nan,
                         "pvalue": np.nan, "ci.lower": np.nan, "ci.upper": np.nan, "std.lv": np.nan,
                         "std.all": np.nan, "std.nox": np.nan})
            index.append(k)
        table = pd.DataFrame(rows, index=index)
        if not any(prm.label for prm in self.table) and not any(e["op"] == ":=" for e in self._extras):
            table = table.drop(columns="label")  # lavaan has the column only when the model has labels
        if not any(prm.exo for prm in self.table):
            table = table.drop(columns="std.nox")  # lavaan has the column only with fixed predictors
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
        std_nox = np.array([s[2] for s in standardized], dtype=float)
        names = [e["lhs"] for e in self._extras if e["op"] == ":="]
        estimates = self._definitions(values)
        defined_lv, defined_all = self._definitions(std_lv), self._definitions(std_all)
        defined_nox = self._definitions(std_nox)
        se = {}
        if names:
            jacobian = self._definition_jacobian(names)
            variances = np.diag(jacobian @ self.vcov @ jacobian.T)
            se = dict(zip(names, np.sqrt(np.clip(variances, 0, None))))
        labels, labels_lv, labels_all, labels_nox = (self._label_values(v) for v in (values, std_lv, std_all, std_nox))
        rows = []
        for extra in self._extras:
            if extra["op"] == ":=":
                name = extra["lhs"]
                est, error = float(np.real(estimates[name])), se[name]
                z = est / error if error > 0 else np.nan
                rows.append({"lhs": name, "op": ":=", "rhs": extra["rhs"], "label": name, "est": est, "se": error,
                             "z": z, "pvalue": 2 * stats.norm.sf(abs(z)) if error > 0 else np.nan,
                             "ci.lower": est - z_crit * error, "ci.upper": est + z_crit * error,
                             "std.lv": float(np.real(defined_lv[name])), "std.all": float(np.real(defined_all[name])),
                             "std.nox": float(np.real(defined_nox[name]))})
            else:  # lavaan reports the constraint's value, lhs - rhs (0 when it holds), with a zero standard error
                coefficients, constant = extra["linear"]

                def difference(label_values):
                    return sum(c * label_values[label] for label, c in coefficients.items()) - constant

                est = difference(labels)
                rows.append({"lhs": extra["lhs"], "op": "==", "rhs": extra["rhs"], "label": "", "est": est,
                             "se": 0.0, "z": np.nan, "pvalue": np.nan, "ci.lower": est, "ci.upper": est,
                             "std.lv": difference(labels_lv), "std.all": difference(labels_all),
                             "std.nox": difference(labels_nox)})
        return rows

    def fit_measures(self):
        """lavaan::fitMeasures(fit) for ML on complete data, in lavaan's order and names."""
        n, p = self.n, self.p
        s, sigma = self.sample_cov, self.sigma
        npar = self.npar  # the free parameters less the equality constraints, as lavaan's fitMeasures
        # with fixed.x the predictors' own moments are no data (lavaan's ndat), and the baseline model keeps them
        x = self._fixed_x()
        nx = len(x)
        moments = p * (p + 1) / 2 + (p if self.meanstructure else 0) - nx * (nx + 1) / 2 - (nx if self.meanstructure
                                                                                            else 0)
        df = moments - npar
        # the baseline model: the predictors' block (lavaan's ov.x, fixed or not) saturated, the other variables
        # uncorrelated with everything
        bx = [self._index["ov"][v] for v in self.ov_x]
        y = [i for i in range(p) if i not in bx]
        free_x = 0 if self.fixed_x else len(bx) * (len(bx) + 1) / 2 + (len(bx) if self.meanstructure else 0)
        if self.missing == "ml":
            unrestricted = self._pattern_logl(self.sigma_h1(), self.sample_mean)
            chisq = 2 * (unrestricted - self.logl)
            self.fmin = chisq / (2 * n)
            baseline_chisq = 2 * (unrestricted - self._independence_logl())
            baseline_df = moments - 2 * len(y) - free_x
        else:
            chisq = 2 * n * self.fmin
            logdet_x = np.linalg.slogdet(s[np.ix_(bx, bx)])[1] if bx else 0.0
            baseline_f = np.sum(np.log(np.diag(s)[y])) + logdet_x - self._logdet_s
            # the baseline model's free variances, and with a mean structure its free (saturated) means
            baseline_chisq = n * baseline_f
            baseline_df = moments - len(y) * (2 if self.meanstructure else 1) - free_x
        out = {"npar": npar, "fmin": self.fmin, "chisq": chisq, "df": df,
               "pvalue": stats.chi2.sf(chisq, df) if df > 0 else np.nan,
               "baseline.chisq": baseline_chisq, "baseline.df": baseline_df,
               "baseline.pvalue": stats.chi2.sf(baseline_chisq, baseline_df)}
        d, db = max(chisq - df, 0), max(baseline_chisq - baseline_df, 0)
        out["cfi"] = 1 - d / max(db, d) if max(db, d) > 0 else 1.0
        tli = ((baseline_chisq / baseline_df) - (chisq / df)) / ((baseline_chisq / baseline_df) - 1) if df > 0 else 1.0
        out["tli"] = out["nnfi"] = tli
        out["rfi"] = _rfi(chisq, df, baseline_chisq, baseline_df)
        out["nfi"] = _nfi(chisq, df, baseline_chisq, baseline_df)
        out["pnfi"] = (df / baseline_df) * ((baseline_chisq - chisq) / baseline_chisq) if baseline_df > 0 and \
            baseline_chisq > 0 else np.nan
        out["ifi"] = (baseline_chisq - chisq) / (baseline_chisq - df)
        out["rni"] = 1 - (chisq - df) / (baseline_chisq - baseline_df)
        sign, logdet_sigma = np.linalg.slogdet(sigma)
        inv = np.linalg.inv(sigma)
        if self.missing == "ml":
            logl = self.logl
        else:
            logl = -n / 2 * (p * np.log(2 * np.pi) + logdet_sigma + np.trace(s @ inv))
            if self.meanstructure:
                diff = self.sample_mean - self.implied_mean()
                logl -= n / 2 * (diff @ inv @ diff)
            unrestricted = -n / 2 * (p * np.log(2 * np.pi) + self._logdet_s + p)
        out["logl"], out["unrestricted.logl"] = logl - self._x_logl(), unrestricted - self._x_logl()
        logl = out["logl"]
        out["aic"] = -2 * logl + 2 * npar
        out["bic"] = -2 * logl + npar * np.log(n)
        out["ntotal"] = n
        out["bic2"] = -2 * logl + npar * np.log((n + 2) / 24)
        out["rmsea"] = np.sqrt(max(chisq / n / df - 1 / n, 0)) if df > 0 else 0.0

        lambda_lower, lambda_upper = _ncp_interval(chisq, df, n)
        out["rmsea.ci.lower"] = np.sqrt(lambda_lower / (n * df)) if lambda_lower > 0 else 0.0
        out["rmsea.ci.upper"] = np.sqrt(lambda_upper / (n * df)) if lambda_upper > 0 else 0.0
        out["rmsea.ci.level"] = 0.90
        out["rmsea.pvalue"] = stats.ncx2.sf(chisq, df, n * df * 0.05 ** 2) if df > 0 else np.nan
        out["rmsea.close.h0"] = 0.05
        out["rmsea.notclose.pvalue"] = stats.ncx2.cdf(chisq, df, n * df * 0.08 ** 2) if df > 0 else np.nan
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
        # lavaan's crmr (lav_residuals_summary) scales the mean residuals by the observed standard deviations
        out["crmr"] = root_mean_square((corr_s - corr_sigma)[strict_lower], mean_residual / sd_s if self.meanstructure
                                       else np.zeros(0))
        out["crmr_nomean"] = root_mean_square((corr_s - corr_sigma)[strict_lower], np.zeros(0))
        mplus = (corr_s - corr_sigma).copy()
        np.fill_diagonal(mplus, (np.diag(s) - np.diag(sigma)) / np.diag(s))
        out["srmr_mplus"] = root_mean_square(mplus[lower], crmr_means)
        out["srmr_mplus_nomean"] = root_mean_square(mplus[lower], np.zeros(0))
        # lavaan 0.7's gfi is the population estimate n p / (n p + 2 x2 - 2 df), x2 Browne's residual-based
        # statistic (lavTest's "browne.residual.nt.model"; the chi square with FIML); gfi_lisrel is the classic GFI
        np_ = n * p
        x2_rls = chisq if self.missing == "ml" else self._browne_residual_statistic()
        denominator = np_ + 2 * x2_rls - 2 * df
        out["gfi"] = min(np_ / denominator, 1.0) if denominator > 0 else np.nan
        lambda_lower, lambda_upper = _ncp_interval(x2_rls, df, n)
        out["gfi.ci.lower"] = min(np_ / (np_ + 2 * lambda_upper), 1.0)
        out["gfi.ci.upper"] = min(np_ / (np_ + 2 * lambda_lower), 1.0)
        out["gfi.ci.level"] = 0.90
        out["cn_05"], out["cn_01"] = (np.nan, np.nan) if df == 0 and chisq < np.sqrt(np.finfo(float).eps) else (
            stats.chi2.ppf(0.95, df) / (chisq / n) + 1, stats.chi2.ppf(0.99, df) / (chisq / n) + 1)

        # lavaan's 1 - d'Vd / s'Vs (lav_fit_gfi_lisrel) over the means (with a mean structure) and covariances,
        # V the ML weights bdiag(Sigma^-1, D'(Sigma^-1 x Sigma^-1)D / 2) with the fixed predictors' rows zeroed
        def weighted(matrix, means):
            matrix = matrix.copy()
            matrix[np.ix_(x, x)] = 0.0
            value = 0.5 * np.trace(inv @ matrix @ inv @ matrix)
            if self.meanstructure:
                means = means.copy()
                means[x] = 0.0
                value += means @ inv @ means
            return value

        gfi_lisrel = 1 - weighted(s - sigma, self.sample_mean - self.implied_mean()) / weighted(s, self.sample_mean)
        out["gfi_lisrel"] = gfi_lisrel
        nel = p * (p + 1) / 2 + (p if self.meanstructure else 0)  # every sample moment, the predictors' too
        out["agfi_lisrel"] = 1 - (nel / df) * (1 - gfi_lisrel) if df > 0 else 1.0
        out["pgfi"] = (df / nel) * gfi_lisrel
        out["mfi"] = np.exp(-0.5 * (chisq - df) / n)
        out["ecvi"] = chisq / n + 2 * npar / n
        if self.missing == "ml":
            # with missing data lavaan weights the moments of the classic GFI differently (its FIML weight
            # matrix is not ported), so these are missing rather than different from lavaan
            for name in ["gfi_lisrel", "agfi_lisrel", "pgfi"]:
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

    def _browne_residual_statistic(self):
        """
        lavaan's lavTest(fit, "browne.residual.nt.model") (lav_test_browne, normal theory, model based):
        n [r' G r - b' (D' G D)^-1 b], b = D' G r, with r the residuals of the means (with a mean structure)
        and the covariances (vech), D their derivatives by the free parameters (times the constraints' basis),
        and G the inverse of their normal-theory covariance at the model's Sigma (lav_samp_gamma_nt; with
        fixed predictors Gamma(Sigma) - Gamma(R), R = Sigma less the outcomes' covariance given the predictors,
        and that conditional covariance for the means). Without constraints the fixed predictors' moments are
        left out first; with them lavaan inverts the whole (then singular) matrix by a pseudo-inverse.
        """
        p = self.p
        pairs = [(i, j) for j in range(p) for i in range(j, p)]
        sigma, mu = self.sigma, self.implied_mean()
        parameters = [(prm.op, prm.lhs, prm.rhs) for prm in self.free]
        d_sigma = self._derivatives(parameters)
        residual = [self.sample_cov[i, j] - sigma[i, j] for i, j in pairs]
        delta = [[d[i, j] for d in d_sigma] for i, j in pairs]

        def gamma_nt(cov):  # 2 D+ (cov x cov) D+' in vech coordinates
            return np.array([[cov[i, k] * cov[j, l] + cov[i, l] * cov[j, k] for k, l in pairs] for i, j in pairs])

        x = self._fixed_x()
        if x:
            y = [i for i in range(p) if i not in x]
            conditional = np.zeros((p, p))
            conditional[np.ix_(y, y)] = sigma[np.ix_(y, y)] - sigma[np.ix_(y, x)] @ np.linalg.solve(
                sigma[np.ix_(x, x)], sigma[np.ix_(x, y)])
            gamma, mean_block = gamma_nt(sigma) - gamma_nt(sigma - conditional), conditional
        else:
            gamma, mean_block = gamma_nt(sigma), sigma
        drop = [i in x and j in x for i, j in pairs]
        if self.meanstructure:
            d_mu = self._mean_derivatives(parameters)
            residual = list(self.sample_mean - mu) + residual
            delta = [[d[i] for d in d_mu] for i in range(p)] + delta
            q = len(pairs)
            gamma = np.block([[mean_block, np.zeros((p, q))], [np.zeros((q, p)), gamma]])
            drop = [i in x for i in range(p)] + drop
        residual, delta = np.array(residual), np.array(delta).reshape(len(residual), len(parameters))
        if self._constrained:
            # lavaan's equality-constraint branch keeps the predictors' moments: solve(), else MASS::ginv()
            delta = delta @ self._jac
            try:
                inverse = np.linalg.inv(gamma)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(gamma)
        else:
            keep = ~np.array(drop)
            residual, delta, gamma = residual[keep], delta[keep], gamma[np.ix_(keep, keep)]
            inverse = np.linalg.inv(gamma)
        weights_residual = inverse @ residual
        a = delta.T @ inverse @ delta
        b = delta.T @ weights_residual
        try:
            correction = b @ np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            correction = b @ np.linalg.pinv(a) @ b
        statistic = residual @ weights_residual - correction
        if -1e-8 < statistic < 0:
            statistic = 0.0
        return self.n * statistic

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
        """
        Log-likelihood of the baseline model (free means and variances, no covariances) with FIML; with fixed.x
        the predictors (complete: lavaan drops cases without them) keep their sample means and covariances.
        """
        data = self.data.to_numpy(dtype=float)
        x = [self._index["ov"][v] for v in self.ov_x]
        total = 0.0
        for k, column in enumerate(data.T):
            if k in x:
                continue
            values = column[~np.isnan(column)]
            variance = values.var()
            total -= 0.5 * len(values) * (np.log(2 * np.pi) + np.log(variance) + 1)
        if x:  # the predictors' block saturated (by EM when a predictor has missing values)
            block = data[:, x]
            block = block[~np.isnan(block).all(axis=1)]
            patterns = _missing_patterns(block)
            mean, cov = _em_saturated(block, patterns)
            for observed, n_k, mean_k, cov_k in patterns:
                sub = cov[np.ix_(observed, observed)]
                inv = np.linalg.inv(sub)
                diff = mean_k - mean[observed]
                total -= 0.5 * n_k * (len(observed) * np.log(2 * np.pi) + np.linalg.slogdet(sub)[1]
                                      + np.sum(inv * cov_k) + diff @ inv @ diff)
        return total

    def _candidates(self):
        """lavaan's full parameter list (lav_pt_full, strict.exo = FALSE) for modification indices."""
        lv = self.latent
        eqs_y = list(dict.fromkeys(p.lhs for p in self.table if p.op == "~"))
        eqs_x = list(dict.fromkeys(p.rhs for p in self.table if p.op == "~"))
        names = list(dict.fromkeys(eqs_y + eqs_x))
        indicators = {p.rhs for p in self.table if p.op == "=~"}
        nox = [v for v in self.observed if v not in self.ov_x]
        # no loadings on variables in regressions
        rows = [("=~", f, v) for f in lv for v in nox if v not in names]
        # no covariances with variables named only in covariances
        other = {v for v in self.observed if v not in indicators and v not in self.ov_x and v not in names}
        rows += [("~~", nox[j], nox[i]) for j in range(len(nox)) for i in range(j, len(nox))
                 if nox[j] not in other and nox[i] not in other]
        x = self.ov_x
        rows += [("~~", x[j], x[i]) for j in range(len(x)) for i in range(j, len(x))]
        rows += [("~~", lv[j], lv[i]) for j in range(len(lv)) for i in range(j, len(lv))]
        # no regressions of an indicator on a latent variable
        rows += [("~", a, b) for a in names for b in names if a != b and not (a in indicators and b in lv)]
        if self.meanstructure:
            rows += [("~1", v, "") for v in self.observed + lv]
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
        values = {key(p.op, p.lhs, p.rhs): p.value for p in self.table if not p.exo}
        rows = []
        for (op, lhs, rhs), m, e in zip(model_rows + extra_rows, mi, epc):
            size = e
            if op == "~~" and lhs == rhs:
                # cov.std: a variance is standardized by itself (not a fixed predictor's, as lavaan's exo == 0)
                size = values.get(key(op, lhs, rhs), e)
                if abs(size) < np.sqrt(np.finfo(float).eps):
                    size = np.nan
            sepc_lv, sepc_all, sepc_nox = self._standardized(Parameter(lhs, op, rhs, True, abs(size), False))
            rows.append({"lhs": lhs, "op": op, "rhs": rhs, "mi": m, "epc": e, "sepc.lv": np.sign(e) * sepc_lv,
                         "sepc.all": np.sign(e) * sepc_all, "sepc.nox": np.sign(e) * sepc_nox,
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
        lavaan::inspect(fit, "est") or (what="std") the completely standardized estimates, as a dict of
        lavaan's model matrices: lambda, theta, psi, beta (when a parameter is in it: regressions, or the
        loading of an indicator that is also an outcome) and, with a mean structure, nu and alpha. Observed
        variables in regressions are lavaan's dummy latent variables (lambda 1, their (co)variances in psi).
        """
        if what not in ("est", "std"):
            raise ValueError('what must be "est" or "std"')
        mats = {name: m.copy() for name, m in self._all_matrices().items()}
        if what == "std":
            # the same standardization as parameter_estimates' std.all, parameter by parameter; the dummies'
            # loadings of 1 stay 1
            for prm in self.table:
                kind, i, j = prm.where
                value = self._standardized(prm)[1]
                if kind in ("nu", "alpha"):
                    mats[kind][i] = value
                else:
                    mats[kind][i, j] = value
                    if kind in ("theta", "psi"):
                        mats[kind][j, i] = value
        result = {
            "lambda": pd.DataFrame(mats["lambda"], index=self.observed, columns=self.lv_names),
            "theta": pd.DataFrame(mats["theta"], index=self.observed, columns=self.observed),
            "psi": pd.DataFrame(mats["psi"], index=self.lv_names, columns=self.lv_names),
        }
        if any(prm.where[0] == "beta" for prm in self.table):
            result["beta"] = pd.DataFrame(mats["beta"], index=self.lv_names, columns=self.lv_names)
        if self.meanstructure:
            result["nu"] = pd.DataFrame({"intercept": mats["nu"]}, index=self.observed)
            result["alpha"] = pd.DataFrame({"intercept": mats["alpha"]}, index=self.lv_names)
        return result

    def predict(self):
        """
        Factor scores of the latent variables (not lavaan's dummies) by the regression method, as
        lavaan::lavPredict(fit) (lav_predict_eta_normal): E(eta) + VETA Lambda*' Sigma^-1 (y - E(y)), where
        VETA is the latent covariance, Lambda* is lambda with each dummy outcome's row replaced by its row of
        beta (lav_lisrel_lambda), and without a mean structure the dummies' intercepts reproduce their sample
        means (lav_lisrel_alpha0) and E(y) follows lav_lisrel_nu0. With missing data each case is scored from
        the variables it has.
        """
        mats = self._all_matrices()
        lam, psi, beta = mats["lambda"], mats["psi"], mats["beta"]
        c = np.linalg.inv(np.eye(self.m) - beta)
        k = len(self.latent)  # the model's latent variables come first, the dummies after them
        veta = c @ psi @ c.T
        ov, lv = self._index["ov"], self._index["lv"]
        dummy_ov = [ov[v] for v in self.dummies]
        dummy_lv = [lv[v] for v in self.dummies]
        x = self.data.to_numpy(dtype=float)
        if self.meanstructure:
            alpha, nu = mats["alpha"], mats["nu"]
        else:
            sample_mean = x.mean(axis=0)
            alpha = np.zeros(self.m)
            if self.dummies:
                alpha[dummy_lv] = np.linalg.solve((lam @ c)[np.ix_(dummy_ov, dummy_lv)], sample_mean[dummy_ov])
            if any(v in self.ov_x for v in self.dummies):
                nu = sample_mean - lam @ c @ alpha
                nu[dummy_ov] = 0.0
            else:
                nu = sample_mean
        eta = c @ alpha
        expected_y = nu + lam @ eta
        lam_star = lam.copy()
        for name in self.dummies:
            if name not in self.ov_x:  # a dummy outcome: its regression on the other latent variables
                lam_star[ov[name]] = beta[lv[name]]
        if self.missing != "ml":
            weights = veta @ lam_star.T @ np.linalg.inv(self.sigma)
            scores = eta + (x - expected_y) @ weights.T
        else:
            scores = np.empty((len(x), self.m))
            for i, row in enumerate(x):
                observed = np.flatnonzero(~np.isnan(row))
                weights = veta @ lam_star[observed].T @ np.linalg.inv(self.sigma[np.ix_(observed, observed)])
                scores[i] = eta + (row[observed] - expected_y[observed]) @ weights.T
        return pd.DataFrame(scores[:, :k], columns=self.latent, index=self.data.index)

    def rsquare(self):
        """
        lavaan's R squared (lav_inspect_rsquare): 1 - std.all of the (residual) variance of every indicator and
        every variable on the left of a regression, in the order of those variances in the parameter table;
        above 1 (after a negative residual variance) it is missing.
        """
        outcomes = {p.rhs for p in self.table if p.op == "=~"} | {p.lhs for p in self.table if p.op == "~"}
        result = {}
        for prm in self.table:
            if prm.op == "~~" and prm.lhs == prm.rhs and prm.lhs in outcomes and prm.lhs not in result:
                value = 1 - self._standardized(prm)[1]
                result[prm.lhs] = np.nan if value > 1 else value
        return result


def sem(model, data, std_lv=False, missing="listwise", meanstructure=False, fixed_x=True):
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
    meanstructure (bool, optional): Estimate intercepts (and the latent
        means the syntax frees with F ~ 1); implied by missing="ml" and by
        intercepts in the syntax (y ~ 1), as lavaan. Defaults to False.
    fixed_x (bool, optional): R's fixed.x. True (default) fixes the
        variances, covariances (and means) of observed predictors at their
        sample values, as lavaan; with missing="ml" cases without a
        predictor are then dropped. False estimates them.

    Returns:
    SEMFit: The fitted model.

    Examples:
    >>> fit = sem("E =~ E1 + E2 + E3; N =~ N1 + N2 + N3", df_ocean)
    >>> fit.parameter_estimates()
    >>> sem("E =~ E1 + E2 + E3", df_with_missing_values, missing="ml")
    >>> sem("E =~ E1 + E2 + E3; E ~ age + gender", df_ocean)
    """
    fit = SEMFit(model, data, std_lv=std_lv, missing=missing, meanstructure=meanstructure, fixed_x=fixed_x)
    fit.r_call = _lavaan_call("sem", inspect.currentframe().f_back, model, data, std_lv, missing, meanstructure,
                              fixed_x)
    return fit


def cfa(model, data, std_lv=False, missing="listwise", meanstructure=False, fixed_x=True):
    """
    lavaan::cfa: the same fit as sem() (the defaults are the same for these
    models); only the call text rwf reports says cfa.
    """
    fit = SEMFit(model, data, std_lv=std_lv, missing=missing, meanstructure=meanstructure, fixed_x=fixed_x)
    fit.r_call = _lavaan_call("cfa", inspect.currentframe().f_back, model, data, std_lv, missing, meanstructure,
                              fixed_x)
    return fit


def _lavaan_call(kind, frame, model, data, std_lv, missing, meanstructure, fixed_x=True):
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
    arguments += [("fixed.x", "FALSE")] if not fixed_x else []
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
