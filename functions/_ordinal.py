# -*- coding: utf-8 -*-
"""
Ordinal (categorical) structural equation modelling as lavaan's WLSMV:
sample statistics of ordered indicators and diagonally weighted least
squares estimation of factor models.

Sample statistics follow lavaan's two-step approach (Muthen, 1984): each
variable's thresholds from its marginal proportions (probit), each pair's
polychoric correlation by maximum likelihood with the thresholds fixed
(lavaan's rules for 2 x 2 tables included), and the asymptotic covariance
matrix Gamma of thresholds and correlations from the case-wise scores of
the univariate and bivariate likelihoods (lavaan's muthen1984).
"""
import re

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats

_SQRT_2PI = np.sqrt(2 * np.pi)
_INF = np.inf


def _phi(x):
    """Standard normal density, 0 at +-Inf."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(np.isfinite(x), np.exp(-0.5 * np.square(np.where(np.isfinite(x), x, 0))) / _SQRT_2PI, 0.0)


def _phi2(x, y, rho):
    """Standard bivariate normal density with correlation rho, 0 when either argument is infinite."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    finite = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.where(finite, x, 0), np.where(finite, y, 0)
    one_minus = 1 - rho * rho
    value = np.exp(-(xs * xs - 2 * rho * xs * ys + ys * ys) / (2 * one_minus)) / (2 * np.pi * np.sqrt(one_minus))
    return np.where(finite, value, 0.0)


def _bvn_upper(h, k, r):
    """P(X > h, Y > k) of a standard bivariate normal with correlation r (Genz's BVNU)."""
    try:
        from .glm_correlation import _bvn_upper as upper
    except ImportError:
        from glm_correlation import _bvn_upper as upper
    return upper(h, k, r)


def _bvn_cdf(x, y, rho):
    """P(X <= x, Y <= y) of a standard bivariate normal (x, y may be +-Inf)."""
    return _bvn_upper(-x, -y, rho)


def _cdf_grid(row_cuts, column_cuts, rho):
    rows = np.concatenate([[-_INF], row_cuts, [_INF]])
    columns = np.concatenate([[-_INF], column_cuts, [_INF]])
    return rows, columns, np.array([[_bvn_cdf(a, b, rho) for b in columns] for a in rows])


def _cell_probabilities(row_cuts, column_cuts, rho):
    _, _, cdf = _cdf_grid(row_cuts, column_cuts, rho)
    return cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
##########################################################################################
# UNIVARIATE: THRESHOLDS AND THEIR SCORES
##########################################################################################
class _Variable:
    """One ordered variable: category codes 0..K-1 (NaN missing), thresholds and case-wise scores."""

    def __init__(self, values, name):
        series = pd.Series(values)
        levels = sorted(series.dropna().unique())
        self.name, self.levels = name, levels
        codes = series.map({level: k for k, level in enumerate(levels)})
        self.codes = codes.to_numpy(dtype=float)
        observed = self.codes[~np.isnan(self.codes)].astype(int)
        counts = np.bincount(observed, minlength=len(levels))
        self.proportions = counts / counts.sum()
        self.thresholds = scipy.stats.norm.ppf(np.cumsum(self.proportions)[:-1])
        self.cuts = np.concatenate([[-_INF], self.thresholds, [_INF]])

    @property
    def n_thresholds(self):
        return len(self.thresholds)

    def scores(self):
        """d log P(category) / d threshold for every case (lavaan's lav_uvord_sc): n x (K - 1)."""
        n = len(self.codes)
        result = np.zeros((n, self.n_thresholds))
        valid = ~np.isnan(self.codes)
        k = self.codes[valid].astype(int)
        p = scipy.special.ndtr(self.cuts[k + 1]) - scipy.special.ndtr(self.cuts[k])
        upper = _phi(self.cuts[k + 1]) / p
        lower = _phi(self.cuts[k]) / p
        rows = np.flatnonzero(valid)
        inside_upper = k < self.n_thresholds
        result[rows[inside_upper], k[inside_upper]] += upper[inside_upper]
        inside_lower = k > 0
        result[rows[inside_lower], k[inside_lower] - 1] -= lower[inside_lower]
        return result
##########################################################################################
# BIVARIATE: POLYCHORIC CORRELATION AND ITS SCORES
##########################################################################################
def _pair_table(first, second):
    valid = ~(np.isnan(first.codes) | np.isnan(second.codes))
    table = np.zeros((len(first.levels), len(second.levels)))
    np.add.at(table, (first.codes[valid].astype(int), second.codes[valid].astype(int)), 1)
    return table


def _polychoric(first, second, zero_add=(0.5, 0.0), keep_margins=True):
    """lavaan's lav_bvord_cor_twostep_fit: ML correlation of the pair's table with the thresholds fixed."""
    table = _pair_table(first, second)
    if table.shape == (2, 2):
        zeros = np.flatnonzero(table.ravel(order="F") == 0)
        if len(zeros) == 2:
            return 1.0 if table[0, 0] > 0 else -1.0
        if len(zeros) == 1 and zero_add[0] > 0:
            add = zero_add[0]
            if keep_margins:
                sign = 1 if zeros[0] in (0, 3) else -1
                table = table + sign * add * np.array([[1, -1], [-1, 1]])
            else:
                table = table.copy()
                table.ravel(order="F")[zeros[0]] += add
        if table[0, 0] * table[1, 1] == table[0, 1] * table[1, 0]:
            return 0.0
        if first.thresholds[0] == 0 and second.thresholds[0] == 0:
            return -np.cos(2 * np.pi * table[0, 0] / table.sum())
    elif zero_add[1] > 0:
        table = np.where(table == 0, zero_add[1], table)

    def objective(rho):
        p = _cell_probabilities(first.thresholds, second.thresholds, rho)
        with np.errstate(divide="ignore"):
            log_p = np.log(np.maximum(p, 1e-300))
        return -np.sum(table * log_p)

    result = scipy.optimize.minimize_scalar(objective, bounds=(-0.999, 0.999), method="bounded",
                                            options={"xatol": 1e-12})
    return float(result.x)


def _pair_scores(first, second, rho):
    """
    Case-wise scores of the pair's log-likelihood: with respect to rho (n) and to the thresholds of
    each variable (n x K1-1, n x K2-1), lavaan's lav_bvord_cor_sc / lav_m84_pair_sc.
    """
    n = len(first.codes)
    valid = ~(np.isnan(first.codes) | np.isnan(second.codes))
    a = first.codes[valid].astype(int)
    b = second.codes[valid].astype(int)
    x, y = first.cuts, second.cuts
    s = np.sqrt(1 - rho * rho)
    p = _cell_probabilities(first.thresholds, second.thresholds, rho)[a, b]

    d_rho = (_phi2(x[a + 1], y[b + 1], rho) - _phi2(x[a], y[b + 1], rho)
             - _phi2(x[a + 1], y[b], rho) + _phi2(x[a], y[b], rho)) / p

    def g(u, v):
        """d F(u, v) / d u: phi(u) * Phi((v - rho u) / s), 0 for infinite u."""
        with np.errstate(invalid="ignore"):
            inner = np.where(np.isinf(v), np.where(v > 0, 1.0, 0.0),
                             scipy.special.ndtr((v - rho * np.where(np.isfinite(u), u, 0)) / s))
        return _phi(u) * inner

    upper_first = (g(x[a + 1], y[b + 1]) - g(x[a + 1], y[b])) / p
    lower_first = (g(x[a], y[b + 1]) - g(x[a], y[b])) / p
    upper_second = (g(y[b + 1], x[a + 1]) - g(y[b + 1], x[a])) / p
    lower_second = (g(y[b], x[a + 1]) - g(y[b], x[a])) / p

    rows = np.flatnonzero(valid)
    score_rho = np.zeros(n)
    score_rho[rows] = d_rho
    score_first = np.zeros((n, first.n_thresholds))
    score_second = np.zeros((n, second.n_thresholds))
    for score, k, upper, lower, nth in [(score_first, a, upper_first, lower_first, first.n_thresholds),
                                        (score_second, b, upper_second, lower_second, second.n_thresholds)]:
        inside = k < nth
        score[rows[inside], k[inside]] += upper[inside]
        inside = k > 0
        score[rows[inside], k[inside] - 1] -= lower[inside]
    return score_rho, score_first, score_second
##########################################################################################
# SAMPLE STATISTICS
##########################################################################################
class OrdinalSampleStats:
    """
    Thresholds, polychoric correlations and their asymptotic covariance of
    ordered variables, as lavaan's sample statistics for WLSMV.

    Attributes:
    names: variable names. th: thresholds (variable by variable).
    th_names: lavaan's names ("x|t1", ...). cor: polychoric correlation
    matrix. nacov: N times the asymptotic covariance of (thresholds,
    correlations below the diagonal by columns), lavaan's "gamma".
    wls_v: diagonal DWLS weights, 1 / diag(nacov). n: number of cases.
    """

    def __init__(self, data):
        data = pd.DataFrame(data)
        self.names = list(data.columns)
        self.variables = [_Variable(data[c], c) for c in self.names]
        self.n = len(data)
        self.th = np.concatenate([v.thresholds for v in self.variables])
        self.th_names = [f"{v.name}|t{k + 1}" for v in self.variables for k in range(v.n_thresholds)]
        p = len(self.variables)
        self.cor = np.eye(p)
        self.pairs = [(i, j) for j in range(p) for i in range(j + 1, p)]
        for i, j in self.pairs:
            self.cor[i, j] = self.cor[j, i] = _polychoric(self.variables[j], self.variables[i])
        self.nacov = self._gamma()
        self.wls_v = 1 / np.diag(self.nacov)

    def _gamma(self):
        """lavaan's muthen1984: Gamma = B^-1 (S'S) B^-T with the case-wise scores S."""
        variables, n = self.variables, self.n
        starts = np.cumsum([0] + [v.n_thresholds for v in variables])
        nth = starts[-1]
        npairs = len(self.pairs)
        sc_th = np.hstack([v.scores() for v in variables])
        sc_cor = np.zeros((n, npairs))
        a21 = np.zeros((npairs, nth))
        for k, (i, j) in enumerate(self.pairs):
            score_rho, score_j, score_i = _pair_scores(variables[j], variables[i], self.cor[i, j])
            sc_cor[:, k] = score_rho
            a21[k, starts[i]:starts[i + 1]] = score_rho @ score_i
            a21[k, starts[j]:starts[j + 1]] = score_rho @ score_j
        scores = np.hstack([sc_th, sc_cor])
        inner = scores.T @ scores
        a11 = np.zeros((nth, nth))
        for v, start in enumerate(starts[:-1]):
            block = slice(start, starts[v + 1])
            a11[block, block] = inner[block, block]
        a22_inverse = np.diag(1 / np.sum(sc_cor ** 2, axis=0))
        a11_inverse = np.linalg.inv(a11)
        b_inverse = np.block([[a11_inverse, np.zeros((nth, npairs))],
                              [-a22_inverse @ a21 @ a11_inverse, a22_inverse]])
        return n * (b_inverse @ inner @ b_inverse.T)

    @property
    def cor_vector(self):
        """Correlations below the diagonal, column by column (lavaan's order: x1~~x2, x1~~x3, ...)."""
        return np.array([self.cor[i, j] for i, j in self.pairs])

    @property
    def cor_names(self):
        return [f"{self.names[j]}~~{self.names[i]}" for i, j in self.pairs]

    @property
    def vector(self):
        """The sample statistics DWLS fits: thresholds then correlations."""
        return np.concatenate([self.th, self.cor_vector])
##########################################################################################
# LAVAAN SYNTAX WITH LABELS, START VALUES AND LINEAR CONSTRAINTS
##########################################################################################
_NAME = r"[A-Za-z.][A-Za-z0-9._]*"


def _parse_term(term):
    """One right-hand term "mod1 * mod2 * name": (name, fixed value, label, start value, freed by NA)."""
    parts = [part.strip() for part in term.split("*")]
    name, modifiers = parts[-1], parts[:-1]
    fixed = label = start = None
    free = False
    for modifier in modifiers:
        start_match = re.fullmatch(r"start\(\s*([-+0-9.eE]+)\s*\)", modifier)
        if start_match:
            start = float(start_match.group(1))
        elif modifier.upper() == "NA":
            free = True
        elif re.fullmatch(r"[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?", modifier):
            fixed = float(modifier)
        elif re.fullmatch(_NAME, modifier):
            label = modifier
        else:
            raise NotImplementedError(f"modifier {modifier!r} is not supported")
    if not re.fullmatch(_NAME, name):
        raise ValueError(f"invalid variable name {name!r}")
    return name, fixed, label, start, free


def _linear(expression):
    """A linear expression of labels and numbers as ({label: coefficient}, constant)."""
    coefficients, constant = {}, 0.0
    text = expression.replace(" ", "")
    if text and text[0] not in "+-":
        text = "+" + text
    for sign, body in re.findall(r"([+-])([^+-]+)", text):
        factor = -1.0 if sign == "-" else 1.0
        pieces = body.split("*")
        number, label = 1.0, None
        for piece in pieces:
            if re.fullmatch(r"[0-9.]+(?:[eE][0-9]+)?", piece):
                number *= float(piece)
            elif re.fullmatch(_NAME, piece) and label is None:
                label = piece
            else:
                raise NotImplementedError(f"constraint term {body!r} is not linear")
        if label is None:
            constant += factor * number
        else:
            coefficients[label] = coefficients.get(label, 0.0) + factor * number
    return coefficients, constant


def parse_lavaan(model):
    """
    lavaan model syntax as statements and linear equality constraints.

    Statements are dicts lhs, op ("=~", "~~", "~", "|"), rhs, fixed, label,
    start and free; a variable named twice on one left-hand side ("start(1)
    * x + a * x") is one parameter with all the modifiers. Constraints
    "lhs == rhs" are linear: ({label: coefficient}, constant) meaning
    sum(coefficient * label) == constant.
    """
    statements, constraints, index = [], [], {}
    for raw in re.split(r"[;\n]", model):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if "==" in line:
            left, right = line.split("==", 1)
            a, ca = _linear(left)
            b, cb = _linear(right)
            coefficients = dict(a)
            for label, value in b.items():
                coefficients[label] = coefficients.get(label, 0.0) - value
            constraints.append((coefficients, cb - ca))
            continue
        op = next((o for o in ("=~", "~~", "|", "~") if o in line), None)
        if op is None:
            raise ValueError(f"cannot read the model statement {line!r}")
        lhs, rhs = (part.strip() for part in line.split(op, 1))
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
                statements.append({"lhs": lhs, "op": op, "rhs": name, "fixed": fixed, "label": label,
                                   "start": start, "free": free})
    return statements, constraints
##########################################################################################
# DWLS / ULS STRUCTURAL MODEL OF ORDERED INDICATORS
##########################################################################################
class OrdinalSEM:
    """
    Factor model of ordered indicators fitted as lavaan's WLSMV (diagonally
    weighted least squares) or ULSMV (unweighted least squares) estimator:
    least squares on the thresholds and polychoric correlations, robust
    standard errors with lavaan's observed-information bread, and the
    mean- and variance-adjusted (scaled and shifted) test statistic.

    The model is lavaan syntax with factor loadings (=~), variances and
    covariances (~~), fixed values (1*x), free values (NA*x), labels (a*x,
    equal when shared), start values (start(1)*x) and linear equality
    constraints (a == b + c, a == -b, a == 1). auto=True adds lavaan::cfa()'s
    defaults (first loading fixed to 1, or factor variances fixed with
    std_lv; residual and factor variances; factor covariances); auto=False
    is lavaan::lavaan(): only the thresholds, scales and fixed intercepts
    are added.

    parameterization "delta" fixes the scale of every latent response at 1
    (residual variances are 1 minus the common variance); "theta" uses the
    residual variances (fixed at 1 by cfa defaults) and derives the scales.
    """

    def __init__(self, model, data, estimator="WLSMV", parameterization="delta", std_lv=False, auto=True):
        if parameterization not in ("delta", "theta"):
            raise ValueError('parameterization must be "delta" or "theta"')
        if estimator.upper() not in ("WLSMV", "ULSMV"):
            raise ValueError('estimator must be "WLSMV" or "ULSMV"')
        statements, constraints = parse_lavaan(model)
        if any(s["op"] == "~" for s in statements):
            raise NotImplementedError("regressions are not supported with ordered indicators")
        latent = list(dict.fromkeys(s["lhs"] for s in statements if s["op"] == "=~"))
        indicators = [s["rhs"] for s in statements if s["op"] == "=~"]
        names = list(dict.fromkeys(n for s in statements for n in (s["lhs"], s["rhs"]) if s["op"] != "|"))
        observed = list(dict.fromkeys(indicators + [n for n in names if n not in latent]))
        self.latent, self.observed = latent, observed
        self.estimator, self.parameterization, self.std_lv = estimator.upper(), parameterization, std_lv
        self.stats = OrdinalSampleStats(pd.DataFrame(data)[observed])
        self.n = self.stats.n
        self.p, self.m = len(observed), len(latent)
        self._ov = {v: i for i, v in enumerate(observed)}
        self._lv = {v: i for i, v in enumerate(latent)}
        self.table = self._table(statements, auto)
        self._constraints(constraints)
        self._start()
        self._estimate()

    # ------------------------------------------------------------------ parameter table
    def _table(self, statements, auto):
        table = []
        seen_factor = set()
        for s in statements:
            fixed = s["fixed"]
            if auto and s["op"] == "=~" and s["lhs"] not in seen_factor:
                seen_factor.add(s["lhs"])
                if fixed is None and not s["free"] and not self.std_lv:
                    fixed = 1.0
            table.append({"lhs": s["lhs"], "op": s["op"], "rhs": s["rhs"], "label": s["label"] or "",
                          "free": fixed is None, "value": fixed if fixed is not None else s["start"],
                          "start": s["start"], "user": True})
        existing = {(r["lhs"], r["op"], r["rhs"]) for r in table} | \
                   {(r["rhs"], r["op"], r["lhs"]) for r in table if r["op"] == "~~"}
        user_thresholds = {(r["lhs"], r["rhs"]) for r in table if r["op"] == "|"}
        thresholds = []
        for variable in self.stats.variables:
            for k in range(variable.n_thresholds):
                if (variable.name, f"t{k + 1}") not in user_thresholds:
                    thresholds.append({"lhs": variable.name, "op": "|", "rhs": f"t{k + 1}", "label": "",
                                       "free": True, "value": None, "start": None, "user": False})
        auto_rows = []
        if auto:
            for name in self.observed:
                if (name, "~~", name) not in existing:
                    auto_rows.append({"lhs": name, "op": "~~", "rhs": name, "label": "", "free": False,
                                      "value": 1.0, "start": None, "user": False})
            for name in self.latent:
                if (name, "~~", name) not in existing:
                    auto_rows.append({"lhs": name, "op": "~~", "rhs": name, "label": "", "free": not self.std_lv,
                                      "value": 1.0 if self.std_lv else None, "start": None, "user": False})
            for k, a in enumerate(self.latent):
                for b in self.latent[k + 1:]:
                    if (a, "~~", b) not in existing:
                        auto_rows.append({"lhs": a, "op": "~~", "rhs": b, "label": "", "free": True, "value": None,
                                          "start": None, "user": False})
            table = table + thresholds + auto_rows
        else:
            table = table + thresholds
        for name in self.observed:
            table.append({"lhs": name, "op": "~*~", "rhs": name, "label": "", "free": False, "value": 1.0,
                          "start": None, "user": False})
        for name in self.observed + self.latent:
            table.append({"lhs": name, "op": "~1", "rhs": "", "label": "", "free": False, "value": 0.0,
                          "start": None, "user": False})
        return table

    def _constraints(self, constraints):
        """Free parameters (labels shared) and the reparameterisation slots = p0 + K theta of the constraints."""
        slots, keys = [], {}
        for k, row in enumerate(self.table):
            if not row["free"]:
                row["slot"] = None
                continue
            key = row["label"] or f"#{k}"
            if key not in keys:
                keys[key] = len(slots)
                slots.append(key)
            row["slot"] = keys[key]
        self.slots = slots
        n_slots = len(slots)
        if constraints:
            a = np.zeros((len(constraints), n_slots))
            c = np.zeros(len(constraints))
            for i, (coefficients, constant) in enumerate(constraints):
                for label, value in coefficients.items():
                    if label not in keys:
                        raise ValueError(f"constraint on unknown label {label!r}")
                    a[i, keys[label]] += value
                c[i] = constant
            u, singular, vt = np.linalg.svd(a)
            rank = int(np.sum(singular > 1e-10 * max(singular.max(), 1)))
            self._p0 = np.linalg.pinv(a) @ c
            self._k = vt[rank:].T
            self._rank = rank
        else:
            self._p0 = np.zeros(n_slots)
            self._k = np.eye(n_slots)
            self._rank = 0
        self.npar = n_slots - self._rank

    def _slots_from_theta(self, theta):
        return self._p0 + self._k @ theta

    def _set(self, theta):
        values = self._slots_from_theta(theta)
        for row in self.table:
            if row["slot"] is not None:
                row["value"] = values[row["slot"]]
        return values

    # ------------------------------------------------------------------ model
    def _matrices(self, theta=None):
        if theta is not None:
            self._set(theta)
        lam = np.zeros((self.p, self.m))
        psi = np.zeros((self.m, self.m))
        resid = np.zeros((self.p, self.p))
        tau = {}
        for row in self.table:
            op, lhs, rhs, value = row["op"], row["lhs"], row["rhs"], row["value"]
            if op == "=~":
                lam[self._ov[rhs], self._lv[lhs]] = value
            elif op == "|":
                tau[(lhs, rhs)] = value
            elif op == "~~" and lhs in self._lv:
                psi[self._lv[lhs], self._lv[rhs]] = psi[self._lv[rhs], self._lv[lhs]] = value
            elif op == "~~":
                resid[self._ov[lhs], self._ov[rhs]] = resid[self._ov[rhs], self._ov[lhs]] = value
        common = lam @ psi @ lam.T
        if self.parameterization == "delta":
            np.fill_diagonal(resid, 1 - np.diag(common))
            scale = np.ones(self.p)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = 1 / np.sqrt(np.diag(common + resid))
        return lam, psi, resid, tau, scale

    def implied(self, theta=None):
        """Model thresholds and correlations of the latent responses, in the order of the sample statistics."""
        lam, psi, resid, tau, scale = self._matrices(theta)
        correlations = np.outer(scale, scale) * (lam @ psi @ lam.T + resid)
        thresholds = np.array([tau[(v.name, f"t{k + 1}")] * scale[i]
                               for i, v in enumerate(self.stats.variables) for k in range(v.n_thresholds)])
        return np.concatenate([thresholds, [correlations[i, j] for i, j in self.stats.pairs]])

    def _jacobian(self, theta):
        """d implied / d theta by central differences."""
        theta = np.asarray(theta, dtype=float)
        columns = []
        for k in range(len(theta)):
            step = 1e-6 * max(1.0, abs(theta[k]))
            up, down = theta.copy(), theta.copy()
            up[k] += step
            down[k] -= step
            columns.append((self.implied(up) - self.implied(down)) / (2 * step))
        self._set(theta)
        return np.column_stack(columns)

    @property
    def weights(self):
        """DWLS weights 1 / diag(Gamma) for WLSMV, ones for ULSMV."""
        return self.stats.wls_v if self.estimator == "WLSMV" else np.ones(len(self.stats.vector))

    def _objective(self, theta):
        residual = self.stats.vector - self.implied(theta)
        return float(np.sum(self.weights * residual ** 2))

    # ------------------------------------------------------------------ estimation
    def _start(self):
        """Start values: start() where given, the sample thresholds, 0.5 (with the sign of the correlation to
        the factor's first indicator) for loadings, 1 for variances and 0 for covariances."""
        r = self.stats.cor
        marker = {}
        for row in self.table:
            if row["op"] == "=~" and row["lhs"] not in marker:
                marker[row["lhs"]] = row["rhs"]
        th_index = {name: k for k, name in enumerate(self.stats.th_names)}
        values = np.zeros(len(self.slots))
        for row in self.table:
            if row["slot"] is None:
                continue
            if row["start"] is not None:
                value = row["start"]
            elif row["op"] == "|":
                value = self.stats.th[th_index[f"{row['lhs']}|{row['rhs']}"]]
            elif row["op"] == "=~":
                first = marker[row["lhs"]]
                value = 0.5 * (np.sign(r[self._ov[row["rhs"]], self._ov[first]]) or 1.0) \
                    if row["rhs"] != first else 1.0
            elif row["op"] == "~~" and row["lhs"] == row["rhs"]:
                value = 1.0
            else:
                value = 0.0
            values[row["slot"]] = value
        theta = self._k.T @ (values - self._p0)
        self._set(theta)
        self._theta_start = theta

    def _estimate(self, tol=1e-12, max_iter=1000):
        theta = self._theta_start.copy()
        w = self.weights
        s = self.stats.vector
        value = self._objective(theta)
        self.converged = False
        for _ in range(max_iter):
            jac = self._jacobian(theta)
            residual = s - self.implied(theta)
            step = np.linalg.lstsq(jac.T @ (w[:, None] * jac), jac.T @ (w * residual), rcond=None)[0]
            factor = 1.0
            while True:
                candidate = theta + factor * step
                new_value = self._objective(candidate)
                if np.isfinite(new_value) and new_value <= value or factor < 1e-10:
                    break
                factor /= 2
            if not np.isfinite(new_value) or new_value > value:
                break
            small = np.max(np.abs(candidate - theta)) < tol or abs(value - new_value) <= 1e-16 * max(1, value)
            theta, value = candidate, new_value
            if small:
                self.converged = True
                break
        self._set(theta)
        self.theta = theta
        self.fdwls = value
        n = self.n
        # lavaan's least squares objective is fx = (N - 1) / N * F / 2, its test statistic 2 N fx = (N - 1) F
        self.fmin = 0.5 * (n - 1) / n * value
        jac_theta = self._jacobian(theta)
        self._jac = jac_theta
        weighted = w[:, None] * jac_theta
        self.information_expected = jac_theta.T @ weighted
        # lavaan 0.7's WLSMV/ULSMV bread is the observed information, the Hessian of fx
        residual = s - self.implied(theta)
        hessian = jac_theta.T @ weighted - self._second_derivative_term(theta, w * residual)
        self.information_observed = (n - 1) / n * hessian
        bread = np.linalg.pinv(self.information_observed)
        meat = weighted.T @ self.stats.nacov @ weighted
        vcov_theta = bread @ meat @ bread / (n - 1)
        self.vcov = self._k @ vcov_theta @ self._k.T

    def _second_derivative_term(self, theta, weights):
        """sum_k weights_k d2 implied_k / d theta d theta' by central differences."""
        theta = np.asarray(theta, dtype=float)
        k = len(theta)
        steps = 1e-4 * np.maximum(1.0, np.abs(theta))
        result = np.zeros((k, k))
        base = weights @ self.implied(theta)

        def f(delta):
            return weights @ self.implied(theta + delta)

        unit = np.eye(k) * steps
        for i in range(k):
            result[i, i] = (f(unit[i]) - 2 * base + f(-unit[i])) / steps[i] ** 2
            for j in range(i + 1, k):
                value = f(unit[i] + unit[j]) - f(unit[i] - unit[j]) - f(-unit[i] + unit[j]) + f(-unit[i] - unit[j])
                result[i, j] = result[j, i] = value / (4 * steps[i] * steps[j])
        self._set(theta)
        return result

    # ------------------------------------------------------------------ results
    @staticmethod
    def _scaled_shifted(jac, w, gamma, df):
        """lavaan's scaled.shifted constants a (1 / scaling factor) and b (shift) for a model's Jacobian."""
        u = np.diag(w)
        if jac is not None and jac.size:
            weighted = w[:, None] * jac
            u = u - weighted @ np.linalg.pinv(jac.T @ weighted) @ weighted.T
        ug = u @ gamma
        trace1, trace2 = np.trace(ug), np.sum(ug * ug.T)
        a = np.sqrt(df / trace2)
        return a, df - a * trace1

    def _derived_values(self):
        """Fill in the residual variances (delta) and scales (theta) lavaan reports but does not estimate."""
        lam, psi, resid, tau, scale = self._matrices()
        for row in self.table:
            if row["op"] == "~~" and row["lhs"] in self._ov and row["lhs"] == row["rhs"] \
                    and self.parameterization == "delta":
                row["value"] = resid[self._ov[row["lhs"]], self._ov[row["lhs"]]]
            elif row["op"] == "~*~":
                row["value"] = scale[self._ov[row["lhs"]]]

    def _standardized(self):
        """std.lv and std.all of every row of the parameter table."""
        lam, psi, resid, tau, scale = self._matrices()
        sd_lv = np.sqrt(np.diag(psi))
        sd_ov = np.sqrt(np.diag(lam @ psi @ lam.T + resid))
        std_lv, std_all = [], []
        for row in self.table:
            op, lhs, rhs, value = row["op"], row["lhs"], row["rhs"], row["value"]
            if op == "=~":
                lv_value = value * sd_lv[self._lv[lhs]]
                std_lv.append(lv_value)
                std_all.append(lv_value / sd_ov[self._ov[rhs]])
            elif op == "|":
                std_lv.append(value)
                std_all.append(value / sd_ov[self._ov[lhs]])
            elif op == "~~" and lhs in self._lv:
                i, j = self._lv[lhs], self._lv[rhs]
                std_lv.append(value / (sd_lv[i] * sd_lv[j]))
                std_all.append(value / (sd_lv[i] * sd_lv[j]))
            elif op == "~~":
                i, j = self._ov[lhs], self._ov[rhs]
                std_lv.append(value)
                std_all.append(value / (sd_ov[i] * sd_ov[j]))
            elif op == "~1":
                std_lv.append(value)
                std_all.append(value)
            else:
                std_lv.append(value)
                std_all.append(1.0)
        return std_lv, std_all

    def parameter_estimates(self, level=0.95, standardized=True):
        """
        lavaan's parameterEstimates(): lhs, op, rhs, label (when the model has labels), est, se, z, pvalue,
        ci.lower, ci.upper and, with standardized, std.lv and std.all.
        """
        self._derived_values()
        se_slot = np.sqrt(np.clip(np.diag(self.vcov), 0, None))
        z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        rows = []
        for row in self.table:
            value = row["value"]
            error = se_slot[row["slot"]] if row["slot"] is not None else 0.0
            error = 0.0 if error < 1e-12 else error
            z = value / error if error > 0 else np.nan
            rows.append({"lhs": row["lhs"], "op": row["op"], "rhs": row["rhs"], "label": row["label"], "est": value,
                         "se": error, "z": z, "pvalue": 2 * scipy.stats.norm.sf(abs(z)) if error > 0 else np.nan,
                         "ci.lower": value - z_crit * error, "ci.upper": value + z_crit * error})
        table = pd.DataFrame(rows)
        if not table["label"].astype(bool).any():
            table = table.drop(columns="label")
        if standardized:
            table["std.lv"], table["std.all"] = self._standardized()
        return table

    def inspect(self, what="est"):
        """lavaan's lavInspect(fit, "est"): lambda, theta, psi, tau (thresholds), nu and delta (scales)."""
        if what != "est":
            raise ValueError('only what="est" is supported')
        self._derived_values()
        lam, psi, resid, tau, scale = self._matrices()
        names = self.stats.th_names
        return {"lambda": pd.DataFrame(lam, index=self.observed, columns=self.latent),
                "theta": pd.DataFrame(resid, index=self.observed, columns=self.observed),
                "psi": pd.DataFrame(psi, index=self.latent, columns=self.latent),
                "tau": pd.DataFrame({"threshold": [tau[tuple(name.split("|"))] for name in names]}, index=names),
                "nu": pd.DataFrame({"intercept": np.zeros(self.p)}, index=self.observed),
                "delta": pd.DataFrame({"scales": scale}, index=self.observed)}

    # ------------------------------------------------------------------ fit measures
    def fit_measures(self):
        """
        lavaan's fitMeasures() of a WLSMV fit: the DWLS test, its scaled
        and shifted version, the baseline (independence) model and the
        indices built from them. The robust indices (cfi.robust, ...), the
        GFI family and the CN/MFI are not ported (NaN).
        """
        n = self.n
        w = self.weights
        gamma = self.stats.nacov
        nstat = len(self.stats.vector)
        npar = self.npar
        df = nstat - npar
        chisq = (n - 1) * self.fdwls
        a, b = self._scaled_shifted(self._jac, w, gamma, df)
        chisq_scaled = a * chisq + b

        nth = len(self.stats.th)
        cor_index = np.arange(nth, nstat)
        base_f = float(np.sum(w[cor_index] * self.stats.vector[cor_index] ** 2))
        base_chisq = (n - 1) * base_f
        base_df = len(cor_index)
        base_jac = np.zeros((nstat, nth))
        base_jac[np.arange(nth), np.arange(nth)] = 1.0
        base_a, base_b = self._scaled_shifted(base_jac, w, gamma, base_df)
        base_chisq_scaled = base_a * base_chisq + base_b

        # lavaan reports no p value for the unscaled DWLS statistic
        out = {"npar": npar, "fmin": self.fmin, "chisq": chisq, "df": df, "pvalue": np.nan,
               "chisq.scaled": chisq_scaled, "df.scaled": df,
               "pvalue.scaled": scipy.stats.chi2.sf(chisq_scaled, df),
               "chisq.scaling.factor": 1 / a,
               "baseline.chisq": base_chisq, "baseline.df": base_df, "baseline.pvalue": np.nan,
               "baseline.chisq.scaled": base_chisq_scaled, "baseline.df.scaled": base_df,
               "baseline.pvalue.scaled": scipy.stats.chi2.sf(base_chisq_scaled, base_df),
               "baseline.chisq.scaling.factor": 1 / base_a}
        for suffix, t, tb in [("", chisq, base_chisq), (".scaled", chisq_scaled, base_chisq_scaled)]:
            out.update({f"{k}{suffix}": v for k, v in _incremental(t, df, tb, base_df).items()})
        # lavaan's WLS family uses N - 1 where ML uses N
        out.update(_rmsea(chisq, df, n - 1, ""))
        out.update(_rmsea(chisq_scaled, df, n - 1, ".scaled"))
        out.update(self._robust(df, base_df))
        residual = self.stats.cor - self._model_correlation()
        squared = np.sum(residual[np.tril_indices(self.p, -1)] ** 2)
        squared_th = np.sum((self.stats.th - self.implied()[:nth]) ** 2)
        with_diagonal, without_diagonal = self.p * (self.p + 1) / 2, self.p * (self.p - 1) / 2
        out["rmr"] = out["srmr_bentler"] = np.sqrt((squared + squared_th) / (with_diagonal + nth))
        out["rmr_nomean"] = out["srmr"] = out["srmr_bentler_nomean"] = np.sqrt(squared / with_diagonal)
        out["crmr"] = np.sqrt((squared + squared_th) / (without_diagonal + nth))
        out["crmr_nomean"] = np.sqrt(squared / without_diagonal)
        out["srmr_mplus"] = out["srmr_mplus_nomean"] = np.nan
        out.update(self._gfi(chisq, df, nstat))
        out["mfi"] = np.exp(-0.5 * (chisq - df) / (n - 1))
        out["wrmr"] = np.sqrt((n - 1) * self.fdwls / nstat)
        return pd.Series(out)

    def _robust(self, df, base_df):
        """
        lavaan's robust CFI, TLI, RNI, RMSEA and GFI of a categorical fit (lav_fit_catml_dwls, Savalei):
        the ML discrepancy of the polychoric correlations from the model correlations at the least squares
        estimates (N times, its baseline -N log|R|), corrected by c = tr(U' V U Gamma) / df with the ML
        weights V of the correlations and U = I - Delta E^-1 Delta' W. NaN when the polychoric matrix is
        not positive definite (lavaan's cat_nonpd = "na").
        """
        names = ["cfi.robust", "tli.robust", "nnfi.robust", "rni.robust", "rmsea.robust", "rmsea.ci.lower.robust",
                 "rmsea.ci.upper.robust", "rmsea.pvalue.robust", "rmsea.notclose.pvalue.robust", "gfi.robust",
                 "gfi.ci.lower.robust", "gfi.ci.upper.robust"]
        r = self.stats.cor
        if np.linalg.eigvalsh(r).min() < np.sqrt(np.finfo(float).eps):
            return dict.fromkeys(names, np.nan)
        n, p = self.n, self.p
        sigma = self._model_correlation()
        inverse = np.linalg.inv(sigma)
        x2 = n * (np.linalg.slogdet(sigma)[1] + np.trace(r @ inverse) - np.linalg.slogdet(r)[1] - p)
        x2_null = -n * np.linalg.slogdet(r)[1]
        pairs = self.stats.pairs
        v = np.array([[inverse[i, k] * inverse[j, l] + inverse[i, l] * inverse[j, k] for k, l in pairs]
                      for i, j in pairs])
        nth = len(self.stats.th)
        jac = self._jac[nth:]
        u = np.eye(len(pairs)) - jac @ np.linalg.pinv(self.information_expected) @ jac.T * self.weights[nth:]
        gamma = self.stats.nacov[nth:, nth:]
        c = np.trace(u.T @ v @ u @ gamma) / df
        c_null = np.trace(gamma) / base_df
        x2_scaled = x2 / c
        t1, t2 = x2 - c * df, x2_null - c_null * base_df
        cfi_den = max(t1, t2, 0)
        out = {"cfi.robust": 1 - max(t1, 0) / cfi_den if cfi_den > 0 else 1.0,
               "tli.robust": 1 - t1 * base_df / (t2 * df), "rni.robust": 1 - t1 / t2}
        out["nnfi.robust"] = out["tli.robust"]

        def ncp(probability):
            f = lambda lam: (scipy.stats.ncx2.cdf(x2_scaled, df, lam) if lam > 0
                             else scipy.stats.chi2.cdf(x2_scaled, df)) - probability
            return 0.0 if f(0) < 0 else scipy.optimize.brentq(f, 0, max(n, 4 * x2_scaled))
        lower, upper = ncp(0.95), ncp(0.05)
        out.update({"rmsea.robust": np.sqrt(max((x2 / n) / df - c / n, 0)),
                    "rmsea.ci.lower.robust": np.sqrt(c * lower / (n * df)),
                    "rmsea.ci.upper.robust": np.sqrt(c * upper / (n * df)),
                    "rmsea.pvalue.robust": 1 - scipy.stats.ncx2.cdf(x2_scaled, df, n * df / c * 0.05 ** 2),
                    "rmsea.notclose.pvalue.robust": scipy.stats.ncx2.cdf(x2_scaled, df, n * df / c * 0.08 ** 2)})
        np_ = n * p
        out.update({"gfi.robust": min(np_ / (np_ + 2 * x2 - 2 * c * df), 1.0),
                    "gfi.ci.lower.robust": min(np_ / (np_ + 2 * c * upper), 1.0),
                    "gfi.ci.upper.robust": min(np_ / (np_ + 2 * c * lower), 1.0)})
        return out

    def _gfi(self, chisq, df, nstat):
        """lavaan's GFI family for a WLS fit (N - 1 in place of N); the robust GFI is in _robust."""
        n1, p = self.n - 1, self.p
        np_ = n1 * p

        def ncp(probability):
            if scipy.stats.chi2.cdf(chisq, df) < probability:
                return 0.0
            return scipy.optimize.brentq(lambda lam: scipy.stats.ncx2.cdf(chisq, df, lam) - probability
                                         if lam > 0 else scipy.stats.chi2.cdf(chisq, df) - probability,
                                         0, max(10 * chisq, 100))
        s, w = self.stats.vector, self.weights
        gfi_lisrel = 1 - self.fdwls / np.sum(w * s * s)
        return {"gfi": min(np_ / (np_ + 2 * chisq - 2 * df), 1.0),
                "gfi.ci.lower": min(np_ / (np_ + 2 * ncp(0.05)), 1.0),
                "gfi.ci.upper": min(np_ / (np_ + 2 * ncp(0.95)), 1.0), "gfi.ci.level": 0.90,

                "cn_05": scipy.stats.chi2.ppf(0.95, df) / (chisq / n1) + 1,
                "cn_01": scipy.stats.chi2.ppf(0.99, df) / (chisq / n1) + 1,
                "gfi_lisrel": gfi_lisrel, "agfi_lisrel": 1 - (nstat / df) * (1 - gfi_lisrel),
                "pgfi": (df / nstat) * gfi_lisrel}

    def _model_correlation(self):
        lam, psi, resid, tau, scale = self._matrices()
        sigma = lam @ psi @ lam.T + resid
        return np.outer(scale, scale) * sigma


def OrdinalCFA(model, data, parameterization="delta", std_lv=False, estimator="WLSMV"):
    """lavaan::cfa(model, data, ordered = TRUE, estimator = estimator): OrdinalSEM with cfa()'s defaults."""
    return OrdinalSEM(model, data, estimator=estimator, parameterization=parameterization, std_lv=std_lv, auto=True)


def _incremental(t, df, tb, dfb):
    """lavaan's incremental fit indices from a model and a baseline test statistic."""
    cfi_den = max(tb - dfb, t - df, 0)
    return {"cfi": 1 - max(t - df, 0) / cfi_den if cfi_den > 0 else 1.0,
            "tli": ((tb / dfb) - (t / df)) / ((tb / dfb) - 1),
            "nnfi": ((tb / dfb) - (t / df)) / ((tb / dfb) - 1),
            "rfi": ((tb / dfb) - (t / df)) / (tb / dfb),
            "nfi": (tb - t) / tb,
            "pnfi": (df / dfb) * (tb - t) / tb,
            "ifi": (tb - t) / (tb - df),
            "rni": 1 - (t - df) / (tb - dfb)}


def _rmsea(t, df, n, suffix):
    """RMSEA with its 90% interval and close-fit tests (lavaan, sample size N)."""
    def ncp_bound(probability):
        if scipy.stats.chi2.cdf(t, df) < probability:
            return 0.0
        return scipy.optimize.brentq(lambda ncp: scipy.stats.ncx2.cdf(t, df, ncp) - probability
                                     if ncp > 0 else scipy.stats.chi2.cdf(t, df) - probability, 0, max(10 * t, 100))
    lower = ncp_bound(0.95)
    upper = ncp_bound(0.05)
    out = {f"rmsea{suffix}": np.sqrt(max(t - df, 0) / (df * n)),
           f"rmsea.ci.lower{suffix}": np.sqrt(lower / (df * n)),
           f"rmsea.ci.upper{suffix}": np.sqrt(upper / (df * n)),
           f"rmsea.pvalue{suffix}": 1 - scipy.stats.ncx2.cdf(t, df, 0.05 ** 2 * df * n),
           f"rmsea.notclose.pvalue{suffix}": scipy.stats.ncx2.cdf(t, df, 0.08 ** 2 * df * n)}
    if not suffix:
        out.update({"rmsea.ci.level": 0.90, "rmsea.close.h0": 0.05, "rmsea.notclose.h0": 0.08})
    return out
