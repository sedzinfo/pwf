# -*- coding: utf-8 -*-
"""
Private helpers that reproduce R behaviour the rwf ports depend on: column
selection by name or 1-based position, R's as.character() for numbers and
labels, make.names(), factor levels, and grDevices colour ramps.
"""
import re

import numpy as np
import pandas as pd


def resolve_columns(df, columns=None):
    """
    Column names from names or 1-based positions, as rwf's names(df)[index].
    None selects every column.
    """
    if columns is None:
        return list(df.columns)
    if isinstance(columns, (str, int, np.integer)) or np.isscalar(columns):
        columns = [columns]
    return [df.columns[c - 1] if isinstance(c, (int, np.integer)) and not isinstance(c, bool) else c
            for c in columns]


def r_number(x):
    """R's as.character() of one number: 15 significant digits, scientific when shorter."""
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if np.isnan(x):
        return np.nan
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.14e}".split("e")
    exponent = int(exponent)
    digits = mantissa.lstrip("-").replace(".", "").rstrip("0") or "0"
    sign = "-" if x < 0 else ""
    sci = f"{sign}{digits[0]}{'.' + digits[1:] if len(digits) > 1 else ''}e{'+' if exponent >= 0 else '-'}{abs(exponent):02d}"
    fixed = f"{x:.{max(0, len(digits) - 1 - exponent)}f}"
    return fixed if len(fixed) <= len(sci) else sci


def r_label(value):
    """R's as.character() of a label or factor level: 2.0 -> "2", True -> "TRUE"."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return r_number(value)
    return str(value)


def r_levels(values):
    """Levels of R's factor(values): sorted unique non-missing values (category order for categoricals)."""
    if isinstance(getattr(values, "dtype", None), pd.CategoricalDtype):
        present = set(values.dropna())
        return [c for c in values.cat.categories if c in present]
    observed = pd.Series(values).dropna()
    return sorted(pd.unique(observed), key=_sort_key)


def _sort_key(value):
    if isinstance(value, str):
        return (1, value.casefold(), value.swapcase())
    return (0, value, "")


def r_make_names(names):
    """R's make.names(names, unique = TRUE): syntactically valid, unique names."""
    valid = []
    for name in map(str, names):
        name = re.sub(r"[^0-9A-Za-z._]", ".", name)
        if not re.match(r"^([A-Za-z]|\.(?![0-9]))", name):
            name = "X" + name
        valid.append(name)
    seen, result = {}, []
    for name in valid:
        if name in seen:
            seen[name] += 1
            candidate = f"{name}.{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}.{seen[name]}"
            seen[candidate] = 0
            result.append(candidate)
        else:
            seen[name] = 0
            result.append(name)
    return result


def r_color_ramp(low, high, n):
    """
    grDevices::colorRampPalette(c(low, high))(n): linear RGB steps as upper-case
    hex. R's rgb(..., maxColorValue = 255) truncates fractional channel values.
    """
    low = np.array([int(low[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    high = np.array([int(high[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    steps = np.linspace(0, 1, n)[:, None]
    rgb = np.floor(low + (high - low) * steps + 1e-9).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*row) for row in rgb]


def r_recycle(values, length):
    """R's recycling of a vector to a given length."""
    values = list(values)
    if not values:
        return [np.nan] * length
    return [values[i % len(values)] for i in range(length)]


def r_round(x, digits=0):
    """
    R's round() for one number, a port of R 4's fround() (src/nmath/fround.c).

    Of the candidates floor(x * 10^d) / 10^d and ceil(x * 10^d) / 10^d it
    takes the one closer to x in double arithmetic, and the even one on a
    tie. Python's round() rounds the exact binary value instead, so it gives
    -0.1 for round(-0.05, 1) where R gives 0 (157 of 53,010 random cases
    differed; this port matched R 4.6.2 on all of them).
    """
    if x is None or pd.isna(x):
        return np.nan
    x = float(x)
    if not np.isfinite(x) or digits > 323 or x == 0.0:
        return x
    if digits < -308:
        return 0.0
    if digits == 0:
        return float(np.rint(x))
    dig = int(np.floor(digits + 0.5))
    sign = 1.0
    if x < 0:
        sign, x = -1.0, -x
    if dig > 0 and (np.floor(np.log2(x)) + 0.5) * np.log10(2) + dig > 15:  # more digits than a double holds
        return sign * x
    pow10 = 10.0 ** dig if dig >= 0 else 1.0 / 10.0 ** -dig
    x10 = pow10 * x
    i10 = np.floor(x10)
    down, up = i10 / pow10, np.ceil(x10) / pow10
    return sign * (up if up - x < x - down or (up - x == x - down and i10 % 2 == 1) else down)


def r_p_adjust(p, method="holm"):
    """
    R's stats::p.adjust(p, method): "holm", "hochberg", "hommel", "bonferroni",
    "BH" (alias "fdr"), "BY" or "none". Missing values stay missing and are not
    counted.
    """
    p = np.asarray(p, dtype=float)
    result = p.copy()
    keep = ~np.isnan(p)
    values = p[keep]
    n = len(values)
    if n <= 1 or method == "none":
        return result
    if n == 2 and method == "hommel":
        method = "hochberg"
    if method == "bonferroni":
        adjusted = np.minimum(1, n * values)
    elif method == "holm":
        order = np.argsort(values, kind="stable")
        adjusted = np.empty(n)
        adjusted[order] = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * values[order]))
    elif method in ("hochberg", "BH", "fdr", "BY"):
        order = np.argsort(-values, kind="stable")
        i = np.arange(n, 0, -1)
        if method == "hochberg":
            scaled = (n + 1 - i) * values[order]
        elif method == "BY":
            scaled = np.sum(1 / np.arange(1, n + 1)) * n / i * values[order]
        else:
            scaled = n / i * values[order]
        adjusted = np.empty(n)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(scaled))
    elif method == "hommel":
        order = np.argsort(values, kind="stable")
        sorted_p = values[order]
        i = np.arange(1, n + 1)
        q = np.full(n, np.min(n * sorted_p / i))
        pa = q.copy()
        for m in range(n - 1, 1, -1):
            i1 = np.arange(n - m + 1)
            i2 = np.arange(n - m + 1, n)
            q1 = np.min(m * sorted_p[i2] / np.arange(2, m + 1))
            q[i1] = np.minimum(m * sorted_p[i1], q1)
            q[i2] = q[n - m]
            pa = np.maximum(pa, q)
        adjusted = np.empty(n)
        adjusted[order] = np.maximum(pa, sorted_p)
    else:
        raise ValueError(f"unknown p.adjust method {method!r}")
    result[keep] = adjusted
    return result


def _r_strip_chars(name, minlength):
    """The C routine behind R's abbreviate() (use.classes = TRUE) for one name."""
    chars = list(name.lstrip())
    if len(chars) < minlength:
        return "".join(chars)

    def first_char(i):
        return chars[i - 1].isspace()

    def last_char(i):
        return not chars[i - 1].isspace() and (i + 1 >= len(chars) or chars[i + 1].isspace())

    def lower_vowel(i):
        return chars[i] in "aeiou"

    n_space = 0
    upper = len(chars) - 1
    done = False
    trailing = True
    for i in range(upper, 0, -1):
        if i < len(chars) and chars[i].isspace():
            if trailing:
                del chars[i:]
            else:
                n_space += 1
        else:
            trailing = False
        if len(chars) - n_space <= minlength:
            done = True
            break
    tests = [lambda i: lower_vowel(i) and last_char(i), lambda i: lower_vowel(i) and not first_char(i),
             lambda i: chars[i].islower() and last_char(i), lambda i: chars[i].islower() and not first_char(i),
             lambda i: not first_char(i) and not chars[i].isspace()]
    for test in tests:
        if done:
            break
        for i in range(len(chars) - 1, 0, -1):
            if i < len(chars) and test(i):
                del chars[i]
            if len(chars) - n_space <= minlength:
                done = True
                break
    if len(chars) > minlength:
        chars = [chars[0]] + [c for c in chars[1:] if not c.isspace()]
    return "".join(chars)


def r_abbreviate(names, minlength=4):
    """R's abbreviate(names, minlength): drop lower-case vowels, then lower-case letters, from the end
    until the names are minlength characters long, lengthening duplicates until they are unique."""
    names = [str(name).strip(" ") for name in names]
    unique = list(dict.fromkeys(names))
    x = list(unique)
    redo = [True] * len(unique)
    length = minlength
    while True:
        for k, flag in enumerate(redo):
            if flag:
                x[k] = _r_strip_chars(unique[k], length)
        seen, duplicated = set(), []
        for value in x:
            duplicated.append(value in seen)
            seen.add(value)
        if not any(duplicated):
            break
        length += 1
        repeated = {x[k] for k, flag in enumerate(duplicated) if flag}
        redo = [value in repeated for value in x]
    mapping = dict(zip(unique, x))
    return [mapping[name] for name in names]


def r_mixedsort_key(text):
    """Sort key of gtools::mixedsort: numeric chunks by value before text, text as R's collation (a < A < b)."""
    key = []
    for chunk in re.findall(r"\d+(?:\.\d+)?|\D+", str(text)):
        if chunk[0].isdigit():
            key.append((0, float(chunk), "", ""))
        else:
            key.append((1, 0.0, chunk.casefold(), chunk.swapcase()))
    return key


def r_optim_bfgs(fn, gr, par, reltol=1.490116119384765625e-08, abstol=-np.inf, maxit=100):
    """
    R's optim(par, fn, gr, method = "BFGS"), a line-by-line port of vmmin()
    (Nash's variable metric method, R's src/appl/optim.c), so an estimate
    stops where R's does. reltol, abstol and maxit default to optim's.

    Returns:
    dict: par, value, convergence (0, or 1 when maxit was reached) and
    counts (function, gradient), as optim's.
    """
    stepredn, acctol, reltest = 0.2, 0.0001, 10.0
    b = np.array(par, dtype=float)
    n = b.size
    f = float(fn(b))
    if not np.isfinite(f):
        raise ValueError("initial value in 'vmmin' is not finite")
    fmin = f
    funcount = gradcount = 1
    g = np.array(gr(b), dtype=float)
    iteration = 1
    ilast = gradcount
    B = np.zeros((n, n))
    X, c, t = np.zeros(n), np.zeros(n), np.zeros(n)
    while True:
        if ilast == gradcount:
            B = np.eye(n)
        X[:], c[:] = b, g
        gradproj = 0.0
        for i in range(n):  # t = -B g, B kept in its lower triangle
            s = 0.0
            for j in range(i + 1):
                s -= B[i, j] * g[j]
            for j in range(i + 1, n):
                s -= B[j, i] * g[j]
            t[i] = s
            gradproj += s * g[i]
        if gradproj < 0.0:  # downhill: backtrack from a unit step
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if reltest + X[i] == reltest + b[i]:
                        count += 1
                if count < n:
                    f = float(fn(b))
                    funcount += 1
                    accpoint = bool(np.isfinite(f)) and f <= fmin + gradproj * steplength * acctol
                    if not accpoint:
                        steplength *= stepredn
                if count == n or accpoint:
                    break
            enough = f > abstol and abs(f - fmin) > reltol * (abs(fmin) + reltol)
            if not enough:
                count = n
                fmin = f
            if count < n:  # progress: BFGS update of B
                fmin = f
                g = np.array(gr(b), dtype=float)
                gradcount += 1
                iteration += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[i] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(i + 1):
                            s += B[i, j] * c[j]
                        for j in range(i + 1, n):
                            s += B[j, i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(i + 1):
                            B[i, j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1
                else:
                    ilast = gradcount
            elif ilast < gradcount:  # no progress: restart from the gradient
                count = 0
                ilast = gradcount
        else:  # uphill: restart unless just restarted
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if iteration >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount  # periodic restart
        if count == n and ilast == gradcount:
            break
    return {"par": b, "value": fmin, "convergence": 0 if iteration < maxit else 1,
            "counts": (funcount, gradcount)}


# R's (X11's) levels of gray0 ... gray100, from grDevices::col2rgb(); no rounding rule gives all of them
_R_GRAY = (
    0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 26, 28, 31, 33, 36, 38, 41, 43, 46, 48, 51, 54, 56, 59, 61,
    64, 66, 69, 71, 74, 77, 79, 82, 84, 87, 89, 92, 94, 97, 99, 102, 105, 107, 110, 112, 115, 117,
    120, 122, 125, 127, 130, 133, 135, 138, 140, 143, 145, 148, 150, 153, 156, 158, 161, 163, 166,
    168, 171, 173, 176, 179, 181, 184, 186, 189, 191, 194, 196, 199, 201, 204, 207, 209, 212, 214,
    217, 219, 222, 224, 227, 229, 232, 235, 237, 240, 242, 245, 247, 250, 252, 255
)


def r_color(name):
    """An R colour name for matplotlib/plotnine: R's "gray25"/"grey25" become R's hex, others pass through."""
    match = re.fullmatch(r"gr[ae]y(\d{1,3})", str(name))
    if match and int(match.group(1)) <= 100:
        return "#{0:02X}{0:02X}{0:02X}".format(_R_GRAY[int(match.group(1))])
    return name


def r_uniroot(f, lower, upper, f_lower=None, f_upper=None, tol=np.finfo(float).eps ** 0.25, maxiter=1000):
    """
    R's uniroot(f, c(lower, upper), tol = tol)$root, a line-by-line port of
    R_zeroin2() (Brent's method, R's src/library/stats/src/zeroin.c), so a
    loose tol stops where R's does. f_lower/f_upper default to f(lower) and
    f(upper); as uniroot, their signs must differ.
    """
    eps = np.finfo(float).eps
    a, b = float(lower), float(upper)
    fa = float(f(a)) if f_lower is None else float(f_lower)
    fb = float(f(b)) if f_upper is None else float(f_upper)
    if fa * fb > 0:
        raise ValueError("f() values at end points not of opposite sign")
    c, fc = a, fa
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(maxiter + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):  # b becomes the best approximation
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * eps * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):  # try interpolation
            cb = c - b
            if a == c:  # linear
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:  # inverse quadratic
                q, t1, t2 = fa / fc, fb / fc, fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = float(f(b))
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def r_make_unique(names, sep="."):
    """
    R's make.unique(names, sep): a repeated name gets sep and the first count
    (1, 2, ...) that makes it unique, as rbind() does to row names with
    sep = "" (E1, E1 -> E1, E11).
    """
    names = [str(name) for name in names]
    used, seen, counts, result = set(names), set(), {}, []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
            continue
        count = counts.get(name, 0)
        while True:
            count += 1
            candidate = f"{name}{sep}{count}"
            if candidate not in used:
                break
        counts[name] = count
        used.add(candidate)
        result.append(candidate)
    return result
