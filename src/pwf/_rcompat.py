# -*- coding: utf-8 -*-
"""
Private helpers that reproduce R behaviour the rwf ports depend on: column
selection by name or 1-based position, R's as.character() for numbers and
labels, make.names(), factor levels, and grDevices colour ramps.
"""
import collections
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


def r_optimize(f, lower, upper, tol=np.finfo(float).eps ** 0.25):
    """
    R's optimize(f, c(lower, upper), tol = tol)$minimum, a line-by-line port
    of Brent_fmin() (R's src/library/stats/src/optimize.c), so a loose tol
    stops where R's does. As R, a non-finite f is replaced by the largest
    double.
    """
    def value(x):
        result = float(f(x))
        return result if np.isfinite(result) else np.finfo(float).max

    c = (3.0 - np.sqrt(5.0)) * 0.5  # squared inverse of the golden ratio
    eps = np.sqrt(np.finfo(float).eps)
    a, b = float(lower), float(upper)
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = fv = fw = value(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):  # golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:  # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = value(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def _r_dnorm(x, sd):
    """R's dnorm(x, sd = sd) (dnorm4 in nmath/dnorm.c, with its accurate tail for |x / sd| >= 5)."""
    one_over_sqrt_2pi = 0.398942280401432677939946059934
    z = np.abs(np.asarray(x, dtype=float) / sd)
    result = one_over_sqrt_2pi * np.exp(-0.5 * z * z) / sd
    tail = z >= 5
    if tail.any():
        zt = z[tail]
        x1 = np.ldexp(np.rint(np.ldexp(zt, 16)), -16)
        x2 = zt - x1
        values = one_over_sqrt_2pi / sd * (np.exp(-0.5 * x1 * x1) * np.exp((-0.5 * x2 - x1) * x2))
        values[zt > np.sqrt(-2 * 0.693147180559945309417232121458 * (-1021 + 1 - 53))] = 0.0
        result[tail] = values
    return result


def r_bw_nrd0(x):
    """R's bw.nrd0(x): 0.9 * min(sd, IQR / 1.34) * n^-0.2, IQR from R's type 7 quantiles."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    ordered = np.sort(x)

    def quantile7(p):  # R's quantile(type = 7), its own formula
        index = 1 + (len(ordered) - 1) * p
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        q = ordered[lo - 1]
        if index > lo and ordered[hi - 1] != q:
            h = index - lo
            q = (1 - h) * q + h * ordered[hi - 1]
        return q

    hi = float(np.std(x, ddof=1))
    lo = min(hi, (quantile7(0.75) - quantile7(0.25)) / 1.34)
    if not lo:
        lo = hi or abs(x[0]) or 1.0
    return 0.9 * lo * len(x) ** -0.2


def r_density(x, bw="nrd0", adjust=1, n=512, from_=None, to=None, cut=3, ext=4):
    """
    R's density(x, bw, adjust, n = n, from, to, cut, ext) with the Gaussian
    kernel: R's linear binning (BinDist) on its grid, the FFT convolution and
    linear interpolation onto n points (R >= 4.4 coordinates). Missing values
    must be removed first (R's na.rm); non-finite values are dropped as R's.

    Returns:
    tuple: (x, y, bw) as density()'s $x, $y and $bw.
    """
    x = np.asarray(x, dtype=float)
    total = len(x)
    x = x[np.isfinite(x)]
    nx = len(x)
    weights = np.full(nx, 1 / nx)
    n_user = n
    n = max(n, 512)
    if n > 512:
        n = 2 ** int(np.ceil(np.log2(n)))
    if isinstance(bw, str):
        if bw.lower() != "nrd0":
            raise ValueError("only bw = 'nrd0' (R's default) is supported")
        bw = r_bw_nrd0(x)
    bw = adjust * bw
    if bw <= 0:
        raise ValueError("'bw' is not positive.")
    from_ = x.min() - cut * bw if from_ is None else from_
    to = x.max() + cut * bw if to is None else to
    lo, up = from_ - ext * bw, to + ext * bw
    # BinDist (massdist.c): each point's weight split between its two grid neighbours, added in data order
    delta = (up - lo) / (n - 1)
    position = (x - lo) / delta
    ix = np.floor(position).astype(int)
    fx = position - ix
    index = np.column_stack([ix, ix + 1]).ravel()
    mass = np.column_stack([weights * (1 - fx), weights * fx]).ravel()
    keep = np.column_stack([(ix >= 0) & (ix <= n - 1), (ix >= -1) & (ix <= n - 2)]).ravel()
    y = np.zeros(2 * n)
    np.add.at(y, index[keep], mass[keep])
    y *= nx / total
    # seq.int(from, to, length.out) is from + i * by with the last point set to `to`: np.linspace, bit for bit
    kords = np.linspace(0, (2 * n - 1) / (n - 1) * (up - lo), 2 * n)
    kords[n + 1:] = -kords[n - 1:0:-1]
    kords = _r_dnorm(kords, bw)
    # R's fft(z, inverse = TRUE) / length(y) is numpy's normalised ifft
    kords = np.maximum(0, np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))).real[:n])
    xords = np.linspace(lo, up, n)
    grid = np.linspace(from_, to, n_user)
    # approx(): y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i])), exact at the nodes
    i = np.clip(np.searchsorted(xords, grid, side="right") - 1, 0, n - 2)
    j = i + 1
    values = kords[i] + (kords[j] - kords[i]) * ((grid - xords[i]) / (xords[j] - xords[i]))
    values = np.where(grid == xords[j], kords[j], np.where(grid == xords[i], kords[i], values))
    return grid, values, bw


def r_pretty(x, n=5, min_n=None, shrink_sml=0.75, high_u_bias=1.5, u5_bias=None, eps_correct=0, f_min=2 ** -20):
    """
    R's pretty(x, n, min.n, ...) (R >= 4.2, bounds = TRUE): a port of R_pretty()
    (src/appl/pretty.c) and pretty.default(), giving R's round breakpoints.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return x
    n = int(n)
    min_n = n // 3 if min_n is None else int(min_n)
    h, h5 = high_u_bias, (0.5 + 1.5 * high_u_bias if u5_bias is None else u5_bias)
    eps, rounding_eps = np.finfo(float).eps, 1e-10
    lo = lo_ = float(x.min())
    up = up_ = float(x.max())
    dx = up_ - lo_
    if dx == 0 and up_ == 0:
        cell, i_small = 1.0, True
    else:
        cell = max(abs(lo_), abs(up_))
        bound = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        bound *= max(1, n) * eps
        i_small = dx < cell * bound * 3
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if n > 1:
            cell /= n
    subsmall = f_min * np.finfo(float).tiny or np.finfo(float).tiny
    if cell < subsmall:
        cell = subsmall
    elif cell > np.finfo(float).max / 1.25:
        cell = np.finfo(float).max / 1.25
    base = 10.0 ** np.floor(np.log10(cell))
    unit = base
    if 2 * base - cell < h * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < h5 * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < h * (cell - unit):
                unit = 10 * base
    ns = np.floor(lo_ / unit + rounding_eps)
    nu = np.ceil(up_ / unit - rounding_eps)
    if eps_correct and (eps_correct > 1 or not i_small):
        lo = lo_ * (1 - eps) if lo_ != 0 else -np.finfo(float).tiny
        up = up_ * (1 + eps) if up_ != 0 else np.finfo(float).tiny
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while not np.isfinite(ns * unit):
        ns += 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1
    while not np.isfinite(nu * unit):
        nu -= 1
    k = int(0.5 + nu - ns)
    if k < min_n:  # add divisions to reach min_n; a range ending at 0 keeps 0 as its bound (as R 4.6)
        k = min_n - k
        if lo_ == 0 and ns == 0 and up_ != 0:
            nu += k
        elif up_ == 0 and nu == 0 and lo_ != 0:
            ns -= k
        elif ns >= 0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2
        ndiv = min_n
    else:
        ndiv = k
    if ns * unit < lo:
        lo = ns * unit
    if nu * unit > up:
        up = nu * unit
    s = np.linspace(lo, up, ndiv + 1)  # seq.int(l, u, length.out = n + 1)
    if not eps_correct and ndiv:
        s[np.abs(s) < 1e-14 * (up / ndiv - lo / ndiv)] = 0.0
    return s


def r_hist(x, breaks="Sturges"):
    """
    R's hist(x, breaks)$breaks and $counts: breaks from pretty() on Sturges'
    (or Scott's, Freedman-Diaconis') number of classes, a number of classes, or
    the breakpoints themselves; right-closed classes with R's fuzz.

    Returns:
    tuple: (breaks, counts) as numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if isinstance(breaks, str):
        kind = breaks.lower()
        if kind == "sturges":
            breaks = int(np.ceil(np.log2(len(x)) + 1))
        elif kind == "scott":
            width = 3.5 * np.sqrt(np.var(x, ddof=1)) * len(x) ** (-1 / 3)
            breaks = max(1, int(np.ceil(np.ptp(x) / width))) if width > 0 else 1
        elif kind in ("fd", "freedman-diaconis"):
            rounded = np.array([float(f"{v:.5g}") for v in x])  # signif(x, 5)
            width = 2 * (np.quantile(rounded, 0.75) - np.quantile(rounded, 0.25))
            alpha = 1 / 4
            while width == 0 and (alpha := alpha / 2) >= 1 / 512:
                width = (np.quantile(rounded, 1 - alpha) - np.quantile(rounded, alpha)) / (1 - 2 * alpha)
            breaks = max(1, int(np.ceil(np.ptp(x) / (width * len(x) ** (-1 / 3))))) if width > 0 else 1
        else:
            raise ValueError("unknown 'breaks' algorithm")
    breaks = np.atleast_1d(np.asarray(breaks, dtype=float))
    if len(breaks) == 1:
        breaks = r_pretty([x.min(), x.max()], n=int(breaks[0]), min_n=1)
    else:
        breaks = np.sort(breaks)
    widths = np.diff(breaks)
    nb = len(breaks)
    diddle = 1e-7 * (np.median(widths) if nb > 5 else np.ptp(x) if nb <= 3 else widths[widths > 0].min())
    fuzzy = breaks + np.r_[-diddle, np.full(nb - 1, diddle)]  # right = TRUE, include.lowest = TRUE
    inside = (fuzzy[0] <= x) & (x <= fuzzy[-1])
    bins = np.clip(np.searchsorted(fuzzy, x[inside], side="left") - 1, 0, nb - 2)
    return breaks, np.bincount(bins, minlength=nb - 1)


def r_ppoints(n):
    """R's ppoints(n): (1:n - a) / (n + 1 - 2a), a = 3/8 for n <= 10, else 1/2."""
    a = 3 / 8 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def r_boxplot_stats(x, coef=1.5):
    """
    R's boxplot.stats(x, coef): Tukey's five numbers (fivenum: hinges, not
    quantiles), whiskers at the most extreme values within coef * IQR of the
    hinges, and the values beyond them.

    Returns:
    dict: stats (whisker, hinge, median, hinge, whisker), n, conf and out.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    ordered = np.sort(x)
    n = len(ordered)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1
    five = 0.5 * (ordered[np.floor(d).astype(int)] + ordered[np.ceil(d).astype(int)])
    iqr = five[3] - five[1]
    out = (x < five[1] - coef * iqr) | (x > five[3] + coef * iqr)
    if out.any():
        five[[0, 4]] = x[~out].min(), x[~out].max()
    return {"stats": five, "n": n, "conf": five[2] + np.array([-1.58, 1.58]) * iqr / np.sqrt(n), "out": x[out]}


def r_call_text(function, arguments, frame=None):
    """
    rwf's call text: R's deparse(match.call()) with quotes and spaces removed,
    built from the arguments that were given, (name, value) pairs in order.
    None and empty lists are left out; a data frame is written by the name it has in frame's
    callers ("df" when not found); lists are c(...), dicts list(...).
    """
    def name_of(value):
        caller = frame
        while caller is not None:
            found = [k for k, v in {**caller.f_globals, **caller.f_locals}.items()
                     if v is value and not k.startswith("_")]
            if found:
                return found[0]
            caller = caller.f_back
        return "df"

    def text(value):
        if isinstance(value, pd.DataFrame):
            return name_of(value)
        if isinstance(value, dict):
            return "list(" + ",".join(f"{k}={text(v)}" for k, v in value.items()) + ")"
        if isinstance(value, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            values = list(value)
            return text(values[0]) if len(values) == 1 else "c(" + ",".join(text(v) for v in values) + ")"
        return r_label(value)

    def given(value):
        return value is not None and not (isinstance(value, (list, tuple, dict)) and len(value) == 0)

    parts = [f"{name}={text(value)}" for name, value in arguments if given(value)]
    return f"{function}({','.join(parts)})".replace(" ", "").replace('"', "").replace("'", "")


def r_deparse_call(function, arguments, cutoff=60):
    """
    Lines of R's deparse() of a call function(name = value, ...): after each argument's ", " and after
    each binary operator of a formula (" ~ ", " + ", ...), deparse ends the line once it is longer than
    width.cutoff (60) characters; continuation lines are indented by 4 spaces.
    """
    lines, line = [], f"{function}("

    def check():
        nonlocal line
        if len(line) > cutoff:
            lines.append(line)
            line = "    "

    for k, (name, value) in enumerate(arguments):
        line += f"{name} = "
        for piece in re.split(r"( [~+\-*/] )", value):
            line += piece
            if re.fullmatch(r" [~+\-*/] ", piece):
                check()
        if k < len(arguments) - 1:
            line += ", "
            check()
    lines.append(line + ")")
    return lines


# ggplot2 gives text sizes in mm and line widths in its own unit (about 0.75 mm); plotnine takes text sizes in
# points and multiplies line sizes by sqrt(pi) points
GG_PT = 72.27 / 25.4  # ggplot2's .pt (points per mm)


def gg_text_size(size):
    """plotnine text size (points) for a ggplot2 text size (mm): size * .pt, the fontsize grid draws with."""
    return size * GG_PT


def gg_linewidth(linewidth):
    """
    plotnine line size for a ggplot2 linewidth: ggplot2 draws lwd = linewidth * .pt, R's lwd 1 is 1/96 inch,
    and plotnine draws size * sqrt(pi) points (1/72 inch).
    """
    return linewidth * GG_PT * 72 / 96 / np.sqrt(np.pi)


def gg_unit_pt(value):
    """Inches of grid's unit(value, "pt") (1/72.27 inch), e.g. ggplot2's arrow(length = unit(6, "pt"))."""
    return value / 72.27


def r_layout_in_circle(n):
    """igraph's layout_in_circle(): vertex i at angle 2 * pi / n * i on the unit circle (cigraph layout_circle)."""
    phi = 2 * np.pi / n * np.arange(n)
    return np.column_stack([np.cos(phi), np.sin(phi)])


def r_layout_as_tree(n, edges, roots, flip_y=True):
    """
    igraph's layout_as_tree(graph, root = roots, mode = "out", flip.y) for a directed graph with vertices
    0, ..., n - 1 and edges (from, to): a port of the Reingold-Tilford layout in cigraph's
    layout/reingold_tilford.c (igraph 2.x). With several roots a virtual root above them is added (and removed
    from the result); vertices it cannot reach get an edge from it; children are taken in vertex order, each
    subtree is placed one unit from its left neighbour's contour and parents are centred on the mean of their
    children. Returns an (n, 2) array of x, y.
    """
    edges = [(int(a), int(b)) for a, b in edges]
    roots = [int(r) for r in roots]
    total = n
    if len(roots) == 1:
        real_root = roots[0]
    else:
        real_root = total
        total += 1
        edges = edges + [(real_root, r) for r in roots]

    def adjacency(edge_list):  # igraph_adjlist_init(mode = OUT, no loops, no multiple edges): sorted neighbours
        neighbours = [set() for _ in range(total)]
        for a, b in edge_list:
            if a != b:
                neighbours[a].add(b)
        return [sorted(s) for s in neighbours]

    adjacent = adjacency(edges)
    visited = [False] * total
    queue = collections.deque([real_root])
    while queue:
        node = queue.popleft()
        visited[node] = True
        queue.extend(v for v in adjacent[node] if not visited[v])
    unreachable = [(real_root, v) for v in range(total) if not visited[v]]
    if unreachable:
        adjacent = adjacency(edges + unreachable)

    parent, y = [-1] * total, [0.0] * total
    parent[real_root] = real_root
    queue = collections.deque([(real_root, 0)])
    while queue:  # step 1: levels (y) by breadth-first search, and each vertex's parent
        node, distance = queue.popleft()
        for v in adjacent[node]:
            if parent[v] >= 0:
                continue
            y[v] = distance + 1
            queue.append((v, distance + 1))
            parent[v] = node
    children = [[] for _ in range(total)]
    for v in range(total):
        if parent[v] >= 0 and parent[v] != v:
            children[parent[v]].append(v)

    minsep = 1.0
    offset = [0.0] * total
    left_contour, right_contour = [-1] * total, [-1] * total
    to_left_contour, to_right_contour = [0.0] * total, [0.0] * total
    left_extreme, right_extreme = list(range(total)), list(range(total))
    to_left_extreme, to_right_extreme = [0.0] * total, [0.0] * total

    def postorder(node):  # step 2: offsets of each subtree from its parent
        for child in children[node]:
            postorder(child)
        if not children[node]:
            return
        leftroot, average = -1, 0.0
        for j, i in enumerate(children[node]):
            if leftroot >= 0:
                lnode, rnode = leftroot, i
                rootsep = offset[leftroot] + minsep
                loffset = offset[leftroot]
                roffset = loffset + minsep
                right_contour[node], to_right_contour[node] = i, rootsep
                while lnode >= 0 and rnode >= 0:
                    if right_contour[lnode] >= 0:
                        loffset += to_right_contour[lnode]
                        lnode = right_contour[lnode]
                    else:
                        if left_contour[rnode] >= 0:
                            aux = left_extreme[node]
                            new_offset = (to_right_extreme[node] - to_left_extreme[node]) + minsep \
                                + to_left_contour[rnode]
                            left_contour[aux] = right_contour[aux] = left_contour[rnode]
                            to_left_contour[aux] = to_right_contour[aux] = new_offset
                            left_extreme[node], right_extreme[node] = left_extreme[i], right_extreme[i]
                            to_left_extreme[node] = to_left_extreme[i] + rootsep
                            to_right_extreme[node] = to_right_extreme[i] + rootsep
                        else:
                            right_extreme[node] = right_extreme[i]
                            to_right_extreme[node] = to_right_extreme[i] + rootsep
                        lnode = -1
                    if left_contour[rnode] >= 0:
                        roffset += to_left_contour[rnode]
                        rnode = left_contour[rnode]
                    else:
                        if lnode >= 0:
                            aux = right_extreme[i]
                            new_offset = loffset - rootsep - to_right_extreme[i]
                            left_contour[aux] = right_contour[aux] = lnode
                            to_left_contour[aux] = to_right_contour[aux] = new_offset
                        rnode = -1
                    if lnode >= 0 and rnode >= 0 and roffset - loffset < minsep:
                        rootsep += minsep - roffset + loffset
                        roffset = loffset + minsep
                        to_right_contour[node] = rootsep
                offset[i] = rootsep
                to_right_contour[node] = rootsep
                average = (average * j) / (j + 1) + rootsep / (j + 1)
                leftroot = i
            else:
                leftroot = i
                left_contour[node] = right_contour[node] = i
                to_left_contour[node] = to_right_contour[node] = 0.0
                left_extreme[node], right_extreme[node] = left_extreme[i], right_extreme[i]
                to_left_extreme[node], to_right_extreme[node] = to_left_extreme[i], to_right_extreme[i]
                average = offset[i]
        to_left_contour[node] -= average
        to_right_contour[node] -= average
        to_left_extreme[node] -= average
        to_right_extreme[node] -= average
        for child in children[node]:
            offset[child] -= average

    x = [0.0] * total

    def coordinates(node, position):  # step 3: x from the offsets
        x[node] = position
        for child in children[node]:
            coordinates(child, position + offset[child])

    postorder(real_root)
    coordinates(real_root, offset[real_root])
    layout = np.column_stack([x[:n], y[:n]]).astype(float)
    if flip_y and n > 0:
        layout[:, 1] = layout[:, 1].max() - layout[:, 1]
    return layout


def _r_curve_origin(x1, y1, x2, y2, origin):
    """grid's calcOrigin(): the centre of the arc's rotation in the 'square' region."""
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    slope = (y2 - y1) / (x2 - x1)
    oslope = -1 / slope
    tmpox = xm if not np.isfinite(slope) else xm + origin * (x2 - x1) / 2
    if not np.isfinite(slope):
        tmpoy = ym + origin * (y2 - y1) / 2
    elif not np.isfinite(oslope):
        tmpoy = ym
    else:
        tmpoy = ym + origin * (y2 - y1) / 2
    sintheta = -1
    return xm - (tmpoy - ym) * sintheta, ym + (tmpox - xm) * sintheta


def r_curve_control_points(x1, y1, x2, y2, curvature, angle=90, ncp=5):
    """
    grid's calcControlPoints(x1, y1, x2, y2, curvature, angle, ncp) for one curve: the ncp control points
    curveGrob() (ggplot2's geom_curve) passes to its X-spline, rotated from the start point around an origin
    that depends on the curvature (positive: a right-hand curve).
    """
    x1, y1, x2, y2 = (np.float64(v) for v in (x1, y1, x2, y2))
    with np.errstate(divide="ignore", invalid="ignore"):
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        angle = angle / 180 * np.pi
        sina, cosa = np.sin(angle), np.cos(angle)
        cornerx = xm + (x1 - xm) * cosa - (y1 - ym) * sina
        cornery = ym + (y1 - ym) * cosa + (x1 - xm) * sina
        beta = -np.arctan((cornery - y1) / (cornerx - x1))
        sinb, cosb = np.sin(beta), np.cos(beta)
        newx2 = x1 + dx * cosb - dy * sinb
        newy2 = y1 + dy * cosb + dx * sinb
        scalex = (newy2 - y1) / (newx2 - x1)
        newx1 = x1 * scalex
        newx2 = newx2 * scalex
        ratio = 2 * np.sin(np.arctan(curvature)) ** 2
        origin = curvature - curvature / ratio
        direction = 1 if curvature > 0 else -1
        ox, oy = _r_curve_origin(newx1, y1, newx2, newy2, origin)
        maxtheta = np.pi + np.sign(origin * direction) * 2 * np.arctan(abs(origin))
        by = direction * maxtheta / (ncp + 1)
        steps = int((direction * maxtheta) / by + 1e-10)  # seq(0, dir * maxtheta, by), less its two ends
        theta = (np.arange(steps + 1) * by)[1:ncp + 1]
        costheta, sintheta = np.cos(theta), np.sin(theta)
        cpx = ox + (newx1 - ox) * costheta - (y1 - oy) * sintheta
        cpy = oy + (y1 - oy) * costheta + (newx1 - ox) * sintheta
        cpx = cpx / scalex
        sinnb, cosnb = np.sin(-beta), np.cos(-beta)
        return x1 + (cpx - x1) * cosnb - (cpy - y1) * sinnb, y1 + (cpy - y1) * cosnb + (cpx - x1) * sinnb


def _xspline_f(numerator, denominator):
    p = 2 * denominator * denominator
    u = numerator / denominator
    u2 = u * u
    return u * u2 * (10 - p + (2 * p - 15) * u + (6 - p) * u2)


def _xspline_g(u, q):
    return u * (q + u * (2 * q + u * (8 - 12 * q + u * (14 * q - 11 + u * (4 - 5 * q)))))


def _xspline_h(u, q):
    u2 = u * u
    return u * (q + u * (2 * q + u2 * (-2 * q - u * q)))


def _xspline_blend(k, t, s1, s2, negative_s2=False):
    """
    The four blending weights of X-spline segment k at t (R's xspline.c *_influence functions); a shape
    below 0 uses the negative influence, as R's (negative_s2 also for s2 = 0, as R's middle point).
    """
    if s1 < 0:
        a0, a2 = _xspline_h(-t, -s1), _xspline_g(t, -s1)
    else:
        tk = k + 1 + s1
        a0 = _xspline_f(t + k + 1 - tk, k - tk) if t + k + 1 < tk else 0.0
        tk = k + 1 - s1
        a2 = _xspline_f(t + k + 1 - tk, k + 2 - tk)
    if s2 < 0 or negative_s2:
        a1, a3 = _xspline_g(1 - t, -s2), _xspline_h(t - 1, -s2)
    else:
        tk = k + 2 + s2
        a1 = _xspline_f(t + k + 1 - tk, k + 1 - tk)
        tk = k + 2 - s2
        a3 = _xspline_f(t + k + 1 - tk, k + 3 - tk) if t + k + 1 > tk else 0.0
    return a0, a1, a2, a3


def _xspline_point(weights, px, py):
    total = weights[0] + weights[1] + weights[2] + weights[3]
    return (sum(w * p for w, p in zip(weights, px)) / total, sum(w * p for w, p in zip(weights, py)) / total)


def _xspline_step(k, px, py, s1, s2, precision, device_diagonal):
    """R's step_computing(): the t step for one segment, from its length and bend (in 1200 ppi units)."""
    if s1 == 0 and s2 == 0:
        return 1.0
    xstart, ystart = _xspline_point(_xspline_blend(k, 0.0, s1, s2), px, py) if s1 > 0 else (px[1], py[1])
    xend, yend = _xspline_point(_xspline_blend(k, 1.0, s1, s2), px, py) if s2 > 0 else (px[2], py[2])
    xmid, ymid = _xspline_point(_xspline_blend(k, 0.5, s1, s2, negative_s2=s2 <= 0), px, py)
    xv1, yv1, xv2, yv2 = xstart - xmid, ystart - ymid, xend - xmid, yend - ymid
    sides = np.sqrt((xv1 * xv1 + yv1 * yv1) * (xv2 * xv2 + yv2 * yv2))
    angle_cos = 0.0 if sides == 0 else (xv1 * xv2 + yv1 * yv2) / sides
    distance = min(np.sqrt((xend - xstart) ** 2 + (yend - ystart) ** 2), device_diagonal)
    steps = np.sqrt(distance) / 2 + int((1 + angle_cos) * 10)
    step = 1.0 if steps == 0 else precision / steps
    return 0.2 if step > 0.2 or step == 0 else step


def r_xspline(x, y, shape, inches_per_unit=1.0, device_diagonal=np.inf):
    """
    The points of grid's open X-spline with repeated end points (xsplineGrob(open = TRUE, repEnds = TRUE)):
    a port of compute_open_spline() in R's src/main/xspline.c (Blanc and Schlick's X-splines). R samples each
    segment in steps that depend on its size on the device (1200 points per inch); inches_per_unit converts
    x and y to inches for that, and device_diagonal (in 1200ths of an inch) caps the length R uses.
    Returns arrays of x and y in the units of x and y.
    """
    scale = 1200 * inches_per_unit
    xs, ys, ss = np.asarray(x, dtype=float) * scale, np.asarray(y, dtype=float) * scale, list(shape)
    n = len(xs)
    if n < 2:
        raise ValueError("there must be at least two control points")
    out_x, out_y = [], []

    def add(px_, py_):
        if out_x and out_x[-1] == px_ and out_y[-1] == py_:
            return
        out_x.append(px_)
        out_y.append(py_)

    def segment(k, px, py, s1, s2):
        step = _xspline_step(k, px, py, s1, s2, 1.0, device_diagonal)  # GEXspline uses LOW_PRECISION
        t = 0.0
        while t < 1:
            add(*_xspline_point(_xspline_blend(k, t, s1, s2), px, py))
            t += step

    def control(indices):
        return [xs[i % n] for i in indices], [ys[i % n] for i in indices], [ss[i % n] for i in indices]

    px, py, ps = control([0, 0, 1, 1 if n == 2 else 2])
    k = 0
    while True:
        segment(k, px, py, ps[1], ps[2])
        if k + 3 >= n:
            break
        px, py, ps = control([k, k + 1, k + 2, k + 3])
        k += 1
    px, py, ps = control([n - 2 if n == 2 else n - 3, n - 2, n - 1, n - 1])
    segment(k, px, py, ps[1], ps[2])
    add(px[3], py[3])
    return np.array(out_x) / scale, np.array(out_y) / scale


def r_curve(x1, y1, x2, y2, curvature=0.5, angle=90, ncp=5, shape=0.5, inches_per_unit=1.0):
    """
    The line ggplot2's geom_curve() draws between two points (grid's curveGrob() with square = FALSE and
    inflect = FALSE): an X-spline from the start through calcControlPoints()'s control points (shape 0.5)
    to the end. Computed in the units of the points, which must be equally scaled on both axes (as with
    coord_fixed()); returns arrays of x and y.
    """
    if x1 == x2 and y1 == y2:
        raise ValueError("end points must not be identical")
    if curvature == 0 or angle < 1 or angle > 179:
        return np.array([x1, x2], dtype=float), np.array([y1, y2], dtype=float)
    cpx, cpy = r_curve_control_points(x1, y1, x2, y2, curvature, angle, ncp)
    return r_xspline(np.concatenate([[x1], cpx, [x2]]), np.concatenate([[y1], cpy, [y2]]),
                     [0] + [shape] * ncp + [0], inches_per_unit=inches_per_unit)


def _sequential_sum(values):
    """A C loop's sum (s += v[i], in order): np.cumsum adds sequentially, np.sum pairwise."""
    return float(np.cumsum(values)[-1]) if len(values) else 0.0


def _lowess_fit(x, y, n, xs, nleft, nright, userw, rw):
    """R's lowest() (stats/src/lowess.c), 0-based: the locally weighted fit at xs, or None when all weights are 0."""
    span = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9, h1 = 0.999 * h, 0.001 * h
    r = np.abs(x[nleft:] - xs)
    stop = np.flatnonzero((r > h9) & (x[nleft:] > xs))  # pick up all ties on the right
    nrt = nleft + (stop[0] if len(stop) else n - nleft) - 1
    r = r[:nrt - nleft + 1]
    with np.errstate(divide="ignore", invalid="ignore"):  # h = 0 when the window holds ties only
        q = r / h
        cube = q * q * q
        w = np.where(r <= h9, np.where(r <= h1, 1.0, (1.0 - cube) * (1.0 - cube) * (1.0 - cube)), 0.0)
    if userw:
        w = np.where(r <= h9, w * rw[nleft:nrt + 1], 0.0)
    a = _sequential_sum(w)
    if a <= 0:
        return None
    w = w / a
    xw = x[nleft:nrt + 1]
    if h > 0:
        a = _sequential_sum(w * xw)
        b = xs - a
        d = xw - a
        c = _sequential_sum(w * (d * d))
        if np.sqrt(c) > 0.001 * span:
            b /= c
            w = w * (b * (xw - a) + 1.0)
    return _sequential_sum(w * y[nleft:nrt + 1])


def r_lowess(x, y, f=2 / 3, iter=3, delta=None):
    """
    R's lowess(x, y, f, iter, delta): a port of clowess() (R's stats/src/lowess.c), Cleveland's robust locally
    weighted regression. Points closer than delta (default 1% of the range of x) to the last fitted point are
    interpolated; ties share one fit. Returns (x sorted, fitted y), as lowess()'s list(x, y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if delta is None:
        delta = 0.01 * (x.max() - x.min())
    order = np.argsort(x, kind="stable")  # R's order(): ties keep their order
    x, y = x[order], y[order]
    n = len(x)
    ys = np.empty(n)
    if n < 2:
        return x, y.copy()
    ns = max(2, min(n, int(f * n + 1e-7)))
    rw = np.ones(n)
    res = np.zeros(n)
    iteration = 1
    while iteration <= iter + 1:
        nleft, nright, last, i = 0, ns - 1, -1, 0
        while True:
            if nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            fitted = _lowess_fit(x, y, n, x[i], nleft, nright, iteration > 1, rw)
            ys[i] = y[i] if fitted is None else fitted
            if last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        res = y - ys
        sc = _sequential_sum(np.abs(res)) / n
        if iteration > iter:
            break
        ordered = np.sort(np.abs(res))
        m1 = n // 2
        cmad = 3.0 * (ordered[m1] + ordered[n - m1 - 1]) if n % 2 == 0 else 6.0 * ordered[m1]
        if cmad < 1e-7 * sc:
            break
        c9, c1 = 0.999 * cmad, 0.001 * cmad
        r = np.abs(res)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = r / cmad
            square = 1.0 - q * q
            rw = np.where(r <= c1, 1.0, np.where(r <= c9, square * square, 0.0))
        iteration += 1
    return x, ys


def r_qqnorm(y):
    """R's qqnorm(y, plot.it = FALSE): (qnorm(ppoints(n))[order(order(y))], y), missing values kept as NaN."""
    from scipy import stats
    y = np.asarray(y, dtype=float)
    x = np.full(len(y), np.nan)
    present = ~np.isnan(y)
    values = y[present]
    ranks = np.empty(len(values), dtype=int)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    x[present] = stats.norm.ppf(r_ppoints(len(values)))[ranks]
    return x, y


def r_quantile7(x, probs):
    """R's quantile(x, probs, type = 7, na.rm = TRUE), with R's own (1 - h) * x[lo] + h * x[hi]."""
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    n = len(x)
    result = []
    for p in np.atleast_1d(probs):
        index = 1 + max(n - 1, 0) * p
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        q = x[lo - 1]
        h = index - lo
        if index > lo and x[hi - 1] != q:
            q = (1 - h) * q + h * x[hi - 1]
        result.append(q)
    return np.array(result)
