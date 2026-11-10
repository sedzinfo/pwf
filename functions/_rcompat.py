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
    """R's round() for one number (IEC 60559 half-to-even on the decimal representation)."""
    if x is None or pd.isna(x):
        return np.nan
    return float(round(float(x), digits))


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
