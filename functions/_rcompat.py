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
