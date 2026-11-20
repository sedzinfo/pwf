# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_STRINGS.R.

Notes:
  - R's file defines str_wrap() twice; the second definition (in the
    "BASE R REPLACEMENTS FOR stringr FUNCTIONS" section) overwrites the
    first when the file is sourced, so only that final, vectorized
    version is actually reachable in R. Only that one is ported here.
  - fixed()/str_replace_all()/str_replace()/str_count()/str_split_fixed()
    exist in R because base R doesn't cleanly separate "literal string"
    from "regex" the way Python's str.replace()/re.sub() already do —
    Python doesn't need a fixed() marker trick for its own sake, but
    it's ported anyway (as a str subclass marker) for 1:1 R-name parity
    with the rest of this project.
  - call_to_string: R models keep the call that fitted them (model$call);
    Python models don't. pwf uses a call an object keeps (pwf's lda()
    does), else rebuilds the R call of a statsmodels formula model (lm or
    glm), else gives "NULL" as rwf does for a model without a call.
  - str_aes: tracing through the R source, its intermediate
    `trimws()` step is dead code — whichever branch (proper=TRUE/FALSE)
    is taken, the final str_squish() call re-trims anyway, so the net
    result is unaffected either way. This port skips the redundant step.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import re
import shutil
import warnings

import numpy as np
import pandas as pd

try:
    from ._rcompat import r_label, r_make_names, r_row_names, resolve_columns
except ImportError:
    from _rcompat import r_label, r_make_names, r_row_names, resolve_columns
##########################################################################################
# MULTIPLE GSUB
##########################################################################################
def str_mgsub(mydata, pattern, replacement, fixed=False):
    """
    Apply a single replacement for each of several patterns, sequentially,
    as rwf::str_mgsub (R's gsub for each pattern).

    Parameters:
    mydata (str or list of str): String(s) to search within.
    pattern (list of str): Patterns to search for.
    replacement (str): Replacement string applied for every pattern.
    fixed (bool, optional): Match the patterns literally (R's gsub(..., fixed = TRUE)).
        Defaults to False: regular expressions, as R's gsub.

    Returns:
    Same type as `mydata` (str if given a str, else list of str).

    Examples:
    >>> str_mgsub("#$%^&*_+", ["%", "[*]"], "REPLACE")
    """
    is_scalar = isinstance(mydata, str)
    items = [mydata] if is_scalar else list(mydata)
    for p in pattern:
        if fixed:
            items = [s.replace(p, replacement) for s in items]
        else:
            items = [re.sub(p, replacement, s) for s in items]
    return items[0] if is_scalar else items
##########################################################################################
# SPLIT STRING
##########################################################################################
def _r_strsplit(text, split):
    """R's strsplit(text, split, fixed = TRUE) for one string: a trailing empty piece is dropped, so "a/b/"
    gives a and b and "" nothing; a missing value gives one missing piece; split "" gives the characters."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return [np.nan]
    pieces = list(text) if split == "" else text.split(split)
    return pieces[:-1] if pieces and pieces[-1] == "" else pieces


def _r_matrix_by_row(values, ncol):
    """R's matrix(values, byrow = TRUE, ncol = ncol): ceiling(length / ncol) rows, the values recycled to fill
    them, with R's warning when the length doesn't fit."""
    n = len(values)
    if ncol == 0:
        if n:
            raise ValueError("nc = 0 for non-null data")
        return np.empty((0, 0), dtype=object)
    nrow = int(np.ceil(n / ncol))
    if n > 1 and (nrow * ncol) % n:
        if (n > nrow and n % nrow) or (n < nrow and nrow % n):
            warnings.warn(f"data length [{n}] is not a sub-multiple or multiple of the number of rows [{nrow}]")
        elif (n > ncol and n % ncol) or (n < ncol and ncol % n):
            warnings.warn(f"data length [{n}] is not a sub-multiple or multiple of the number of columns [{ncol}]")
        else:
            warnings.warn(f"data length differs from size of matrix: [{n} != {nrow} x {ncol}]")
    return np.array([values[k % n] for k in range(nrow * ncol)], dtype=object).reshape(nrow, ncol)


# the name R's data.frame() gives rwf's one-column matrix: its deparsed expression through make.names()
_ONE_PIECE_NAME = "matrix.unlist.str_split...byrow...TRUE..ncol...length.str_split..1...."


def str_split(vector, split="/", include_original=False):
    """
    Split each element of a string vector by a separator into a data
    frame of parts (one row per input element), as rwf: R's
    strsplit(fixed = TRUE), so a trailing separator adds no empty part,
    and the parts filled row by row into as many columns as the first
    element has parts. Elements with a different number of parts shift
    the rows, with R's warning, as rwf's matrix() does.

    Parameters:
    vector (str or list of str): Strings to split.
    split (str, optional): Separator (matched literally). Defaults to "/".
    include_original (bool, optional): If True, appends the original
        input as a final "vector" column. Defaults to False.

    Returns:
    pandas.DataFrame: One row per element, columns "X1".."Xn" (plus
    "vector" if include_original=True). One part per element gives the
    column R names after rwf's expression,
    "matrix.unlist.str_split...byrow...TRUE..ncol...length.str_split..1....".

    Examples:
    >>> string = [f"{i}/aa/bb/cc" for i in range(1, 11)]
    >>> str_split(string, split="/")
    """
    items = [vector] if isinstance(vector, str) else list(vector)
    parts = [_r_strsplit(s, split) for s in items]
    ncol = len(parts[0]) if parts else 0
    matrix = _r_matrix_by_row([piece for pieces in parts for piece in pieces], ncol)
    columns = [_ONE_PIECE_NAME] if ncol == 1 else [f"X{i + 1}" for i in range(ncol)]
    result = pd.DataFrame(matrix, columns=columns)
    if include_original:
        if len(result) != len(items):
            raise ValueError(f"arguments imply differing number of rows: {len(result)}, {len(items)}")
        result["vector"] = items
    return result
##########################################################################################
# SPLIT STRING IN DATAFRAME
##########################################################################################
def str_split_df(df, split="/", type="row", index=None, **kwargs):
    """
    Split a delimited string — from the data frame's row labels or from
    a specified column — and prepend the resulting parts as new columns.

    Parameters:
    df (pandas.DataFrame)
    split (str, optional): Separator. Defaults to "/".
    type (str, optional): "row" splits the row names: R's, so pandas'
        default 0, 1, ... index is split as "1", "2", ...; "collumn"
        (rwf's spelling) splits the column at `index`, as R's
        as.character() writes it (10.0 is "10"). Any other type fails,
        as in rwf. Defaults to "row".
    index (int or str, optional): Column to split when type="collumn": a
        1-based position, as in R, or a name.
    **kwargs: Forwarded to str_split.

    Returns:
    pandas.DataFrame: Split-part columns prepended to the original columns
    of df, all names made unique and valid as R's data.frame() does (a
    column X1 of df becomes X1.1 after the parts' X1).

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2]}, index=["1/aa/bb", "2/cc/dd"])
    >>> str_split_df(df, split="/", type="row")
    """
    if type == "row":
        source = r_row_names(df.index)
    elif type == "collumn":
        source = [value if pd.isna(value) else r_label(value) for value in df[resolve_columns(df, index)[0]]]
    else:  # rwf has no branch for it and fails on its undefined result
        raise ValueError(f'type must be "row" or "collumn", not {type!r} (rwf: object \'result\' not found)')
    split_df = str_split(source, split=split, **kwargs)
    if len(split_df) != len(df):
        raise ValueError(f"arguments imply differing number of rows: {len(split_df)}, {len(df)}")
    split_df.index = df.index
    result = pd.concat([split_df, df], axis=1)
    result.columns = r_make_names(result.columns)
    return result
##########################################################################################
# RETURN RIGHT LEFT CHARACTERS
##########################################################################################
def str_sub(x, n=2, type=None):
    """
    Extract n characters from the left or right of a string.

    Parameters:
    x (str or list of str)
    n (int, optional): Number of characters to extract. Defaults to 2.
    type (str): "left" or "right".

    Returns:
    Same type as `x`.

    Examples:
    >>> str_sub("12345", n=2, type="right")
    >>> str_sub("12345", n=2, type="left")
    """
    is_scalar = isinstance(x, str)
    items = [x] if is_scalar else list(x)
    if type == "right":
        result = [s[-n:] if n > 0 else "" for s in items]
    elif type == "left":
        result = [s[:n] for s in items]
    else:
        raise ValueError("type must be 'left' or 'right'")
    return result[0] if is_scalar else result
##########################################################################################
# PROPER
##########################################################################################
def str_proper(x):
    """
    Capitalize the first character and lowercase the rest of each
    string (whole-string case, not per-word title case).

    Parameters:
    x (str or list of str)

    Returns:
    Same type as `x`.

    Examples:
    >>> str_proper("HELLO WORLD")
    """
    is_scalar = isinstance(x, str)
    items = [x] if is_scalar else list(x)
    result = [s[:1].upper() + s[1:].lower() for s in items]
    return result[0] if is_scalar else result
##########################################################################################
# TRIM DATAFRAME
##########################################################################################
def str_trim_df(df):
    """
    Trim and collapse whitespace in every string cell of a data frame,
    as R's strwrap(), which rwf applies to each cell: runs of spaces,
    tabs and newlines become one space, except that two spaces after a
    sentence end (".", "?" or "!") stay two.

    Deviations, where rwf's apply() over the cells breaks the data frame:
    rwf returns a mixed data frame's numbers as formatted text ("10.0"),
    drops every column of an all-numeric one, turns missing values into
    the text "NA" and garbles the frame when a cell is longer than
    strwrap's line (72 characters in an 80-column console). pwf leaves
    the cells that aren't text as they are and keeps a long text on one
    line.

    Parameters:
    df (pandas.DataFrame)

    Returns:
    pandas.DataFrame: Same shape, string cells cleaned.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'s': ['  a   b  ', ' c']})
    >>> str_trim_df(df)
    """
    def clean_cell(x):
        return " ".join(line for line in _strwrap(x, np.inf) if line) if isinstance(x, str) else x
    return df.map(clean_cell)
##########################################################################################
# ADJUST STRING AESTHETICS
##########################################################################################
def str_aes(vector, characterlist=None, proper=True):
    """
    Replace separator characters/HTML tags with spaces, normalize
    whitespace, and optionally apply proper casing.

    Parameters:
    vector (str or list of str)
    characterlist (list of str, optional): Strings treated as
        separators, each replaced by a space. Defaults to common
        punctuation and HTML tags (".", "_", "-", ",", "$", "<p>",
        "</p>", "<br>", "<br/>", "<B>", "</B>", "<BR/>", "|", "/", "&nbsp").
    proper (bool, optional): If True (default), apply str_proper.

    Returns:
    Same type as `vector`.

    Examples:
    >>> str_aes(["TES.T", "TES<p>T", "TES&nbspT"])
    >>> str_aes(["TES.T", "TES<p>T", "TES&nbspT"], proper=False)
    """
    if characterlist is None:
        characterlist = [".", "_", "-", ",", "$", "<p>", "</p>", "<br>", "<br/>",
                          "<B>", "</B>", "<BR/>", "|", "/", "&nbsp"]
    is_scalar = isinstance(vector, str)
    items = [vector] if is_scalar else list(vector)
    for ch in characterlist:
        items = [s.replace(ch, " ") for s in items]

    result = str_proper(items) if proper else items
    result = str_squish(result)
    return result[0] if is_scalar else result
##########################################################################################
# MODEL CALL TO STRING
##########################################################################################
def call_to_string(model):
    """
    The call that fitted a model as one string, as rwf: R's deparse() of
    the call, its lines joined by ", ", every space removed.

    R keeps the call in the model (model$call, else model$Call). Here:
    - an object that keeps one (a `call` or `Call` attribute or dict
      entry: text, or deparse() lines as pwf's lda() keeps) gives it;
    - a statsmodels formula model gives the R call fitting the same
      model, lm(formula = ..., data = ...) for OLS and glm(formula = ...,
      family = ..., data = ...) for a GLM or Logit, the data named by its
      variable in the calling code ("data" when not found), as
      report_logistic's model_call;
    - anything else gives "NULL", as rwf for a model without a call.

    Parameters:
    model: A fitted model object.

    Returns:
    str: The call, spaces removed.

    Examples:
    >>> import statsmodels.formula.api as smf
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    >>> model = smf.ols("y ~ x", data=df).fit()
    >>> call_to_string(model)  # "lm(formula=y~x,data=df)"
    """
    lines = None
    for name in ("call", "Call"):  # R's model$call, then model$Call
        call = model.get(name) if isinstance(model, dict) else getattr(model, name, None)
        if call is not None and not callable(call):
            lines = [call] if isinstance(call, str) else [str(line) for line in call]
            break
    if lines is None:
        lines = _statsmodels_call(model)
    return (", ".join(lines) if lines else "NULL").replace(" ", "")


def _statsmodels_call(model):
    """deparse() lines of the R call fitting a statsmodels formula model (lm for OLS, glm for a GLM or
    Logit), or None."""
    try:
        from ._rcompat import r_deparse_call
        from .glm_linear_regression import _data_name, _deparse_formula
    except ImportError:
        from _rcompat import r_deparse_call
        from glm_linear_regression import _data_name, _deparse_formula
    fitted = getattr(model, "model", None)
    formula = getattr(fitted, "formula", None)
    if formula is None:
        return None
    kind = type(fitted).__name__
    if kind == "OLS":
        function, family = "lm", None
    elif kind == "Logit":
        function, family = "glm", "binomial"
    elif kind == "GLM":
        function = "glm"
        family = {"Binomial": "binomial", "Poisson": "poisson", "Gaussian": "gaussian", "Gamma": "Gamma",
                  "InverseGaussian": "inverse.gaussian"}.get(type(fitted.family).__name__)
        if family is None:
            return None
    else:
        return None
    # _data_name looks for the data's variable from its caller's caller up: call_to_string, then the code calling it
    arguments = [("formula", _deparse_formula(formula))] + ([("family", family)] if family else []) + \
        [("data", _data_name(fitted.data.frame))]
    return r_deparse_call(function, arguments)
##########################################################################################
# WRAP
##########################################################################################
def str_wrap(string, width=80):
    """
    Wrap each string as R's strwrap(x, width) and join the lines with
    "\\n", as rwf::str_wrap: whitespace runs become one space (two after
    a sentence end that had two), lines are shorter than width, a word
    longer than that gets its own line, and blank lines separate
    paragraphs.

    Parameters:
    string (str or list of str)
    width (int, optional): Lines are at most width - 1 characters. Defaults to 80.

    Returns:
    Same type as `string`.

    Examples:
    >>> str_wrap("The quick brown fox jumped over the lazy dog", width=30)
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    result = ["\n".join(_strwrap(s, width)) for s in items]
    return result[0] if is_scalar else result


def _strwrap(text, width):
    """R's strwrap(text, width) for one string (indent and exdent 0)."""
    lines = []
    for number, paragraph in enumerate(re.split(r"\n[ \t\n]*\n", text)):
        if number:
            lines.append("")
        tokens = re.split(r"[ \t\n]", paragraph)
        words = []  # [word, followed by a kept second space]
        for k, token in enumerate(tokens):
            if token:
                words.append([token, False])
            elif (words and k > 0 and tokens[k - 1] and not words[-1][1]
                  and re.search(r"[.?!][)\"']*$", words[-1][0])):
                words[-1][1] = True
        line, extra = "", False
        for word, word_extra in words:
            candidate = word if not line else line + (" " if extra else "") + " " + word
            if line and len(candidate) > width - 1:
                lines.append(line)
                candidate = word
            line, extra = candidate, word_extra
        if line or not words:
            lines.append(line)
    return lines
##########################################################################################
# OUTPUT SEPARATOR
##########################################################################################
def output_separator(string, output=None, instruction=None, length=None):
    """
    Print a heading, optional instructions, and optional output,
    surrounded by "#" separator lines.

    Parameters:
    string (str): Title, printed between the main separators.
    output (optional): Content printed below the heading, if given.
    instruction (str, optional): Text printed between the heading and
        output, followed by a shorter separator.
    length (float, optional): Width of the main separator, and half of it
        the subtitle's, both cut to whole characters as R's rep(). Defaults
        to a quarter of the terminal width, as R's getOption("width") / 4.

    Returns:
    None. R's print() shows each text as [1] "..."; pwf prints the text.

    Examples:
    >>> output_separator(string="TEST", output="TEST", instruction="TEST", length=40)
    """
    if length is None:
        length = shutil.get_terminal_size().columns / 4  # R: getOption("width") / 4
    separator_title = "#" * int(length)
    separator_subtitle = "#" * int(length / 2)
    print(separator_title)
    print(string)
    print(separator_title)
    if instruction is not None:
        print(instruction)
        print(separator_subtitle)
    if output is not None:
        print(output)
##########################################################################################
# BASE R REPLACEMENTS FOR stringr FUNCTIONS
##########################################################################################
class _FixedPattern(str):
    """Marker subclass of str: matched literally, not as a regex."""
    pass


def fixed(pattern):
    """
    Mark a pattern as a literal string rather than a regular
    expression, for use with str_replace/str_replace_all/str_count/str_split_fixed.

    Parameters:
    pattern (str): String to match literally.

    Returns:
    _FixedPattern: `pattern`, tagged for literal matching.

    Examples:
    >>> str_replace_all("a.b.c", ".", "-")        # "." as regex matches any char
    >>> str_replace_all("a.b.c", fixed("."), "-")  # "." matched literally
    """
    return _FixedPattern(pattern)


def str_replace_all(string, pattern, replacement=None):
    """
    Replace every occurrence of `pattern` in `string`.

    Parameters:
    string (str or list of str)
    pattern: A regex string, a literal string from fixed(), or a dict
        of {regex_pattern: replacement} applied sequentially (the
        Python analogue of R's named-character-vector form).
    replacement (str, optional): Replacement string. Ignored if
        `pattern` is a dict.

    Returns:
    Same type as `string`.

    Examples:
    >>> str_replace_all("hello world", "o", "0")
    >>> str_replace_all("a.b.c", fixed("."), "-")
    >>> str_replace_all("aabbcc", {"a": "X", "b": "Y"})
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    if isinstance(pattern, _FixedPattern):
        items = [s.replace(str(pattern), replacement) for s in items]
    elif isinstance(pattern, dict):
        for pat, repl in pattern.items():
            items = [re.sub(pat, repl, s) for s in items]
    else:
        items = [re.sub(pattern, replacement, s) for s in items]
    return items[0] if is_scalar else items


def str_replace(string, pattern, replacement):
    """
    Replace only the first occurrence of `pattern` in each element.

    Parameters:
    string (str or list of str)
    pattern: A regex string or a literal string from fixed().
    replacement (str)

    Returns:
    Same type as `string`.

    Examples:
    >>> str_replace("hello world", "o", "0")
    >>> str_replace("007 bond", "^0+", "")
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    if isinstance(pattern, _FixedPattern):
        items = [s.replace(str(pattern), replacement, 1) for s in items]
    else:
        items = [re.sub(pattern, replacement, s, count=1) for s in items]
    return items[0] if is_scalar else items


def str_split_fixed(string, pattern, n):
    """
    Split each element of `string` by `pattern` into exactly `n` pieces
    (short results padded with "").

    Parameters:
    string (str or list of str)
    pattern: A regex string or a literal string from fixed().
    n (int): Number of output columns.

    Returns:
    numpy.ndarray: shape (len(string), n).

    Examples:
    >>> str_split_fixed(["speed.run", "height.jump"], fixed("."), 2)
    >>> str_split_fixed(["a.b.c", "x.y"], fixed("."), 3)
    """
    items = [string] if isinstance(string, str) else list(string)
    if isinstance(pattern, _FixedPattern):
        parts_list = [s.split(str(pattern)) for s in items]
    else:
        parts_list = [re.split(pattern, s) for s in items]
    result = np.full((len(items), n), "", dtype=object)
    for i, parts in enumerate(parts_list):
        for j in range(min(n, len(parts))):
            result[i, j] = parts[j]
    return result


def str_count(string, pattern):
    """
    Count occurrences of `pattern` in each element of `string`.

    Parameters:
    string (str or list of str)
    pattern: A regex string or a literal string from fixed().

    Returns:
    Same type as `string` (int, or list of int).

    Examples:
    >>> str_count(["banana", "apple", "cherry"], "[aeiou]")
    >>> str_count(["a;b;c", "x;y", "z"], fixed(";"))
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    if isinstance(pattern, _FixedPattern):
        result = [s.count(str(pattern)) for s in items]
    else:
        result = [len(re.findall(pattern, s)) for s in items]
    return result[0] if is_scalar else result


def str_pad(string, width, side="right", pad=" "):
    """
    Pad each string to at least `width` characters.

    Parameters:
    string (str or list of str)
    width (int): Minimum total width.
    side (str, optional): "right" (default), "left", or "both".
    pad (str, optional): Single padding character. Defaults to " ".

    Returns:
    Same type as `string`.

    Examples:
    >>> str_pad(["1", "10", "100"], width=3, side="left", pad="0")
    >>> str_pad("hello", width=11, side="both")
    """
    is_scalar = isinstance(string, str)
    items = [str(s) for s in ([string] if is_scalar else list(string))]
    result = []
    for s in items:
        n = width - len(s)
        if n <= 0:
            result.append(s)
        elif side == "right":
            result.append(s + pad * n)
        elif side == "left":
            result.append(pad * n + s)
        else:
            lpad = pad * (n // 2)
            rpad = pad * (n - n // 2)
            result.append(lpad + s + rpad)
    return result[0] if is_scalar else result


def str_squish(string):
    """
    Trim leading/trailing whitespace and collapse internal whitespace
    runs to a single space.

    Parameters:
    string (str or list of str)

    Returns:
    Same type as `string`.

    Examples:
    >>> str_squish("  hello   world  ")
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    result = [re.sub(r"\s+", " ", s).strip() for s in items]
    return result[0] if is_scalar else result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nstr_mgsub\n", "=" * 80, sep="")
    print(str_mgsub("#$%^&*_+", ["%", "[*]"], "REPLACE"))

    print("\n" + "=" * 80, "\nstr_split\n", "=" * 80, sep="")
    string = [f"{i}/aa/bb/cc" for i in range(1, 11)]
    print(str_split(string, split="/"))

    print("\n" + "=" * 80, "\nstr_split_df\n", "=" * 80, sep="")
    df = pd.DataFrame({'val': [10, 20]}, index=["1/aa/bb", "2/cc/dd"])
    print(str_split_df(df, split="/", type="row"))

    print("\n" + "=" * 80, "\nstr_sub\n", "=" * 80, sep="")
    print(str_sub("12345", n=2, type="right"))
    print(str_sub("12345", n=2, type="left"))

    print("\n" + "=" * 80, "\nstr_proper\n", "=" * 80, sep="")
    print(str_proper(["HELLO", "wORLD"]))

    print("\n" + "=" * 80, "\nstr_trim_df\n", "=" * 80, sep="")
    df2 = pd.DataFrame({'str1': ['  A  B  ', 'C   D'], 'num1': [1.5, 2.5]})
    print(str_trim_df(df2))

    print("\n" + "=" * 80, "\nstr_aes\n", "=" * 80, sep="")
    vector = ["TES.T", "TES<p>T", "TES&nbspT"]
    print(str_aes(vector))
    print(str_aes(vector, proper=False))

    print("\n" + "=" * 80, "\ncall_to_string\n", "=" * 80, sep="")
    import statsmodels.formula.api as smf
    dfm = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1.1, 2.2, 2.9, 4.1]})
    model = smf.ols("y ~ x", data=dfm).fit()
    print(call_to_string(model))

    print("\n" + "=" * 80, "\nstr_wrap\n", "=" * 80, sep="")
    print(str_wrap("The quick brown fox jumped over the lazy dog", width=30))

    print("\n" + "=" * 80, "\noutput_separator\n", "=" * 80, sep="")
    output_separator(string="TEST", output="TEST", instruction="TEST", length=40)

    print("\n" + "=" * 80, "\nfixed / str_replace_all / str_replace\n", "=" * 80, sep="")
    print(str_replace_all("a.b.c", ".", "-"))
    print(str_replace_all("a.b.c", fixed("."), "-"))
    print(str_replace_all("aabbcc", {"a": "X", "b": "Y"}))
    print(str_replace("hello world", "o", "0"))

    print("\n" + "=" * 80, "\nstr_split_fixed\n", "=" * 80, sep="")
    print(str_split_fixed(["speed.run", "height.jump"], fixed("."), 2))
    print(str_split_fixed(["a.b.c", "x.y"], fixed("."), 3))

    print("\n" + "=" * 80, "\nstr_count\n", "=" * 80, sep="")
    print(str_count(["banana", "apple", "cherry"], "[aeiou]"))
    print(str_count(["a;b;c", "x;y", "z"], fixed(";")))

    print("\n" + "=" * 80, "\nstr_pad\n", "=" * 80, sep="")
    print(str_pad(["1", "10", "100"], width=3, side="left", pad="0"))
    print(str_pad("hello", width=11, side="both"))

    print("\n" + "=" * 80, "\nstr_squish\n", "=" * 80, sep="")
    print(repr(str_squish("  hello   world  ")))
