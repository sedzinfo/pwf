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
  - call_to_string has no true Python equivalent: R's model$call
    captures the literal, unevaluated call expression via language
    reflection, which Python doesn't have. This is a best-effort
    analog using a statsmodels formula-API model's .model.formula
    when available, falling back to repr(model) otherwise.
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
import textwrap
import numpy as np
import pandas as pd
##########################################################################################
# MULTIPLE GSUB
##########################################################################################
def str_mgsub(mydata, pattern, replacement, fixed=True):
    """
    Apply a single replacement for each of several patterns, sequentially.

    Parameters:
    mydata (str or list of str): String(s) to search within.
    pattern (list of str): Patterns to search for.
    replacement (str): Replacement string applied for every pattern.
    fixed (bool, optional): If True (default), patterns are matched
        literally; if False, as regular expressions.

    Returns:
    Same type as `mydata` (str if given a str, else list of str).

    Examples:
    >>> str_mgsub("#$%^&*_+", ["%", "*"], "REPLACE")
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
def str_split(vector, split="/", include_original=False):
    """
    Split each element of a string vector by a separator into a data
    frame of parts (one row per input element). Assumes every element
    produces the same number of parts.

    Parameters:
    vector (str or list of str): Strings to split.
    split (str, optional): Separator (matched literally). Defaults to "/".
    include_original (bool, optional): If True, appends the original
        input as a final "vector" column. Defaults to False.

    Returns:
    pandas.DataFrame: One row per element, columns "X1".."Xn" (plus
    "vector" if include_original=True).

    Examples:
    >>> string = [f"{i}/aa/bb/cc" for i in range(1, 11)]
    >>> str_split(string, split="/")
    """
    items = [vector] if isinstance(vector, str) else list(vector)
    parts = [s.split(split) for s in items]
    ncols = len(parts[0])
    result = pd.DataFrame(parts, columns=[f"X{i+1}" for i in range(ncols)])
    if include_original:
        result['vector'] = items
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
    type (str, optional): "row" splits df.index; "collumn" (matching
        R's own typo for "column") splits the column at `index`.
        Defaults to "row".
    index (int, optional): Column position to split when type="collumn".
    **kwargs: Forwarded to str_split.

    Returns:
    pandas.DataFrame: Split-part columns prepended to the original columns of df.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2]}, index=["1/aa/bb", "2/cc/dd"])
    >>> str_split_df(df, split="/", type="row")
    """
    if type == "row":
        source = df.index.astype(str).tolist()
    else:
        source = df.iloc[:, index].astype(str).tolist()
    split_df = str_split(source, split=split, **kwargs)
    split_df.index = df.index
    return pd.concat([split_df, df], axis=1)
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
    Trim and collapse whitespace in every string cell of a data frame;
    non-string cells are unchanged. Matches R's actual behavior (which
    uses strwrap(), collapsing internal whitespace too, not just
    trimming leading/trailing).

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
        return re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
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
    Best-effort description of a fitted model's specification as a
    whitespace-free string.

    Note: R's version extracts model$call, the literal unevaluated call
    expression via R's language reflection — Python has no equivalent.
    This tries a statsmodels formula-API model's .model.formula first,
    falling back to repr(model).

    Parameters:
    model: A fitted model object.

    Returns:
    str: Description of the model, whitespace removed.

    Examples:
    >>> import statsmodels.formula.api as smf
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    >>> model = smf.ols("y ~ x", data=df).fit()
    >>> call_to_string(model)
    """
    formula = getattr(getattr(model, "model", model), "formula", None)
    result = str(formula) if formula is not None else repr(model)
    return result.replace(" ", "")
##########################################################################################
# WRAP
##########################################################################################
def str_wrap(string, width=80):
    """
    Wrap each string to a line width, joining wrapped lines with "\\n".

    Parameters:
    string (str or list of str)
    width (int, optional): Maximum characters per line. Defaults to 80.

    Returns:
    Same type as `string`.

    Examples:
    >>> str_wrap("The quick brown fox jumped over the lazy dog", width=30)
    """
    is_scalar = isinstance(string, str)
    items = [string] if is_scalar else list(string)
    result = [textwrap.fill(s, width=width) for s in items]
    return result[0] if is_scalar else result
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
    length (int, optional): Width of the main separator. Defaults to
        half the current terminal width.

    Returns:
    None.

    Examples:
    >>> output_separator(string="TEST", output="TEST", instruction="TEST", length=40)
    """
    if length is None:
        length = shutil.get_terminal_size().columns // 2
    separator_title = "#" * length
    separator_subtitle = "#" * (length // 2)
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
    print(str_mgsub("#$%^&*_+", ["%", "*"], "REPLACE"))

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
