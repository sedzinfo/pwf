# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_GENERATE_DATA.R. Random draws use numpy's
global RNG (np.random.seed() makes them reproducible); the values differ from
R's, but the shapes, column names, types and every deterministic part match
rwf (checked in tests/test_functions_generate_data.py).

Note: `functions_generic.py` already has differently-shaped versions of
several of these ideas (generate_normal/generate_uniform instead of a
single generate_data(type=...) dispatcher, generate_factor_exact/
generate_factor_randomized instead of generate_factor(type=...), etc.).
This file instead mirrors the R names/signatures exactly, matching the
"new self-contained file with full R parity" pattern used for the other
recent ports (functions.py, functions_environment.py,
functions_train_test_full.py) — these names will shadow the
functions_generic.py versions if both are wildcard-imported together.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import string
import numpy as np
import pandas as pd
def _r_colon(start, end):
    """R's start:end: steps of 1 from start towards end (integers when start is whole)."""
    step = 1 if end >= start else -1
    values = start + step * np.arange(int(np.floor(abs(end - start) + 1e-10)) + 1)
    return values.astype(int) if float(start).is_integer() else values
##########################################################################################
# GENERATE RANDOM NUMBERS
##########################################################################################
def generate_data(nrows=10, ncols=5, mean=0, sd=1, min=1, max=5, type="normal"):
    """
    Generate a data frame of random numbers, normally or uniformly
    distributed.

    Parameters:
    nrows (int, optional): Number of rows. Defaults to 10.
    ncols (int, optional): Number of columns. Defaults to 5.
    mean (float, optional): Mean of the normal distribution. Only used
        with type="normal". Defaults to 0.
    sd (float, optional): Standard deviation of the normal distribution.
        Only used with type="normal". Defaults to 1.
    min (int, optional): Minimum of the uniform distribution (inclusive).
        Only used with type="uniform". Defaults to 1.
    max (int, optional): Maximum of the uniform distribution (inclusive).
        Only used with type="uniform". Defaults to 5. As in rwf the values
        are drawn from R's min:max (min, min + 1, ... up to max), so
        min=1.5, max=4 draws from 1.5, 2.5, 3.5, and, as R's sample(), a
        single value n >= 1 (min == max) draws from 1..n.
    type (str, optional): "normal" or "uniform". Defaults to "normal".

    Returns:
    pandas.DataFrame: nrows x ncols of random values, columns named
    "X1".."Xncols". Any other type gives all-missing columns, as in rwf.

    Examples:
    >>> generate_data(nrows=10, ncols=5, mean=0, sd=1, type="normal")
    >>> generate_data(nrows=10, ncols=5, min=1, max=5, type="uniform")
    """
    columns = [f"X{i+1}" for i in range(ncols)]
    if type == "normal":
        data = np.random.normal(loc=mean, scale=sd, size=(nrows, ncols))
    elif type == "uniform":
        pool = _r_sample_pool(_r_colon(min, max))
        data = np.random.choice(np.asarray(pool), size=(nrows, ncols), replace=True)
    else:
        data = np.full((nrows, ncols), np.nan)
    return pd.DataFrame(data, columns=columns)
##########################################################################################
# GENERATE FACTOR
##########################################################################################
def generate_factor(vector=None, nrows=2, ncols=10, type="random"):
    """
    Generate a data frame (or single factor) of categorical values
    sampled from a pool, either randomly or balanced across levels.

    Parameters:
    vector (list, optional): Pool of factor levels. Defaults to ["A".."E"].
    nrows (int, optional): Number of rows. For type="balanced", nrows
        should be divisible by len(vector). Defaults to 2.
    ncols (int, optional): Number of columns. When ncols=1, a single
        categorical Series (no name) is returned instead of a DataFrame, as
        R returns a single factor.
        Defaults to 10.
    type (str, optional): "random" (sample independently with
        replacement) or "balanced" (each level appears exactly
        nrows/len(vector) times per column). Defaults to "random".

    Returns:
    pandas.DataFrame or pandas.Series: Categorical columns "X1".."Xncols",
    nrows x ncols, with categories in the order of `vector` (or a single
    Series when ncols=1).

    Examples:
    >>> generate_factor(vector=list("ABCDE"), ncols=5, nrows=10, type="random")
    >>> generate_factor(vector=list("ABCDE"), ncols=5, nrows=10, type="balanced")
    >>> generate_factor(vector=list("ABCDE"), ncols=1, nrows=10, type="balanced")
    """
    if vector is None:
        vector = list(string.ascii_uppercase[:5])

    data = {}
    for j in range(ncols):
        if type == "balanced":
            col = np.repeat(vector, nrows // len(vector))
        else:
            col = np.random.choice(vector, size=nrows, replace=True)
        data[f"X{j+1}"] = pd.Categorical(col, categories=vector)

    result = pd.DataFrame(data)
    if ncols == 1:
        return result.iloc[:, 0].rename(None)
    return result
##########################################################################################
# GENERATE RANDOM STRING
##########################################################################################
def generate_string(vector=None, vector_length=1, nchar=5):
    """
    Generate random strings by sampling characters from a pool.

    Parameters:
    vector (list, optional): Character pool. Defaults to uppercase +
        lowercase letters + digits.
    vector_length (int, optional): Number of strings to generate. Defaults to 1.
    nchar (int, optional): Length of each string. Defaults to 5.

    Returns:
    list of str: `vector_length` random strings, each `nchar` characters long.

    Examples:
    >>> generate_string(nchar=10)
    >>> generate_string(nchar=10, vector_length=10)
    """
    if vector is None:
        vector = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    return ["".join(np.random.choice(vector, size=nchar, replace=True)) for _ in range(vector_length)]
##########################################################################################
# GENERATE MULTIPLE RESPONSE VECTOR
##########################################################################################
def _r_sample_pool(x):
    """The pool R's sample(x) draws from: 1..x when x is a single number >= 1, else x itself."""
    x = list(np.atleast_1d(x))
    if len(x) == 1 and isinstance(x[0], (int, float, np.integer, np.floating)) and x[0] >= 1:
        return list(range(1, int(x[0]) + 1))
    return x


def _r_as_character(value):
    """R's as.character for a scalar, for whole numbers: 2.0 -> "2"."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def generate_multiple_response_vector(responses=range(1, 5), responded=range(1, 5), length=10):
    """
    Simulate a multiple-response survey item (port of
    rwf::generate_multiple_response_vector): each element is a comma-separated
    string of distinct responses, drawn without replacement, with how many
    responses drawn from `responded`. Draws use numpy's global RNG, so
    np.random.seed() makes them reproducible (the values differ from R's).

    Parameters:
    responses (sequence, optional): Possible responses. Defaults to 1..4. As
        in R's sample(), a single number n means 1..n.
    responded (sequence, optional): Possible numbers of responses per
        element. Defaults to 1..4. A single number n means 1..n.
    length (int, optional): Number of elements. Defaults to 10.

    Returns:
    list of str: length strings such as "3, 1".

    Examples:
    >>> generate_multiple_response_vector(responses=range(1, 5), responded=range(1, 5), length=10)
    >>> generate_multiple_response_vector(responses=["a", "b", "c"], responded=[1, 2], length=5)
    """
    responses = _r_sample_pool(responses)
    responded = _r_sample_pool(responded)
    result = []
    for _ in range(length):
        n = np.random.choice(responded)
        drawn = np.random.choice(len(responses), size=n, replace=False)
        result.append(", ".join(_r_as_character(responses[i]) for i in drawn))
    return result


def generate_multiple_responce_vector(responces=None, responded=None, length=10):
    """Earlier pwf name of generate_multiple_response_vector, kept so existing code keeps working."""
    return generate_multiple_response_vector(responses=range(1, 5) if responces is None else responces,
                                             responded=range(1, 5) if responded is None else responded,
                                             length=length)
##########################################################################################
# SIMULATE CORRELATION MATRIX
##########################################################################################
def _default_correlation_matrix(n):
    """
    rwf's default target: symmetric_matrix(as.matrix(generate_data(ncols = n,
    nrows = n, min = 0.1, max = 1, type = "uniform"))) with a unit diagonal.
    R's 0.1:1 is the single value 0.1, so every off-diagonal cell is 0.1.
    """
    m = np.full((n, n), 0.1)
    np.fill_diagonal(m, 1)
    return m


def generate_correlation_matrix(correlation_martix=None, nrows=10, correlation_matrix=None):
    """
    Generate a data frame whose columns reproduce a target correlation
    structure, via Cholesky decomposition (data = t(chol(R)) %*% Z).

    Parameters:
    correlation_martix (array-like or pandas.DataFrame, optional): Symmetric
        positive-definite matrix of the desired correlations (rwf's spelling
        of the argument). If None, rwf's default is used: an nrows x nrows
        matrix with 0.1 off the diagonal and 1 on it (rwf builds it from
        generate_data(min = 0.1, max = 1, type = "uniform"), and R's 0.1:1
        is the single value 0.1).
    nrows (int, optional): Number of observations to generate. Defaults to 10.
    correlation_matrix (array-like, optional): Correctly spelled alias of
        correlation_martix, kept for existing pwf code.

    Returns:
    pandas.DataFrame: nrows x ncol(correlation_martix), columns named after
    the matrix's columns when it is a DataFrame (R keeps its colnames),
    "X1".."Xn" otherwise. Reproduced correlations approximate the target matrix,
    with accuracy improving as nrows increases.

    Examples:
    >>> df = pd.DataFrame([[1, .999], [.999, 1]])
    >>> data = generate_correlation_matrix(df.to_numpy(), nrows=100)
    >>> data.corr()
    """
    if correlation_martix is None:
        correlation_martix = correlation_matrix
    if correlation_martix is None:
        correlation_martix = _default_correlation_matrix(nrows)
    columns = list(correlation_martix.columns) if isinstance(correlation_martix, pd.DataFrame) else None
    correlation_matrix = np.asarray(correlation_martix, dtype=float)

    # numpy's cholesky returns lower-triangular L (L @ L.T = Sigma), the
    # transpose of R's chol() upper-triangular convention (t(U) %*% U = Sigma)
    # — using the wrong one would silently reproduce the wrong correlations.
    L = np.linalg.cholesky(correlation_matrix)
    nvars = L.shape[0]
    z = np.random.normal(size=(nvars, nrows))
    r = L @ z
    return pd.DataFrame(r.T, columns=columns or [f"X{i+1}" for i in range(nvars)])
##########################################################################################
# SIMULATE DATA FROM SAMPLE
##########################################################################################
def simulate_correlation_from_sample(cordata, nrows=10):
    """
    Simulate data that preserves the correlation structure of an input
    data frame, by drawing from a multivariate normal distribution
    parameterised by its sample covariance and column means.

    Parameters:
    cordata (pandas.DataFrame or array-like): Source numeric data.
        Missing values handled pairwise (pandas' default .cov() behavior).
    nrows (int, optional): Number of observations to simulate. Defaults to 10.

    Returns:
    pandas.DataFrame: nrows x ncol(cordata), with simulated values whose
    correlation structure approximates that of cordata.

    Examples:
    >>> correlation_matrix = generate_correlation_matrix(nrows=100)
    >>> correlation_matrix.corr()
    >>> simulate_correlation_from_sample(correlation_matrix, nrows=1000).corr()
    """
    cordata = pd.DataFrame(cordata)
    cov = cordata.cov()
    means = cordata.mean()
    sim = np.random.multivariate_normal(mean=means.to_numpy(), cov=cov.to_numpy(), size=nrows)
    return pd.DataFrame(sim, columns=cordata.columns)
##########################################################################################
# SIMULATE MISSING DATA
##########################################################################################
def generate_missing(df, missing=5):
    """
    Introduce missing values into a numeric vector or data frame: a
    fixed number of positions are replaced with NaN, independently per
    column for a data frame.

    Parameters:
    df (array-like, pandas.Series or pandas.DataFrame): Vector or data
        frame; text columns are allowed, as in R.
    missing (int, optional): Number of values to replace with NaN, per
        vector or per column. Must not exceed the vector length /
        nrow(df). Defaults to 5.

    Returns:
    Same type as `df` (a numpy array for a list), with `missing` values per
    vector or column replaced by NaN; integer columns become floats, text
    keeps its type. A copy: the input is not modified.

    Examples:
    >>> import numpy as np
    >>> generate_missing(np.random.normal(size=10), missing=5)
    >>> generate_missing(generate_data(nrows=10, ncols=2), missing=5)
    """
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        n = len(result)
        for j in range(result.shape[1]):
            idx = np.random.choice(n, size=missing, replace=False)
            column = result.iloc[:, j].copy()
            column = column.astype(float) if column.dtype.kind in "iub" else column.astype(object)
            column.iloc[idx] = np.nan
            result.isetitem(j, column.infer_objects())
        return result
    if isinstance(df, pd.Series):
        result = df.astype(float) if df.dtype.kind in "iub" else df.astype(object)
        result.iloc[np.random.choice(len(result), size=missing, replace=False)] = np.nan
        return result.infer_objects()
    arr = np.array(df)
    arr = arr.astype(float) if arr.dtype.kind in "iufb" else arr.astype(object)
    idx = np.random.choice(len(arr), size=missing, replace=False)
    arr[idx] = np.nan
    return arr
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80, "\ngenerate_data\n", "=" * 80, sep="")
    print(generate_data(nrows=5, ncols=3, mean=0, sd=1, type="normal"))
    print()
    print(generate_data(nrows=5, ncols=3, min=1, max=5, type="uniform"))

    print("\n" + "=" * 80, "\ngenerate_factor\n", "=" * 80, sep="")
    print(generate_factor(vector=list("ABCDE"), ncols=3, nrows=10, type="random"))
    print()
    print(generate_factor(vector=list("ABCDE"), ncols=3, nrows=10, type="balanced"))
    print()
    print(generate_factor(vector=list("ABCDE"), ncols=1, nrows=10, type="balanced"))

    print("\n" + "=" * 80, "\ngenerate_string\n", "=" * 80, sep="")
    print(generate_string(nchar=10, vector_length=5))

    print("\n" + "=" * 80, "\ngenerate_multiple_response_vector\n", "=" * 80, sep="")
    print(generate_multiple_response_vector(responses=[1, 2, 3, 4], responded=[1, 2, 3, 4], length=10))

    print("\n" + "=" * 80, "\ngenerate_correlation_matrix\n", "=" * 80, sep="")
    target = np.array([[1, .9], [.9, 1]])
    simulated = generate_correlation_matrix(target, nrows=2000)
    print("target:\n", target)
    print("reproduced (should be close to target):\n", simulated.corr().to_numpy())

    print("\n" + "=" * 80, "\nsimulate_correlation_from_sample\n", "=" * 80, sep="")
    sampled = simulate_correlation_from_sample(simulated, nrows=2000)
    print("reproduced from sample (should also be close to target):\n", sampled.corr().to_numpy())

    print("\n" + "=" * 80, "\ngenerate_missing\n", "=" * 80, sep="")
    print(generate_missing(generate_data(nrows=10, ncols=2), missing=3))
