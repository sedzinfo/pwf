# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_GENERATEDATA.R.

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
        Only used with type="uniform". Defaults to 5.
    type (str, optional): "normal" or "uniform". Defaults to "normal".

    Returns:
    pandas.DataFrame: nrows x ncols of random values, columns named
    "X1".."Xncols".

    Examples:
    >>> generate_data(nrows=10, ncols=5, mean=0, sd=1, type="normal")
    >>> generate_data(nrows=10, ncols=5, min=1, max=5, type="uniform")
    """
    columns = [f"X{i+1}" for i in range(ncols)]
    if type == "normal":
        data = np.random.normal(loc=mean, scale=sd, size=(nrows, ncols))
    elif type == "uniform":
        data = np.random.randint(min, max + 1, size=(nrows, ncols))
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
        pandas.Categorical Series is returned instead of a DataFrame.
        Defaults to 10.
    type (str, optional): "random" (sample independently with
        replacement) or "balanced" (each level appears exactly
        nrows/len(vector) times per column). Defaults to "random".

    Returns:
    pandas.DataFrame or pandas.Series: Categorical data, nrows x ncols
    (or a single Series when ncols=1).

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
        return result.iloc[:, 0]
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
def generate_multiple_responce_vector(responces=None, responded=None, length=10):
    """
    Generate a multiple-response survey-style vector: each element is a
    comma-separated string of a random number of distinct sampled
    categories.

    Parameters:
    responces (list, optional): Pool of unique response categories.
        Defaults to [1, 2, 3, 4].
    responded (list, optional): Candidate counts of how many categories
        to select per observation; one value is drawn from this list at
        each iteration. Defaults to [1, 2, 3, 4].
    length (int, optional): Number of observations. Defaults to 10.

    Returns:
    list of str: `length` comma-separated strings of sampled categories.

    Examples:
    >>> generate_multiple_responce_vector(responces=[1,2,3,4], responded=[1,2,3,4], length=10)
    """
    if responces is None:
        responces = list(range(1, 5))
    if responded is None:
        responded = list(range(1, 5))

    result = []
    for _ in range(length):
        n = np.random.choice(responded)
        # R's sample() defaults to replace=FALSE
        sampled = np.random.choice(responces, size=n, replace=False)
        result.append(", ".join(str(x) for x in sampled))
    return result
##########################################################################################
# SIMULATE CORRELATION MATRIX
##########################################################################################
def _random_symmetric_matrix(n):
    """Random symmetric matrix with unit diagonal, used as a convenience default."""
    m = np.random.uniform(0.1, 1, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1)
    return m


def generate_correlation_matrix(correlation_matrix=None, nrows=10):
    """
    Generate a data frame whose columns reproduce a target correlation
    structure, via Cholesky decomposition. If no matrix is given, a
    random symmetric matrix (unit diagonal) is generated automatically.

    Note: R's parameter is named `correlation_martix` (a typo in the R
    source); the Python parameter is spelled correctly.

    Parameters:
    correlation_matrix (array-like, optional): Symmetric positive-definite
        matrix of the desired correlations. If None, a random one is used.
    nrows (int, optional): Number of observations to generate. Defaults to 10.

    Returns:
    pandas.DataFrame: nrows x ncol(correlation_matrix), columns named
    "X1".."Xn". Reproduced correlations approximate the target matrix,
    with accuracy improving as nrows increases.

    Examples:
    >>> df = pd.DataFrame([[1, .999], [.999, 1]])
    >>> data = generate_correlation_matrix(df.to_numpy(), nrows=100)
    >>> data.corr()
    """
    if correlation_matrix is None:
        correlation_matrix = _random_symmetric_matrix(nrows)
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)

    # numpy's cholesky returns lower-triangular L (L @ L.T = Sigma), the
    # transpose of R's chol() upper-triangular convention (t(U) %*% U = Sigma)
    # — using the wrong one would silently reproduce the wrong correlations.
    L = np.linalg.cholesky(correlation_matrix)
    nvars = L.shape[0]
    z = np.random.normal(size=(nvars, nrows))
    r = L @ z
    return pd.DataFrame(r.T, columns=[f"X{i+1}" for i in range(nvars)])
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
    df (array-like or pandas.DataFrame): Numeric vector or data frame.
    missing (int, optional): Number of values to replace with NaN, per
        vector or per column. Must not exceed the vector length /
        nrow(df). Defaults to 5.

    Returns:
    Same type as `df`, with `missing` values replaced by NaN (a fresh
    copy — the input is not modified in place).

    Examples:
    >>> import numpy as np
    >>> generate_missing(np.random.normal(size=10), missing=5)
    >>> generate_missing(generate_data(nrows=10, ncols=2), missing=5)
    """
    if isinstance(df, pd.DataFrame):
        result = df.astype(float)
        n = len(result)
        for col in result.columns:
            idx = np.random.choice(n, size=missing, replace=False)
            result.iloc[idx, result.columns.get_loc(col)] = np.nan
        return result
    else:
        arr = np.array(df, dtype=float)
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

    print("\n" + "=" * 80, "\ngenerate_multiple_responce_vector\n", "=" * 80, sep="")
    print(generate_multiple_responce_vector(responces=[1, 2, 3, 4], responded=[1, 2, 3, 4], length=10))

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
