# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_STATISTICAL.R.

Note: compute_confidence_inteval keeps R's misspelled name (missing an
"r" in "interval") rather than correcting it, since the whole point of
this port is 1:1 R-name parity for cross-referencing; a correctly
spelled compute_confidence_interval alias is provided alongside it for
convenience.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import scipy.stats
##########################################################################################
# ADJUST
##########################################################################################
def compute_adjustment(a, ntests):
    """
    Compute Bonferroni and Sidak corrected alpha thresholds for multiple
    comparisons.

    Parameters:
    a (float): Desired family-wise alpha level (e.g. 0.05).
    ntests (int): Number of tests/comparisons being performed.

    Returns:
    dict: {'sidak': 1 - (1-a)**(1/ntests), 'bonferroni': a/ntests}.

    Examples:
    >>> compute_adjustment(0.05, 100)
    """
    sidak = 1 - ((1 - a) ** (1 / ntests))
    bonferroni = a / ntests
    return {'sidak': sidak, 'bonferroni': bonferroni}
##########################################################################################
# STANDARDIZE
##########################################################################################
def compute_standard(vector, mean=0, sd=1, type="z", input="non_standard"):
    """
    Transform a numeric vector into one of several standard score
    formats. Can operate on raw scores or pre-standardized z-scores.

    Parameters:
    vector (array-like): Raw scores, or z-scores when input="standard".
    mean (float, optional): Population mean, used by type="uz" and
        type="normal_density". Defaults to 0.
    sd (float, optional): Population standard deviation, used by
        type="uz" and type="normal_density". Defaults to 1.
    type (str, optional): One of:
        - "z": z-scores (mean=0, sd=1).
        - "uz": un-standardize z-scores back to raw scores using the
          given mean/sd.
        - "sten": sten scores (1-10, mean=5.5, sd=2).
        - "t": T-scores (mean=50, sd=10).
        - "stanine": stanine scores (1-9, mean=5, sd=2).
        - "center": mean-centered scores.
        - "center_reversed": reversed mean-centered scores.
        - "percent": percentage of the maximum observed value.
        - "percentile": cumulative normal percentile (0-100).
        - "scale_zero_one": min-max scaled (0-1).
        - "normal_density": normal density values. Note: uses the same
          (non-standard) formula as the R source, 1/sqrt(sd*pi) rather
          than the textbook normal PDF's 1/(sd*sqrt(2*pi)) — kept as-is
          for parity rather than "corrected", since the goal is
          matching R's actual behavior.
        - "cumulative_density": cumulative sum of the input vector.
        - "all": data frame with every score type, sorted by z-score.
        Defaults to "z".
    input (str, optional): "non_standard" (vector holds raw scores) or
        "standard" (vector already holds z-scores). Defaults to
        "non_standard".

    Returns:
    numpy.ndarray of transformed scores, or a pandas.DataFrame when type="all".

    Examples:
    >>> import numpy as np
    >>> vector = np.concatenate([np.random.normal(size=10), [np.nan], np.random.normal(size=10)])
    >>> compute_standard(vector, type="z")
    >>> compute_standard(vector, mean=0, sd=1, type="uz")
    >>> compute_standard(vector, type="all")
    """
    vector = np.asarray(vector, dtype=float)
    if input == "non_standard":
        z = (vector - np.nanmean(vector)) / np.nanstd(vector, ddof=1)
    else:
        z = vector

    if type == "z":
        result = z
    elif type == "uz":
        result = vector * sd + mean
    elif type == "sten":
        result = np.clip(np.round(z * 2 + 5.5), 1, 10)
    elif type == "t":
        result = z * 10 + 50
    elif type == "stanine":
        result = np.round(np.clip(z * 2 + 5, 1, 9))
    elif type == "center":
        result = vector - np.nanmean(vector)
    elif type == "center_reversed":
        result = np.nanmean(vector) - vector
    elif type == "percent":
        result = (vector / np.nanmax(vector)) * 100
    elif type == "percentile":
        result = scipy.stats.norm.cdf(z) * 100
    elif type == "scale_zero_one":
        result = (vector - np.nanmin(vector)) / (np.nanmax(vector) - np.nanmin(vector))
    elif type == "normal_density":
        result = (1 / np.sqrt(sd * np.pi)) * np.exp(-0.5 * ((vector - mean) / sd) ** 2)
    elif type == "cumulative_density":
        result = np.cumsum(vector)
    elif type == "all":
        mydata = pd.DataFrame({'score': vector})
        for col, t in [('z', 'z'), ('sten', 'sten'), ('t', 't'), ('stanine', 'stanine'),
                       ('percent', 'percent'), ('percentile', 'percentile'),
                       ('scale_0_1', 'scale_zero_one')]:
            mydata[col] = compute_standard(mydata['score'].to_numpy(), mean=mean, sd=sd, type=t, input=input)
        result = mydata.sort_values('z').reset_index(drop=True)
    else:
        raise ValueError(f"Unknown type: {type!r}")
    return result
##########################################################################################
# COMPUTE DISSATENUATION
##########################################################################################
def compute_dissatenuation(variable1, error1, variable2, error2):
    """
    Correct the observed correlation between two variables for
    attenuation due to measurement error in both.

    Parameters:
    variable1 (array-like): True scores for the first variable.
    error1 (array-like): Measurement error for variable1 (same length).
    variable2 (array-like): True scores for the second variable.
    error2 (array-like): Measurement error for variable2 (same length).

    Returns:
    float: The disattenuated (corrected) correlation.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> compute_dissatenuation(np.random.normal(size=10), np.random.normal(size=10),
    ...                        np.random.normal(size=10), np.random.normal(size=10))
    """
    variable1, error1 = np.asarray(variable1), np.asarray(error1)
    variable2, error2 = np.asarray(variable2), np.asarray(error2)
    observed1 = variable1 + error1
    observed2 = variable2 + error2

    correlation = np.cov(observed1, observed2, ddof=1)[0, 1] / np.sqrt(
        np.var(observed1, ddof=1) * np.var(observed2, ddof=1))
    reliability1 = np.var(variable1, ddof=1) / (np.var(variable1, ddof=1) + np.var(error1, ddof=1))
    reliability2 = np.var(variable2, ddof=1) / (np.var(variable2, ddof=1) + np.var(error2, ddof=1))
    return correlation / np.sqrt(reliability1 * reliability2)
##########################################################################################
# COMPUTE SKEWNESS
##########################################################################################
def compute_skewness(vector):
    """
    Compute skewness using the b1 formula consistent with MINITAB and
    BMDP (matches e1071::skewness() with type=2). NaNs are removed
    before computation.

    Parameters:
    vector (array-like): Numeric vector.

    Returns:
    float: Positive = right skew, negative = left skew.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> compute_skewness(np.random.normal(size=1000))
    """
    vector = pd.Series(vector).dropna().to_numpy()
    n = len(vector)
    x = vector - vector.mean()
    y = np.sqrt(n) * np.sum(x ** 3) / (np.sum(x ** 2) ** 1.5)
    return y * (1 - 1 / n) ** 1.5
##########################################################################################
# COMPUTE KURTOSIS
##########################################################################################
def compute_kurtosis(vector):
    """
    Compute excess kurtosis using the b2 formula consistent with
    MINITAB and BMDP (matches e1071::kurtosis() with type=2). NaNs are
    removed before computation.

    Parameters:
    vector (array-like): Numeric vector.

    Returns:
    float: 0 = normal distribution, positive = heavier tails
    (leptokurtic), negative = lighter tails (platykurtic).

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> compute_kurtosis(np.random.normal(size=1000))
    """
    vector = pd.Series(vector).dropna().to_numpy()
    n = len(vector)
    x = vector - vector.mean()
    r = n * np.sum(x ** 4) / (np.sum(x ** 2) ** 2)
    return r * (1 - 1 / n) ** 2 - 3
##########################################################################################
# COMPUTE STANDARD ERROR
##########################################################################################
def compute_standard_error(vector):
    """
    Compute the standard error of the mean. NaNs are removed before computation.

    Parameters:
    vector (array-like): Numeric vector.

    Returns:
    float: Standard error of the mean.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> compute_standard_error(np.random.normal(size=1000))
    """
    x = pd.Series(vector).dropna().to_numpy()
    return np.sqrt(np.var(x, ddof=1) / len(x))
##########################################################################################
# COMPUTE CONFIDENCE INTERVAL
##########################################################################################
def compute_confidence_inteval(vector):
    """
    Compute the half-width of the 95% confidence interval of the mean
    (i.e. the margin of error: z * se). NaNs are removed before
    computation.

    Parameters:
    vector (array-like): Numeric vector.

    Returns:
    float: Half-width of the 95% CI.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> compute_confidence_inteval(np.random.normal(size=1000))
    """
    x = pd.Series(vector).dropna().to_numpy()
    n = len(x)
    s = np.std(x, ddof=1)
    return scipy.stats.norm.ppf(0.975) * s / np.sqrt(n)


compute_confidence_interval = compute_confidence_inteval
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    np.random.seed(1)

    print("=" * 80, "\ncompute_adjustment\n", "=" * 80, sep="")
    print(compute_adjustment(0.05, 100))

    print("\n" + "=" * 80, "\ncompute_standard\n", "=" * 80, sep="")
    vector = np.concatenate([np.random.normal(size=10), [np.nan], np.random.normal(size=10)])
    for t in ["z", "sten", "t", "stanine", "center", "center_reversed", "percent", "scale_zero_one"]:
        print(t, "->", np.round(compute_standard(vector, type=t), 3))
    print()
    print(compute_standard(vector, type="all"))

    print("\n" + "=" * 80, "\ncompute_dissatenuation\n", "=" * 80, sep="")
    print(compute_dissatenuation(np.random.normal(size=10), np.random.normal(size=10),
                                  np.random.normal(size=10), np.random.normal(size=10)))

    print("\n" + "=" * 80, "\ncompute_skewness / compute_kurtosis (cross-check vs scipy)\n", "=" * 80, sep="")
    sample = np.random.normal(size=1000)
    print("compute_skewness:", compute_skewness(sample), "| scipy.stats.skew(bias=False):", scipy.stats.skew(sample, bias=False))
    print("compute_kurtosis:", compute_kurtosis(sample), "| scipy.stats.kurtosis(bias=False):", scipy.stats.kurtosis(sample, bias=False))

    print("\n" + "=" * 80, "\ncompute_standard_error\n", "=" * 80, sep="")
    print(compute_standard_error(sample))

    print("\n" + "=" * 80, "\ncompute_confidence_inteval\n", "=" * 80, sep="")
    print(compute_confidence_inteval(sample))
