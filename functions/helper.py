# -*- coding: utf-8 -*-
'''helper functions conversion between moments
contains:
* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix
Author: Josef Perktold
License: BSD-3'''
from statsmodels.compat.python import range
import numpy as np
from scipy.misc import comb
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
## start moment helpers
def mc2mnc(mc):
    '''
    convert central to non-central moments, uses recursive formula optionally adjusts first moment to return mean
    
    This function transforms a list of central moments into non-central moments 
    while optionally adjusting the first moment to return the mean.

    Parameters
    ----------
    mc : list of float
        A list of central moments, where `mc[0]` is the mean, and subsequent elements 
        represent higher-order central moments.

    Returns
    -------
    list of float
        A list of non-central moments. The first element is the mean, and the rest 
        are the higher-order non-central moments.

    Notes
    -----
    The conversion follows the recursive formula:
    
    .. math::
        m_n = \sum_{k=0}^{n} \binom{n}{k} \mu_k \cdot \text{mean}^{(n-k)}

    where:
        - \( m_n \) is the non-central moment of order \( n \),
        - \( \mu_k \) is the central moment of order \( k \),
        - \( \text{mean} \) is the first moment (mean),
        - \( \binom{n}{k} \) is the binomial coefficient.

    Examples
    --------
    >>> mc = [3, 2, 4]  # Mean = 3, central moments: variance=2, skewness=4
    >>> mc2mnc(mc)
    [3, 11, 57]
    '''
    n = len(mc)
    mean = mc[0]
    mc = [1] + list(mc)    # add zero moment = 1
    mc[1] = 0              # define central mean as zero for formula
    mnc = [1, mean]        # zero and first raw moments
    for nn,m in enumerate(mc[2:]):
        n=nn+2
        mnc.append(0)
        for k in range(n+1):
            mnc[n] += comb(n,k,exact=1) * mc[k] * mean**(n-k)
    return mnc[1:]
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mnc2mc(mnc, wmean = True):
    '''
    convert non-central to central moments, uses recursive formula optionally adjusts first moment to return mean
    
    This function transforms a list of non-central moments into central moments.
    Optionally, it adjusts the first moment to return the mean.

    Parameters
    ----------
    mnc : list of float
        A list of non-central moments, where `mnc[0]` is the mean, and subsequent elements 
        represent higher-order non-central moments.
    wmean : bool, optional (default=True)
        If True, sets the first central moment (mean) to the input mean.

    Returns
    -------
    list of float
        A list of central moments. The first element is the variance, and the rest 
        are the higher-order central moments.

    Notes
    -----
    The conversion follows the recursive formula:

    .. math::
        \mu_n = \sum_{k=0}^{n} (-1)^{(n-k)} \binom{n}{k} m_k \cdot \text{mean}^{(n-k)}

    where:
        - \( \mu_n \) is the central moment of order \( n \),
        - \( m_k \) is the non-central moment of order \( k \),
        - \( \text{mean} \) is the first moment (mean),
        - \( \binom{n}{k} \) is the binomial coefficient.

    Examples
    --------
    >>> mnc = [3, 11, 57]  # Mean = 3, non-central moments: variance-related = 11, skewness-related = 57
    >>> mnc2mc(mnc)
    [2, 4]

    '''
    n = len(mnc)
    mean = mnc[0]
    mnc = [1] + list(mnc)       # add zero moment = 1
    mu = []                     # np.zeros(n+1)
    for n,m in enumerate(mnc):
        mu.append(0)
                                # [comb(n-1,k,exact=1) for k in range(n)]
        for k in range(n+1):
            mu[n] += (-1)**(n-k) * comb(n,k,exact=1) * mnc[k] * mean**(n-k)
    if wmean:
        mu[1] = mean
    return mu[1:]
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def cum2mc(kappa):
    '''
    convert non-central moments to cumulants recursive formula produces as many cumulants as moments

    This function takes a list of cumulants and converts them into 
    non-central moments using the recursive formula.

    Parameters
    ----------
    kappa : list of float
        A list of cumulants, where `kappa[0]` is the mean, and subsequent elements 
        represent higher-order cumulants.

    Returns
    -------
    list of float
        A list of non-central moments. The first element is the mean, followed by 
        higher-order non-central moments.

    Notes
    -----
    The recursive formula used for conversion is:

    .. math::
        m_n = \sum_{k=0}^{n-1} \binom{n-1}{k} \kappa_{n-k} m_k

    where:
        - \( m_n \) is the non-central moment of order \( n \),
        - \( \kappa_{n-k} \) is the cumulant of order \( n-k \),
        - \( \binom{n-1}{k} \) is the binomial coefficient.

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40  
    [Google Books Link](http://books.google.ca/books?id=gm7kwttyRT0C&pg=PA40&lpg=PA40&dq=convert+cumulants+to+moments&source=web&ots=qyIaY6oaWH&sig=cShTDWl-YrWAzV7NlcMTRQV6y0A&hl=en&sa=X&oi=book_result&resnum=1&ct=result)

    Examples
    --------
    >>> kappa = [3, 2, 4]  # Mean = 3, cumulants related to variance and skewness
    >>> cum2mc(kappa)
    [3, 2, 10]
    
    '''
    mc = [1,0.0] #_kappa[0]]  #insert 0-moment and mean
    kappa0 = kappa[0]
    kappa = [1] + list(kappa)
    for nn,m in enumerate(kappa[2:]):
        n = nn+2
        mc.append(0)
        for k in range(n-1):
            mc[n] += comb(n-1,k,exact=1) * kappa[n-k]*mc[k]

    mc[1] = kappa0 # insert mean as first moments by convention
    return mc[1:]
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mnc2cum(mnc):
    '''
    convert non-central moments to cumulants recursive formula produces as many cumulants as moments
    
    This function converts a list of non-central moments into cumulants
    using a well-established recursive formula.

    Parameters
    ----------
    mnc : list of float
        A list of non-central moments, where `mnc[0]` represents the mean,
        and subsequent elements are higher-order moments.

    Returns
    -------
    list of float
        A list of cumulants corresponding to the input non-central moments.

    Notes
    -----
    The conversion is based on the recursive formula:

    .. math::
        \kappa_n = m_n - \sum_{k=1}^{n-1} \binom{n-1}{k-1} \kappa_k m_{n-k}

    where:
        - \( \kappa_n \) is the cumulant of order \( n \),
        - \( m_n \) is the non-central moment of order \( n \),
        - \( \binom{n-1}{k-1} \) is the binomial coefficient.

    References
    ----------
    Wikipedia: [Cumulants and Moments](http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments)

    Examples
    --------
    >>> mnc = [3, 2, 10]  # Mean = 3, variance-related moment, skewness-related moment
    >>> mnc2cum(mnc)
    [3, 2, 4]

    '''
    mnc = [1] + list(mnc)
    kappa = [1]
    for nn,m in enumerate(mnc[1:]):
        n = nn+1
        kappa.append(m)
        for k in range(1,n):
            kappa[n] -= comb(n-1,k-1,exact=1) * kappa[k]*mnc[n-k]
    return kappa[1:]
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mc2cum(mc):
    '''
    Convert central moments to cumulants.

    This function converts a list of central moments to cumulants by first 
    converting them to non-central moments and then applying the recursive 
    cumulant conversion.

    Parameters
    ----------
    mc : list of float
        A list of central moments, where `mc[0]` represents the mean, 
        and subsequent elements are higher-order moments.

    Returns
    -------
    list of float
        A list of cumulants corresponding to the input central moments.

    Notes
    -----
    This function performs the transformation in two steps:
    1. Convert central moments to non-central moments using `mc2mnc`.
    2. Convert non-central moments to cumulants using `mnc2cum`.

    Examples
    --------
    >>> mc = [3, 2, 5]  # Mean = 3, second and third central moments
    >>> mc2cum(mc)
    [3, 2, 4]
    '''
    return mnc2cum(mc2mnc(mc))
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mvsk2mc(args):
    '''
    convert mean, variance, skew, kurtosis to central moments
    
    Convert mean, variance, skewness, and kurtosis to central moments.

    This function takes statistical measures—mean, variance, skewness, 
    and kurtosis—and converts them into the corresponding central moments.

    Parameters
    ----------
    args : tuple of float
        A tuple containing:
        - `mu` (float) : Mean of the distribution.
        - `sig2` (float) : Variance of the distribution.
        - `sk` (float) : Skewness (third standardized moment).
        - `kur` (float) : Kurtosis (excess kurtosis; does not include the baseline 3).

    Returns
    -------
    tuple of float
        A tuple containing the first four central moments:
        - First central moment (mean) `mu`
        - Second central moment (variance) `sig2`
        - Third central moment (skew * variance^(3/2))
        - Fourth central moment ((kurtosis + 3) * variance^2)

    Notes
    -----
    - The skewness is multiplied by `variance^(3/2)` to obtain the third central moment.
    - The fourth central moment is obtained by scaling kurtosis and adding the normal distribution constant (3).

    Examples
    --------
    >>> mvsk2mc((2.0, 4.0, 0.5, 1.2))
    (2.0, 4.0, 2.0, 16.0)

    '''
    mu,sig2,sk,kur = args
    cnt = [None]*4
    cnt[0] = mu
    cnt[1] = sig2
    cnt[2] = sk * sig2**1.5
    cnt[3] = (kur+3.0) * sig2**2.0
    return tuple(cnt)
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mvsk2mnc(args):
    '''
    convert mean, variance, skew, kurtosis to non-central moments
    
    This function computes non-central moments from statistical measures: 
    mean, variance, skewness, and kurtosis.

    Parameters
    ----------
    args : tuple of float
        A tuple containing:
        - `mc` (float) : Mean of the distribution.
        - `mc2` (float) : Variance of the distribution.
        - `skew` (float) : Skewness (third standardized moment).
        - `kurt` (float) : Kurtosis (excess kurtosis; does not include the baseline 3).

    Returns
    -------
    tuple of float
        A tuple containing the first four non-central moments:
        - First non-central moment (mean) `mnc`
        - Second non-central moment `mnc2`
        - Third non-central moment `mnc3`
        - Fourth non-central moment `mnc4`

    Notes
    -----
    - The second non-central moment is computed as `variance + mean^2`.
    - The third and fourth non-central moments are derived using standard formulas 
      involving central moments.

    Examples
    --------
    >>> mvsk2mnc((2.0, 4.0, 0.5, 1.2))
    (2.0, 8.0, 18.0, 86.0)
    '''
    mc, mc2, skew, kurt = args
    mnc = mc
    mnc2 = mc2 + mc*mc
    mc3  = skew*(mc2**1.5) # 3rd central moment
    mnc3 = mc3+3*mc*mc2+mc**3 # 3rd non-central moment
    mc4  = (kurt+3.0)*(mc2**2.0) # 4th central moment
    mnc4 = mc4+4*mc*mc3+6*mc*mc*mc2+mc**4
    return (mnc, mnc2, mnc3, mnc4)
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mc2mvsk(args):
    '''convert central moments to mean, variance, skew, kurtosis
    
    This function computes the standard statistical measures (mean, variance, 
    skewness, and kurtosis) from the central moments of a distribution.

    Parameters
    ----------
    args : tuple of float
        A tuple containing:
        - `mc` (float) : First central moment (mean).
        - `mc2` (float) : Second central moment (variance).
        - `mc3` (float) : Third central moment.
        - `mc4` (float) : Fourth central moment.

    Returns
    -------
    tuple of float
        A tuple containing:
        - Mean (`mc`)
        - Variance (`mc2`)
        - Skewness (`skew`)
        - Kurtosis (`kurt`, excess kurtosis where normal distribution has a value of 0)

    Notes
    -----
    - Skewness is computed as `mc3 / mc2^(3/2)`.
    - Kurtosis (excess kurtosis) is computed as `mc4 / mc2^2 - 3`, where the 
      normal distribution has an excess kurtosis of 0.

    Examples
    --------
    >>> mc2mvsk((2.0, 4.0, 3.0, 10.0))
    (2.0, 4.0, 0.375, -0.5)
    
    '''
    mc, mc2, mc3, mc4 = args
    skew = np.divide(mc3, mc2**1.5)
    kurt = np.divide(mc4, mc2**2.0) - 3.0
    return (mc, mc2, skew, kurt)
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def mnc2mvsk(args):
    '''
    convert central moments to mean, variance, skew, kurtosis
    
    This function first converts non-central moments to central moments and then 
    computes the standard statistical measures (mean, variance, skewness, and kurtosis) 
    from those central moments.

    Parameters
    ----------
    args : tuple of float
        A tuple containing:
        - `mnc` (float) : First non-central moment.
        - `mnc2` (float) : Second non-central moment.
        - `mnc3` (float) : Third non-central moment.
        - `mnc4` (float) : Fourth non-central moment.

    Returns
    -------
    tuple of float
        A tuple containing:
        - Mean (`mc`)
        - Variance (`mc2`)
        - Skewness (`skew`)
        - Kurtosis (`kurt`, excess kurtosis where normal distribution has a value of 0)

    Notes
    -----
    - Central moments are first computed from non-central moments using the recursive formulas:
      - Second central moment: `mc2 = mnc2 - mnc^2`
      - Third central moment: `mc3 = mnc3 - (3*mc*mc2 + mc^3)`
      - Fourth central moment: `mc4 = mnc4 - (4*mc*mc3 + 6*mc*mc*mc2 + mc^4)`
    - Then the skewness and kurtosis are computed from the central moments:
      - Skewness is `mc3 / mc2^(3/2)`
      - Kurtosis is `(mc4 / mc2^2) - 3` (excess kurtosis).

    Examples
    --------
    >>> mnc2mvsk((2.0, 4.0, 3.0, 10.0))
    (2.0, 4.0, 0.375, -0.5)

    '''
    #convert four non-central moments to central moments
    mnc, mnc2, mnc3, mnc4 = args
    mc = mnc
    mc2 = mnc2 - mnc*mnc
    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)
    return mc2mvsk((mc, mc2, mc3, mc4))
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
#def mnc2mc(args):
#    '''convert four non-central moments to central moments
#    '''
#    mnc, mnc2, mnc3, mnc4 = args
#    mc = mnc
#    mc2 = mnc2 - mnc*mnc
#    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
#    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)
#    return mc, mc2, mc
# TODO: no return, did it get lost in cut-paste?
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def cov2corr(cov, return_std=False):
    '''convert covariance matrix to correlation matrix
    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes
    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.
    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.'''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def corr2cov(corr, std):
    '''convert correlation matrix to covariance matrix given standard deviation
    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation
    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix
    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that multiplication is defined elementwise. np.ma.array are allowed, but not matrices.'''
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov
#############################################################################################################################################################################################################
#
#############################################################################################################################################################################################################
def se_cov(cov):
    '''get standard deviation from covariance matrix just a shorthand function np.sqrt(np.diag(cov))
    Parameters
    ----------
    cov : array_like, square
        covariance matrix
    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov'''
    return np.sqrt(np.diag(cov))
