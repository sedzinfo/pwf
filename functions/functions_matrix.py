# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_MATRIX.R.

Deviations from the R originals, by design:
  - symmetric_matrix drops R's `rownames(matrix)<-colnames(matrix)` step,
    since a plain numpy array (unlike an R matrix with dimnames, or a
    labeled data frame) has no row/column names to sync.
  - display_upper_lower_triangle's `diagonal` argument mirrors R's
    dynamic typing (sentinel strings "upper"/"lower", NaN, a scalar, or
    a list of values/labels) by promoting the result to an object-dtype
    array when the diagonal replacement isn't numeric. R itself goes
    further and silently coerces the *entire* matrix to character mode
    in that case; this only promotes dtype, it doesn't stringify the
    other (numeric) cells — a reasonable approximation, not an exact
    match, for what's a fairly unusual display-formatting edge case.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
##########################################################################################
# MATRIX DISPLAY DIAGONAL
##########################################################################################
def matrix_triangle(m, off_diagonal=np.nan, diagonal=None, type="lower"):
    """
    Extract the upper or lower triangle of a matrix, replacing the
    off-triangle values with a fill value and optionally overriding the
    diagonal. Useful for displaying correlation/covariance matrices
    without redundant values.

    Parameters:
    m (array-like): Numeric matrix.
    off_diagonal (optional): Fill value for the suppressed triangle.
        Defaults to NaN.
    diagonal (optional): Value(s) for the diagonal. If None (default),
        the original diagonal of m is preserved.
    type (str, optional): "lower" or "upper" — which triangle to retain.
        Defaults to "lower".

    Returns:
    numpy.ndarray: Same shape as m, off-triangle filled with
    off_diagonal, diagonal set per `diagonal`.

    Examples:
    >>> m = np.arange(1, 10).reshape(3, 3, order='F')
    >>> matrix_triangle(m)
    >>> matrix_triangle(m, diagonal=np.nan, type="lower")
    >>> matrix_triangle(m, diagonal=np.nan, type="upper")
    """
    m = np.array(m, dtype=float, copy=True)
    if type == "lower":
        mask = np.triu(np.ones(m.shape, dtype=bool), k=1)
    else:
        mask = np.tril(np.ones(m.shape, dtype=bool), k=-1)
    m[mask] = off_diagonal
    if diagonal is not None:
        np.fill_diagonal(m, diagonal)
    return m
##########################################################################################
# MATRIX DISPLAY UPPER LOWER TRIANGLE
##########################################################################################
def display_upper_lower_triangle(m_upper, m_lower, diagonal=np.nan):
    """
    Combine the upper triangle of one matrix with the lower triangle of
    another.

    Parameters:
    m_upper (array-like): Matrix supplying the upper-triangle values.
    m_lower (array-like): Matrix supplying the lower-triangle values.
    diagonal (optional): One of:
        - "upper": use m_upper's own diagonal.
        - "lower": use m_lower's own diagonal.
        - NaN/None (default): diagonal set to NaN.
        - any other scalar or list: used directly as the diagonal
          (e.g. a list of labels — the result is promoted to an
          object-dtype array in that case).

    Returns:
    numpy.ndarray: Same shape as the inputs, upper triangle from
    m_upper, lower triangle from m_lower, diagonal per `diagonal`.

    Examples:
    >>> m1 = np.arange(1, 10).reshape(3, 3, order='F')
    >>> m2 = np.arange(11, 20).reshape(3, 3, order='F')
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="upper")
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="lower")
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=[1, 2, 3])
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=["X1", "X2", "X3"])
    """
    upper = matrix_triangle(m_upper, diagonal=None, type="upper")
    lower = matrix_triangle(m_lower, diagonal=None, type="lower")
    upper_mask = np.triu(np.ones(lower.shape, dtype=bool), k=1)
    combined = lower.copy()
    combined[upper_mask] = upper[upper_mask]

    if isinstance(diagonal, str) and diagonal == "upper":
        diag_values = np.diag(np.array(m_upper, dtype=float))
    elif isinstance(diagonal, str) and diagonal == "lower":
        diag_values = np.diag(np.array(m_lower, dtype=float))
    elif diagonal is None or (np.isscalar(diagonal) and pd.isna(diagonal)):
        diag_values = np.nan
    else:
        diag_values = diagonal

    values = np.atleast_1d(diag_values)
    if values.dtype.kind in ('U', 'S', 'O') and any(isinstance(v, str) for v in values):
        combined = combined.astype(object)

    np.fill_diagonal(combined, diag_values)
    return combined
##########################################################################################
# MAKE SYMMETRIC MATRIX
##########################################################################################
_UNSET = object()


def symmetric_matrix(matrix, duplicate="lower", diagonal=_UNSET):
    """
    Make a matrix symmetric by mirroring one triangle onto the other.

    Parameters:
    matrix (array-like): Square numeric matrix.
    duplicate (str, optional): "lower" mirrors the lower triangle onto
        the upper; "upper" mirrors the upper triangle onto the lower.
        Defaults to "lower".
    diagonal (optional): Value(s) for the diagonal. If not passed at all
        (the default), the original diagonal of `matrix` is preserved.
        Pass NaN explicitly to fill the diagonal with NaN.

    Returns:
    numpy.ndarray: Symmetric matrix, same shape as `matrix`.

    Examples:
    >>> m_lower = matrix_triangle(np.arange(1, 10).reshape(3, 3, order='F'), type="lower", diagonal=np.nan)
    >>> symmetric_matrix(matrix=m_lower, duplicate="lower", diagonal=np.nan)
    """
    matrix = np.array(matrix, dtype=float, copy=True)
    if diagonal is _UNSET:
        diagonal = np.diag(matrix).copy()
    if duplicate == "lower":
        iu = np.triu_indices_from(matrix, k=1)
        matrix[iu] = matrix.T[iu]
    elif duplicate == "upper":
        il = np.tril_indices_from(matrix, k=-1)
        matrix[il] = matrix.T[il]
    np.fill_diagonal(matrix, diagonal)
    return matrix
##########################################################################################
# INDEX OFF DIAGONAL
##########################################################################################
def off_diagonal_index(length):
    """
    Get off-diagonal neighbour indices for a square matrix of a given size.

    Parameters:
    length (int): Size of the diagonal (number of rows/columns).

    Returns:
    pandas.DataFrame: `length` rows, columns x1 (row index), x2 (column
    index, same as x1), x3 (index just above, i+1), x4 (index just
    below, i-1). Indices are 1-based, matching R.

    Examples:
    >>> off_diagonal_index(length=6)
    """
    i = np.arange(1, length + 1)
    return pd.DataFrame({'x1': i, 'x2': i, 'x3': i + 1, 'x4': i - 1})
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    m = np.arange(1, 10).reshape(3, 3, order='F')
    print("m:\n", m)

    print("\n" + "=" * 80, "\nmatrix_triangle\n", "=" * 80, sep="")
    print(matrix_triangle(m=m))
    print()
    print(matrix_triangle(m=m, diagonal=np.nan, type="lower"))
    print()
    print(matrix_triangle(m=m, diagonal=None, type="lower"))
    print()
    print(matrix_triangle(m=m, diagonal=np.nan, type="upper"))

    print("\n" + "=" * 80, "\ndisplay_upper_lower_triangle\n", "=" * 80, sep="")
    m1 = np.arange(1, 10).reshape(3, 3, order='F')
    m2 = np.arange(11, 20).reshape(3, 3, order='F')
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="upper"))
    print()
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="lower"))
    print()
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=np.nan))
    print()
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=1))
    print()
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=["X1", "X2", "X3"]))
    print()
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=[1, 2, 3]))

    print("\n" + "=" * 80, "\nsymmetric_matrix\n", "=" * 80, sep="")
    m_lower = matrix_triangle(np.arange(1, 10).reshape(3, 3, order='F'), type="lower", diagonal=np.nan)
    m_upper = matrix_triangle(np.arange(11, 20).reshape(3, 3, order='F'), type="upper", diagonal=np.nan)
    print(symmetric_matrix(matrix=m_lower, duplicate="lower", diagonal=np.nan))
    print()
    print(symmetric_matrix(matrix=m_upper, duplicate="upper", diagonal=np.nan))

    print("\n" + "=" * 80, "\noff_diagonal_index\n", "=" * 80, sep="")
    print(off_diagonal_index(length=6))
