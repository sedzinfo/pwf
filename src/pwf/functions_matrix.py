# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_MATRIX.R (matrix_triangle,
display_upper_lower_triangle, symmetric_matrix, off_diagonal_index).
Outputs match rwf, checked against R reference output in
tests/test_functions_matrix.py.

R matrices carry row/column names; numpy arrays do not. So: a numpy array
or list in gives a numpy array out, and a pandas DataFrame in gives a
DataFrame out with the labels R would keep (R's dimnames).

Differences from R that cannot be avoided:
  - display_upper_lower_triangle(diagonal="upper"/"lower") also accepts
    DataFrames; R's diag() fails on a data frame there.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd

try:
    from ._rcompat import r_number
except ImportError:
    from _rcompat import r_number


def _as_matrix(m):
    """R's as.matrix(): a float 2-D array (a vector becomes one column) and its labels, if any."""
    labels = (m.index, m.columns) if isinstance(m, pd.DataFrame) else None
    values = np.array(m, dtype=float, copy=True)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values, labels


def _with_labels(values, labels):
    if labels is None:
        return values
    return pd.DataFrame(values, index=labels[0], columns=labels[1])
##########################################################################################
# MATRIX DISPLAY DIAGONAL
##########################################################################################
def matrix_triangle(m, off_diagonal=np.nan, diagonal=None, type="lower"):
    """
    Keep the lower or upper triangle of a matrix, including the diagonal,
    and fill the other triangle with `off_diagonal`; optionally replace the
    diagonal. Useful for displaying correlation or covariance matrices
    without repeated values.

    Parameters:
    m (array-like or pandas.DataFrame): Numeric matrix (need not be square).
        A vector is treated as a one-column matrix, as R's as.matrix() does.
    off_diagonal (optional): Value used to fill the dropped triangle.
        Defaults to NaN (R's NA).
    diagonal (optional): Value(s) for the diagonal: a single value or one
        per diagonal element. If None (default), the original diagonal is
        kept.
    type (str, optional): "lower" (default) or "upper": the triangle to keep.

    Returns:
    numpy.ndarray or pandas.DataFrame: Same shape as m (a DataFrame with
    m's labels when m is a DataFrame).

    Examples:
    >>> m = np.arange(1, 10).reshape(3, 3, order='F')
    >>> matrix_triangle(m=m)
    >>> matrix_triangle(m=m, diagonal=np.nan, type="lower")
    >>> matrix_triangle(m=m, diagonal=None, type="lower")
    >>> matrix_triangle(m=m, diagonal=np.nan, type="upper")
    >>> matrix_triangle(m=m, diagonal=None, type="upper")
    """
    if type not in ("lower", "upper"):
        raise ValueError('type must be "lower" or "upper"')
    values, labels = _as_matrix(m)
    ones = np.ones(values.shape, dtype=bool)
    mask = np.triu(ones, k=1) if type == "lower" else np.tril(ones, k=-1)
    values[mask] = off_diagonal
    if diagonal is not None:
        np.fill_diagonal(values, diagonal)
    return _with_labels(values, labels)
##########################################################################################
# MATRIX DISPLAY UPPER LOWER TRIANGLE
##########################################################################################
def _is_text(x):
    return isinstance(x, str) or (np.ndim(x) > 0 and any(isinstance(v, str) for v in np.ravel(np.asarray(x, dtype=object))))


def display_upper_lower_triangle(m_upper, m_lower, diagonal=np.nan):
    """
    Combine the upper triangle of one matrix with the lower triangle of
    another, e.g. to show two correlation matrices, or correlations and p
    values, in a single table.

    Parameters:
    m_upper (array-like or pandas.DataFrame): Square numeric matrix
        supplying the upper triangle.
    m_lower (array-like or pandas.DataFrame): Square numeric matrix of the
        same size supplying the lower triangle.
    diagonal (optional): What to place on the diagonal: NaN/None (default)
        for missing, "upper" for the diagonal of m_upper, "lower" for the
        diagonal of m_lower, or any other value(s), a single value or one
        per diagonal element.

    Returns:
    numpy.ndarray or pandas.DataFrame: The combined matrix, with the labels
    of m_lower when it is a DataFrame. It is numeric unless `diagonal`
    contains text: then, as in R, the whole matrix becomes text (numbers
    written as R's as.character() writes them, missing cells stay missing).

    Examples:
    >>> m1 = np.arange(1, 10).reshape(3, 3, order='F')
    >>> m2 = np.arange(11, 20).reshape(3, 3, order='F')
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="upper")
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="lower")
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=np.nan)
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=1)
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=["X1", "X2", "X3"])
    >>> display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=[1, 2, 3])
    """
    upper, _ = _as_matrix(matrix_triangle(m_upper, diagonal=None, type="upper"))
    combined, labels = _as_matrix(m_lower)
    combined = _as_matrix(matrix_triangle(combined, diagonal=None, type="lower"))[0]
    upper_mask = np.triu(np.ones(combined.shape, dtype=bool), k=1)
    combined[upper_mask] = upper[upper_mask]

    if isinstance(diagonal, str) and diagonal == "upper":
        diag_values = np.diag(_as_matrix(m_upper)[0])
    elif isinstance(diagonal, str) and diagonal == "lower":
        diag_values = np.diag(_as_matrix(m_lower)[0])
    elif diagonal is None or (np.ndim(diagonal) == 0 and pd.isna(diagonal)):
        diag_values = np.nan
    else:
        diag_values = diagonal

    if _is_text(diag_values):
        text = np.empty(combined.shape, dtype=object)
        for idx, v in np.ndenumerate(combined):
            text[idx] = r_number(v)
        n = min(text.shape)
        text[np.arange(n), np.arange(n)] = np.resize(np.asarray(diag_values, dtype=object), n)
        combined = text
    else:
        np.fill_diagonal(combined, diag_values)
    return _with_labels(combined, labels)
##########################################################################################
# MAKE SYMMETRIC MATRIX
##########################################################################################
_UNSET = object()


def symmetric_matrix(matrix, duplicate="lower", diagonal=_UNSET):
    """
    Make a square matrix symmetric by copying one triangle onto the other;
    optionally replace the diagonal. As in rwf, the row names are set to the
    column names, so both sides carry the same labels.

    Parameters:
    matrix (array-like or pandas.DataFrame): Square numeric matrix.
    duplicate (str, optional): "lower" (default) copies the lower triangle
        into the upper, "upper" copies the upper triangle into the lower.
    diagonal (optional): Value(s) for the diagonal: a single value (e.g.
        NaN) or one per diagonal element. Leave it out to keep the original
        diagonal.

    Returns:
    numpy.ndarray or pandas.DataFrame: The symmetric matrix; for a DataFrame
    its index is set to its columns (R: rownames(matrix) <- colnames(matrix)).

    Examples:
    >>> m_lower = matrix_triangle(np.arange(1, 10).reshape(3, 3, order='F'), type="lower", diagonal=np.nan)
    >>> m_upper = matrix_triangle(np.arange(11, 20).reshape(3, 3, order='F'), type="upper", diagonal=np.nan)
    >>> symmetric_matrix(matrix=m_lower, duplicate="lower", diagonal=np.nan)
    >>> symmetric_matrix(matrix=m_upper, duplicate="upper", diagonal=np.nan)
    """
    values, labels = _as_matrix(matrix)
    if diagonal is _UNSET:
        diagonal = np.diag(values).copy()
    if duplicate == "lower":
        iu = np.triu_indices_from(values, k=1)
        values[iu] = values.T[iu]
    if duplicate == "upper":
        il = np.tril_indices_from(values, k=-1)
        values[il] = values.T[il]
    np.fill_diagonal(values, diagonal)
    if labels is not None:
        labels = (labels[1], labels[1])
    return _with_labels(values, labels)
##########################################################################################
# INDEX OFF DIAGONAL
##########################################################################################
def off_diagonal_index(length):
    """
    Row and column indices around the diagonal of a square matrix: for each
    diagonal position i, the diagonal cell and its two neighbours in the
    same row (column i + 1 to the right, column i - 1 to the left).

    Parameters:
    length (int): Number of rows and columns of the square matrix (at
        least 1).

    Returns:
    pandas.DataFrame: `length` rows and numeric columns x1 (row index i),
    x2 (column of the diagonal cell, same as x1), x3 (i + 1; length + 1 in
    the last row, outside the matrix) and x4 (i - 1; 0 in the first row).
    Indices are 1-based, as in R.

    Examples:
    >>> off_diagonal_index(length=6)
    """
    i = np.arange(1, length + 1, dtype=float)
    return pd.DataFrame({'x1': i, 'x2': i, 'x3': i + 1, 'x4': i - 1})
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    m = np.arange(1, 10).reshape(3, 3, order='F')
    print("=" * 80, "\nmatrix_triangle\n", "=" * 80, sep="")
    print(matrix_triangle(m=m))
    print(matrix_triangle(m=m, diagonal=np.nan, type="upper"))

    print("\n" + "=" * 80, "\ndisplay_upper_lower_triangle\n", "=" * 80, sep="")
    m1 = np.arange(1, 10).reshape(3, 3, order='F')
    m2 = np.arange(11, 20).reshape(3, 3, order='F')
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal="upper"))
    print(display_upper_lower_triangle(m_upper=m1, m_lower=m2, diagonal=["X1", "X2", "X3"]))

    print("\n" + "=" * 80, "\nsymmetric_matrix\n", "=" * 80, sep="")
    m_lower = matrix_triangle(m1, type="lower", diagonal=np.nan)
    print(symmetric_matrix(matrix=m_lower, duplicate="lower", diagonal=np.nan))

    print("\n" + "=" * 80, "\noff_diagonal_index\n", "=" * 80, sep="")
    print(off_diagonal_index(length=6))
