# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_MATHEMATICAL.R. Like R, both functions are
vectorised: a number gives a float, a list or array gives a numpy array.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import math

import numpy as np


def _as_numeric(x):
    """A number stays a number; a list/array/Series becomes a float numpy array (R vector)."""
    if np.ndim(x) == 0:
        return float(x)
    return np.asarray(x, dtype=float)
##########################################################################################
# ANGLE RADIANS TO DEGREES
##########################################################################################
def rad2deg(radians):
    """
    Convert radians to degrees: (radians * 180) / pi.

    Parameters:
    radians (float or array-like): Angle(s) in radians.

    Returns:
    float or numpy.ndarray: Angle(s) in degrees (an array for array input).

    Examples:
    >>> rad2deg(math.pi)
    >>> rad2deg([0, math.pi / 2, math.pi])
    """
    return (_as_numeric(radians) * 180) / math.pi
##########################################################################################
# ANGLE DEGREES TO RADIANS
##########################################################################################
def deg2rad(degrees):
    """
    Convert degrees to radians: (degrees * pi) / 180.

    Parameters:
    degrees (float or array-like): Angle(s) in degrees.

    Returns:
    float or numpy.ndarray: Angle(s) in radians (an array for array input).

    Examples:
    >>> deg2rad(180)
    >>> deg2rad([0, 90, 180])
    """
    return (_as_numeric(degrees) * math.pi) / 180
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nrad2deg\n", "=" * 80, sep="")
    print(rad2deg(math.pi))

    print("\n" + "=" * 80, "\ndeg2rad\n", "=" * 80, sep="")
    print(deg2rad(180))
