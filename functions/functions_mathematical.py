# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_MATHEMATICAL.R.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import math
##########################################################################################
# ANGLE RADIANS TO DEGREES
##########################################################################################
def rad2deg(radians):
    """
    Convert radians to degrees.

    Parameters:
    radians (float): Angle in radians.

    Returns:
    float: Angle in degrees.

    Examples:
    >>> rad2deg(math.pi)
    """
    return (radians * 180) / math.pi
##########################################################################################
# ANGLE DEGREES TO RADIANS
##########################################################################################
def deg2rad(degrees):
    """
    Convert degrees to radians.

    Parameters:
    degrees (float): Angle in degrees.

    Returns:
    float: Angle in radians.

    Examples:
    >>> deg2rad(180)
    """
    return (degrees * math.pi) / 180
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nrad2deg\n", "=" * 80, sep="")
    print(rad2deg(math.pi))

    print("\n" + "=" * 80, "\ndeg2rad\n", "=" * 80, sep="")
    print(deg2rad(180))
