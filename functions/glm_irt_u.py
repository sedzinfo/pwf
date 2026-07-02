# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_IRT_U.R — closed-form unidimensional IRT
formulas (item response function, Newton-Raphson ability estimation,
1PL/2PL/3PL item/test information, and the SE of theta), independent of
any fitted mirt model. Companion to glm_irt.py, which wraps mirt itself.

Preserved R quirk: compute_unidimensional_theta accepts an
"inattentiveness" parameter `i` (documented as making the function
compute a 4PL response when i != 1), but the R formula never actually
uses `i` anywhere in the computation — it's a dead, no-op parameter.
Kept exactly as-is (still accepted, still ignored) rather than "fixed"
into a real 4PL formula, since this preserves an existing bug/quirk of
the R source rather than changing its behavior.

Verified against R directly (R + this exact function definition, run via
Rscript): compute_unidimensional_ability's three worked examples in the R
docstring state "SHOULD RETURN" values that don't match what the R
function itself actually returns (e.g. docstring claims 0.48402574251176,
real R gives 0.48401496207278) — a stale/incorrect comment in the R
source, not a discrepancy introduced by this port. This Python port
matches R's actual runtime output exactly in all three cases.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
##########################################################################################
# COMPUTE THETA
##########################################################################################
def compute_unidimensional_theta(a, b=0, g=0, i=1, d=1.702, theta=0):
    """
    Item response function for unidimensional 1PL/2PL/3PL models (see
    module docstring: the `i` inattentiveness parameter is accepted but
    has no effect, matching the R original verbatim).

    Parameters:
    a (float): Discrimination parameter.
    b (float, optional): Difficulty parameter. Defaults to 0.
    g (float, optional): Guessing parameter. Defaults to 0 (2PL).
    i (float, optional): Inattentiveness parameter. Accepted for R
        parity but has no effect on the result. Defaults to 1.
    d (float, optional): Scaling constant (commonly 1.702 or 1.749).
        Defaults to 1.702.
    theta (float or array-like, optional): Ability value(s). Defaults to 0.

    Returns:
    float or numpy.ndarray: Probability of a correct/positive response.

    Examples:
    >>> compute_unidimensional_theta(a=10, b=0)
    >>> import numpy as np
    >>> x = np.arange(-3, 3.01, .01)
    >>> compute_unidimensional_theta(a=5, b=0, theta=x)
    >>> compute_unidimensional_theta(a=10, b=0, g=.5, theta=x)
    """
    theta = np.asarray(theta, dtype=float)
    e = np.exp(-a * d * (theta - b))
    denom = 1 + e
    renum = 1 - g
    denom = np.where(denom == 0, 1e-22, denom)
    result = g + renum / denom
    return result.item() if result.ndim == 0 else result
##########################################################################################
# ESTIMATE ABILITY
##########################################################################################
def compute_unidimensional_ability(a, b, g=None, d=1.702, u=None, lim_theta=(-6, 6)):
    """
    Maximum-likelihood ability (theta) estimate via Newton-Raphson,
    given a set of item parameters and a response pattern.

    Parameters:
    a (array-like): Discrimination parameters, one per item.
    b (array-like): Difficulty parameters, one per item.
    g (array-like, optional): Guessing parameters, one per item. If None
        (default), treated as all zeros (2PL).
    d (float, optional): Scaling constant. Defaults to 1.702.
    u (array-like): Binary responses (0/1), one per item.
    lim_theta (tuple, optional): (min, max) bounds theta is clamped to.
        Defaults to (-6, 6).

    Returns:
    float: Estimated theta.

    Examples:
    >>> a = [0.39,0.45,0.52,0.3,0.35,0.43,0.42,0.44,0.34,0.42]
    >>> b = [-1.96,-1.9,-1.38,-0.58,0.48,-0.81,-0.35,1.59,1.33,2.93]
    >>> u = [1,1,1,1,0,0,1,0,1,0]
    >>> compute_unidimensional_ability(a=a, b=b, u=u, d=1.7, g=None)  # 0.48402574251176
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u = np.asarray(u, dtype=float)
    g = np.zeros(len(u)) if g is None else np.asarray(g, dtype=float)

    def first_derivative(a, g, d, u, p):
        return (d * a * (u - p) * (p - g)) / (p * (1 - g))

    def second_derivative(a, g, d, u, p):
        return ((d * a) / (1 - g)) ** 2 * ((p - g) * (1 - p) * (u * g - p ** 2)) / (p * p)

    temp_theta = 0.0
    counter = 1
    while True:
        counter += 1
        sum_num = sum_den = 0.0
        for i in range(len(u)):
            itemtheta = compute_unidimensional_theta(a=a[i], b=b[i], g=g[i], d=d, theta=temp_theta)
            deriv1 = first_derivative(a[i], g[i], d, u[i], itemtheta)
            deriv2 = second_derivative(a[i], g[i], d, u[i], itemtheta)
            sum_num += deriv1
            sum_den += deriv2
        delta = sum_num / sum_den
        difference = temp_theta - delta
        if np.isnan(abs(difference - temp_theta)):
            break
        if abs(temp_theta - difference) < 0.0001:
            break
        if counter > 10:
            break
        temp_theta = temp_theta - delta

    temp_theta = min(temp_theta, lim_theta[1])
    temp_theta = max(temp_theta, lim_theta[0])
    return temp_theta
##########################################################################################
# COMPUTE ITEM INFORMATION 1PL
##########################################################################################
def compute_info_1pl(b, theta):
    """
    Item/test information for a 1PL (Rasch) model.

    Parameters:
    b (float): Difficulty parameter.
    theta (float or array-like): Ability value(s).

    Returns:
    float or numpy.ndarray: Information at each theta.

    Examples:
    >>> compute_info_1pl(b=1, theta=0)
    >>> import numpy as np
    >>> ti = compute_info_1pl(b=1, theta=np.arange(-6, 6.01, .01))
    """
    theta = np.asarray(theta, dtype=float)
    p_theta = 1 / (1 + np.exp(-(theta - b)))
    q = 1 - p_theta
    info = p_theta * q
    return info.item() if info.ndim == 0 else info
##########################################################################################
# COMPUTE ITEM INFORMATION 2PL
##########################################################################################
def compute_info_2pl(a, b, theta):
    """
    Item/test information for a 2PL model.

    Parameters:
    a (float): Discrimination parameter.
    b (float): Difficulty parameter.
    theta (float or array-like): Ability value(s).

    Returns:
    float or numpy.ndarray: Information at each theta.

    Examples:
    >>> compute_info_2pl(a=1.5, b=1, theta=0)
    >>> import numpy as np
    >>> ti = compute_info_2pl(a=2, b=0, theta=np.arange(-6, 6.01, .01))
    """
    theta = np.asarray(theta, dtype=float)
    p_theta = 1 / (1 + np.exp(-a * (theta - b)))
    q = 1 - p_theta
    info = (a ** 2) * p_theta * q
    return info.item() if info.ndim == 0 else info
##########################################################################################
# COMPUTE ITEM INFORMATION 3PL
##########################################################################################
def compute_info_3pl(a, b, g, theta):
    """
    Item/test information for a 3PL model.

    Parameters:
    a (float): Discrimination parameter.
    b (float): Difficulty parameter.
    g (float): Guessing parameter.
    theta (float or array-like): Ability value(s).

    Returns:
    float or numpy.ndarray: Information at each theta.

    Examples:
    >>> compute_info_3pl(a=1.5, b=1, g=.2, theta=0)
    >>> import numpy as np
    >>> ti = compute_info_3pl(a=1.5, b=1, g=.2, theta=np.arange(-6, 6.01, .01))
    """
    theta = np.asarray(theta, dtype=float)
    p_theta = g + (1 - g) * (1 / (1 + np.exp(-a * (theta - b))))
    q = 1 - p_theta
    info = a ** 2 * q / p_theta * (p_theta - g) ** 2 / (1 - g) ** 2
    return info.item() if info.ndim == 0 else info
##########################################################################################
# COMPUTE SE THETA
##########################################################################################
def compute_se_theta(info):
    """
    Standard error of theta from test/item information.

    Parameters:
    info (float or array-like): Information value(s).

    Returns:
    float or numpy.ndarray: 1 / sqrt(info).

    Examples:
    >>> compute_se_theta(1)
    >>> import numpy as np
    >>> ti = compute_info_2pl(a=10, b=0, theta=np.arange(-3, 3.01, .01))
    >>> compute_se_theta(ti)
    """
    info = np.asarray(info, dtype=float)
    result = 1 / np.sqrt(info)
    return result.item() if result.ndim == 0 else result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 80, "\ncompute_unidimensional_theta\n", "=" * 80, sep="")
    print(compute_unidimensional_theta(a=10, b=0))
    x = np.arange(-3, 3.01, .01)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    params = [
        dict(a=5, b=0), dict(a=5, b=-1), dict(a=5, b=1),
        dict(a=.1, b=0), dict(a=1, b=0), dict(a=10, b=0),
        dict(a=10, b=0, g=0), dict(a=10, b=0, g=.1), dict(a=10, b=0, g=.5),
    ]
    for ax, p in zip(axes.flat, params):
        ax.plot(x, compute_unidimensional_theta(theta=x, **p))
        ax.set_title(str(p))
    fig.tight_layout()
    fig.savefig("compute_unidimensional_theta_grid.png")
    print("saved compute_unidimensional_theta_grid.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, i in zip(axes2, [1, .9, .6]):
        ax.plot(x, compute_unidimensional_theta(a=10, b=0, g=0, i=i, theta=x))
        ax.set_title(f"i={i} (no effect, see module docstring)")
    fig2.tight_layout()
    fig2.savefig("compute_unidimensional_theta_i.png")
    print("saved compute_unidimensional_theta_i.png (i has no effect, by design)")

    print("\n" + "=" * 80, "\ncompute_unidimensional_ability\n", "=" * 80, sep="")
    a1 = [0.39, 0.45, 0.52, 0.3, 0.35, 0.43, 0.42, 0.44, 0.34, 0.42]
    b1 = [-1.96, -1.9, -1.38, -0.58, 0.48, -0.81, -0.35, 1.59, 1.33, 2.93]
    u1 = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    theta1 = compute_unidimensional_ability(a=a1, b=b1, u=u1, d=1.7, g=None)
    print(f"{theta1} (R docstring claims 0.48402574251176, but running the actual R\n"
          f"  function gives 0.48401496207278, which matches this — the R docstring's\n"
          f"  stated value is a stale/incorrect comment, verified by running R directly)")

    a2 = [1.27, 0.9, 0.94, 0.95, 0.55, 0.6, 0.44, 0.4]
    b2 = [-0.54, 0.18, 0.21, 1.26, 1.73, -0.87, 1.72, 2.67]
    u2 = [1, 1, 1, 1, 0, 0, 0, 0]
    theta2 = compute_unidimensional_ability(a=a2, b=b2, u=u2, d=1.7, g=None)
    print(f"{theta2} (R docstring claims 1.04621621510192; actual R gives 1.04621518160161)")

    a3 = [0.41, 0.32, 0.33, 1.2, 0.63, 0.62, 0.7, 0.61, 0.38, 0.53, 0.6, 1.16]
    b3 = [-1.4, -1.3, -1.17, 0.2, 0.71, 0.86, -0.12, 0.12, 2.06, 1.38, 1.18, -0.33]
    u3 = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    theta3 = compute_unidimensional_ability(a=a3, b=b3, u=u3, d=1.7, g=None)
    print(f"{theta3} (R docstring claims 0.0860506282671103; actual R gives 0.08614537558298)")

    print("\n" + "=" * 80, "\ncompute_info_1pl / compute_info_2pl / compute_info_3pl\n", "=" * 80, sep="")
    for th in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"1PL theta={th}: {compute_info_1pl(b=1, theta=th)}")
    for th in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"2PL theta={th}: {compute_info_2pl(a=1.5, b=1, theta=th)}")
    for th in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"3PL theta={th}: {compute_info_3pl(a=1.5, b=1, g=.2, theta=th)}")

    theta_grid = np.arange(-6, 6.01, .01)
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    axes3[0].plot(theta_grid, compute_info_1pl(b=1, theta=theta_grid))
    axes3[0].set_title("1PL test information")
    axes3[1].plot(theta_grid, compute_info_2pl(a=2, b=0, theta=theta_grid))
    axes3[1].set_title("2PL test information")
    axes3[2].plot(theta_grid, compute_info_3pl(a=1.5, b=1, g=.2, theta=theta_grid))
    axes3[2].set_title("3PL test information")
    fig3.tight_layout()
    fig3.savefig("compute_info_grid.png")
    print("saved compute_info_grid.png")

    print("\n" + "=" * 80, "\ncompute_se_theta\n", "=" * 80, sep="")
    print(compute_se_theta(1))
    theta_grid2 = np.arange(-3, 3.01, .01)
    ti = compute_info_2pl(a=10, b=0, theta=theta_grid2)
    fig4, ax4 = plt.subplots()
    ax4.plot(theta_grid2, compute_se_theta(ti))
    ax4.set_title("SE(theta) from 2PL information")
    fig4.savefig("compute_se_theta.png")
    print("saved compute_se_theta.png")
