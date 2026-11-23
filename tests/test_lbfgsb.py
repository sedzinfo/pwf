"""
pwf._lbfgsb.r_optim_lbfgsb against R's optim(method = "L-BFGS-B") itself (make_reference.R, "R'S OPTIM
L-BFGS-B"): the end point, value, counts, convergence code and message of the same problems. The functions
below compute what the R functions compute; R's sum() adds in long double, math.fsum is the closest Python
sum, so par and value are compared to 1e-10 and the rest exactly.
"""
import math

import numpy as np
import pytest

from pwf._lbfgsb import r_optim_lbfgsb


def rosen(x):
    return math.fsum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosen_gr(x):
    g = np.zeros(len(x))
    g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:] = g[1:] + 200 * (x[1:] - x[:-1] ** 2)
    return g


def quad(x):
    index = np.arange(1, len(x) + 1)
    return math.fsum(index * (x - np.cos(index)) ** 2)


def quad_gr(x):
    index = np.arange(1, len(x) + 1)
    return 2 * index * (x - np.cos(index))


CASES = {
    "rosen2": dict(par=[-1.2, 1], fn=rosen, gr=rosen_gr),
    "rosen5_bounds": dict(par=[-1] * 5, fn=rosen, gr=rosen_gr, lower=-0.5, upper=0.8),
    "rosen4_numerical": dict(par=[0.5] * 4, fn=rosen, lower=0),
    "rosen6_maxit": dict(par=[-1] * 6, fn=rosen, gr=rosen_gr, maxit=5),
    "quad8_scaled": dict(par=[0] * 8, fn=quad, gr=quad_gr, lower=[-np.inf, 0, 0.5, -np.inf, -1, 0, -np.inf, 0.2],
                         upper=[np.inf, np.inf, 1, 0, 1, 0.1, np.inf, 0.3], parscale=[1, 2, 0.5, 1, 1, 10, 1, 0.01],
                         fnscale=2),
    "quad3_maximize": dict(par=[1, 1, 1], fn=lambda x: -quad(x), gr=lambda x: -quad_gr(x), fnscale=-1, upper=0.4),
    "quad4_numerical_on_bound": dict(par=[0, 1, 0, 0], fn=quad, lower=0, upper=1, ndeps=[1e-4] * 4),
    "rosen3_lmm_factr": dict(par=[-1, 2, -1], fn=rosen, gr=rosen_gr, lmm=3, factr=1e3, pgtol=1e-8),
}


@pytest.mark.parametrize("case", CASES)
def test_r_optim_lbfgsb_matches_optim(reference, case):
    arguments = dict(CASES[case])
    result = r_optim_lbfgsb(arguments.pop("fn"), arguments.pop("par"), **arguments)
    expected = reference("optim_lbfgsb_results").set_index("case").loc[case]
    par = reference("optim_lbfgsb_par")
    np.testing.assert_allclose(result["par"], par[par["case"] == case]["par"].to_numpy(), rtol=1e-10, atol=1e-12)
    assert result["value"] == pytest.approx(expected["value"], rel=1e-10, abs=1e-15)
    assert list(result["counts"]) == [expected["fn_count"], expected["gr_count"]]  # (function, gradient)
    assert result["convergence"] == expected["convergence"]
    assert result["message"] == expected["message"]


def test_scalar_ndeps_and_parscale_are_recycled():
    # R's optim wants one ndeps per parameter; pwf also takes one number for all
    one = r_optim_lbfgsb(quad, [0, 1, 0, 0], lower=0, upper=1, ndeps=1e-4)
    each = r_optim_lbfgsb(quad, [0, 1, 0, 0], lower=0, upper=1, ndeps=[1e-4] * 4)
    np.testing.assert_array_equal(one["par"], each["par"])
    with pytest.raises(ValueError, match="parscale"):
        r_optim_lbfgsb(quad, [0, 1, 0, 0], parscale=[1, 2])
