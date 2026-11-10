import numpy as np
import pandas as pd

from conftest import assert_matches_reference
from pwf import glm_irt_u as irt


def test_response_and_information_functions_match_rwf(reference):
    expected = reference("irt_u_theta")
    theta = expected["theta"].to_numpy()
    result = pd.DataFrame({
        "theta": theta,
        "a5_b0": irt.compute_unidimensional_theta(a=5, b=0, theta=theta),
        "a1_bm1": irt.compute_unidimensional_theta(a=1, b=-1, theta=theta),
        "a10_g1": irt.compute_unidimensional_theta(a=10, b=0, g=.1, theta=theta),
        "a2_g2_i9": irt.compute_unidimensional_theta(a=2, b=.5, g=.2, i=.9, theta=theta),
        "a1_d1": irt.compute_unidimensional_theta(a=1, b=0, d=1, theta=theta),
        "info_1pl": irt.compute_info_1pl(b=1, theta=theta),
        "info_2pl": irt.compute_info_2pl(a=1.5, b=1, theta=theta),
        "info_3pl": irt.compute_info_3pl(a=1.5, b=1, g=.2, theta=theta),
        "se_2pl": irt.compute_se_theta(irt.compute_info_2pl(a=1.5, b=1, theta=theta)),
    })
    assert_matches_reference(result, expected, rtol=1e-12)


def test_response_function_is_vectorised_over_parameters(reference):
    result = irt.compute_unidimensional_theta(a=[1, 2, 0.5], b=[0, 1, -1], g=[0, .1, .2], i=[1, .9, .8], theta=0.5)
    assert_matches_reference(pd.DataFrame({"value": result}), reference("irt_u_theta_vectorised"), rtol=1e-12)


def test_ability_estimates_match_rwf(reference):
    result = [
        irt.compute_unidimensional_ability(a=[0.39, 0.45, 0.52, 0.3, 0.35, 0.43, 0.42, 0.44, 0.34, 0.42],
                                           b=[-1.96, -1.9, -1.38, -0.58, 0.48, -0.81, -0.35, 1.59, 1.33, 2.93],
                                           u=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0], d=1.7, g=None),
        irt.compute_unidimensional_ability(a=[1.27, 0.9, 0.94, 0.95, 0.55, 0.6, 0.44, 0.4],
                                           b=[-0.54, 0.18, 0.21, 1.26, 1.73, -0.87, 1.72, 2.67],
                                           u=[1, 1, 1, 1, 0, 0, 0, 0], d=1.7, g=None),
        irt.compute_unidimensional_ability(a=[0.41, 0.32, 0.33, 1.2, 0.63, 0.62, 0.7, 0.61, 0.38, 0.53, 0.6, 1.16],
                                           b=[-1.4, -1.3, -1.17, 0.2, 0.71, 0.86, -0.12, 0.12, 2.06, 1.38, 1.18, -0.33],
                                           u=[1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0], d=1.7, g=None),
        irt.compute_unidimensional_ability(a=[1, 1.5, 2, 0.8], b=[-1, 0, 1, 2], g=[.2, .1, .25, .15], u=[1, 1, 0, 1]),
    ]
    assert_matches_reference(pd.DataFrame({"example": [1, 2, 3, 4], "theta": result}),
                             reference("irt_u_ability"), rtol=1e-10)


def test_ability_limits_match_rwf(reference):
    result = pd.DataFrame({"case": ["all_correct", "all_wrong"], "theta": [
        irt.compute_unidimensional_ability(a=[1, 1, 1], b=[0, 0, 0], u=[1, 1, 1]),
        irt.compute_unidimensional_ability(a=[1, 1, 1], b=[0, 0, 0], u=[0, 0, 0], lim_theta=(-4, 4)),
    ]})
    assert_matches_reference(result, reference("irt_u_ability_limits"), rtol=1e-10)


def test_scalars_give_floats():
    assert isinstance(irt.compute_unidimensional_theta(a=10, b=0), float)
    assert isinstance(irt.compute_info_3pl(a=1.5, b=1, g=.2, theta=0), float)
    assert np.isclose(irt.compute_se_theta(4), 0.5)
