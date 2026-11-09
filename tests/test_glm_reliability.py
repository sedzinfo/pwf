import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import assert_matches_reference  # noqa: E402
from pwf import glm_reliability  # noqa: E402

FORMULA = "response ~ r(time) * r(person) + r(item) * r(person)"


@pytest.fixture(scope="module")
def model():
    from conftest import _load_reference
    return glm_reliability.random_anova(FORMULA, _load_reference("vc_design"))


def test_random_anova_matches_mixlm(model, reference):
    expected = reference("extract_components_anova").set_index("term")
    assert list(model.anova.index) == list(expected.index)
    np.testing.assert_allclose(model.anova.to_numpy(), expected.to_numpy(), rtol=1e-6)


def test_extract_components_matches_rwf(model, reference):
    result = glm_reliability.extract_components(model, title="components")
    assert_matches_reference(result["components"], reference("extract_components"))
    assert result["plot"].labels.title == "components"


def test_extract_components_rejects_other_models():
    with pytest.raises(TypeError):
        glm_reliability.extract_components(object())
