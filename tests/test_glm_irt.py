import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import _load_reference  # noqa: E402
from pwf import glm_irt  # noqa: E402

# glm_irt runs R's mirt through rpy2; without R, mirt and rpy2 (as on CI) these tests are skipped
pytestmark = pytest.mark.skipif(glm_irt._R_ERROR is not None, reason="needs R with mirt and the rpy2 package")


def test_plot_irt_onefactor_matches_rwf(reference):
    """rwf's series levels (information, expected_score, then the items, which set the colours), grep()'s item
    selection (V1 also takes V10 and V11) and the curves on R's seq(-6, 6, .1)."""
    model = glm_irt.fit_irt(_load_reference("plot_irt_onefactor_items"), n_factors=1, verbose=False)
    p = glm_irt.plot_irt_onefactor(model, title="T")
    levels = reference("plot_irt_onefactor_levels")
    assert list(p.data["variable"].cat.categories) == levels["level"].tolist()
    assert p.labels.get("title", None) == levels["title"].iloc[0]
    expected = reference("plot_irt_onefactor_data")
    assert p.data["type"].tolist() == expected["type"].tolist()
    assert p.data["variable"].astype(str).tolist() == expected["variable"].tolist()
    np.testing.assert_allclose(p.data["theta"], expected["theta"], rtol=0, atol=1e-14)  # write.csv keeps 15 digits
    np.testing.assert_allclose(p.data["value"], expected["value"], rtol=1e-10, atol=1e-12)
