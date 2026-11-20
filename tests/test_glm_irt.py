import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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


def test_report_irt_matches_rwf(reference):
    """Every table of rwf's report on the same mirt model: names, row names and values (mirt itself runs in R,
    so they agree to the 15 digits write.csv keeps), and rwf's model_call, the text "NULL"."""
    model = glm_irt.fit_irt(_load_reference("plot_irt_onefactor_items"), n_factors=1, verbose=False)
    result = glm_irt.report_irt(model)
    for part in ["model_coefficients", "model_coefficients_oblimin", "model_options", "q3_matrix", "exp_residuals",
                 "item_fit", "g2_fit", "m2_fit"]:
        expected = reference(f"report_irt_{part}", keep_default_na=False, na_values=["NA"], dtype={"row": str})
        table = result[part]
        assert [str(label) for label in table.index] == expected["row"].tolist(), part
        assert list(table.columns) == list(expected.columns[1:]), part
        for column in table.columns:
            want = expected[column]
            if pd.api.types.is_numeric_dtype(want):
                np.testing.assert_allclose(table[column].to_numpy(float), want.to_numpy(float), rtol=1e-12,
                                           atol=1e-14, err_msg=f"{part} {column}")
            else:
                assert table[column].astype(str).tolist() == want.astype(str).tolist(), f"{part} {column}"
    assert result["model_call"] == reference("report_irt_model_call", keep_default_na=False)["call"].iloc[0]


def test_report_irt_writes_rwfs_sheets(tmp_path):
    model = glm_irt.fit_irt(_load_reference("plot_irt_onefactor_items"), n_factors=1, verbose=False)
    glm_irt.report_irt(model, file=str(tmp_path / "irt"))
    assert list(pd.read_excel(tmp_path / "irt.xlsx", sheet_name=None)) == [
        "Coefficients", "Coefficients Oblimin", "item fit", "G2", "M2", "Q3", "Residuals", "Model Options"]
