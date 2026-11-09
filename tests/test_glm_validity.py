import numpy as np
import pandas as pd
import pytest

from conftest import _load_reference
from pwf import glm_validity

# Same dimensions as efa_dimensions in tests/reference/make_reference.R
ITEMS = [f"E{i}" for i in range(1, 11)] + [f"N{i}" for i in range(1, 11)]
REVERSE = ["E2", "E4", "E6", "E8", "E10", "N2", "N4"]
TEXT_COLUMNS = ["dimension", "variable", "index", "index_cfa"]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    dimensions = pd.DataFrame({"dimension": [i[0] for i in ITEMS], "subquestion_text": ITEMS,
                               "key": [1 if i in REVERSE else -1 for i in ITEMS]})
    file = tmp_path_factory.mktemp("efa_cfa") / "efa_cfa.xlsx"
    out = glm_validity.compute_efa_cfa(dimensions, _load_reference("efa_cfa_data"), filename=str(file))
    out["file"] = file
    return out


def test_efa_cfa_models(result, reference):
    expected = reference("compute_efa_cfa_models")
    got = result["efa_cfa_models"]
    assert list(got.columns) == list(expected.columns)
    for column in expected.columns:
        if column in TEXT_COLUMNS:
            assert got[column].astype(str).tolist() == expected[column].astype(str).tolist(), column
        else:
            # EFA loadings are rounded to 2 decimals as in rwf; CFA values agree to optimizer tolerance
            np.testing.assert_allclose(got[column].to_numpy(float), expected[column].to_numpy(float), atol=2e-4,
                                       err_msg=column)


def test_cfa_scores_and_coefficients(result, reference):
    np.testing.assert_allclose(result["cfa_scores"].to_numpy(), reference("compute_efa_cfa_scores").to_numpy(),
                               atol=5e-5)
    expected = reference("compute_efa_cfa_coefficients")
    got = result["cfa_coefficients"]
    assert got[["dimension", "q"]].values.tolist() == expected[["dimension", "q"]].values.tolist()
    np.testing.assert_allclose(got.iloc[:, 2:].to_numpy(float), expected.iloc[:, 2:].to_numpy(float), atol=5e-5)


def test_workbook(result):
    sheets = list(pd.read_excel(result["file"], sheet_name=None))
    assert sheets == ["critical", "CFA_Modification_Indexes", "EFA1F_Residuals", "CFA_Coefficients",
                      "CFA_RI_Coefficients", "used_items", "used_items_item_correlation", "used_items_mean_score",
                      "used_items_cfa_score"]


def test_input_checks():
    with pytest.raises(ValueError, match="missing the column"):
        glm_validity.compute_efa_cfa(pd.DataFrame({"dimension": ["A"]}), pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="key must be"):
        glm_validity.compute_efa_cfa(pd.DataFrame({"dimension": ["A"], "subquestion_text": ["x"], "key": [2]}),
                                     pd.DataFrame({"x": [1, 2]}))
