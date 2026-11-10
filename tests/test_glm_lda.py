import numpy as np
import pandas as pd
import pytest

from conftest import _load_reference, assert_matches_reference
from pwf import glm_lda

# Same models as lda_models in tests/reference/make_reference.R
CASES = {"infert": ("case ~ .", "infert"), "iris": ("Species ~ .", "iris"),
         "iris_two": ("Species ~ Sepal.Length + Petal.Width", "iris")}


@pytest.mark.parametrize("case", list(CASES))
def test_report_lda_matches_rwf(case, reference):
    formula, dataset = CASES[case]
    # the call names the data by its variable, as R's reference does
    if dataset == "infert":
        infert_data = _load_reference("infert")
        model = glm_lda.lda(formula, data=infert_data)
    else:
        iris_data = _load_reference("iris")
        model = glm_lda.lda(formula, data=iris_data)
    result = glm_lda.report_lda(model)
    assert_matches_reference(result["prior_counts"].rename_axis("class").reset_index(),
                             reference(f"report_lda_{case}_prior_counts", dtype={"class": str}), rtol=1e-10)
    assert_matches_reference(result["coeficients"].rename_axis("variable").reset_index(),
                             reference(f"report_lda_{case}_coeficients"), rtol=1e-8, atol=1e-12)
    assert_matches_reference(result["model_description"], reference(f"report_lda_{case}_model_description"),
                             rtol=1e-8)
    expected = reference(f"report_lda_{case}_cmatrix", dtype=str, keep_default_na=False)
    assert result["cmatrix"].index.tolist() == expected["row"].tolist()
    assert result["cmatrix"].to_numpy().tolist() == expected.iloc[:, 1:].to_numpy().tolist()
    call = reference(f"report_lda_{case}_call")["call"].iloc[0]
    assert result["call"]["call"].iloc[0] == call
    posterior = model.predict()["posterior"]
    np.testing.assert_allclose(posterior.to_numpy(), reference(f"report_lda_{case}_posterior").to_numpy(),
                               rtol=1e-6, atol=1e-12)


def test_report_lda_writes_file(tmp_path):
    iris = _load_reference("iris")
    glm_lda.report_lda(glm_lda.lda("Species ~ .", data=iris), file=str(tmp_path / "lda"))
    assert list(pd.read_excel(tmp_path / "lda.xlsx", sheet_name=None)) == [
        "Coefficients", "Confusion Matrix", "Priors and Counts", "Means", "Descriptives", "Call"]
