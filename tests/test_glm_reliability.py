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


ALPHA_KEY = {"E": [f"E{i}" for i in range(1, 6)], "N": [f"N{i}" for i in range(1, 6)], "single": ["E1"]}
ALPHA_PARTS = ["total", "item_statistics", "dropped"]


def _report_alpha(*args, **kwargs):
    import contextlib
    import io
    import warnings
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glm_reliability.report_alpha(*args, **kwargs)


@pytest.mark.parametrize("case,data", [("complete", "lavaan_data"), ("missing", "lavaan_fiml_data")])
def test_report_alpha_matches_rwf(reference, case, data):
    # psych::alpha; a scale of one item is skipped; with missing values the frequency columns are "1".."5"
    result = _report_alpha(reference(data), key=ALPHA_KEY)
    for part in ALPHA_PARTS:
        expected = reference(f"report_alpha_{case}_{part}", keep_default_na=False, na_values=["NA"])
        assert_matches_reference(result[f"result_{part}"], expected, rtol=1e-10)


def test_report_alpha_reverse_cumulative_matches_rwf(reference):
    # psych::reverse.code renames a reversed item "E2-", which rwf's data.frame() makes "E2."
    result = _report_alpha(reference("lavaan_data"), key={k: ALPHA_KEY[k] for k in "EN"},
                           reverse={"E": [1, -1, 1, -1, 1], "N": [1, 1, 1, 1, -1]}, mini=1, maxi=5, cumulative=True)
    for part in ALPHA_PARTS:
        expected = reference(f"report_alpha_reverse_{part}", keep_default_na=False, na_values=["NA"])
        assert_matches_reference(result[f"result_{part}"], expected, rtol=1e-10)


def test_alpha_helpers_match_rwf(reference):
    items = reference("lavaan_fiml_data")[[f"E{i}" for i in range(1, 6)]]
    assert glm_reliability.compute_raw_alpha(items) == pytest.approx(reference("compute_raw_alpha")["alpha"][0],
                                                                     rel=1e-12)
    assert_matches_reference(glm_reliability.compute_alpha_diagnostics(items).reset_index(drop=True),
                             reference("compute_alpha_diagnostics"), rtol=1e-12)
    assert_matches_reference(glm_reliability.compute_mean_sd_alpha(items), reference("compute_mean_sd_alpha"),
                             rtol=1e-12)
    assert_matches_reference(glm_reliability.compute_mean_sd_alpha(items, divisor=5),
                             reference("compute_mean_sd_alpha_divisor"), rtol=1e-12)


def test_key_to_cfa_model_and_compute_shrout_match_rwf(reference):
    model = glm_reliability.key_to_cfa_model({k: ALPHA_KEY[k] for k in "EN"})
    assert model == reference("key_to_cfa_model")["model"][0]
    result = glm_reliability.compute_shrout(sperson=0.5, spersonitem=0.2, stime=0.1, spersontime=0.15, serror=0.3,
                                            m=4, k=3)
    assert_matches_reference(result.reset_index(drop=True), reference("compute_shrout"), rtol=1e-12)


def test_report_alpha_workbook_matches_rwf(reference, tmp_path):
    # rwf's sheets, and its comments on the headers they name (10, 9 and 7 on the three result sheets)
    import re
    import zipfile
    file = str(tmp_path / "alpha")
    _report_alpha(reference("lavaan_data"), key={k: ALPHA_KEY[k] for k in "EN"}, file=file)
    assert list(pd.read_excel(f"{file}.xlsx", sheet_name=None)) == ["total statistics", "item statistics",
                                                                      "if dropped", "call"]
    with zipfile.ZipFile(f"{file}.xlsx") as archive:
        counts = [len(re.findall(r"<comment ", archive.read(name).decode("utf-8")))
                  for name in sorted(n for n in archive.namelist() if re.search(r"comments\d+\.xml$", n))]
    assert counts == [10, 9, 7]


def test_plot_mtmm_matches_rwf(reference):
    """R's collation for traits and methods (A, b, c; m1, m2, M3), rwf's lower-case names and cell labels."""
    data = reference("plot_mtmm_data")
    key = {"b": ["x1", "x2", "x3"], "A": ["x4", "x5", "x6"], "c": ["x7", "x8", "x9"]}
    p = glm_reliability.plot_mtmm(data, key=key, method="method", subject="id", title="T")
    levels = reference("plot_mtmm_levels")
    assert ",".join(p.data["trait_x"].cat.categories) == levels["trait_x"].iloc[0]
    assert ",".join(p.data["trait_y"].cat.categories) == levels["trait_y"].iloc[0]
    assert ",".join(p.data["method_x"].cat.categories) == levels["method_x"].iloc[0]
    expected = reference("plot_mtmm_plot_data", keep_default_na=False)
    result = p.data.astype({c: str for c in ("trait_x", "method_x", "trait_y", "method_y")})
    assert_matches_reference(result, expected, rtol=1e-12)
