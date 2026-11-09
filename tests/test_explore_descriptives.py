import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import DATA_DIR, assert_matches_reference  # noqa: E402
from pwf import explore_descriptives as ed  # noqa: E402


@pytest.fixture
def df_insurance(load_data):
    return load_data("insurance")


@pytest.fixture
def df_blood_pressure_na(df_blood_pressure):
    """Same NAs as df_blood_pressure_na in tests/reference/make_reference.R (1-based rows there)."""
    df = df_blood_pressure.copy()
    df.loc[[2, 49, 98], "bp_before"] = np.nan
    df.loc[[9, 69], "agegrp"] = np.nan
    return df


@pytest.fixture
def df_ocean_N():
    ocean = pd.read_csv(DATA_DIR / "ocean.csv", sep="\t")
    return ocean[[c for c in ocean.columns if c.startswith("N")]]


def text_reference(reference, name, text_columns):
    """Reference with the given columns read as text (R's "" kept, NA as NaN)."""
    return reference(name, keep_default_na=False, na_values=["NA"], dtype={c: str for c in text_columns})


def test_compute_descriptives(df_insurance, reference):
    result = ed.compute_descriptives(df=df_insurance, dv=[1, 3, 7])
    assert_matches_reference(result, reference("compute_descriptives"))


def test_compute_descriptives_iv(df_insurance, reference):
    result = ed.compute_descriptives(df=df_insurance, dv=[1, 7], iv=[2, 4])
    assert_matches_reference(result, text_reference(reference, "compute_descriptives_iv", ["levels"]))


def test_compute_descriptives_na(df_blood_pressure_na, reference):
    result = ed.compute_descriptives(df=df_blood_pressure_na, dv=[4, 5], iv=3)
    assert_matches_reference(result, reference("compute_descriptives_na"))


def test_compute_aggregate(df_insurance, reference):
    assert_matches_reference(ed.compute_aggregate(df=df_insurance, iv=[2, 5]), reference("compute_aggregate"))


def test_compute_aggregate_na_group(df_blood_pressure_na, reference):
    result = ed.compute_aggregate(df=df_blood_pressure_na.iloc[:, 1:], iv=2)
    assert_matches_reference(result, reference("compute_aggregate_na"))


def test_compute_aggregate_numeric_iv(df_insurance, reference):
    result = ed.compute_aggregate(df=df_insurance[["age", "children", "charges"]], iv=2)
    assert_matches_reference(result, reference("compute_aggregate_numeric_iv"))


def test_compute_frequencies(df_insurance, reference):
    result = ed.compute_frequencies(df=df_insurance[["sex", "children", "smoker", "region"]])
    assert_matches_reference(result, text_reference(reference, "compute_frequencies", ["Observation"]))


def test_compute_frequencies_na(df_blood_pressure_na, reference):
    result = ed.compute_frequencies(df=df_blood_pressure_na[["sex", "agegrp"]])
    assert_matches_reference(result, reference("compute_frequencies_na"))


def test_response_frequency_default_items(df_ocean_N, reference):
    assert_matches_reference(ed.response_frequency(df_ocean_N), reference("response_frequency"))


@pytest.mark.parametrize("type", ["frequency", "proportion", "percent", "all"])
def test_response_frequency_types(df_ocean_N, reference, type):
    result = ed.response_frequency(df_ocean_N, uniqueitems=range(1, 6), type=type)
    assert_matches_reference(result, reference(f"response_frequency_{type}"))


def test_response_frequency_missing_and_names(load_data, reference):
    responses = load_data("responses")
    result = ed.response_frequency(responses.iloc[:, 1:12], uniqueitems=range(1, 6), type="all")
    assert_matches_reference(result, reference("response_frequency_responses_na"))


def test_response_frequency_rejects_continuous(df_insurance):
    assert ed.response_frequency(df_insurance[["charges"]]) is None


def test_compute_crosstable(df_insurance, reference):
    result = ed.compute_crosstable(df=df_insurance, factor_index=[2, 5, 6])
    assert_matches_reference(result, reference("compute_crosstable"))


def test_compute_crosstable_combinations(df_insurance, reference):
    combinations = pd.DataFrame({"index1": ["sex", "smoker"], "index2": ["region", "children"]})
    result = ed.compute_crosstable(df=df_insurance, combinations=combinations)
    assert_matches_reference(result, text_reference(reference, "compute_crosstable_combinations", ["l1", "l2"]))


def test_compute_crosstable_na(df_blood_pressure_na, reference):
    result = ed.compute_crosstable(df=df_blood_pressure_na, factor_index=[2, 3])
    assert_matches_reference(result, reference("compute_crosstable_na"))


def test_plot_crosstable(df_insurance, reference):
    plots = ed.plot_crosstable(df=df_insurance, factor_index=[2, 5, 6])
    assert list(plots) == reference("plot_crosstable_names")["name"].tolist()
    assert_matches_reference(plots["region_sex"].data, reference("plot_crosstable_region_sex"))


def test_plot_mosaic(df_insurance, df_blood_pressure_na, reference):
    plots = ed.plot_mosaic(df=df_insurance, factor_index=[2, 5, 6])
    assert list(plots) == reference("plot_mosaic_names")["name"].tolist()
    assert_matches_reference(plots["region smoker"].data, reference("plot_mosaic_region_smoker"))
    with_na = ed.plot_mosaic(df=df_blood_pressure_na, factor_index=[2, 3])
    assert_matches_reference(with_na["agegrp sex"].data, reference("plot_mosaic_na_agegrp_sex"))


def test_plot_response_frequencies(df_ocean_N, reference):
    plots = ed.plot_response_frequencies(df=df_ocean_N, factor_index=[1, 2])
    assert_matches_reference(plots["N1"].data, text_reference(reference, "plot_response_frequencies_N1", ["tempdata"]))
    titles = reference("plot_response_frequencies_titles", keep_default_na=False)
    assert list(plots) == titles["name"].tolist()
    assert [p.labels.title for p in plots.values()] == titles["title"].tolist()
    assert [p.labels.caption for p in plots.values()] == titles["caption"].tolist()


def test_plots_draw(df_insurance, df_ocean_N, tmp_path):
    plots = [
        ed.plot_crosstable(df=df_insurance, factor_index=["sex", "region"])["region_sex"],
        ed.plot_mosaic(df=df_insurance, factor_index=["sex", "region"])["region sex"],
        ed.plot_response_frequencies(df=df_ocean_N, factor_index=["N1"], reorder=True)["N1"],
    ]
    for i, p in enumerate(plots):
        p.save(tmp_path / f"plot{i}.png", width=5, height=5, verbose=False)
