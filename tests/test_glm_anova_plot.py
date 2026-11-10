import matplotlib

matplotlib.use("Agg")

from conftest import assert_matches_reference  # noqa: E402
from pwf import glm_anova_plot  # noqa: E402


def test_plot_oneway_data_matches_rwf(load_data, reference):
    result = glm_anova_plot.plot_oneway(df=load_data("insurance"), dv=[3, 7], iv=[2, 5, 6])
    assert list(result["plot_data"]) == reference("plot_oneway_keys")["key"].tolist()
    assert_matches_reference(result["plot_data_df"], reference("plot_oneway_plot_data_df"), rtol=1e-10)


def test_plot_interaction_data_matches_rwf(load_data, reference):
    result = glm_anova_plot.plot_interaction(df=load_data("insurance"), dv=7, iv=[2, 5, 6])
    assert list(result["plot_data"]) == reference("plot_interaction_keys")["key"].tolist()
    assert_matches_reference(result["plot_data_df"], reference("plot_interaction_plot_data_df"), rtol=1e-10)
    for plot in result["plots"].values():
        plot.draw()


def test_plot_oneway_diagnostics_by_position(load_data):
    by_position = glm_anova_plot.plot_oneway_diagnostics(df=load_data("insurance"), dv=[7], iv=[5])
    by_name = glm_anova_plot.plot_oneway_diagnostics(df=load_data("insurance"), dv=["charges"], iv=["smoker"])
    assert list(by_position) == list(by_name)
