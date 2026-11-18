import matplotlib

matplotlib.use("Agg")

from conftest import assert_autoplot_matches, assert_matches_reference, autoplot_panels  # noqa: E402
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


def test_plot_oneway_diagnostics_matches_rwf(load_data, reference):
    """ggfortify's six panels: R's lowess, qqnorm and its quartile line, Cook's contours, the 3 extreme row names."""
    plots = glm_anova_plot.plot_oneway_diagnostics(df=load_data("insurance"), dv=[3, 7], iv=[5, 6, 4])
    assert list(plots) == reference("plot_oneway_diagnostics_keys")["key"].tolist()
    assert_autoplot_matches(plots["smoker_charges"], reference("plot_oneway_diagnostics_smoker_charges_layers", dtype={"label": str}),
                            reference("plot_oneway_diagnostics_smoker_charges_labels"))


def test_plot_oneway_diagnostics_constant_leverage_matches_rwf(load_data, reference):
    """Equal group sizes: panel 5 plots the residuals against the factor levels, with the x axis widened to 0."""
    plots = glm_anova_plot.plot_oneway_diagnostics(df=load_data("blood_pressure"), dv=[4], iv=[3])
    assert_autoplot_matches(plots["agegrp_bp_before"], reference("plot_oneway_diagnostics_balanced_layers", dtype={"label": str}),
                            reference("plot_oneway_diagnostics_balanced_labels"))
    assert len(autoplot_panels(plots["agegrp_bp_before"])) == 6
