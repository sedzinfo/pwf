import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import _load_reference, assert_matches_reference  # noqa: E402
from pwf import glm_sem  # noqa: E402

SEM_MODEL = "E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2"


@pytest.fixture(scope="module")
def fit():
    return glm_sem.sem(SEM_MODEL, _load_reference("lavaan_data"))


@pytest.mark.parametrize("what", ["est", "std"])
def test_inspect_matches_lavaan(fit, reference, what):
    matrices = fit.inspect(what)
    for name in ["lambda", "theta", "psi", "beta"]:
        expected = reference(f"lavaan_sem_inspect_{what}_{name}")
        np.testing.assert_allclose(matrices[name].to_numpy(), expected.to_numpy(), atol=5e-5, err_msg=name)
        assert list(matrices[name].columns) == list(expected.columns)


def test_report_cfa(fit, reference, tmp_path):
    file = str(tmp_path / "cfa")
    result = glm_sem.report_cfa(fit, file=file)
    assert set(result) == {"r_squared", "fit_indices", "parameters", "modification_indices", "sample_covariance",
                           "unstandardized_estimates", "standardized_estimates", "group", "predict", "call"}
    fit_expected = reference("lavaan_sem_fit")
    # lavaan writes chi-square p values below about 1e-20 as 0
    np.testing.assert_allclose(result["fit_indices"].loc[fit_expected["measure"], "fit"], fit_expected["value"],
                               rtol=1e-3, atol=1e-20)
    parameters = reference("lavaan_sem_parameters")
    r2 = parameters[parameters["op"] == "r2"].set_index("lhs")["est"]
    np.testing.assert_allclose(result["r_squared"].loc[r2.index, "r_squared"], r2, rtol=1e-4)
    assert list(result["predict"].columns[-3:]) == ["E", "N", "A"]
    assert result["predict"].shape[0] == fit.n
    for suffix in ["_diagram.pdf", ".log", ".xlsx"]:
        assert (tmp_path / f"cfa{suffix}").exists()
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["R_Squared", "Fit_Indices", "Parameters", "Modification_Indices", "Groups", "Sample_Covariance",
                      "Scores", "Call"]


def test_plot_cfa_keys_match_rwf(fit, reference, capsys):
    """rwf's plot_cfa skips the "eq" diagrams of a model without labels (plot_cfa_gg fails on them)."""
    plots = glm_sem.plot_cfa(fit)
    assert list(plots) == reference("plot_cfa_keys")["key"].tolist()
    assert capsys.readouterr().out.count("Skipping") == 3


def test_plot_cfa_gg_eq_needs_labels(fit):
    with pytest.raises(ValueError, match="labels"):
        glm_sem.plot_cfa_gg(fit, what="eq")


_CFA_GEOMS = {"geom_segment": "GeomSegment", "geom_label": "GeomLabel", "geom_rect": "GeomRect",
              "geom_text": "GeomText", "geom_path": "GeomCurve"}


def _assert_cfa_layers_match(p, expected):
    """plot_cfa_gg's layers as make_reference.R's plots_layer_rows(): rectangles by their centres, arcs by their ends."""
    p.draw()
    assert [_CFA_GEOMS[type(layer.geom).__name__] for layer in p.layers] == \
        expected.drop_duplicates("layer")["geom"].tolist()
    for i, layer in enumerate(p.layers, start=1):
        data, rows = layer.data, expected[expected["layer"] == i]
        if isinstance(layer.geom, glm_sem.geom_rect):
            data = pd.DataFrame({"x": (data["xmin"] + data["xmax"]) / 2, "y": (data["ymin"] + data["ymax"]) / 2})
        elif isinstance(layer.geom, glm_sem.geom_path):  # grid's curve, drawn as a path from (x, y) to (xend, yend)
            ends = data.groupby("group", sort=True)
            data = pd.DataFrame({"x": ends["x"].first(), "y": ends["y"].first(), "xend": ends["x"].last(),
                                 "yend": ends["y"].last()})
        data = data.reset_index(drop=True)
        assert len(data) == len(rows), i
        for column in ["x", "y", "xend", "yend"]:
            if rows[column].notna().any():
                np.testing.assert_allclose(data[column], rows[column], rtol=0, atol=1e-12, err_msg=f"{i} {column}")
        if rows["label"].notna().any():
            assert data["label"].astype(str).tolist() == rows["label"].astype(str).tolist(), i


@pytest.mark.parametrize("layout,what", [("tree", "std"), ("tree", "est"), ("circle", "std"), ("circle", "est")])
def test_plot_cfa_gg_matches_rwf(fit, reference, layout, what):
    """igraph's tree and circle layouts, R's merge() order of the edges, formatC labels and grid's covariance arc."""
    expected = reference(f"plot_cfa_gg_{layout}_{what}", keep_default_na=False, na_values=["NA"])
    p = glm_sem.plot_cfa_gg(fit, what=what, layout=layout)
    assert p.labels.get("title", None) == expected["title"].iloc[0]
    _assert_cfa_layers_match(p, expected)


def test_plot_cfa_gg_cross_loadings_match_rwf(reference):
    """A cross-loading indicator: R's merge() orders tied rows by lhs, factors a, b, C in R's collation."""
    model = glm_sem.cfa("b =~ E1 + E2 + E3 + N1; a =~ N1 + N2 + N3 + E3; C =~ A1 + A2 + A3",
                        _load_reference("lavaan_data"))
    expected = reference("plot_cfa_gg_cross_tree_std", keep_default_na=False, na_values=["NA"])
    _assert_cfa_layers_match(glm_sem.plot_cfa_gg(model, what="std", layout="tree"), expected)


@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened")  # one plot per fit index, as in rwf
def test_simulate_cfa_fit():
    sim_results, plots = glm_sem.simulate_cfa_fit(model_sim="F =~ 1*X1 + 0.8*X2 + 0.6*X3 + 0.9*X4",
                                                  model="F =~ X1 + X2 + X3 + X4", minnobs=100, maxnobs=300,
                                                  stepping=100)
    assert sim_results["observations"].tolist() == [100, 200, 300]
    assert {"chisq", "cfi", "rmsea", "srmr"} <= set(sim_results.columns)
    assert len(plots) > 0


def test_simulate_data_recovers_model():
    data = glm_sem.simulate_data("F =~ 1*X1 + 0.8*X2 + 0.6*X3 + 0.9*X4", sample_nobs=20000, seed=1)
    loadings = glm_sem.cfa("F =~ X1 + X2 + X3 + X4", data).parameter_estimates()
    loadings = loadings[loadings["op"] == "=~"].set_index("rhs")["est"]
    np.testing.assert_allclose(loadings, [1, 0.8, 0.6, 0.9], atol=0.05)


def test_report_cfa_rejects_other_models():
    with pytest.raises(TypeError):
        glm_sem.report_cfa(object())


@pytest.mark.parametrize("kind,std_lv,case", [("sem", False, "sem"), ("cfa", False, "cfa"),
                                              ("cfa", True, "cfa_std_lv")])
def test_report_cfa_call_matches_rwf(reference, capsys, kind, std_lv, case):
    # rwf's Call sheet: lavaan's fit@call deparsed (with the arguments lavaan::sem/cfa add), spaces removed;
    # the model and the data appear under the names they have here (SEM_MODEL, data)
    data = _load_reference("lavaan_data")
    fit = getattr(glm_sem, kind)(SEM_MODEL, data, std_lv=std_lv)
    result = glm_sem.report_cfa(fit)
    assert result["call"]["call"].tolist() == reference(f"report_cfa_call_{case}")["call"].tolist()


def test_report_cfa_modification_rows_match_rwf(reference, capsys):
    # lavaan numbers modification indices by their place in its parameter table, then its added candidates
    data = _load_reference("lavaan_data")
    result = glm_sem.report_cfa(glm_sem.cfa(SEM_MODEL, data, std_lv=True))["modification_indices"]
    expected = reference("report_cfa_modification_rows")
    rows = dict(zip(zip(expected["lhs"], expected["op"], expected["rhs"]), expected["row"]))
    ours = dict(zip(zip(result["lhs"], result["op"], result["rhs"]), result.index))
    assert ours == rows


def test_std_lv_fixes_endogenous_latent_variance(reference):
    # lavaan's std.lv fixes every latent (residual) variance to 1: with A ~ E + N, pwf left A ~~ A free
    # (an under-identified model: 40 df instead of 41, missing standard errors)
    fit = glm_sem.sem(SEM_MODEL, _load_reference("lavaan_data"), std_lv=True)
    mine = fit.parameter_estimates()
    expected = reference("lavaan_sem_std_lv_parameters")
    merged = expected.merge(mine[["lhs", "op", "rhs", "est", "se"]], on=["lhs", "op", "rhs"], suffixes=("", "_pwf"))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged["est_pwf"], merged["est"], atol=5e-5)
    np.testing.assert_allclose(merged["se_pwf"], merged["se"], atol=5e-5)
    fit_expected = reference("lavaan_sem_std_lv_fit").set_index("measure")["value"]
    assert fit.fit_measures()["df"] == fit_expected["df"]
    assert fit.fit_measures()["chisq"] == pytest.approx(fit_expected["chisq"], rel=1e-6)


LABELLED_MODEL = ("E =~ E1 + a*E2 + a*E3 + l4*E4; N =~ N1 + N2 + l3*N3 + N4; A =~ A1 + A2 + A3; def1 := l4*2; "
                  "N ~ r*E; l4 == -l3 - 0.4; A ~ s*N; ind := r*s; N2 ~~ v*N2; N4 ~~ v*N4")
DIAGRAM_MODEL = "E =~ E1 + a*E3 + a*E5 + E2; N =~ N1 + b*N3 + l5*N5 + N2; E ~~ c*N; l5 == b - 0.2"


def test_report_cfa_parameters_of_a_labelled_model_match_rwf(reference, capsys):
    """rwf's parameterEstimates(remove.eq = FALSE, remove.system.eq = TRUE): the label and exo columns, the :=
    rows and the user's == row (its est the constraint's value), with lavaan's row numbers."""
    model = glm_sem.sem(LABELLED_MODEL, _load_reference("lavaan_data"))
    result = glm_sem.report_cfa(model)["parameters"]
    capsys.readouterr()
    expected = reference("labels_combined_report_parameters", keep_default_na=False, na_values=["NA"])
    assert [str(i) for i in result.index] == expected["row"].astype(str).tolist()
    assert_matches_reference(result, expected.drop(columns="row"), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("layout", ["tree", "circle"])
def test_plot_cfa_gg_eq_matches_rwf(reference, layout):
    """The "eq" diagram of a labelled model: the labels (a, b, l5; c on the covariance arc) and the estimates
    of the other parameters, formatC'd."""
    model = glm_sem.sem(DIAGRAM_MODEL, _load_reference("lavaan_data"))
    expected = reference(f"labels_plot_cfa_gg_{layout}_eq", keep_default_na=False, na_values=["NA"])
    p = glm_sem.plot_cfa_gg(model, what="eq", layout=layout)
    assert p.labels.get("title", None) == expected["title"].iloc[0]
    _assert_cfa_layers_match(p, expected)


def test_plot_cfa_keys_of_a_labelled_model_match_rwf(reference, capsys):
    """With labels rwf's plot_cfa draws all nine diagrams, the "eq" ones included."""
    model = glm_sem.sem(DIAGRAM_MODEL, _load_reference("lavaan_data"))
    assert list(glm_sem.plot_cfa(model)) == reference("labels_plot_cfa_keys")["key"].tolist()
    assert "Skipping" not in capsys.readouterr().out
