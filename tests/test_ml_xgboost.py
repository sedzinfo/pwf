import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xgboost as xgb  # noqa: E402

from conftest import REFERENCE_DIR, assert_matches_reference  # noqa: E402
from pwf import ml_xgboost  # noqa: E402

# Same splits and features as xgb_cases in tests/reference/make_reference.R; the models themselves
# were trained in R and saved as xgboost JSON, so both sides read identical trees.
XGB_CASES = {
    "classification": ("admission", "admit", ["gre", "gpa", "rank"]),
    "regression": ("insurance", "charges", ["age", "bmi", "children"]),
}


@pytest.fixture(params=list(XGB_CASES))
def case(request, load_data):
    name = request.param
    data, label, features = XGB_CASES[name]
    df = load_data(data)
    booster = xgb.Booster(model_file=str(REFERENCE_DIR / f"xgb_{name}.json"))
    return name, booster, df.iloc[300:400].reset_index(drop=True), label, features


def evals_from_reference(log):
    """The evaluation log R stores in the model, as the evals_result dict xgboost.train fills in Python."""
    result = {}
    for column in log.columns.drop("iter"):
        data, metric = column.split("_", 1)
        result.setdefault(data, {})[metric] = log[column].tolist()
    return result


def test_report_xgboost(case, reference):
    name, booster, test, label, _ = case
    log = reference(f"report_xgboost_{name}_evaluation_log")
    report = ml_xgboost.report_xgboost(booster, validation_data=test, label=label, file=None,
                                       evals_result=evals_from_reference(log))
    result = report["result"]
    # split gains are float32 in xgboost; R and Python read them from different dump formats
    assert_matches_reference(result["importance"], reference(f"report_xgboost_{name}_importance"), rtol=1e-6)
    assert_matches_reference(result["model_summary"],
                             reference(f"report_xgboost_{name}_model_summary", dtype={"value": str}))
    # predictions are float32 in xgboost
    assert_matches_reference(result["validation_metrics"], reference(f"report_xgboost_{name}_validation_metrics"),
                             rtol=1e-5)
    predicted = reference(f"report_xgboost_{name}_predicted")
    np.testing.assert_allclose(report["predicted"], predicted["predicted"], rtol=1e-6)
    np.testing.assert_array_equal(report["observed"], predicted["observed"])
    assert_matches_reference(result["evaluation_log"], log)
    assert report["feature_names"] == XGB_CASES[name][2]
    assert {"regression", "depth", "cover", "weight", "error", "importance"} <= set(report["plots"])
    assert ("performance" in report["plots"]) == (name == "classification")


def test_leaf_depths(case, reference):
    name, booster, *_ = case
    assert_matches_reference(ml_xgboost._leaf_depths(booster), reference(f"report_xgboost_{name}_depth"),
                             rtol=1e-6)


def test_plot_trees_xgboost(case, reference, tmp_path):
    name, booster, test, _, features = case
    out = ml_xgboost.plot_trees_xgboost(booster, train=test[features], file=str(tmp_path / "trees"))
    expected_nodes = reference(f"plot_trees_xgboost_{name}_nodes", keep_default_na=False)
    assert out["nodes"]["abs.node.position"].tolist() == expected_nodes["abs.node.position"].astype(str).tolist()
    for got, want in zip(out["nodes"]["Text"], expected_nodes["Text"]):
        got_lines, want_lines = got.split("\n"), want.split("\n")
        assert [g.split(" (")[0] for g in got_lines] == [w.split(" (")[0] for w in want_lines]
        # same notation as R's format(digits = 5); values agree to float32 precision
        assert ["e+" in g for g in got_lines] == ["e+" in w for w in want_lines]
        got_values = [float(g.split(" (")[1].rstrip(")")) for g in got_lines]
        want_values = [float(w.split(" (")[1].rstrip(")")) for w in want_lines]
        np.testing.assert_allclose(got_values, want_values, rtol=1e-6)
    expected_edges = reference(f"plot_trees_xgboost_{name}_edges")
    assert list(zip(out["edges"]["From"], out["edges"]["To"])) == list(zip(expected_edges["From"].astype(str),
                                                                           expected_edges["To"].astype(str)))
    with open(out["file"], encoding="utf-8") as handle:
        page = handle.read()
    assert page.count("<rect") == len(expected_nodes)


def test_report_xgboost_writes_files(load_data, tmp_path):
    df = load_data("admission")
    train, test = df.iloc[:300], df.iloc[300:400]
    dtrain = xgb.DMatrix(train[["gre", "gpa", "rank"]], label=train["admit"])
    evals = {}
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2, "nthread": 1}, dtrain, 5,
                        evals=[(dtrain, "train")], evals_result=evals, verbose_eval=False)
    file = str(tmp_path / "xgb")
    ml_xgboost.report_xgboost(booster, validation_data=test, label="admit", file=file, title="test",
                              evals_result=evals)
    sheets = list(pd.read_excel(f"{file}.xlsx", sheet_name=None))
    assert sheets == ["Confusion Matrix", "Model Summary", "Validation Metrics", "Feature Importance",
                      "Hyperparameters", "Evaluation Log"]
    assert (tmp_path / "xgb_test.pdf").exists()
