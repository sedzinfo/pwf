import matplotlib

matplotlib.use("Agg")

import re  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import assert_matches_reference  # noqa: E402
from pwf import functions_train_test  # noqa: E402

# Same cases as confusion_cases in tests/reference/make_reference.R
CONFUSION_CASES = {
    "perfect": dict(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 1]),
    "positive_0": dict(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0], positive=0),
    "positive_1": dict(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0], positive=1),
    "majority": dict(observed=["no"] * 40 + ["yes"] * 10,
                     predicted=["no"] * 35 + ["yes"] * 5 + ["no"] * 3 + ["yes"] * 7),
    "no_positive_predictions": dict(observed=["a", "a", "b", "b"], predicted=["a", "a", "a", "a"]),
}


@pytest.mark.parametrize("case", CONFUSION_CASES)
def test_plot_confusion_extended_data(case, reference):
    p = functions_train_test.plot_confusion_extended(**CONFUSION_CASES[case])
    expected = reference(f"plot_confusion_extended_{case}", keep_default_na=False, na_values=["NA"])
    assert_matches_reference(p.data, expected)


def test_plot_confusion_extended_draws(tmp_path):
    p = functions_train_test.plot_confusion_extended(**CONFUSION_CASES["majority"], title="test")
    p.save(tmp_path / "confusion.png", width=5, height=5, verbose=False)
    assert (tmp_path / "confusion.png").stat().st_size > 0


def test_plot_confusion_extended_needs_two_classes():
    with pytest.raises(ValueError, match="binary"):
        functions_train_test.plot_confusion_extended(observed=[1, 2, 3], predicted=[1, 2, 3])


TRAIN_TEST_CASES = {
    "numbers": dict(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11]),
    "ordinal": dict(observed=[1, 2, 2, 2, 2, 3, 3, 1, 2], predicted=[1, 1, 2, 2, 2, 3, 2, 2, 3]),
    "labels": dict(observed=["no", "no", "yes", "yes", "no"], predicted=["no", "yes", "yes", "no", "no"]),
}


@pytest.mark.parametrize("case", TRAIN_TEST_CASES)
def test_confusion_matrix_percent_matches_rwf(case, reference):
    table = functions_train_test.confusion_matrix_percent(**TRAIN_TEST_CASES[case])
    expected = reference(f"confusion_matrix_percent_{case}", dtype=str, keep_default_na=False)
    assert table.index.tolist() == expected["row"].tolist()
    assert table.columns.tolist() == expected.columns[1:].tolist()
    assert table.to_numpy().tolist() == expected.iloc[:, 1:].to_numpy().tolist()


@pytest.mark.parametrize("case", TRAIN_TEST_CASES)
def test_proportion_accurate_matches_rwf(case, reference):
    result = functions_train_test.proportion_accurate(**TRAIN_TEST_CASES[case])
    assert_matches_reference(result, reference(f"proportion_accurate_{case}"), rtol=1e-12)


@pytest.mark.parametrize("step", [0.1, 0.05])
def test_result_confusion_performance_matches_rwf(step, reference):
    data = reference("confusion_performance_data")
    result = functions_train_test.result_confusion_performance(data["observed"], data["predicted"], step=step)
    label = round(step * 100)
    assert_matches_reference(result["cut_performance"], reference(f"confusion_performance_{label}_cut_performance"),
                             rtol=1e-12, atol=1e-12)
    assert result["cut"].tolist() == pytest.approx(reference(f"confusion_performance_{label}_cut")["cut"].tolist())


def test_plot_roc_matches_proc(reference):
    data = reference("confusion_performance_data")
    expected = reference("plot_roc_auc")
    observed = data["observed"].astype(str).to_numpy()
    for _, row in expected.iterrows():
        control = str(int(row["control"])) if row["control"] >= 0 else "0"
        case = "0" if control == "1" else "1"
        predicted = data["predicted"].to_numpy() if row["control"] >= 0 else 1 - data["predicted"].to_numpy()
        roc = functions_train_test._roc(observed, predicted, control, case)
        assert roc["auc"] == pytest.approx(row["auc"], rel=1e-12)
        assert roc["direction"] == row["direction"]
    plots = functions_train_test.plot_roc(data["observed"], data["predicted"])
    assert list(plots) == ["1, 0", "0, 1"]


@pytest.mark.parametrize("case", ["two", "one"])
def test_recode_scale_dummy_matches_rwf(case, reference):
    data = reference("recode_scale_dummy_data")
    data = data if case == "two" else data.iloc[:, :3]
    result = functions_train_test.recode_scale_dummy(data)
    assert_matches_reference(result, reference(f"recode_scale_dummy_{case}"), rtol=1e-12)


def test_plot_separability_matches_rwf(reference):
    """as.factor() levels (-1, 1, 2, 10) and R's density of each class over the range of all predictions."""
    import numpy as np
    data = reference("plot_separability_data")
    p = functions_train_test.plot_separability(data["observed"], data["predicted"], title="T")
    labels = reference("plot_separability_labels")
    assert p.labels.get("title", None) == labels["title"].iloc[0]
    assert p.labels.get("caption", None) == labels["caption"].iloc[0]
    expected = reference("plot_separability_density", dtype={"observed": str})
    assert list(p.data["observed"].cat.categories) == list(dict.fromkeys(expected["observed"]))
    p.draw()
    density = p.layers[0].data.sort_values(["group", "x"], kind="stable")
    levels = list(p.data["observed"].cat.categories)
    assert [levels[g - 1] for g in density["group"]] == expected["observed"].tolist()
    np.testing.assert_allclose(density["x"], expected["x"], rtol=1e-13)
    np.testing.assert_allclose(density["y"], expected["density"], rtol=1e-11, atol=1e-14)


def test_r_cut_labels_match_rwf(reference):
    """R's cut(1:n, breaks = k, labels = FALSE), which k_fold and k_sample shuffle into folds."""
    import numpy as np
    for (n, k), labels in reference("cut_labels").groupby(["n", "k"], sort=False):
        np.testing.assert_array_equal(functions_train_test._r_cut_labels(n, k), labels["label"], err_msg=f"{n} {k}")


def test_k_fold_matches_rwf(reference, load_data, capsys):
    """The folds are random, their sizes are not: the printed lines are rwf's."""
    import numpy as np
    result = functions_train_test.k_fold(load_data("insurance"), "charges ~ age + bmi + smoker", k=7)
    assert capsys.readouterr().out.splitlines() == reference("k_fold_printed")["line"].tolist()
    parts = reference("k_fold_parts").iloc[0]
    assert ",".join(result["variables"]) == parts["variables"] and ",".join(result["predictors"]) == parts["predictors"]
    assert result["outcome"] == parts["outcome"] and len(result["index"]) == parts["n_index"]
    rows = np.sort(np.concatenate([result["f"]["index"][f"f{i}"] for i in range(1, 8)]))
    np.testing.assert_array_equal(rows, np.arange(1, 1339))  # every row is in one test set
    assert result["xgb"]["f1"]["test"].num_row() == len(result["f"]["test"]["f1"])


def test_k_sample_matches_rwf(reference, capsys):
    """k_sample's halving into train, test and validation (and its "Validation:" count, the test set's), and
    R's failure on a fold too small to split."""
    for _, row in reference("k_sample_printed").iterrows():
        n = int(row["n"])
        frame = pd.DataFrame({"y": range(1, n + 1), "x": [2 * v for v in range(1, n + 1)]})
        if row["line"].startswith("ERROR: "):
            with pytest.raises(IndexError, match=re.escape(row["line"][len("ERROR: "):])):
                functions_train_test.k_sample(frame, "y ~ x", k=1)
            capsys.readouterr()
        else:
            functions_train_test.k_sample(frame, "y ~ x", k=1)
            assert capsys.readouterr().out == row["line"] + "\n", n


def test_plot_confusion_matches_rwf(reference):
    """rwf's tiles, counts, class order (numbers by value, text by R's collation) and caption, and its colours:
    ggplot2's scale_fill_continuous(low = "#fafaff") interpolated in Lab."""
    cases = {"numbers": dict(observed=[1, 2, 3, 1, 2, 3, 10, 2], predicted=[1, 2, 3, 1, 1, 3, 10, 2]),
             "text": dict(observed=["male"] * 10 + ["female"] * 10 + ["male", "male"],
                          predicted=["male"] * 10 + ["female"] * 10 + ["female", "female"]),
             "unequal": dict(observed=["a", "b", "c", "a", "B"], predicted=["a", "b", "b", "a", "a"])}
    expected = reference("plot_confusion_tiles")
    for case, arguments in cases.items():
        want = expected[expected["case"] == case]
        p = functions_train_test.plot_confusion(**arguments, title="T")
        assert ",".join(p.data["observed"].cat.categories) == want["observed"].iloc[0], case
        assert ",".join(p.data["predicted"].cat.categories) == want["predicted"].iloc[0], case
        assert p.labels.get("title", None) == want["title"].iloc[0]
        assert p.labels.get("caption", None) == want["caption"].iloc[0]
        p.draw()
        tiles = p.layers[0].data.merge(p.layers[1].data[["x", "y", "label"]], on=["x", "y"]).sort_values(["x", "y"])
        assert tiles[["x", "y"]].to_numpy().tolist() == want[["x", "y"]].to_numpy().tolist(), case
        assert tiles["fill"].str.upper().tolist() == want["fill"].tolist(), case
        assert tiles["label"].astype(int).tolist() == want["label"].tolist(), case
