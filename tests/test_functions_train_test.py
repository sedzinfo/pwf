import matplotlib

matplotlib.use("Agg")

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
