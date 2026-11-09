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
