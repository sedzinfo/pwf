import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from conftest import assert_matches_reference  # noqa: E402
from pwf import explore_time_series  # noqa: E402


def test_plot_ts_data_matches_rwf(reference):
    lh = reference("lh")["value"]
    assert_matches_reference(explore_time_series.plot_ts(lh.to_numpy()).data, reference("plot_ts_lh"))


def test_plot_acf_matches_rwf(reference):
    lh = reference("lh")["value"].to_numpy()
    plot = explore_time_series.plot_acf(lh)
    data = plot.data.copy()
    data["type"] = data["type"].astype(str)
    assert_matches_reference(data, reference("plot_acf_lh"), rtol=1e-10, atol=1e-14)
    ci = reference("plot_acf_lh_ci")
    lines = explore_time_series._acf_ci(lh, len(lh))
    np.testing.assert_allclose(lines[["upper", "mean", "lower"]].to_numpy(), ci[["upper", "mean", "lower"]].to_numpy(),
                               rtol=1e-10, atol=1e-14)
    short = explore_time_series.plot_acf(lh, lag_max=10).data.copy()
    short["type"] = short["type"].astype(str)
    assert_matches_reference(short, reference("plot_acf_lh_10"), rtol=1e-10, atol=1e-14)


def test_compute_moving_average_matches_rwf(reference):
    lh = reference("lh")["value"]
    result = explore_time_series.compute_moving_average(pd.DataFrame({"a": lh, "b": lh ** 2}), 2)
    assert_matches_reference(result, reference("compute_moving_average_lh"), rtol=1e-12)


def test_ts_smoothing_draws():
    lh = np.sin(np.linspace(0, 6, 40))
    for kind in ["default", "kernel", "lowess", "friedman", "splines", "polynomial", "linear"]:
        explore_time_series.ts_smoothing(lh, start=0.1, stop=0.9, step=0.4, type=kind)
