import pathlib

import numpy as np
import pandas as pd
import pytest


def assert_matches_reference(result, expected, rtol=1e-8, atol=0.0, column_tol=None):
    """
    Assert a pwf result equals an rwf reference: same column names in the same
    order, same row count, equal text columns, and numeric columns within
    tolerance. column_tol maps column name -> (rtol, atol) overrides.

    R writes logicals as TRUE/FALSE, which pandas reads back as bools; they are
    compared as the strings "TRUE"/"FALSE", R's as.character() spelling.
    """
    column_tol = column_tol or {}
    assert list(result.columns) == list(expected.columns)
    assert len(result) == len(expected)
    for col in expected.columns:
        exp, res = expected[col], result[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(exp):
            exp = exp.map({True: "TRUE", False: "FALSE"})
        if pd.api.types.is_bool_dtype(res):
            res = res.map({True: "TRUE", False: "FALSE"})
        if pd.api.types.is_numeric_dtype(exp):
            col_rtol, col_atol = column_tol.get(col, (rtol, atol))
            np.testing.assert_allclose(res.to_numpy(dtype=float), exp.to_numpy(dtype=float),
                                       rtol=col_rtol, atol=col_atol, err_msg=col)
        else:
            assert res.astype(str).tolist() == exp.astype(str).tolist(), col


_AUTOPLOT_GEOMS = {"geom_point": "GeomPoint", "geom_line": "GeomLine", "geom_hline": "GeomHline", "geom_text": "GeomText",
                   "geom_abline": "GeomAbline", "geom_linerange": "GeomLinerange"}


def autoplot_panels(composition):
    """The panels of a plotnine composition (ggfortify's ggmultiplot), in order, without spacers."""
    from plotnine import ggplot
    if isinstance(composition, ggplot):
        return [composition] if composition.layers else []
    return [panel for item in composition.items for panel in autoplot_panels(item)]


def assert_autoplot_matches(composition, layers, labels, rtol=1e-8, atol=1e-10):
    """
    Panels of pwf's ggfortify::autoplot port against make_reference.R's autoplot_rows(): titles, captions,
    x ranges and every layer's values (geom_blank layers left out). The Q-Q points are compared as sorted
    pairs: residuals tied in value take the tied quantiles in an order R's last-bit noise decides.
    """
    panels = autoplot_panels(composition)
    for _, label in labels.iterrows():
        k = int(label["panel"])
        panel = panels[k - 1]
        expected_title = None if pd.isna(label["title"]) else label["title"]
        assert panel.labels.get("title", None) == expected_title, k
        assert panel.labels.get("caption", None) == label["caption"], k
        panel.draw()
        np.testing.assert_allclose(panel.layout.panel_params[0].x.range, [label["x_min"], label["x_max"]],
                                   rtol=1e-7, err_msg=f"panel {k} x range")
        drawn = [layer for layer in panel.layers if type(layer.geom).__name__ != "geom_blank"]
        expected = layers[(layers["panel"] == k) & (layers["geom"] != "GeomBlank")]
        assert [_AUTOPLOT_GEOMS[type(layer.geom).__name__] for layer in drawn] == \
            expected.drop_duplicates("layer")["geom"].tolist(), k
        for layer, (_, rows) in zip(drawn, expected.groupby("layer", sort=True)):
            data = layer.data.reset_index(drop=True)
            assert len(data) == len(rows), k
            if k == 2 and rows["geom"].iloc[0] == "GeomPoint":
                got = np.array(sorted(zip(data["x"], data["y"])))
                want = np.array(sorted(zip(rows["x"], rows["y"])))
                np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"panel {k} Q-Q points")
                continue
            for column in ["x", "y", "ymax", "slope", "intercept", "yintercept"]:
                if rows[column].notna().any():
                    np.testing.assert_allclose(data[column].to_numpy(float), rows[column].to_numpy(float), rtol=rtol,
                                               atol=atol, err_msg=f"panel {k} {rows['geom'].iloc[0]} {column}")
            if rows["label"].notna().any():
                assert data["label"].astype(str).tolist() == rows["label"].astype(str).tolist(), k

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "pwf" / "data"
REFERENCE_DIR = pathlib.Path(__file__).resolve().parent / "reference"


def _load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def _load_reference(name, **read_csv_kwargs):
    path = REFERENCE_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"no R reference {path.name}; run tests/reference/make_reference.R")
    # Read as text, so the CRLF line endings of a Git checkout on Windows become "\n" again, also inside quoted
    # cells that span lines (make_reference.R writes "\n"); read_csv(path) would keep their "\r"
    with open(path, encoding="utf-8", newline=None) as handle:
        return pd.read_csv(handle, **read_csv_kwargs)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure a test opened (matplotlib warns beyond 20 open figures)."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def load_data():
    """Load a bundled dataset by name, e.g. load_data("co2")."""
    return _load_data


@pytest.fixture
def reference():
    """Load an rwf reference output by name (see tests/reference/make_reference.R)."""
    return _load_reference


@pytest.fixture
def df_blood_pressure():
    return _load_data("blood_pressure")
