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


_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
_XLSX_FORMATS = {0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00", 9: "0%", 10: "0.00%", 49: "@"}


def xlsx_summary(path):
    """
    A workbook read from its XML, the same way for openxlsx's files (rwf) and xlsxwriter's (pwf): per sheet
    its cells (value, number format, fill, left border and font), the style of its columns and of a cell
    without one, its comments, conditional formats and frozen pane.
    """
    import posixpath
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    ns = _XLSX_NS
    main = "{" + ns["m"] + "}"

    def text(element):
        return "".join(t.text or "" for t in element.iter(main + "t"))

    def color(element):
        if element is None:
            return None
        return element.get("rgb") or ("theme" + element.get("theme") if element.get("theme") else None)

    def rels(archive, part):
        folder, name = posixpath.split(part)
        path = posixpath.join(folder, "_rels", name + ".rels")
        if path not in archive.namelist():
            return {}
        return {rel.get("Id"): posixpath.normpath(posixpath.join(folder, rel.get("Target")))
                for rel in ET.fromstring(archive.read(path)).findall("pr:Relationship", ns)}

    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        workbook_rels = rels(archive, "xl/workbook.xml")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [text(si) for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        styles = ET.fromstring(archive.read("xl/styles.xml"))
        formats = dict(_XLSX_FORMATS)
        formats.update({int(f.get("numFmtId")): f.get("formatCode") for f in styles.findall("m:numFmts/m:numFmt", ns)})
        fills = []
        for fill in styles.findall("m:fills/m:fill", ns):
            pattern = fill.find("m:patternFill", ns)
            solid = pattern is not None and pattern.get("patternType") == "solid"
            fills.append(color(pattern.find("m:fgColor", ns)) if solid else None)
        fonts = [(font.find("m:name", ns).get("val"), float(font.find("m:sz", ns).get("val")))
                 for font in styles.findall("m:fonts/m:font", ns)]
        borders = []
        for border in styles.findall("m:borders/m:border", ns):
            left = border.find("m:left", ns)
            borders.append(None if left is None or left.get("style") is None else
                           (left.get("style"), color(left.find("m:color", ns))))
        xfs = [(int(xf.get("numFmtId", 0)), int(xf.get("fontId", 0)), int(xf.get("fillId", 0)),
                int(xf.get("borderId", 0))) for xf in styles.findall("m:cellXfs/m:xf", ns)]
        dxfs = []
        for dxf in styles.findall("m:dxfs/m:dxf", ns):
            pattern = dxf.find("m:fill/m:patternFill", ns)
            dxfs.append(None if pattern is None else
                        color(pattern.find("m:bgColor", ns)) or color(pattern.find("m:fgColor", ns)))

        def style(index):
            num_fmt, font, fill, border = xfs[index]
            return {"format": formats.get(num_fmt, f"id{num_fmt}"), "font": fonts[font], "fill": fills[fill],
                    "border": borders[border]}

        summary = {}
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            part = workbook_rels[sheet.get("{" + ns["r"] + "}id")]
            root = ET.fromstring(archive.read(part))
            columns = {}
            for col in root.findall("m:cols/m:col", ns):
                if col.get("style") is not None:
                    columns.update({k: style(int(col.get("style")))
                                    for k in range(int(col.get("min")), int(col.get("max")) + 1)})
            cells = {}
            for c in root.iter(main + "c"):
                kind, v = c.get("t", "n"), c.find("m:v", ns)
                if kind == "s":
                    value = strings[int(v.text)]
                elif kind == "inlineStr":
                    value = text(c.find("m:is", ns))
                elif kind in ("str", "e"):
                    value = None if v is None else v.text
                elif kind == "b":
                    value = v.text == "1"
                else:
                    value = None if v is None else float(v.text)
                cells[c.get("r")] = {"value": value, **style(int(c.get("s", 0)))}
            comments = {}
            for target in rels(archive, part).values():
                if re.search(r"comments\d*\.xml$", target):
                    for comment in ET.fromstring(archive.read(target)).findall("m:commentList/m:comment", ns):
                        comments[comment.get("ref")] = text(comment.find("m:text", ns))
            conditional = []
            for block in root.findall("m:conditionalFormatting", ns):
                sqref = re.sub(r"^([A-Z]+[0-9]+):\1$", r"\1", block.get("sqref"))  # F2:F2 is F2
                for rule in block.findall("m:cfRule", ns):
                    if rule.get("type") == "colorScale":
                        detail = tuple(color(c) for c in rule.findall("m:colorScale/m:color", ns))
                    else:
                        detail = tuple(f.text for f in rule.findall("m:formula", ns))
                    dxf = rule.get("dxfId")
                    conditional.append((sqref, rule.get("type"), detail, None if dxf is None else dxfs[int(dxf)]))
            pane = root.find("m:sheetViews/m:sheetView/m:pane", ns)
            summary[sheet.get("name")] = {
                "cells": cells, "columns": columns, "default": style(0), "comments": comments,
                "conditional": sorted(conditional, key=str),
                "pane": None if pane is None else (pane.get("topLeftCell"), pane.get("state"))}
        return summary


def assert_workbook_matches(path, reference_path):
    """
    pwf's workbook against rwf's (an .xlsx make_reference.R saved), as Excel shows them: the same sheets,
    frozen panes, comments and conditional formats, and every cell with the same value, number format, fill,
    border and font (a cell a file leaves out has its column's style, else the workbook's default). Only what
    looks the same may differ: R writes "" where pwf leaves a cell empty; and R's #NUM! for Inf is pandas'
    text "inf", a known difference.
    """
    import re

    mine, expected = xlsx_summary(path), xlsx_summary(reference_path)
    assert list(mine) == list(expected)

    def cell(sheet, ref):
        if ref in sheet["cells"]:
            return sheet["cells"][ref]
        letters = re.match(r"[A-Z]+", ref).group(0)
        column = sum((ord(ch) - 64) * 26 ** k for k, ch in enumerate(reversed(letters)))
        return {"value": None, **sheet["columns"].get(column, sheet["default"])}

    def same(a, b):
        if a in (None, "") and b in (None, ""):
            return True
        if isinstance(a, float) and isinstance(b, float):
            return bool(np.isclose(a, b, rtol=1e-12, atol=0))
        return a == b or (b == "#NUM!" and a in ("inf", "-inf"))

    for name in expected:
        got, want = mine[name], expected[name]
        assert got["pane"] == want["pane"], name
        assert got["comments"] == want["comments"], name
        assert got["conditional"] == want["conditional"], name
        for ref in set(got["cells"]) | set(want["cells"]):
            a, b = cell(got, ref), cell(want, ref)
            assert same(a["value"], b["value"]), (name, ref, a["value"], b["value"])
            for key in ["format", "fill", "border", "font"]:
                assert a[key] == b[key], (name, ref, key, a[key], b[key])
