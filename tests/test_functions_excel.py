import numpy as np
import pandas as pd
import pytest

from conftest import REFERENCE_DIR, _load_data, assert_matches_reference, assert_workbook_matches
from pwf import functions_excel as fe
from pwf.functions_train_test import confusion_matrix_percent

# make_reference.R's excel_df: whole numbers, decimals, whole numbers with a missing value (no style in rwf),
# text with an empty string, and -Inf (rwf writes #NUM!; it and "" get no critical-value rule)
EXCEL_DF = pd.DataFrame({"id": range(1, 7), "score": [1.25, 2.5, np.nan, 4.75, 5, 6.125],
                         "count": [3, np.nan, 5, 6, 7, 8], "group": ["a", "b", "", "a", "b", "c"],
                         "p": [0.01, 0.2, 0.049, 0.5, np.nan, 0.03], "z": [-2.5, 0.3, 1.97, -np.inf, 2.4, np.nan]})


def test_excel_helpers_match_rwf(tmp_path):
    """rwf's styles, comments, colours (R's gray, purple and green), conditional rules and red diagonal."""
    insurance = _load_data("insurance")
    observed = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]
    predicted = [1, 2, 2, 2, 3, 1, 1, 2, 3, 3, 2, 3]
    path = tmp_path / "excel_helpers.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as workbook:
        fe.excel_critical_value(EXCEL_DF, workbook, sheet="critical", title="Critical values",
                                comment={"score": "Score note", "group": "Group note", "nothere": "ignored"},
                                critical={"p": "<0.05", "z": ["<-1.96", ">1.96"], "group": '="a"'})
        fe.excel_matrix(insurance[["age", "bmi", "children", "charges"]].corr(), workbook, sheet="matrix",
                        title="Correlations", comment={"age": "Age note"}, conditional_formatting=True, diagonal=True)
        fe.excel_matrix(insurance.loc[:2, ["age", "bmi", "charges", "children"]], workbook, sheet="rectangle",
                        numFmt="#0.000", diagonal=True)
        fe.excel_confusion_matrix(confusion_matrix_percent(observed, predicted), workbook)
    assert_workbook_matches(path, REFERENCE_DIR / "excel_helpers.xlsx")


def test_report_dataframe_matches_rwf(tmp_path, reference):
    file = tmp_path / "report_dataframe"
    assert fe.report_dataframe(EXCEL_DF.iloc[:3, :4], file=str(file), type="matrix", sheet="data",
                               conditional_formatting=True) is None
    assert_workbook_matches(f"{file}.xlsx", REFERENCE_DIR / "report_dataframe.xlsx")
    assert_matches_reference(fe.report_dataframe(EXCEL_DF, file=None),
                             reference("report_dataframe_no_file", keep_default_na=False, na_values=["NA"]))


def test_excel_critical_value_unknown_column_fails(tmp_path):
    """rwf's df[, i] fails on a column that isn't there."""
    with pd.ExcelWriter(tmp_path / "x.xlsx", engine="xlsxwriter") as workbook:
        with pytest.raises(KeyError, match="undefined columns selected"):
            fe.excel_critical_value(EXCEL_DF, workbook, critical={"nothere": "<0"})


def test_data_frame_index_matches_rwf(reference):
    assert_matches_reference(fe.data_frame_index(3, 2), reference("data_frame_index"))


def test_excel_generic_format_matches_rwf(tmp_path, load_data):
    """On a sheet of its own, written with the row names first, as excel_generic_format's example."""
    data = load_data("insurance").iloc[:6].copy()
    data.loc[[1, 4], "bmi"] = np.nan  # make_reference.R's plots_insurance[1:6, ]
    path = tmp_path / "excel_generic.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as workbook:
        fe.excel_generic_format(data, workbook, sheet="sheet", title="Generic",
                                comment={"age": "Age note", "region": "Region note"}, numFmt="#0.0")
    assert_workbook_matches(path, REFERENCE_DIR / "excel_generic.xlsx")
