# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_EXCEL.R on pandas.ExcelWriter (xlsxwriter
engine), with rwf's argument names: excel_generic_format, excel_matrix,
excel_critical_value and excel_confusion_matrix take (df, workbook, sheet,
title, comment, numFmt, ...), write row names as rwf (1..n for a default
index) and format as rwf (frozen header and row names, Liberation Sans 10,
grey borders, "#0" for whole-number columns, comments from title and
comment); report_dataframe(df, file, type, ...) writes one sheet.

The old pwf keyword names (writer, sheetname, comments, decimals) still
work with a DeprecationWarning.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import os
import warnings

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell
##########################################################################################
# COLUMN WIDTHS
##########################################################################################
def get_col_widths(df):
    """
    This function calculates the maximum width needed for each column in a dataframe including the index.

    The width of a column is defined as the maximum string length of all entries in that column.
    This is useful for formatting the dataframe when exporting it to a fixed-width file (like a text file).

    Parameters:
    df (pandas.DataFrame): The DataFrame for which to calculate column widths.

    Returns:
    list: A list of integers representing the maximum width of each column, with the index column width as the first element.
    """
    # First we find the maximum length of the index column
    idx_max=max([len(str(s)) for s in df.index.values]+[len(str(df.index.name))])
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    result=[idx_max]+[max([len(str(s)) for s in df[col].values]+[len(str(col))]) for col in df.columns]
    return result
##########################################################################################
# GENERIC FORMAT EXCEL
##########################################################################################
_LEGACY_NAMES = {"writer": "workbook", "sheetname": "sheet", "comments": "comment"}


def _legacy(kwargs, function):
    """Map the old pwf keyword names (writer, sheetname, comments, decimals) onto rwf's."""
    mapped = {}
    for key, value in kwargs.items():
        if key == "decimals":
            warnings.warn(f"{function}(decimals=) is deprecated, use numFmt=", DeprecationWarning, stacklevel=3)
            mapped["numFmt"] = "#0" if value == 0 else "#0." + "0" * value
        elif key in _LEGACY_NAMES:
            warnings.warn(f"{function}({key}=) is deprecated, use {_LEGACY_NAMES[key]}=", DeprecationWarning,
                          stacklevel=3)
            mapped[_LEGACY_NAMES[key]] = value
        else:
            raise TypeError(f"{function}() got an unexpected keyword argument {key!r}")
    return mapped


def _r_rows(df):
    """The frame as rwf writes it (openxlsx writeData(rowNames = TRUE)): a default index becomes 1..n."""
    frame = pd.DataFrame(df).copy()
    if isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1:
        frame.index = pd.RangeIndex(1, len(frame) + 1)
    return frame


def _write_sheet(df, workbook, sheet):
    """openxlsx::addWorksheet + writeData(rowNames = TRUE)."""
    frame = _r_rows(df)
    frame.to_excel(workbook, sheet_name=sheet, index=True)
    return frame


def _is_whole(column):
    """rwf's integer test: every value of a numeric column equals its rounded value (text counts as whole)."""
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        return True
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return bool(np.all(finite == np.round(finite)))


def excel_generic_format(df, workbook, sheet="output", title=None, comment=None, numFmt="#0.00", **kwargs):
    """
    Format a worksheet holding df, as rwf::excel_generic_format: frozen
    header row and row-name column, Liberation Sans 10, thin grey cell
    borders, numFmt for numeric columns with decimals and "#0" for
    whole-number columns, black borders on the header and row names,
    automatic column widths, title as a comment on A1 and comment's texts
    on the headers of the columns they name. When the sheet does not exist
    yet, df is first written to it with its row names (as rwf's callers do
    with openxlsx::writeData).

    Parameters:
    df (pandas.DataFrame): The data on the sheet.
    workbook (pandas.ExcelWriter): Open writer with the xlsxwriter engine.
    sheet (str, optional): Worksheet name. Defaults to "output".
    title (str, optional): Comment on cell A1. Defaults to None.
    comment (dict, optional): Column name -> comment on that column's header. Defaults to None.
    numFmt (str, optional): Excel number format of non-integer numeric columns. Defaults to "#0.00".

    Returns:
    None

    Examples:
    >>> with pd.ExcelWriter("out.xlsx", engine="xlsxwriter") as workbook:
    ...     excel_generic_format(df_blood_pressure, workbook, sheet="data", comment={"sex": "M/F"})
    """
    if kwargs:
        mapped = _legacy(kwargs, "excel_generic_format")
        workbook = mapped.get("workbook", workbook)
        sheet = mapped.get("sheet", sheet)
        comment = mapped.get("comment", comment)
    frame = _r_rows(df)
    if sheet not in workbook.sheets:
        _write_sheet(df, workbook, sheet)
    book = workbook.book
    worksheet = workbook.sheets[sheet]
    font = {"font_name": "Liberation Sans", "font_size": 10}
    content = book.add_format({**font, "border": 1, "border_color": "gray", "valign": "bottom", "num_format": numFmt})
    whole = book.add_format({**font, "border": 1, "border_color": "gray", "valign": "bottom", "num_format": "#0"})
    header = book.add_format({**font, "border": 1, "border_color": "black", "valign": "bottom", "bold": True})
    widths = get_col_widths(frame)
    worksheet.set_column(0, 0, widths[0] + 2, header)
    for k, column in enumerate(frame.columns, start=1):
        worksheet.set_column(k, k, widths[k] + 2, whole if _is_whole(frame[column]) else content)
    worksheet.freeze_panes(1, 1)
    if title is not None:
        worksheet.write_comment(0, 0, str(title), {"author": "pwf", "visible": False})
    for column, text in (comment or {}).items():
        if column in frame.columns:
            worksheet.write_comment(0, list(frame.columns).index(column) + 1, str(text),
                                    {"author": "pwf", "visible": False, "x_scale": 2})
##########################################################################################
# MATRIX EXCEL
##########################################################################################
def excel_matrix(df, workbook, sheet="output", title=None, comment=None, numFmt="#0.00",
                 conditional_formatting=False, diagonal=False, diagonal_length=None, **kwargs):
    """
    Write a matrix to a new worksheet, as rwf::excel_matrix: the data with
    its row names, formatted by excel_generic_format, optionally with a
    red-yellow-green colour scale over the values and a red diagonal.

    Parameters:
    df (pandas.DataFrame): The matrix.
    workbook (pandas.ExcelWriter): Open writer with the xlsxwriter engine.
    sheet (str, optional): Worksheet name. Defaults to "output".
    title (str, optional): Comment on cell A1. Defaults to None.
    comment (dict, optional): Column name -> header comment. Defaults to None.
    numFmt (str, optional): Excel number format. Defaults to "#0.00".
    conditional_formatting (bool, optional): Colour scale red - yellow - green. Defaults to False.
    diagonal (bool, optional): Fill the diagonal of a square matrix red. Defaults to False.
    diagonal_length (int, optional): Number of diagonal cells filled. Defaults to the number of rows.

    Returns:
    None

    Examples:
    >>> with pd.ExcelWriter("out.xlsx", engine="xlsxwriter") as workbook:
    ...     excel_matrix(df_personality.select_dtypes("number").corr(), workbook, sheet="r",
    ...                  conditional_formatting=True, diagonal=True)
    """
    if kwargs:
        mapped = _legacy(kwargs, "excel_matrix")
        workbook, sheet = mapped.get("workbook", workbook), mapped.get("sheet", sheet)
        comment, numFmt = mapped.get("comment", comment), mapped.get("numFmt", numFmt)
    frame = _write_sheet(df, workbook, sheet)
    excel_generic_format(frame, workbook, sheet=sheet, title=title, comment=comment, numFmt=numFmt)
    worksheet = workbook.sheets[sheet]
    n_rows, n_cols = frame.shape
    if conditional_formatting and n_rows and n_cols:
        worksheet.conditional_format(1, 1, n_rows, n_cols, {
            "type": "3_color_scale", "min_color": "#FF0000", "mid_color": "#FFFF00", "max_color": "#00FF00"})
    if diagonal and n_rows == n_cols:
        red = workbook.book.add_format({"bg_color": "red"})
        for k in range(1, (n_rows if diagonal_length is None else diagonal_length) + 1):
            worksheet.conditional_format(k, k, k, k, {"type": "formula", "criteria": "TRUE", "format": red})
##########################################################################################
# CRITICAL VALUE EXCEL
##########################################################################################
def excel_critical_value(df, workbook, sheet="output", title=None, comment=None, numFmt="#0.00", critical=None,
                         **kwargs):
    """
    Write a data frame to a new worksheet with highlighted critical values,
    as rwf::excel_critical_value: the data with its row names, formatted by
    excel_generic_format, and for each column named in critical a
    conditional rule on its non-missing cells: one rule (e.g. "<0.05")
    fills matching cells red; two rules (e.g. ["<-1.96", ">1.96"]) fill the
    first red and the second purple.

    Parameters:
    df (pandas.DataFrame): The data.
    workbook (pandas.ExcelWriter): Open writer with the xlsxwriter engine.
    sheet (str, optional): Worksheet name. Defaults to "output".
    title (str, optional): Comment on cell A1. Defaults to None.
    comment (dict, optional): Column name -> header comment. Defaults to None.
    numFmt (str, optional): Excel number format. Defaults to "#0.00".
    critical (dict, optional): Column name -> rule or [rule, rule], Excel comparisons such as "<0.05". Defaults to None.

    Returns:
    None

    Examples:
    >>> with pd.ExcelWriter("out.xlsx", engine="xlsxwriter") as workbook:
    ...     excel_critical_value(df_insurance, workbook, sheet="critical",
    ...                          critical={"charges": ">20000", "age": ["<25", ">50"]})
    """
    if kwargs:
        mapped = _legacy(kwargs, "excel_critical_value")
        workbook, sheet = mapped.get("workbook", workbook), mapped.get("sheet", sheet)
        comment = mapped.get("comment", comment)
    frame = _write_sheet(df, workbook, sheet)
    excel_generic_format(frame, workbook, sheet=sheet, title=title, comment=comment, numFmt=numFmt)
    worksheet = workbook.sheets[sheet]
    for column, rules in (critical or {}).items():
        if column not in frame.columns:
            continue
        rules = list(rules) if isinstance(rules, (list, tuple)) else [rules]
        k = list(frame.columns).index(column) + 1
        values = pd.to_numeric(frame[column], errors="coerce") if pd.api.types.is_numeric_dtype(frame[column]) \
            else frame[column]
        for rule, color in zip(rules, ("red", "purple")):
            fmt = workbook.book.add_format({"bg_color": color, "num_format": numFmt})
            for row in np.flatnonzero(pd.notna(values).to_numpy()) + 1:
                cell = xl_rowcol_to_cell(row, k)
                worksheet.conditional_format(row, k, row, k, {"type": "formula", "criteria": f"={cell}{rule}",
                                                              "format": fmt})
##########################################################################################
# DATAFRAME TO EXCEL CONFUSION MATRIX
##########################################################################################
def excel_confusion_matrix(df, workbook, title="Rows: Expected Collumns: Observed"):
    """
    Write a confusion matrix of confusion_matrix_percent (counts with "sum"
    and "p" margins) to the worksheet "Confusion Matrix", as
    rwf::excel_confusion_matrix: whole-number format, a white-green colour
    scale over the counts, the "sum" margins in yellow with whole numbers
    and the "p" margins in yellow with 2 decimals, and title as a comment
    on A1.

    Parameters:
    df (pandas.DataFrame): The table of confusion_matrix_percent (text or numbers).
    workbook (pandas.ExcelWriter): Open writer with the xlsxwriter engine.
    title (str, optional): Comment on cell A1. Defaults to "Rows: Expected Collumns: Observed" (rwf's text).

    Returns:
    None

    Examples:
    >>> with pd.ExcelWriter("out.xlsx", engine="xlsxwriter") as workbook:
    ...     excel_confusion_matrix(confusion_matrix_percent([0, 1, 1, 0], [0, 1, 0, 0]), workbook)
    """
    sheet = "Confusion Matrix"
    frame = df.apply(pd.to_numeric, errors="coerce")
    frame = _write_sheet(frame, workbook, sheet)
    excel_generic_format(frame, workbook, sheet=sheet, title=title, numFmt="#0")
    worksheet = workbook.sheets[sheet]
    book = workbook.book
    n_rows, n_cols = frame.shape
    if n_rows > 2 and n_cols > 2:
        worksheet.conditional_format(1, 1, n_rows - 2, n_cols - 2,
                                     {"type": "2_color_scale", "min_color": "white", "max_color": "green"})
    base = {"border": 1, "border_color": "gray", "valign": "bottom", "bg_color": "yellow",
            "font_name": "Liberation Sans", "font_size": 10}
    whole = book.add_format({**base, "num_format": "#0"})
    decimal = book.add_format({**base, "num_format": "#0.00"})
    for i in range(n_rows):
        for j in range(n_cols):
            row_label, column_label = frame.index[i], frame.columns[j]
            if "p" in (row_label, column_label):
                fmt = decimal
            elif "sum" in (row_label, column_label):
                fmt = whole
            else:
                continue
            value = frame.iat[i, j]
            worksheet.write(i + 1, j + 1, None if pd.isna(value) else value, fmt)
##########################################################################################
# REPORT DATAFRAME
##########################################################################################
def report_dataframe(df, file=None, type="critical_value", **kwargs):
    """
    Write a data frame to a formatted Excel workbook, or return it unchanged,
    as rwf::report_dataframe.

    Parameters:
    df (pandas.DataFrame): Data to report.
    file (str, optional): Output filename without extension; ".xlsx" is
        appended and an existing file is replaced. If None (default),
        nothing is written and df is returned.
    type (str, optional): "critical_value" (default) writes the sheet with
        excel_critical_value, "matrix" with excel_matrix.
    **kwargs: Passed to excel_critical_value or excel_matrix (sheet, title,
        comment, numFmt, critical, conditional_formatting, ...).

    Returns:
    pandas.DataFrame or None: df when file is None, otherwise None (as in R).

    Examples:
    >>> report_dataframe(df_blood_pressure)
    >>> report_dataframe(df_blood_pressure, file="blood_pressure", sheet="data", critical={"bp_before": ">170"})
    """
    if file is None:
        return df
    filename = f"{file}.xlsx"
    if os.path.exists(filename):
        os.remove(filename)
    with pd.ExcelWriter(filename, engine="xlsxwriter") as workbook:
        if type == "critical_value":
            excel_critical_value(df, workbook, **kwargs)
        if type == "matrix":
            excel_matrix(df, workbook, **kwargs)
    return None
##########################################################################################
# DATA FRAME INDEX
##########################################################################################
def data_frame_index(nrow, ncol):
    """
    Row/column index pairs for every cell of an nrow x ncol table, column by
    column, 1-based (port of rwf::data_frame_index).

    Parameters:
    nrow (int): Number of rows.
    ncol (int): Number of columns.

    Returns:
    numpy.ndarray: nrow*ncol x 2 integer array with columns ri (row) and ci (column).

    Examples:
    >>> data_frame_index(3, 2)
    """
    ri = np.tile(np.arange(1, nrow + 1), ncol)
    ci = np.repeat(np.arange(1, ncol + 1), nrow)
    return np.column_stack([ri, ci])
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os

    df_blood_pressure = pd.read_csv("data/blood_pressure.csv") if os.path.exists("data/blood_pressure.csv") \
        else pd.read_csv("../data/blood_pressure.csv")
    df_personality = pd.read_csv("data/personality.csv") if os.path.exists("data/personality.csv") \
        else pd.read_csv("../data/personality.csv")

    try:
        from .functions_train_test import confusion_matrix_percent
    except ImportError:
        from functions_train_test import confusion_matrix_percent
    with pd.ExcelWriter("functions_excel_example.xlsx", engine="xlsxwriter") as workbook:
        excel_matrix(df_personality.select_dtypes("number").corr(), workbook, sheet="matrix",
                     title="Pearson correlation matrix", conditional_formatting=True, diagonal=True)
        excel_critical_value(df_blood_pressure, workbook, sheet="critical", comment={"sex": "M/F"},
                             critical={"bp_before": ["<110", ">160"]})
        excel_confusion_matrix(confusion_matrix_percent([0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
                                                        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]), workbook)
    print("saved functions_excel_example.xlsx (sheets: matrix, critical, Confusion Matrix)")
    report_dataframe(df_blood_pressure.head(), file="functions_excel_report", sheet="data")
