# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:45:39 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
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
def excel_generic_format(df, writer, sheetname, comments=None):
    import xlsxwriter
    """
    Writes a DataFrame to a new sheet in an existing Excel writer, with a
    header row, a fixed 2-decimal number format on numeric columns, a
    frozen header row, and optional comments on specific column headers.

    Args:
        df (pandas.DataFrame): Data to write. The row index is not written
            (index=False) — only the header row and data rows.
        writer (pandas.ExcelWriter): An already-open ExcelWriter using the
            xlsxwriter engine. `sheetname` must not already exist in it.
        sheetname (str): Name of the new worksheet to create.
        comments (dict, optional): Maps column name -> comment text, added
            to that column's header cell (row 1). Keyed by column name, not
            cell reference — unlike excel_matrix/critical_value_excel,
            which use cell references like "A1". Keys not found in
            df.columns are silently ignored. Defaults to None.

    Returns:
        None

    Notes:
        - Every numeric column gets the same fixed "0.00" number format;
          there's no way to change the decimal places or opt individual
          columns out, unlike excel_matrix's `decimals` argument.
        - Only the header row is frozen (freeze_panes(1, 0)) — no column
          is frozen, unlike excel_matrix, which freezes both.
        - Column widths are not auto-sized here, unlike excel_matrix,
          which calls get_col_widths.

    Examples:
        >>> writer = pd.ExcelWriter("out.xlsx", engine="xlsxwriter")
        >>> excel_generic_format(
        ...     df=df_blood_pressure, writer=writer, sheetname="ge",
        ...     comments={"sex": "M/F", "bp_before": "mmHg before treatment"},
        ... )
        >>> writer._save(); writer.close()
    """
    
    df.to_excel(writer, sheet_name=sheetname, index=False)
    workbook = writer.book 
    sheet = writer.sheets[sheetname]
    sheet.freeze_panes(1, 0)
    
    num_format = workbook.add_format({'num_format': '0.00'})

    # Add comments to header cells if column names match
    if comments:
        for col_idx, col_name in enumerate(df.columns):
            if col_name in comments:
                # Convert column index to Excel-style letter, e.g., 0 → A, 1 → B, etc.
                col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
                cell = f"{col_letter}1"  # Header is always in row 1
                sheet.write_comment(cell, comments[col_name], {'author': 'pwf'})
            # Apply number format if column is numeric
            if pd.api.types.is_numeric_dtype(df[col_name]):
                sheet.set_column(col_idx, col_idx, None, num_format)


# DATA_DIR = pathlib.Path("/home/dimitrios/GitHub/pwf") / "data"
# df_blood_pressure=pd.read_csv(DATA_DIR / "blood_pressure.csv")
# output_file='/home/dimitrios/GitHub/pwf/output/generic.xlsx'
# if os.path.exists(output_file):
#     os.remove(output_file)
# ge=pd.ExcelWriter(output_file,engine='xlsxwriter')
# comments = {
#     'sex': "This is a general comment for the sheet.",
#     'bp_before': "This cell contains important data."
# }
# excel_generic_format(df=df_blood_pressure,writer=ge,sheetname="ge",comments=comments)
# ge._save()
# ge.close()
# os.remove(output_file)
##########################################################################################
# MATRIX EXCEL
##########################################################################################
def excel_matrix(df,writer,sheetname,comments=None,decimals=2):
    """
    Writes a matrix (or any DataFrame) to a new sheet, formatted as a
    heatmap: fixed-decimal number format on every data column, and a
    diverging blue-white-red conditional color scale anchored at zero —
    intended for correlation matrices, but works as a general heatmap for
    any numeric matrix.

    Args:
        df (pandas.DataFrame): The data to write. Unlike excel_generic_format,
            the row index IS written (as the first column) — to_excel is
            called without index=False.
        writer (pandas.ExcelWriter): An already-open ExcelWriter using the
            xlsxwriter engine. `sheetname` must not already exist in it.
        sheetname (str): Name of the new worksheet to create.
        comments (dict, optional): Maps cell reference -> comment text,
            e.g. {"A1": "note"}. Keyed by cell reference, not column name —
            unlike excel_generic_format, which is keyed by column name.
            Defaults to None.
        decimals (int, optional): Number of decimal places for the number
            format applied to every data column (not the index column).
            Use 0 for whole numbers. Defaults to 2.

    Returns:
        None

    Notes:
        - Both the first row and first column are frozen
          (freeze_panes(1, 1)).
        - The color scale's midpoint is fixed at the value 0 (not at the
          data's own midpoint), so it correctly renders negative values as
          blue and positive as red regardless of the data's actual range —
          appropriate for correlation-like data bounded in [-1, 1]. For
          all-positive matrices (e.g. eigenvalues, factor variance) it
          still produces a sensible white-to-red gradient.
        - Column widths are not auto-sized here, unlike some of the other
          Excel helpers in this module that call get_col_widths.

    Examples:
        >>> writer = pd.ExcelWriter("out.xlsx", engine="xlsxwriter")
        >>> excel_matrix(
        ...     df=df_personality.select_dtypes("number").corr(),
        ...     writer=writer, sheetname="correlation",
        ...     comments={"A1": "Pearson correlation matrix"}, decimals=2,
        ... )
        >>> writer._save(); writer.close()
    """
    df.to_excel(writer,sheet_name=sheetname)
    workbook=writer.book
    sheet=writer.sheets[sheetname]
    sheet.freeze_panes(1,1)
    # Add comments to cells if provided
    if comments:
        for cell, comment in comments.items():
            sheet.write_comment(cell, comment, {'author': 'pwf'})
    num_format=workbook.add_format({'num_format': '0.'+'0'*decimals if decimals>0 else '0'})
    for col_idx in range(1,len(df.columns)+1):
        sheet.set_column(col_idx,col_idx,None,num_format)
    dimensions=np.array(df.shape)
    sheet.conditional_format(1,1,dimensions[0],dimensions[1], {
        'type': '3_color_scale',
        'min_color': '#2166AC',
        'mid_color': '#F7F7F7',
        'max_color': '#B2182B',
        'mid_type': 'num',
        'mid_value': 0,
    })

# DATA_DIR = pathlib.Path("/home/dimitrios/GitHub/pwf") / "data"
# df_personality=pd.read_csv(DATA_DIR / "personality.csv")
# output_file='/home/dimitrios/GitHub/pwf/output/matrix.xlsx'
# if os.path.exists(output_file):
#     os.remove(output_file)
# me=pd.ExcelWriter(output_file,engine='xlsxwriter')
# comments={'A1': "This is a general comment for the sheet.",
#           'B2': "This cell contains important data."}
# excel_matrix(df=pd.DataFrame(df_personality.corr()),
#              writer=me,
#              sheetname="me",
#              comments=comments)
# me._save()
# me.close()
# os.remove(output_file)
##########################################################################################
# CRITICAL VALUE EXCEL
##########################################################################################
def _parse_critical_expression(expr):
    """Split an Excel comparison expression like "<0.05" into (criteria, value)."""
    for op in (">=", "<=", "<>", ">", "<", "="):
        if expr.startswith(op):
            return op, float(expr[len(op):])
    raise ValueError(f"Unrecognized critical expression: {expr!r}")


def excel_critical_value(df,writer,sheetname,comments=None,critical=None):
    """
    Writes df to a new sheet via generic_format_excel, then highlights
    cells in specified columns that meet one or two threshold conditions
    (ported from R rwf::excel_critical_value).

    Args:
        df (pandas.DataFrame): The data to write.
        writer (pandas.ExcelWriter): An already-open ExcelWriter using the
            xlsxwriter engine.
        sheetname (str): Name of the new worksheet to create.
        comments (dict, optional): Forwarded to generic_format_excel —
            maps column name -> header comment text.
        critical (dict, optional): Maps column name -> either a single
            comparison expression string (e.g. "<0.05", ">20", "=0";
            matching cells highlighted red), or a 2-item list/tuple of
            expressions (e.g. [">20", "<11"]; first highlighted red,
            second purple). NaN cells in that column are skipped. Column
            names not found in df.columns are silently ignored.

    Returns:
        None

    Examples:
        >>> writer = pd.ExcelWriter("out.xlsx", engine="xlsxwriter")
        >>> excel_critical_value(
        ...     df=df_insurance, writer=writer, sheetname="critical",
        ...     critical={"charges": ">20000", "age": [">50", "<25"]},
        ... )
        >>> writer._save(); writer.close()
    """
    excel_generic_format(df,writer,sheetname,comments)
    workbook=writer.book
    sheet=writer.sheets[sheetname]
    colors=('red','purple')
    if critical:
        for col_name, expr in critical.items():
            if col_name not in df.columns:
                continue
            col_idx=df.columns.get_loc(col_name)
            exprs=expr if isinstance(expr,(list,tuple)) else [expr]
            row_positions=[i+1 for i,v in enumerate(df[col_name]) if pd.notna(v)]
            for e, color in zip(exprs, colors):
                criteria, value=_parse_critical_expression(e)
                fmt=workbook.add_format({'bg_color': color})
                for r in row_positions:
                    sheet.conditional_format(r,col_idx,r,col_idx, {'type':'cell','criteria':criteria,'value':value,'format':fmt})
##########################################################################################
# DATAFRAME TO EXCEL CONFUSION MATRIX
##########################################################################################
def excel_confusion_matrix(df,writer,sheetname="Confusion Matrix",title="Rows: Expected Columns: Observed"):
    """
    Writes a confusion matrix (as produced by confusion_matrix_percent,
    with "sum" and "p" margin row/column) to a new sheet: whole-number
    format plus a white-to-green colour scale on the core counts, and a
    yellow highlight on the "sum" (integer format) and "p" (decimal
    format) margins (ported from R rwf::excel_confusion_matrix).

    Args:
        df (pandas.DataFrame): Square confusion matrix with an appended
            "sum" row/column (totals) and "p" row/column (per-class
            accuracy/precision), as returned by confusion_matrix_percent.
        writer (pandas.ExcelWriter): An already-open ExcelWriter using the
            xlsxwriter engine. `sheetname` must not already exist in it.
        sheetname (str, optional): Name of the new worksheet. Defaults to
            "Confusion Matrix".
        title (str, optional): Comment written to cell A1. Defaults to
            "Rows: Expected Columns: Observed".

    Returns:
        None

    Notes:
        - Unlike excel_generic_format, this writes df with its row index
          intact (like excel_matrix), since the "sum"/"p" row and column
          labels are needed to tell the margins apart from the core counts.

    Examples:
        >>> cm = confusion_matrix_percent(observed, predicted)
        >>> writer = pd.ExcelWriter("out.xlsx", engine="xlsxwriter")
        >>> excel_confusion_matrix(cm, writer)
        >>> writer._save(); writer.close()
    """
    df=df.astype(float)
    df.to_excel(writer,sheet_name=sheetname)
    workbook=writer.book
    sheet=writer.sheets[sheetname]
    sheet.freeze_panes(1,1)
    if title:
        sheet.write_comment('A1',title,{'author': 'pwf'})

    int_format=workbook.add_format({'num_format': '#0'})
    for col_idx, width in enumerate(get_col_widths(df)):
        sheet.set_column(col_idx,col_idx,width,int_format if col_idx>0 else None)

    nrow, ncol=df.shape
    core_rows=[i for i,lbl in enumerate(df.index) if lbl not in ('sum','p')]
    core_cols=[j for j,lbl in enumerate(df.columns) if lbl not in ('sum','p')]
    if core_rows and core_cols:
        sheet.conditional_format(min(core_rows)+1,min(core_cols)+1,max(core_rows)+1,max(core_cols)+1, {
            'type': '2_color_scale', 'min_color': 'white', 'max_color': 'green',
        })

    yellow_int_fmt=workbook.add_format({'border':1,'border_color':'gray','num_format':'#0','bg_color':'yellow'})
    yellow_dec_fmt=workbook.add_format({'border':1,'border_color':'gray','num_format':'#0.00','bg_color':'yellow'})

    for i, lbl in enumerate(df.index):
        fmt=yellow_int_fmt if lbl=='sum' else (yellow_dec_fmt if lbl=='p' else None)
        if fmt is not None:
            for j in range(ncol):
                sheet.write(i+1,j+1,df.iat[i,j],fmt)
    for j, lbl in enumerate(df.columns):
        fmt=yellow_int_fmt if lbl=='sum' else (yellow_dec_fmt if lbl=='p' else None)
        if fmt is not None:
            for i in range(nrow):
                sheet.write(i+1,j+1,df.iat[i,j],fmt)



































































