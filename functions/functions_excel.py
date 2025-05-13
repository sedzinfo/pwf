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
    result=[idx_max]+[max([len(str(s)) for s in df[col].values]+[len(col)]) for col in df.columns]
    return result
##########################################################################################
# GENERIC FORMAT EXCEL
##########################################################################################
def generic_format_excel(df,writer,sheetname,comments=None):
    """
    Writes data to an Excel file.

    Args:
        df (pandas.DataFrame): The data to be written to the Excel file.
        writer: The Excel writer object to use for writing the data.
        sheetname (str): The name of the sheet to write the data to.
        comment (str, optional): A comment to be added to cell A1 in the sheet. Defaults to "".

    Returns:
        None

    Notes:
        - The function uses the `to_excel` method of the `df` DataFrame to write the data to the Excel file.
        - It retrieves the sheet object corresponding to the specified sheetname from the writer object.
        - The function freezes the panes in the sheet at the first row and column (1, 1).
        - If a non-empty comment is provided, it is written to cell A1 in the sheet.
        - The function iterates over each column in the DataFrame and sets the width of the corresponding column in the sheet.
        - The column widths are determined by the `get_col_widths` function, which should be defined elsewhere.

    """
    df.to_excel(writer,sheetname)
    sheet=writer.sheets[sheetname]
    sheet.freeze_panes(1,1)
    if comments:
       for cell, comment in comments.items():
            sheet.write_comment(cell, comment)
    for i, width in enumerate(get_col_widths(df)):
        sheet.set_column(i,i,width)

def generic_format_excel(df, writer, sheetname, comments=None):
    import xlsxwriter
    """
    Writes a DataFrame to an Excel sheet with formatting and optional column header comments.

    Args:
        df (pandas.DataFrame): Data to write.
        writer (pandas.ExcelWriter): ExcelWriter object.
        sheetname (str): Sheet name.
        comments (dict): Optional. Dict of {column_name: comment_text}.

    Returns:
        None
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
                sheet.write_comment(cell, comments[col_name])
            # Apply number format if column is numeric
            if pd.api.types.is_numeric_dtype(df[col_name]):
                sheet.set_column(col_idx, col_idx, None, num_format)

# output_file=path_root+'/output/generic.xlsx'
# if os.path.exists(output_file):
#     os.remove(output_file)
# ge=pd.ExcelWriter(output_file,engine='xlsxwriter')
# comments = {'A1': "This is a general comment for the sheet.",
#             'A2': "This cell contains important data."}
# generic_format_excel(df=df_personality,writer=ge,sheetname="ge",comments=comments)
# ge._save()
# ge.close()
# os.remove(output_file)
##########################################################################################
# MATRIX EXCEL
##########################################################################################
def matrix_excel(df,writer,sheetname,comments=None):
    """
    Writes data to an Excel file using the given writer object and formats the sheet.

    Args:
        df (pandas.DataFrame): The data to be written to the Excel file.
        writer: The Excel writer object to use for writing the data.
        sheetname (str): The name of the sheet to write the data to.
        comment (str, optional): A comment to be added to cell A1 in the sheet. Defaults to "".

    Returns:
        None

    Notes:
        - The function uses the `to_excel` method of the `df` DataFrame to write the data to the Excel file.
        - The sheet is formatted with frozen panes, where the first row and column are frozen.
        - If a comment is provided, it is added to cell A1 in the sheet.
        - The function also adjusts the column widths based on the data in `mydata` using the `set_column` method of the sheet.

    """
    df.to_excel(writer,sheetname)
    sheet=writer.sheets[sheetname]
    sheet.freeze_panes(1,1)
    # Add comments to cells if provided
    if comments:
        for cell, comment in comments.items():
            sheet.write_comment(cell, comment)
    dimensions=np.array(df.shape)
    sheet.conditional_format(1,1,dimensions[0],dimensions[1], {'type': '2_color_scale', 'min_color': 'yellow', 'max_color': 'green'})

# output_file=path_root+'/output/matrix.xlsx'
# if os.path.exists(output_file):
#     os.remove(output_file)
# me=pd.ExcelWriter(output_file,engine='xlsxwriter')
# comments={'A1': "This is a general comment for the sheet.",
#           'B2': "This cell contains important data."}
# matrix_excel(df=pd.DataFrame(df_personality.corr()),
#              writer=me,
#              sheetname="me",
#              comments=comments)
# me._save()
# me.close()
# os.remove(output_file)
##########################################################################################
# CRITICAL VALUE EXCEL
##########################################################################################
def critical_value_excel(df,writer,sheetname,comments="",critical_collumn="",rule="<",value=""):
    """
    Writes data to an Excel file using the given writer object and applies conditional formatting to the specified column.

    Args:
        df (pandas.DataFrame): The data to be written to the Excel file.
        writer: The Excel writer object to use for writing the data.
        sheetname (str): The name of the sheet to write the data to.
        comment (str): A comment to be added to cell A1 in the sheet.

    Returns:
        None

    Notes:
        - The function uses the `to_excel` method of the `mydata` DataFrame to write the data to the Excel file.
        - The function assumes that the writer object has an associated workbook.
        - The function retrieves the sheet object corresponding to the specified sheetname.
        - It creates a format object with a red fill color and adds it to the workbook.
        - The dimensions of the `mydata` DataFrame are used to determine the range for conditional formatting.
        - If a `rule` is provided, the function applies conditional formatting to the specified critical column.
        - The formatting is applied based on the `rule`, `value`, and `format_red_fill` parameters.

    """
    generic_format_excel(df,writer,sheetname,comments)
    workbook=writer.book
    sheet=writer.sheets[sheetname]
    format_red_fill=workbook.add_format()
    format_red_fill.set_bg_color('red')
    dimensions=np.array(df.shape)
    if len(rule)>0:
        sheet.conditional_format(1,critical_collumn,dimensions[0],critical_collumn, {'type':'cell','criteria':rule,'value':value,'format':format_red_fill})

# personality=pd.read_csv(path_root+"/data/personality.csv")
# output_file=path_root+'/output/critical.xlsx'
# if os.path.exists(output_file):
#     os.remove(output_file)
# cv=pd.ExcelWriter(output_file,engine='xlsxwriter')
# comments={'A1': "This is a general comment for the sheet.",
#           'B2': "This cell contains important data."}
# critical_value_excel(df=personality,writer=cv,sheetname="cv",comments=None,critical_collumn=1,rule="<",value=5)
# cv._save()
# cv.close()
# os.remove(output_file)
































































