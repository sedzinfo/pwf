# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:45:39 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import sys
sys.path.insert(1,'/opt/pyrepo/functions/')
from __init__ import *
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
    return [idx_max]+[max([len(str(s)) for s in df[col].values]+[len(col)]) for col in df.columns]
##########################################################################################
# GENERIC FORMAT EXCEL
##########################################################################################
def generic_format_excel(df,writer,sheetname,comment=""):
    """
    Writes the provided data to an Excel file using the given writer object and applies formatting to the worksheet.

    Args:
        df (pandas.DataFrame): The data to be written to the Excel file.
        writer: The Excel writer object to use for writing the data.
        sheetname (str): The name of the sheet to write the data to.
        comment (str, optional): A comment to be added to cell A1 in the sheet. Defaults to "".

    Returns:
        None

    Notes:
        - The function uses the `to_excel` method of the `df` DataFrame to write the data to the Excel file.
        - It retrieves the worksheet object corresponding to the specified sheetname from the writer object.
        - The function freezes the panes in the worksheet at the first row and column (1, 1).
        - If a non-empty comment is provided, it is written to cell A1 in the worksheet.
        - The function iterates over each column in the DataFrame and sets the width of the corresponding column in the worksheet.
        - The column widths are determined by the `get_col_widths` function, which should be defined elsewhere.

    """
    df.to_excel(writer,sheetname)
    worksheet=writer.sheets[sheetname]
    worksheet.freeze_panes(1,1)
    if len(comment)>0:
        worksheet.write_comment('A1',comment)
    for i, width in enumerate(get_col_widths(df)):
        worksheet.set_column(i,i,width)
# writer_generic_format_excel=pd.ExcelWriter('/opt/pyrepo/output/xlsxwriter_generic_format_excel.xlsx',engine='xlsxwriter')
# import pandas as pd
# personality=pd.read_csv("/opt/pyrepo/data/personality.csv")
# generic_format_excel(personality,writer_generic_format_excel,"sheet name","test general comment")
# writer_generic_format_excel._save()
##########################################################################################
# MATRIX EXCEL
##########################################################################################
def matrix_excel(df,writer,sheetname,comment=""):
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
        - The function also adjusts the column widths based on the data in `mydata` using the `set_column` method of the worksheet.

    """
    generic_format_excel(df,writer,sheetname,comment)
    worksheet=writer.sheets[sheetname]
    dimensions=np.array(df.shape)
    worksheet.conditional_format(1,1,dimensions[0],dimensions[1], {'type': '2_color_scale'})
# writer_matrix_excel=pd.ExcelWriter('/opt/pyrepo/output/xlsxwriter_matrix_excel.xlsx',engine='xlsxwriter')
# import pandas as pd
# personality=pd.read_csv("/opt/pyrepo/data/personality.csv")
# matrix_excel(personality,writer_matrix_excel,"sheet name","test general comment")
# writer_matrix_excel._save()
##########################################################################################
# CRITICAL VALUE EXCEL
##########################################################################################
def critical_value_excel(df,writer,sheetname,comment="",critical_collumn="",rule="<",value=""):
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
        - The function retrieves the worksheet object corresponding to the specified sheetname.
        - It creates a format object with a red fill color and adds it to the workbook.
        - The dimensions of the `mydata` DataFrame are used to determine the range for conditional formatting.
        - If a `rule` is provided, the function applies conditional formatting to the specified critical column.
        - The formatting is applied based on the `rule`, `value`, and `format_red_fill` parameters.

    """
    generic_format_excel(df,writer,sheetname,comment)
    workbook=writer.book
    worksheet=writer.sheets[sheetname]
    format_red_fill=workbook.add_format()
    format_red_fill.set_bg_color('red')
    dimensions=np.array(df.shape)
    if len(rule)>0:
        worksheet.conditional_format(1,critical_collumn,dimensions[0],critical_collumn, {'type':'cell','criteria':rule,'value':value,'format':format_red_fill})
# writer_critical_value_excel=pd.ExcelWriter('/opt/pyrepo/output/xlsxwriter_critical_value_excel.xlsx',engine='xlsxwriter')
# import pandas as pd
# personality=pd.read_csv("/opt/pyrepo/data/personality.csv")
# critical_value_excel(personality,writer_critical_value_excel,"DATA",comment="Test Comment",critical_collumn=1,rule="<",value=5)
# writer_critical_value_excel._save()
