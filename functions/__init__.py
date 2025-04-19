#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:20:31 2017
@author: Dimitrios Zacharatos
"""

from . import functions_excel
from .functions_excel import get_col_widths, generic_format_excel, matrix_excel, critical_value_excel
__all__ = ['functions_excel', 'get_col_widths', 'generic_format_excel','matrix_excel', 'critical_value_excel']
