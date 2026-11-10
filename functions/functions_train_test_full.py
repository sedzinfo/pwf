# -*- coding: utf-8 -*-
"""
Deprecated alias of functions_train_test (the port of rwf's
FUNCTIONS_TRAIN_TEST.R). Import from functions_train_test instead.
"""
try:
    from .functions_train_test import *  # noqa: F401,F403
except ImportError:
    from functions_train_test import *  # noqa: F401,F403
