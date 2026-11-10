# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_KEYS.R.

Note on questions_by_keys: rwf computes `which(match(key, i) %in% key)`.
match(key, i) is 1 where key == i (and missing elsewhere), so the result is
which(key == i) whenever the value 1 occurs in `key` (the documented case:
dimensions numbered 1..max(key)); when 1 does not occur, 1 %in% key is FALSE
and every dimension comes back empty. The port reproduces both cases.
rwf returns an unnamed list indexed 1..max(key); the port returns a dict
keyed 1..max(key), so key_list[i] means the same in both.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
##########################################################################################
# KEYS
##########################################################################################
def questions_by_keys(key):
    """
    Convert a key vector to a mapping of dimension -> question indices.

    Parameters:
    key (array-like of int): Each element indicates which dimension the
        corresponding question belongs to. Values must be consecutive
        integers starting from 1 up to the number of dimensions.

    Returns:
    dict: {dimension_number: numpy.ndarray of 1-based question indices},
    one entry per dimension from 1 to max(key) (rwf returns the same as an
    unnamed list). As in rwf, if the value 1 does not occur in `key`, every
    dimension is empty.

    Examples:
    >>> key = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    >>> questions_by_keys(key)
    """
    key = np.asarray(key, dtype=float)
    has_one = bool(np.any(key == 1))
    return {i: np.flatnonzero((key == i) & has_one) + 1 for i in range(1, int(np.max(key)) + 1)}
##########################################################################################
# KEYS
##########################################################################################
def questions_dimensions_dataframe(key, dimensions, elaborate_dimensions, questions):
    """
    Build a question-to-dimension mapping table: one row per question,
    with its order within its dimension, the dimension's short name and
    full description, and the question's own label.

    Parameters:
    key (array-like of int): Each element indicates which dimension the
        corresponding question belongs to (see questions_by_keys).
    dimensions (list of str): Short dimension names, one per dimension;
        length must equal max(key).
    elaborate_dimensions (list of str): Full dimension descriptions, one
        per dimension; length must equal max(key).
    questions (list of str): Question labels in the same order as key;
        length must equal len(key).

    Returns:
    pandas.DataFrame: One row per question, columns "ORDER" (the
    question's 1-based position within the overall key vector),
    "DIMENSION", "ELABORATE DIMENSION", "QUESTION".

    Examples:
    >>> key = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    >>> dimensions = [f"Dimension{i}" for i in range(1, 6)]
    >>> elaborate_dimensions = [f"Elaborated_Dimension{i}" for i in range(1, 6)]
    >>> questions = [f"Question{i}" for i in range(1, 11)]
    >>> questions_dimensions_dataframe(key, dimensions, elaborate_dimensions, questions)
    """
    key_list = questions_by_keys(key)
    blocks = []
    for i in sorted(key_list.keys()):
        order = key_list[i]
        n = len(order)
        blocks.append(pd.DataFrame({
            'ORDER': order,
            'DIMENSION': [dimensions[i - 1]] * n,
            'ELABORATE DIMENSION': [elaborate_dimensions[i - 1]] * n,
            'QUESTION': [questions[j - 1] for j in order],
        }))
    return pd.concat(blocks, ignore_index=True)
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    key = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    print("=" * 80, "\nquestions_by_keys\n", "=" * 80, sep="")
    print(questions_by_keys(key))

    print("\n" + "=" * 80, "\nquestions_dimensions_dataframe\n", "=" * 80, sep="")
    dimensions = [f"Dimension{i}" for i in range(1, 6)]
    elaborate_dimensions = [f"Elaborated_Dimension{i}" for i in range(1, 6)]
    questions = [f"Question{i}" for i in range(1, 11)]
    print(questions_dimensions_dataframe(key, dimensions, elaborate_dimensions, questions))
