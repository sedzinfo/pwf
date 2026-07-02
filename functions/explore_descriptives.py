# -*- coding: utf-8 -*-
"""
Small descriptive-statistics helpers.

Fixed a real bug: describe_by_mean used pandas' private/internal
`DataFrame._append`, which was removed in pandas 3.0 (the public
`.append()` method was removed in pandas 2.0, and `_append` was never a
stable API) -- replaced with `pd.concat()`.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import pandas as pd
##########################################################################################
# FLATTEN
##########################################################################################
def flatten(l):
    """
    Flatten a list of lists into a single flat list.

    Parameters:
    l (list of list): Nested list.

    Returns:
    list: Flattened list.

    Examples:
    >>> flatten([[1, 2], [3], [4, 5]])
    """
    return [item for sublist in l for item in sublist]
##########################################################################################
# DESCRIBE BY MEAN
##########################################################################################
def describe_by_mean(df, factorname=[]):
    """
    Mean/min/max/SD of every numeric column, grouped separately by each
    of several factor columns.

    Parameters:
    df (pandas.DataFrame): Data containing the factor and numeric columns.
    factorname (list of str, optional): Grouping column names. Each is
        summarized independently (not crossed). Defaults to [].

    Returns:
    pandas.DataFrame: Columns Statistic, Factor, Level, then one column
    per numeric variable in `df`.

    Examples:
    >>> import pandas as pd
    >>> titanic = pd.read_csv("data/titanic.csv")
    >>> describe_by_mean(df=titanic, factorname=["survived", "pclass", "sex"])
    """
    result = pd.DataFrame()
    factor = []
    for category in factorname:
        df_mean = df.groupby(category).mean(numeric_only=True)
        df_min = df.groupby(category).min(numeric_only=True)
        df_max = df.groupby(category).max(numeric_only=True)
        df_std = df.groupby(category).std(numeric_only=True)
        df_mean = df_mean.assign(Statistic=["Mean"] * df_mean.shape[0])
        df_min = df_min.assign(Statistic=["Min"] * df_min.shape[0])
        df_max = df_max.assign(Statistic=["Max"] * df_max.shape[0])
        df_std = df_std.assign(Statistic=["SD"] * df_std.shape[0])
        descriptives = pd.concat([df_mean, df_min, df_max, df_std])
        descriptives.reset_index(inplace=True)
        descriptives.rename(columns={list(descriptives)[0]: 'Level'}, inplace=True)
        factor.append([category] * descriptives.shape[0])
        result = pd.concat([result, descriptives], ignore_index=True)
    result.insert(0, "Factor", flatten(factor))
    result.insert(0, "Statistic", result.pop("Statistic"))
    result = result.sort_values(by=["Statistic", "Factor", "Level"])
    return result
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import os

    print("=" * 80, "\nflatten\n", "=" * 80, sep="")
    print(flatten([[1, 2], [3], [4, 5]]))

    print("\n" + "=" * 80, "\ndescribe_by_mean\n", "=" * 80, sep="")
    titanic = pd.read_csv("data/titanic.csv") if os.path.exists("data/titanic.csv") \
        else pd.read_csv("../data/titanic.csv")
    result = describe_by_mean(df=titanic, factorname=["survived", "pclass", "sex"])
    print(result)







