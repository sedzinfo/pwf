# -*- coding: utf-8 -*-
"""
The datasets of rwf (R's data(package = "rwf"): df_admission, df_insurance,
...), as pandas data frames.

load_dataset("insurance") (or "df_insurance") reads the CSV shipped with
pwf the way rwf's data were read in R, utils::read.csv: "NA" is missing,
empty text stays an empty string, and a column whose values are all
numbers (blanks counting as missing) is numeric. df_ocean is
tab-delimited (R's read.delim).

Besides rwf's datasets, pwf ships infert and iris (R's datasets package)
and thurstonianIRT's triplets (tirt_triplets) for its examples.
"""
from importlib import resources
from pathlib import Path

import pandas as pd

DATASETS = ["admission", "automotive_data", "blood_pressure", "co2", "crop_yield", "difficile", "insurance",
            "ocean", "personality", "responses", "responses_state", "sexual_comp", "titanic"]
EXTRA_DATASETS = ["infert", "iris", "tirt_triplets"]


def _data_dir():
    """The data folder: inside the installed package, or the repository's data/ next to the sources."""
    try:
        packaged = resources.files(__package__ or "pwf") / "data"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, TypeError):
        pass
    here = Path(__file__).resolve().parent
    for candidate in (here / "data", here.parent / "data", here.parent.parent / "data"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("pwf's data folder was not found")


def _r_read_csv(path, sep=","):
    """utils::read.csv: "NA" is missing, empty text stays "", columns of numbers (and blanks) are numeric."""
    frame = pd.read_csv(path, sep=sep, keep_default_na=False, na_values=["NA"])
    for column in frame.columns:
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column]):
            values = frame[column].replace("", pd.NA)
            numbers = pd.to_numeric(values.dropna(), errors="coerce")
            if numbers.notna().all() and len(numbers):
                frame[column] = pd.to_numeric(values, errors="coerce")
    return frame


def load_dataset(name):
    """
    One of rwf's datasets (or pwf's extra example data) as a DataFrame.

    Parameters:
    name (str): Dataset name with or without rwf's "df_" prefix: admission,
        automotive_data, blood_pressure, co2, crop_yield, difficile,
        insurance, ocean, personality, responses, responses_state,
        sexual_comp, titanic; or infert, iris, tirt_triplets.

    Returns:
    pandas.DataFrame: The data.

    Examples:
    >>> df_insurance = load_dataset("df_insurance")
    >>> df_ocean = load_dataset("ocean")
    """
    key = name[3:] if name.startswith("df_") else name
    if key not in DATASETS + EXTRA_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(DATASETS + EXTRA_DATASETS)}")
    return _r_read_csv(_data_dir() / f"{key}.csv", sep="\t" if key == "ocean" else ",")


def data_path(filename):
    """
    Path of a file in pwf's data folder, e.g. data_path("tirt_lavaan_model.txt")
    (thurstonianIRT's lavaan syntax for tirt_triplets) or data_path("ocean_codebook.txt").

    Parameters:
    filename (str): File name.

    Returns:
    pathlib.Path: The path.
    """
    return _data_dir() / filename


def list_datasets():
    """Names of the available datasets (rwf's first)."""
    return DATASETS + EXTRA_DATASETS
