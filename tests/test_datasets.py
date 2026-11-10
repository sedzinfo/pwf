import numpy as np
import pandas as pd
import pytest

from pwf import datasets


@pytest.mark.parametrize("name", datasets.DATASETS)
def test_load_dataset_reads_like_r(reference, name):
    summary = reference("datasets_summary")
    expected = summary[summary["dataset"] == name]
    df = datasets.load_dataset(f"df_{name}")
    assert list(df.columns) == expected["column"].tolist()
    numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns]
    assert numeric == expected["numeric"].tolist()
    assert df.isna().sum().tolist() == expected["missing"].astype(int).tolist()
    totals = [df[c].sum() if is_number else np.nan for c, is_number in zip(df.columns, numeric)]
    np.testing.assert_allclose(totals, expected["total"].to_numpy(float), rtol=1e-12)


def test_extra_datasets_and_errors():
    assert datasets.load_dataset("iris").shape == (150, 5)
    assert "tirt_triplets" in datasets.list_datasets()
    with pytest.raises(ValueError):
        datasets.load_dataset("nothing")
