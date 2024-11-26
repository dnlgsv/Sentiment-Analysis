import tempfile
from unittest.mock import patch

import pandas as pd
from datasets import Dataset

from src.dataset import load_and_prepare_dataset


@patch("src.dataset.load_dataset")
def test_load_and_prepare_dataset(mock_load_dataset):
    # create a mock dataset
    data = {"text": ["Sample text"] * 2000, "label": [0] * 1000 + [1] * 1000}
    mock_df = pd.DataFrame(data)
    mock_dataset = Dataset.from_pandas(mock_df)
    mock_load_dataset.return_value = mock_dataset

    with tempfile.NamedTemporaryFile() as temp_file:
        df_subset = load_and_prepare_dataset(
            dataset_name="mock-dataset",
            subset_size=1000,
            output_path=temp_file.name,
            split="test",
            random_state=42,
        )

        assert len(df_subset) == 1000, "Subset size should be 1000"
        assert (
            df_subset["label"].value_counts()[0] == 500
        ), "Should have 500 samples with label 0"
        assert (
            df_subset["label"].value_counts()[1] == 500
        ), "Should have 500 samples with label 1"
