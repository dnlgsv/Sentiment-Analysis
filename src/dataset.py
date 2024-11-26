import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset


def load_and_prepare_dataset(
    dataset_name: str = "ajaykarthick/imdb-movie-reviews",
    subset_size: int = 500,
    output_path: str = "./data/imdb_movie_reviews_subset.csv",
    split: str = "test",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads the dataset and selects a balanced subset.

    Args:
        dataset_name (str): The name of the dataset to load.
        subset_size (int): Total number of samples in the subset.
        output_path (str): Path to save the subset CSV.
        split (str): The dataset split to use ('train', 'test').
        random_state (Optional[int]): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The prepared subset of the dataset.
    """

    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()

        # Check label types and adjust if necessary
        unique_labels = df["label"].unique()
        logger.info(f"Unique labels found: {unique_labels}")

        if set(unique_labels) == {0, 1} or set(unique_labels) == {1, 0}:
            positive_label, negative_label = 0, 1
        else:
            raise ValueError("Unexpected label values in the dataset.")

        logger.info("Preparing balanced subset")
        half_size = subset_size // 2
        df_positive = df[df["label"] == positive_label].sample(
            n=half_size, random_state=random_state
        )
        df_negative = df[df["label"] == negative_label].sample(
            n=half_size, random_state=random_state
        )
        df_subset = (
            pd.concat([df_positive, df_negative])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

        logger.info(f"Saving subset to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_subset.to_csv(output_path, index=False)

        logger.info("Dataset preparation completed successfully.")
        return df_subset

    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Load and prepare a balanced subset of a dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ajaykarthick/imdb-movie-reviews",
        help="The name of the dataset to load.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help="Total number of samples in the subset (must be even).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/imdb_movie_reviews_subset.csv",
        help="Path to save the subset CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The dataset split to use ('train', 'test', etc.).",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    try:
        load_and_prepare_dataset(
            dataset_name=args.dataset_name,
            subset_size=args.subset_size,
            output_path=args.output_path,
            split=args.split,
            random_state=args.random_state,
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(1)
