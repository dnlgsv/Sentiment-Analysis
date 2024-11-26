import logging
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_metrics(
    metrics_df: pd.DataFrame, save_path: str = "./results/metrics_plot.png"
) -> None:
    """
    Plots evaluation metrics for different models and prompts, with bars sorted within each metric.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.
        save_path (str): Path to save the plot image.
    """
    try:
        # Exclude non-numeric or non-plot-friendly columns
        exclude_columns = ["confusion_matrix"]
        metrics_df_filtered = metrics_df.drop(columns=exclude_columns, errors="ignore")

        # Combine 'Model' and 'Prompt' into a single identifier
        metrics_df_filtered["Model_Prompt"] = (
            metrics_df_filtered["Model"] + "_" + metrics_df_filtered["Prompt"]
        )

        # Exclude non-metric columns to get metric columns
        non_metric_columns = ["Model", "Prompt", "Model_Prompt"]
        metric_columns = [
            col for col in metrics_df_filtered.columns if col not in non_metric_columns
        ]

        # Verify that metric_columns is not empty
        if not metric_columns:
            logger.error("No metric columns found for plotting.")
            return

        # Convert metric columns to numeric, handle errors
        for col in metric_columns:
            metrics_df_filtered[col] = pd.to_numeric(
                metrics_df_filtered[col], errors="coerce"
            )

        # Drop rows with NaN values in metric columns
        metrics_df_filtered.dropna(subset=metric_columns, inplace=True)

        # Reshape dataframe for plotting
        metrics_melted = metrics_df_filtered.melt(
            id_vars=["Model_Prompt"],
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Value",
        )

        # Check if metrics_melted is empty
        if metrics_melted.empty:
            logger.error("No data to plot. The melted DataFrame is empty.")
            return

        # Sort the data within each metric
        metrics_melted["Metric"] = metrics_melted["Metric"].astype(str)
        sorted_data = metrics_melted.copy()
        sorted_data["Model_Prompt"] = sorted_data["Model_Prompt"].astype(str)

        # Create a FacetGrid to plot each metric separately
        g = sns.FacetGrid(sorted_data, col="Metric", sharey=False, height=6, aspect=1)

        # For each subplot, sort the bars within the metric
        g.map_dataframe(
            sns.barplot,
            x="Value",
            y="Model_Prompt",
            order=sorted_data.sort_values("Value", ascending=True)["Model_Prompt"],
            palette="viridis",
        )

        # Adjust the titles and labels
        g.set_titles("{col_name}")
        g.set_axis_labels("Value", "Model_Prompt")

        # Adjust layout
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(
            "Model Performance Metrics Sorted within Each Metric", fontsize=16
        )

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved metrics plot to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_metrics: {e}")
        raise


def plot_confusion_matrix(
    cm: List[List[int]], model: str, prompt: str, save_path: str = "../results/"
) -> None:
    """
    Plots and saves the confusion matrix.

    Args:
        cm (list): Confusion matrix.
        model (str): Model name.
        prompt (str): Prompt name.
        save_path (str): Path to save the confusion matrix image.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Positive", "Negative"]
        )
        disp.plot(cmap=plt.cm.Blues)  # type: ignore[attr-defined]
        plt.title(f"Confusion Matrix for {model} with {prompt} Prompt")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_confusion_matrix: {e}")
        raise
