import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_metrics(metrics_df: pd.DataFrame, save_path: str):
    """
    Plots evaluation metrics for different models and prompts.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.
        save_path (str): Path to save the plot image.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metrics_melted = metrics_df.melt(
            id_vars=["Model", "Prompt"], var_name="Metric", value_name="Value"
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_melted)
        plt.title("Model Performance Metrics")
        plt.ylim(0, 1)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved metrics plot to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_metrics: {e}")
        raise


def plot_confusion_matrix(cm: list, model: str, prompt: str, save_path: str):
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
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {model} with {prompt} Prompt")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_confusion_matrix: {e}")
        raise
