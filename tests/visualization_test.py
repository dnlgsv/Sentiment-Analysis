import os

import numpy as np
import pandas as pd

from src.visualization import plot_confusion_matrix, plot_metrics


def test_plot_metrics(tmp_path):
    """
    Test the plot_metrics function by generating a sample plot.
    """
    metrics_data = {
        "Model": ["1.5B", "0.5B"],
        "Prompt": ["simple", "simple"],
        "Accuracy": [0.85, 0.75],
        "Precision": [0.80, 0.70],
        "Recall": [0.90, 0.80],
        "F1-Score": [0.85, 0.75],
    }
    metrics_df = pd.DataFrame(metrics_data)
    save_path = os.path.join(tmp_path, "metrics.png")

    plot_metrics(metrics_df, save_path)
    assert os.path.exists(save_path), "Metrics plot should be saved"


def test_plot_confusion_matrix(tmp_path):
    """
    Test the plot_confusion_matrix function by generating a sample confusion matrix.
    """
    cm = np.array([[50, 10], [5, 35]])
    model = "1.5B"
    prompt = "simple"
    save_path = os.path.join(tmp_path, "cm.png")

    plot_confusion_matrix(cm, model, prompt, save_path)
    assert os.path.exists(save_path), "Confusion matrix plot should be saved"
