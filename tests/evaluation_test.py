import logging
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    pos_label: str = "Positive",
    return_confusion_matrix: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the predictions against the true labels.

    Args:
        y_true (List[str]): True sentiment labels.
        y_pred (List[str]): Predicted sentiment labels.
        pos_label (str): The label considered as positive.
        return_confusion_matrix (bool): Whether to include the confusion matrix.

    Returns:
        Dict[str, Any]: A dictionary of evaluation metrics.
    """
    try:
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

        # Include confusion matrix if requested
        if return_confusion_matrix:
            labels = ["Positive", "Negative"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            metrics["confusion_matrix"] = cm.tolist()  # Convert numpy array to list for consistency

        return metrics

    except Exception as e:
        logging.error(f"Error in evaluate_predictions: {e}")
        raise
