import logging
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    pos_label: str = "positive",
    return_confusion_matrix: bool = True,
) -> Dict[str, float]:
    """
    Evaluates the predictions against the true labels.

    Args:
        y_true (List[str]): True sentiment labels.
        y_pred (List[str]): Predicted sentiment labels.
        pos_label (str): The label considered as positive.
        return_confusion_matrix (bool): Whether to include the confusion matrix.

    Returns:
        Dict[str, float]: A dictionary of evaluation metrics.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred,  pos_label=pos_label, zero_division=0, average='binary')
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0, average='binary')
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0, average='binary')
        cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])

        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

        if return_confusion_matrix:
            metrics["confusion_matrix"] = cm

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluate_predictions: {e}")
        raise