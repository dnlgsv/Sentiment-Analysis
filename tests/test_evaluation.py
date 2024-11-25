from src.evaluation import evaluate_predictions


def test_evaluate_predictions():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "positive", "positive", "negative"]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["Accuracy"] == 0.75, "Accuracy should be 0.75"
    assert metrics["Precision"] == 0.6666666666666666, "Precision should be 2/3"
    assert metrics["Recall"] == 1.0, "Recall should be 1.0"
    assert metrics["F1-Score"] == 0.8, "F1-Score should be 0.8"

    metrics_with_cm = evaluate_predictions(y_true, y_pred, return_confusion_matrix=True)
    assert "confusion_matrix" in metrics_with_cm, "Should include confusion matrix"
    assert metrics_with_cm["confusion_matrix"].tolist() == [
        [2, 0],
        [1, 1],
    ], "Confusion matrix mismatch"
