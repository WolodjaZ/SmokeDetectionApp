# src/evaluate.py
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Performance metrics using ground truths and predictions.

    Args:
        y_true (np.ndarray): Array with ground truths.
        y_pred (np.ndarray): Array with predictions.

    Returns:
        dict: Dictionary with metrics.
    """
    # Performance
    metrics = {}

    # Overall metrics
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)

    return metrics
