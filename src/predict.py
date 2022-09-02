# src/predict.py
from typing import List

import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: float, index: int = 0) -> np.ndarray:
    """Custom predict function that defaults
    to an index if conditions are not met.
    Args:
        y_prob (np.ndarray): Predicted probabilities
        threshold (float): Minimum softmax score to predict majority class
        index (int): Label index to use if custom conditions is not met.
    Returns:
        np.ndarray: Predicted label indices.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(input_data: np.ndarray, artifacts: dict) -> List:
    """Predict smoke for input data.

    Args:
        input_data (np.ndarray): Array with input data.
        artifacts (dict): Dictionary with artifacts.

    Returns:
        List: Predictions for input.
    """
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(input_data),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    predictions = [
        {
            "input_text": input_data[i],
            "Smoke_detected": y_pred[i],
        }
        for i in range(len(y_pred))
    ]
    return predictions
