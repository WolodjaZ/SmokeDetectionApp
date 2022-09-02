# src/predict.py
from typing import List

import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: int) -> np.ndarray:
    """Custom predict with threshold.

    Args:
        y_prob (np.ndarray): Array with probabilities.
        threshold (int): Threshold cutoff.

    Returns:
        np.ndarray: Array with custom condition predictions.
    """
    y_pred = [1 if p > threshold else 0 for p in y_prob]
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
