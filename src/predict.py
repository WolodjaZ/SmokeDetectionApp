# src/predict.py
from typing import List

import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: int, index: int) -> np.ndarray:
    """Custom predict function that defaults
    to an index if conditions are not met.

    Args:
        y_prob (np.ndarray): Array with probabilities.
        threshold (int): Threshold cutoff.
        index (int): Index to return if conditions are not met.

    Returns:
        np.ndarray: Array with custom condition predictions.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(input_data: np.ndarray, artifacts: dict) -> List[dict]:
    """Predict smoke for input data.

    Args:
        input_data (np.ndarray): Array with input data.
        artifacts (dict): Dictionary with artifacts.

    Returns:
        List[dict]: List of dictionaries with predictions.
    """
    pass
