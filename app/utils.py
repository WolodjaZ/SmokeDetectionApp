import logging
import logging.config
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import whylogs as why
from pydantic import BaseModel
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split

from config import config


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


def create_logger() -> logging.Logger:
    """Create logger.

    Returns:
        logging.Logger: Logger.
    """
    logging.config.fileConfig(Path(config.CONFIG_DIR, "logging.config"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    logger = logging.getLogger()
    return logger


def initialize_why_logger():
    """Create whylogs logger.

    Returns:
        Whylogs logger.
    """
    # Initialize session
    n_attempts = 3
    while n_attempts > 0:
        # Initialize logger
        why_logger = why.logger(mode="rolling", interval=5, when="M", base_name="whylogs-smokeapp")
        why_logger.append_writer("local", base_dir=str(config.WHY_LOGS_DIR))

        if why_logger is not None:
            break
        else:
            n_attempts -= 1
    if n_attempts <= 0:
        raise Exception("Could not initialize whylogs session")

    return why_logger


def get_data_splits(X: pd.DataFrame, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generate balanced data splits.
    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


class SmokeFeatures(BaseModel):
    temperature_c: float
    humidity: float
    tvoc_ppb: int
    e_co_2_ppm: int
    raw_h_2: int
    raw_ethanol: int
    pressure_h_pa: float
    pm_1_0: float
    pm_2_5: float
    nc_0_5: float
    nc_1_0: float
    nc_2_5: float
    cnt: int
