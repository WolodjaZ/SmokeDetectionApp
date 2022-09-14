import json
import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import requests
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split

from .schemas import SmokeFeatures

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
WHY_LOGS_DIR = Path(LOGS_DIR, "whylogs")
COLUMNS = [
    "temperature_c",
    "humidity",
    "tvoc_ppb",
    "e_co_2_ppm",
    "raw_h_2",
    "raw_ethanol",
    "pressure_h_pa",
    "pm_1_0",
    "pm_2_5",
    "nc_0_5",
    "nc_1_0",
    "nc_2_5",
    "cnt",
]


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
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Logger
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "minimal": {"format": "%(message)s"},
            "detailed": {
                "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "minimal",
                "level": logging.DEBUG,
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(LOGS_DIR, "info.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.INFO,
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(LOGS_DIR, "error.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.ERROR,
            },
        },
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    }
    logging.config.dictConfig(logging_config)
    bentoml_logger = logging.getLogger()
    bentoml_logger.setLevel(logging.INFO)
    bentoml_logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    return bentoml_logger


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


def predict_api(*args: Any, api_url: str) -> str:
    """Predict function.

    Args:
        args (Any): Input arguments
        api_url (str): API url

    Returns:
        str: Prediction
    """
    # Prepare input
    df = pd.DataFrame([args], columns=COLUMNS)

    # Validate input
    valid_df = SmokeFeatures(**(df.to_dict(orient="records")[0]))

    # Predict
    output = requests.post(
        f"{api_url}/predict",
        headers={"content-type": "application/json"},
        data=json.dumps(valid_df.dict()),
    )
    # Convert output to dict
    if isinstance(output.text, dict):
        output_prediction = output.text
    elif isinstance(output.text, str):
        output_prediction = json.loads(output.text)
    else:
        Exception("Not a valid output")

    return output_prediction["predictions"]
