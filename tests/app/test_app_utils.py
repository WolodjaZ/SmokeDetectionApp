import logging
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

myDir = os.getcwd() + "/app"
sys.path.append(myDir)

from app import app_utils  # noqa: E402

DATA = os.path.join(os.getcwd(), "data", "preprocess_without_outlines.csv")


@pytest.mark.parametrize(
    "threshold, y_pred",
    [
        (0.5, [0]),
        (0.6, [1]),
        (0.75, [1]),
    ],
)
def test_custom_predict(threshold, y_pred):
    y_prob = np.array([[0.6, 0.4]])
    assert np.array_equal(
        app_utils.custom_predict(y_prob=y_prob, threshold=threshold, index=1), np.array(y_pred)
    )


def test_create_logger():
    logger = app_utils.create_logger()
    assert isinstance(logger, logging.Logger)
    shutil.rmtree(Path(os.getcwd(), "app", "logs"))


def test_get_data_splits():
    df_path = Path(DATA)
    df = pd.read_csv(df_path)

    X_train, X_val, X_test, y_train, y_val, y_test = app_utils.get_data_splits(
        df.drop(columns=["fire_alarm"], axis=1), df["fire_alarm"]
    )

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) / float(len(df)) == pytest.approx(0.7, abs=0.11)  # 0.7 ± 0.11
    assert len(X_val) / float(len(df)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05
    assert len(X_test) / float(len(df)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05


def test_predict(mocker):
    data = [22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    output_text = "No fire alarm"
    mocker.patch.object(app_utils, "create_logger", return_value=logging.Logger("test"))
    mocker.patch.object(
        requests, "post", return_value=Namespace(**{"text": {"predictions": output_text}})
    )
    output = app_utils.predict_api(*data, api_url="test")
    assert output == output_text
