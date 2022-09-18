import logging
import os

import numpy as np
import pytest
from hydra import compose, initialize

from src import predict
from src.config import Path as PathConfig
from src.config import Predict as PredictConfig
from src.config import SmokeConfig

MODEL_REGISTRY = "storage/model"


@pytest.fixture()
def Smoke_Config():
    Smoke_Config = SmokeConfig(
        model=None,
        dataset=None,
        secret=None,
        path=PathConfig("", "", "", "", "", "", "", ""),
        test=None,
        predict=PredictConfig("test", "test"),
    )
    return Smoke_Config


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
        predict.custom_predict(y_prob=y_prob, threshold=threshold, index=1), np.array(y_pred)
    )


def test_load_artifacts(Smoke_Config):
    config_path = "../../config"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")
        Smoke_Config.predict = PredictConfig(cfg.predict.input, cfg.predict.run_id)
        Smoke_Config.path.model_registry = os.path.join(os.getcwd(), MODEL_REGISTRY)
        print(Smoke_Config.path.model_registry)
        artifacts = predict.load_artifacts(cfg=Smoke_Config)
        assert len(artifacts)


@pytest.mark.parametrize(
    "data",
    [
        (
            "[20.117000579833984,52.810001373291016,0,400,12448,19155,939.7579956054688,0.0,0.0,0.0,0.0,0.0,8]"
        ),
        (
            "[[20.117000579833984,52.810001373291016,0,400,12448,19155,939.7579956054688,0.0,0.0,0.0,0.0,0.0,8],[20.10300064086914,53.20000076293945,0,400,12439,19114,939.7579956054688,0.0,0.0,0.0,0.0,0.0,7]]"
        ),
    ],
)
def test_predict_tag(mocker, Smoke_Config, data):
    config_path = "../../config"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")
        Smoke_Config.predict = PredictConfig(data, cfg.predict.run_id)
        Smoke_Config.path.model_registry = os.path.join(os.getcwd(), MODEL_REGISTRY)

        mocker.patch("src.utils.get_logger", return_value=logging.getLogger())
        prediction = predict.predict_smoke(Smoke_Config)
        assert len(prediction)
