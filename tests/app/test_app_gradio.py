import logging
from argparse import Namespace

import requests

from app import app_gradio, app_utils


def test_predict(mocker):
    data = [22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    output_text = "No fire alarm"
    mocker.patch.object(app_utils, "create_logger", return_value=logging.Logger("test"))
    mocker.patch.object(
        requests, "post", return_value=Namespace(**{"text": {"predictions": output_text}})
    )
    output = app_gradio.predict(*data, api_url="test")
    assert output == output_text


def test_create_app(mocker):
    mocker.patch.object(app_utils, "create_logger", return_value=logging.Logger("test"))
    app = app_gradio.create_app()
    assert app is not None
