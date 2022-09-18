import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize

from src import utils
from src.config import Dataset as DatasetConfig
from src.config import Path as PathConfig
from src.config import Secret as SecretConfig
from src.config import SmokeConfig


def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello": "world"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, filepath=fp)
        d = utils.load_dict(filepath=fp)
        assert d["hello"] == "world"


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)


def test_get_logger():
    logger = utils.get_logger(f"{os.getcwd()}/config")
    assert isinstance(logger, logging.Logger)


def test_init_smoke():
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        Smoke_Config = SmokeConfig(
            model=None,
            dataset=None,
            secret=None,
            path=PathConfig(
                str(base_path),
                os.path.join(str(base_path), "test_a"),
                os.path.join(str(base_path), "test_b"),
                os.path.join(str(base_path), "test_c"),
                os.path.join(str(base_path), "test_d"),
                os.path.join(str(base_path), "test_e"),
                os.path.join(str(base_path), "test_f"),
                os.path.join(str(base_path), "test_g"),
            ),
            test=None,
            predict=None,
        )
        utils.init_smoke(cfg=Smoke_Config.path)
        assert Path(Smoke_Config.path.base).is_dir()
        assert Path(Smoke_Config.path.data).is_dir()
        assert Path(Smoke_Config.path.log).is_dir()
        assert Path(Smoke_Config.path.model_registry).is_dir()
        assert Path(Smoke_Config.path.results).is_dir()
        assert Path(Smoke_Config.path.storage).is_dir()
        assert Path(Smoke_Config.path.why_logs).is_dir()


@pytest.mark.parametrize(
    "exist_raw",
    [
        (True),
        (False),
    ],
)
def test_download_data(exist_raw):
    config_path = "../../config"
    with tempfile.TemporaryDirectory() as dp:
        with initialize(version_base=None, config_path=config_path):
            cfg = compose(config_name="config")

        base_path = Path(dp)
        Smoke_Config = SmokeConfig(
            model=None,
            dataset=DatasetConfig(
                cfg.dataset.dataset_ref, cfg.dataset.dataset_url, cfg.dataset.raw, "", ""
            ),
            secret=SecretConfig(cfg.secret.kaggle, cfg.secret.secrets),
            path=PathConfig("", f"{os.getcwd()}/config", str(base_path), "", "", "", "", ""),
            test=None,
            predict=None,
        )
        logger = logging.getLogger()
        if exist_raw:
            Path(Smoke_Config.path.data, Smoke_Config.dataset.raw)
        utils.download_data(cfg=Smoke_Config, logger=logger)
        assert (Path(Smoke_Config.path.data) / Path(Smoke_Config.dataset.raw)).is_file()
