import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pretty_errors
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.logging import RichHandler

from src.config import Path as Path_config
from src.config import SmokeConfig


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): Path to json file to be loaded.

    Returns:
        Dict: File json insights as a dictionary.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls: Optional[Dict] = None, sort_keys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): Data in dictionary to be saved as json.
        filepath (str): Path where the json will be saved.
        cls (Optional[Dict], optional): Method to serialize additional types. Defaults to None.
        sort_keys (bool, optional): Sort dictionary by keys. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sort_keys)


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int, optional): Value for seed to be set. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)


def init_smoke(cfg: Path_config) -> None:
    """Initialize smoke app.

    Args:
        cfg (Path_config): Smoke config.
    """
    # Create dirs
    Path(cfg.data).mkdir(parents=True, exist_ok=True)
    Path(cfg.storage).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_registry).mkdir(parents=True, exist_ok=True)
    Path(cfg.results).mkdir(parents=True, exist_ok=True)
    Path(cfg.log).mkdir(parents=True, exist_ok=True)
    Path(cfg.why_logs).mkdir(parents=True, exist_ok=True)

    # Additional parameters
    mlflow.set_tracking_uri("file://" + cfg.model_registry)
    pretty_errors.configure(
        separator_character="*",
        filename_display=pretty_errors.FILENAME_EXTENDED,
        line_number_first=True,
        display_link=True,
        lines_before=5,
        lines_after=2,
        line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
        code_color="  " + pretty_errors.default_config.line_color,
        truncate_code=True,
        display_locals=True,
    )


def get_logger(config_path: str) -> logging.Logger:
    """Get logger object.

    Args:
        config_path (str): Path to config file.

    Returns:
        logging.Logger: Logger object.
    """
    logging.config.fileConfig(Path(config_path, "logging.config"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    return logger


def download_data(cfg: SmokeConfig, logger: logging.Logger) -> None:
    """load data from kaggle.

    Args:
        cfg (SmokeConfig): Smoke config.
        logger (logging.Logger): Logger.
    """
    # Download data
    if not Path(os.path.join(cfg.path.data, cfg.dataset.raw)).is_file():
        # Set Kaggle API to init from file
        os.environ["KAGGLE_CONFIG_DIR"] = os.path.join(cfg.path.config, cfg.secret.kaggle)
        api = KaggleApi()
        api.authenticate()

        # Load
        api.dataset_download_files(cfg.dataset.dataset_ref, path=cfg.path.data, unzip=True)

        logger.info("✅ Data downloaded!")
    else:
        logger.info("✅ Data already downloaded!")
