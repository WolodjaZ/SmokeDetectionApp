import json
import logging
import logging.config
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from rich.logging import RichHandler

from config import config


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
