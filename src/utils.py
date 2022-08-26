import json
import random
from typing import Optional

import numpy as np


def load_dict(filepath: str) -> dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): Path to json file to be loaded.

    Returns:
        dict: File json insights as a dictionary.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, filepath: str, cls: Optional[dict] = None, sort_keys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (dict): Data in dictionary to be saved as json.
        filepath (str): Path where the json will be saved.
        cls (Optional[dict], optional): Method to serialize additional types. Defaults to None.
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
