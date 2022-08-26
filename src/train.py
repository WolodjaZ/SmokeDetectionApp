from argparse import Namespace
from typing import Optional

import pandas as pd


def train(args: dict, df: pd.DataFrame, trial: Optional[int] = None) -> dict:
    """Train model on data.

    Args:
        args (dict): Arguments.
        df (pd.DataFrame): DataFrame with data.
        trial (Optional[int], optional): Trial number. Defaults to None.

    Returns:
        dict: Dictionary with results.
    """
    pass


def objective(args: Namespace, df: pd.DataFrame, trial: int) -> float:
    """Objective function to optimize.

    Args:
        args (Namespace): Arguments.
        df (pd.DataFrame): DataFrame with data.
        trial (int): Trial number.

    Returns:
        float: Loss.
    """
    pass
