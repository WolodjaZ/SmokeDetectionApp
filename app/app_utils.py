import json
import logging
import logging.config
from argparse import Namespace
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import whylogs as why
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


def load_artifacts(run_id: int) -> dict:
    """Load artifacts from run.

    Args:
        run_id (int): run id to load artifacts for prediction.

    Returns:
        dict: Dictionary with artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**load_dict(filepath=Path(artifacts_dir, "args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))

    return {"args": args, "model": model}


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
        why_logger = why.logger(
            mode="rolling", interval=5, when="M", base_name="whylogs-gradio-smokeapp"
        )
        why_logger.append_writer("local", base_dir=str(config.WHY_LOGS_DIR))

        if why_logger is not None:
            break
        else:
            n_attempts -= 1
    if n_attempts <= 0:
        raise Exception("Could not initialize whylogs session")

    return why_logger
