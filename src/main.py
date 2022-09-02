# src/main.py
import os
import warnings
from typing import List, Optional

import typer
from kaggle.api.kaggle_api_extended import KaggleApi

from config import config
from src import utils

app = typer.Typer()
logger = utils.create_logger()

warnings.filterwarnings("ignore")


@app.command()
def download_data() -> None:
    """load data from kaggle."""
    if not (config.DATA_DIR / config.DATA_RAW_NAME).is_file():
        # Set Kaggle API to init from file
        os.environ["KAGGLE_CONFIG_DIR"] = str(config.CONFIG_DIR / config.KAGGLE_FILE)
        api = KaggleApi()
        api.authenticate()

        # Load
        api.dataset_download_files(config.DATASET_REF, path=str(config.DATA_DIR), unzip=True)

        logger.info("✅ Data downloaded!")
    else:
        logger.info("✅ Data already downloaded!")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: bool = False,
) -> None:
    """Train model.

    Args:
        args_fp (str, optional): Path to arguments. Defaults to "config/args.json".
        experiment_name (str, optional): Experiment name. Defaults to "baselines".
        run_name (str, optional): Run name. Defaults to "sgd".
        test_run (bool, optional): Test run. Defaults to False.
    """
    pass


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize model.

    Args:
        args_fp (str, optional): Path to arguments. Defaults to "config/args.json".
        study_name (str, optional): Study name. Defaults to "optimization".
        num_trials (int, optional): Number of trials. Defaults to 20.
    """
    pass


@app.command()
def predict_smoke(data: List = [], run_id: Optional[int] = None) -> None:
    """Predict if smoke is detected for given input data.

    Args:
        data (List): List with data to classify.
        run_id (Optional[int], optional): run id to load artifacts for prediction. Defaults to None.
    """
    pass


def load_artifacts(run_id: int) -> dict:
    """Load artifacts from run.

    Args:
        run_id (int): run id to load artifacts for prediction.

    Returns:
        dict: Dictionary with artifacts.
    """
    pass


if __name__ == "__main__":
    download_data()
