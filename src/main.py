# src/main.py
import json
import os
import tempfile
import warnings
from argparse import Namespace
from ast import literal_eval
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import typer
from kaggle.api.kaggle_api_extended import KaggleApi
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from src import predict, train, utils

app = typer.Typer()
logger = utils.create_logger()

warnings.filterwarnings("ignore")


@app.command()
def download_data():
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
    experiment_name: str = "baseline",
    run_name: str = "bayes",
    test_run: bool = False,
):
    """Train model.

    Args:
        args_fp (str, optional): Path to arguments. Defaults to "config/args.json".
        experiment_name (str, optional): Experiment name. Defaults to "baselines".
        run_name (str, optional): Run name. Defaults to "bayes".
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load data
    df = pd.read_csv(config.DATA_DIR / config.DATA_RAW_NAME)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args, logger=logger)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"accuracy": performance["balanced_accuracy"]})
        mlflow.log_metrics({"precision": performance["precision"]})
        mlflow.log_metrics({"recall": performance["recall"]})
        mlflow.log_metrics({"f1": performance["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.RESULT_DIR, "performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
):
    """Optimize model.

    Args:
        args_fp (str, optional): Path to arguments. Defaults to "config/args.json".
        study_name (str, optional): Study name. Defaults to "optimization".
        num_trials (int, optional): Number of trials. Defaults to 20.
    """
    # Load data
    df = pd.read_csv(config.DATA_DIR / config.DATA_RAW_NAME)

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, logger, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


@app.command()
def predict_smoke(data: str = "", run_id: str = ""):
    """Predict if smoke is detected for given input data.

    Args:
        data (str): String list with data to classify.
        run_id (str): run id to load artifacts for prediction. Defaults to empty string.
    """
    if len(run_id) == 0:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    data = np.array(literal_eval(data))
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    prediction = predict.predict(input_data=data, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2, cls=NumpyEncoder))
    return prediction


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
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "model": model,
        "performance": performance,
    }


if __name__ == "__main__":
    app()  # pragma: no cover, live app
