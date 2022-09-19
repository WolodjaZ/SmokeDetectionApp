import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Union

import bentoml
import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from numpyencoder import NumpyEncoder
from omegaconf import OmegaConf
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import data, evaluate, predict, utils
from src.config import SmokeConfig, SmokeConfigOptimize

cs = ConfigStore.instance()
cs.store(name="smoke_train", node=SmokeConfig)

warnings.filterwarnings("ignore")


def train_smoke(cfg: SmokeConfig):
    """Train model on data.

    Args:
        cfg (SmokeConfig): cfg to use for training.
    """
    # Init smoke
    utils.init_smoke(cfg=cfg.path)

    # Set logger
    logger = utils.get_logger(cfg.path.config)

    # Download dataset
    utils.download_data(cfg=cfg, logger=logger)
    df = pd.read_csv(os.path.join(cfg.path.data, cfg.dataset.raw))

    # Train
    mlflow.set_experiment(experiment_name=cfg.model.experiment_name)
    with mlflow.start_run(run_name=cfg.model.run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train(df=df, cfg=cfg, logger=logger)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"accuracy": performance["balanced_accuracy"]})
        mlflow.log_metrics({"precision": performance["precision"]})
        mlflow.log_metrics({"recall": performance["recall"]})
        mlflow.log_metrics({"f1": performance["f1"]})
        mlflow.log_params(OmegaConf.to_container(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(
                OmegaConf.to_container(artifacts["args"]),
                str(Path(dp, "args.json")),
                cls=NumpyEncoder,
            )
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, str(Path(dp, "performance.json")))
            mlflow.log_artifacts(dp)

    # Save to config
    if cfg.model.test_run:  # pragma: no cover, actual run
        utils.save_dict(performance, os.path.join(cfg.path.results, "performance.json"))

        # Save model to BentoML
        bento_model = bentoml.sklearn.save_model(
            "smoke_clf_model",
            artifacts["model"],
            labels={"Type": cfg.model.run_name},
            signatures={
                "predict": {"batchable": True, "batch_dim": 0},
                "predict_proba": {"batchable": True, "batch_dim": 0},
            },
            metadata={
                "threshold": artifacts["args"].threshold,
                "performance": performance,
                "mlflow_run_id": run_id,
            },
        )
        logger.info(f"✅ Model saved {bento_model}")
        logger.info(f"Change model:threshold to: {cfg.model.threshold}")
        logger.info(f"Change run_id to: {run_id}")


def train(
    cfg: Union[SmokeConfig, SmokeConfigOptimize],
    df: pd.DataFrame,
    logger: logging.Logger,
    optimize: bool = False,
) -> Dict:
    """Train model on data.

    Args:
        cfg (SmokeConfig): cfg to use for training.
        df (pd.DataFrame): DataFrame with data.
        logger (logging.Logger): Logger.
        optimize (bool, optional): Processed is optimized. Defaults to False.

    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.

    Returns:
        dict: Dictionary with results.
    """
    # Setup
    utils.set_seeds(cfg.model.seed)

    # Get data
    params = cfg.model
    if params.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: params.subset]  # None = all samples
    X_train, X_val, X_test, y_train, y_val, y_test = data.preprocess(df, cfg)
    logger.info("✅ Data preprocessed")

    # Creating pipeline
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", GaussianNB(**params.hyperparams)),
        ]
    )

    # Train data
    logger.info("✅ Model training")
    with tqdm(range(params.num_epochs), unit="epoch") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch:02d}")
            pipe.fit(X_train, y_train)

            # Get train metrics
            train_loss = log_loss(y_train, pipe.predict_proba(X_train))
            val_loss = log_loss(y_val, pipe.predict_proba(X_val))
            pbar.set_postfix({"Train loss": f"{train_loss:.4f}", "Val loss": f"{val_loss:.4f}"})

            # Log mlflow
            if not optimize:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        logger.info(
            f"✅ Model finished training with train loss: {train_loss:.5f}, val_loss {val_loss:.5f}"
        )

    # Threshold
    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)
    cfg.model.threshold = float(
        np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)
    )  # Q1

    # Evaluation
    y_prob = pipe.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=cfg.model.threshold)
    performance = evaluate.get_metrics(y_true=y_test, y_pred=y_pred)
    logger.info("✅ Model evaluated")

    return {
        "args": cfg.model,
        "model": pipe,
        "performance": performance,
    }


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/config", config_name="config")
def main(cfg: SmokeConfig):
    train_smoke(cfg=cfg)


if __name__ == "__main__":
    main()
