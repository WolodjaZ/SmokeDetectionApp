# src/main.py
import json
import os
import warnings

import hydra
import mlflow
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src import utils
from src.config import SmokeConfigOptimize
from src.train import train

cs = ConfigStore.instance()
cs.store(name="smoke_optimie", node=SmokeConfigOptimize)

warnings.filterwarnings("ignore")


def optimize(cfg: SmokeConfigOptimize) -> float:
    """main smoke app.

    Args:
        cfg (SmokeConfig): Smoke config.
    """
    # init smoke
    utils.init_smoke(cfg=cfg.path)

    # Set logger
    logger = utils.get_logger(cfg.path.config)

    # Download dataset
    utils.download_data(cfg=cfg, logger=logger)
    df = pd.read_csv(os.path.join(cfg.path.data, cfg.dataset.raw))

    # Train
    mlflow.set_experiment(experiment_name=cfg.secret.study_name)
    with mlflow.start_run(run_name=cfg.model.run_name):
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("model_type", cfg.model.run_name)
        logger.info(f"Run ID: {run_id}")
        artifacts = train(df=df, cfg=cfg, logger=logger, optimize=True)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))
        mlflow.log_metrics({"accuracy": performance["balanced_accuracy"]})
        mlflow.log_metrics({"precision": performance["precision"]})
        mlflow.log_metrics({"recall": performance["recall"]})
        mlflow.log_metrics({"f1": performance["f1"]})
        mlflow.log_params(OmegaConf.to_container(artifacts["args"]))
        logger.info(rf"\Performance value (f1): {performance['f1']}")

    return performance["f1"]


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/config", config_name="config-optimize")
def main(cfg: SmokeConfigOptimize) -> float:
    """main smoke app.

    Args:
        cfg (SmokeConfig): Smoke config.

    Returns:
        float: performance value.
    """
    performance = optimize(cfg=cfg)
    return performance


if __name__ == "__main__":
    main()
