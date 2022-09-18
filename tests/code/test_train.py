import logging
import os
import shutil

import mlflow
import pandas as pd
import pytest
from hydra import compose, initialize

from src import train, utils


def delete_experiment(experiment_name, model_path):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    folder = os.path.join(model_path, ".trash")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


@pytest.mark.training
def test_train_smoke(mocker):
    mocker.patch("src.utils.get_logger", return_value=logging.getLogger())
    config_path = "../../config"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")
        cfg.path.base = os.getcwd()
        cfg.model.shuffle = cfg.test.shuffle
        cfg.model.subset = cfg.test.subset
        cfg.model.use_outlines = cfg.test.use_outlines
        cfg.model.outliers_numb = cfg.test.outliers_numb
        cfg.model.num_epochs = cfg.test.num_epochs
        cfg.model.threshold = cfg.test.threshold
        cfg.model.experiment_name = cfg.test.experiment_name
        cfg.model.run_name = cfg.test.run_name
        cfg.model.test_run = False

    train.train_smoke(cfg)

    # Clean up
    delete_experiment(experiment_name=cfg.model.experiment_name, model_path=cfg.path.model_registry)


@pytest.mark.training
def test_train_model():
    config_path = "../../config"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")
        cfg.path.base = os.getcwd()
        cfg.model.shuffle = cfg.test.shuffle
        cfg.model.subset = cfg.test.subset
        cfg.model.use_outlines = cfg.test.use_outlines
        cfg.model.outliers_numb = cfg.test.outliers_numb
        cfg.model.num_epochs = cfg.test.num_epochs
        cfg.model.threshold = cfg.test.threshold
        cfg.model.experiment_name = cfg.test.experiment_name
        cfg.model.run_name = cfg.test.run_name

    # Init smoke
    utils.init_smoke(cfg=cfg.path)

    # Set logger
    logger = logging.getLogger()

    # Download dataset
    utils.download_data(cfg=cfg, logger=logger)
    df = pd.read_csv(os.path.join(cfg.path.data, cfg.dataset.raw))

    # Train
    mlflow.create_experiment(name=cfg.model.experiment_name)
    with mlflow.start_run(run_name=cfg.model.run_name):
        artifacts = train.train(df=df, cfg=cfg, logger=logger)
    assert len(artifacts)

    # Clean up
    delete_experiment(experiment_name=cfg.model.experiment_name, model_path=cfg.path.model_registry)
