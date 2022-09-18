import logging
import os
import shutil

import mlflow
import pytest
from hydra import compose, initialize

from src import optimize


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
def test_optimize(mocker):
    mocker.patch("src.utils.get_logger", return_value=logging.getLogger())
    config_path = "../../config"
    with initialize(version_base=None, config_path=config_path):
        cfg_test = compose(config_name="config")
        cfg = compose(config_name="config-optimize")
        cfg.path.base = os.getcwd()
        cfg.model.shuffle = cfg_test.test.shuffle
        cfg.model.subset = cfg_test.test.subset
        cfg.model.use_outlines = cfg_test.test.use_outlines
        cfg.model.outliers_numb = cfg_test.test.outliers_numb
        cfg.model.num_epochs = cfg_test.test.num_epochs
        cfg.model.threshold = cfg_test.test.threshold
        cfg.model.experiment_name = cfg_test.test.experiment_name
        cfg.secret.study_name = cfg_test.test.experiment_name

    performance = optimize.optimize(cfg)
    assert performance > 0.0
    # Clean up
    delete_experiment(experiment_name=cfg.secret.study_name, model_path=cfg.path.model_registry)
