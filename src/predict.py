# src/predict.py
import json
import os
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import List

import hydra
import joblib
import mlflow
import numpy as np
from hydra.core.config_store import ConfigStore
from numpyencoder import NumpyEncoder

from src import utils
from src.config import SmokeConfig

cs = ConfigStore.instance()
cs.store(name="smoke_train", node=SmokeConfig)


def custom_predict(y_prob: np.ndarray, threshold: float, index: int = 0) -> np.ndarray:
    """Custom predict function that defaults
    to an index if conditions are not met.
    Args:
        y_prob (np.ndarray): Predicted probabilities
        threshold (float): Minimum softmax score to predict majority class
        index (int): Label index to use if custom conditions is not met.
    Returns:
        np.ndarray: Predicted label indices.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(input_data: np.ndarray, artifacts: dict) -> List:
    """Predict smoke for input data.

    Args:
        input_data (np.ndarray): Array with input data.
        artifacts (dict): Dictionary with artifacts.

    Returns:
        List: Predictions for input.
    """
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(input_data), threshold=artifacts["args"].threshold
    )
    predictions = [
        {
            "input_data": input_data[i],
            "Smoke_detected": y_pred[i],
        }
        for i in range(len(y_pred))
    ]
    return predictions


def load_artifacts(cfg: SmokeConfig) -> dict:
    """Load artifacts from run.

    Args:
        cfg (SmokeConfig): Config object. Run ID is extracted from this object.

    Returns:
        dict: Dictionary with artifacts.
    """
    # Set Mlflow tracking URI
    mlflow.set_tracking_uri("file://" + cfg.path.model_registry)

    # Locate specifics artifacts directory
    experiment_id = str(mlflow.get_run(run_id=cfg.predict.run_id).info.experiment_id)
    artifacts_dir = Path(
        os.path.join(cfg.path.model_registry, experiment_id, cfg.predict.run_id, "artifacts")
    )

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=(artifacts_dir / Path("args.json"))))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir / Path("performance.json")))

    return {
        "args": args,
        "model": model,
        "performance": performance,
    }


def predict_smoke(cfg: SmokeConfig):
    """Predict if smoke is detected for given input data.

    Args:
        cfg (SmokeConfig): Smoke config.
        data (str): String list with data to classify. (Specified in cfg)
        run_id (str): run id to load artifacts for prediction. Defaults to empty string. (Specified in cfg)
    """
    logger = utils.get_logger(cfg.path.config)
    artifacts = load_artifacts(cfg=cfg)
    data = np.array(literal_eval(cfg.predict.input))
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    prediction = predict(input_data=data, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2, cls=NumpyEncoder))
    return prediction


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/config", config_name="config")
def main(cfg: SmokeConfig):
    """Main function for prediction.

    Args:
        cfg (SmokeConfig): Smoke config.
    """
    predict_smoke(cfg=cfg)


if __name__ == "__main__":
    main()
