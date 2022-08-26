# config/config.py
import logging
import logging.config
from pathlib import Path

import mlflow
import pretty_errors  # NOQA: F401 (imported but unused)
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORAGE_DIR = Path(BASE_DIR, "storage")
MODEL_REGISTRY = Path(STORAGE_DIR, "model")
LOGS_DIR = Path(BASE_DIR, "logs")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Assets
DATA_RAW_NAME = "smoke_detection_iot.csv"
DATA_REF = "deepcontractor/smoke-detection-dataset"
DATA_URL = "https://www.kaggle.com/code/stpeteishii/smoke-detection-fastai-with-tabularpandas"

# Additional parameters
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # set rich handler
