# config/config.py
from pathlib import Path

import mlflow
import pretty_errors  # NOQA: F401 (imported but unused)

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
KAGGLE_FILE = "kaggle.json"
SECRETS = "secrets.json"
DATASET_REF = "deepcontractor/smoke-detection-dataset"
DATA_URL = "https://www.kaggle.com/code/stpeteishii/smoke-detection-fastai-with-tabularpandas"
DATA_RAW_NAME = "smoke_detection_iot.csv"
DATA_PREPROCESS_NAME = "preprocess.csv"
DATA_PREPROCESS_WITHOUT_OUTLINES_NAME = "preprocess_without_outlines.csv"

# Additional parameters
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)
