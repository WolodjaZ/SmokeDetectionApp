# config/config.py
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Dataclasses
@dataclass
class Model:
    experiment_name: str
    run_name: str
    test_run: bool
    shuffle: bool
    subset: Optional[int]
    use_outlines: bool
    num_epochs: int
    threshold: float
    oversample: bool
    seed: int
    outliers_numb: int
    hyperparams: Dict[str, Any]


@dataclass
class Dataset:
    dataset_ref: str
    dataset_url: str
    raw: str
    preprocess: str
    preprocess_without_outlines: str


@dataclass
class Secret:
    kaggle: str
    secrets: str


@dataclass
class SecretOptimize:
    kaggle: str
    secrets: str
    study_name: str


@dataclass
class Path:
    base: str
    config: str
    data: str
    storage: str
    model_registry: str
    results: str
    why_logs: str
    log: str


@dataclass
class Test:
    shuffle: bool
    subset: Optional[int]
    use_outlines: bool
    outliers_numb: int
    num_epochs: int
    threshold: float
    experiment_name: str
    run_name: str


@dataclass
class Predict:
    input: str = ""
    run_id: str = ""


@dataclass
class SmokeConfig:
    model: Model
    dataset: Dataset
    secret: Secret
    path: Path
    test: Test
    predict: Predict


@dataclass
class SmokeConfigOptimize:
    model: Model
    dataset: Dataset
    secret: SecretOptimize
    path: Path
