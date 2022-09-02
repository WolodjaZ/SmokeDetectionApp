import tempfile
from pathlib import Path

import mlflow
import pytest
from typer.testing import CliRunner

from config import config
from src import main
from src.main import app

runner = CliRunner()
args_fp = Path(config.BASE_DIR, "tests", "code", "test_args.json")


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


def test_download_data(mocker) -> None:
    """Test download data."""
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        mocker.patch.object(main.config, "DATA_DIR", base_path)

        result = runner.invoke(app, ["download-data"])
        assert result.exit_code == 0


@pytest.mark.training
def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke(
        app,
        [
            "train-model",
            f"--args-fp={args_fp}",
            f"--experiment-name={experiment_name}",
            f"--run-name={run_name}",
            "--test-run",
        ],
    )
    assert result.exit_code == 0

    # Clean up
    delete_experiment(experiment_name=experiment_name)


@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    num_trials = 1
    result = runner.invoke(
        app,
        [
            "optimize",
            f"--args-fp={args_fp}",
            f"--study-name={study_name}",
            f"--num-trials={num_trials}",
        ],
    )
    assert result.exit_code == 0

    # Clean up
    delete_experiment(experiment_name=study_name)


def test_load_artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    assert len(artifacts)


@pytest.mark.parametrize(
    "data",
    [
        (
            "[20.117000579833984,52.810001373291016,0,400,12448,19155,939.7579956054688,0.0,0.0,0.0,0.0,0.0,8]"
        ),
        (
            "[[20.117000579833984,52.810001373291016,0,400,12448,19155,939.7579956054688,0.0,0.0,0.0,0.0,0.0,8],[20.10300064086914,53.20000076293945,0,400,12439,19114,939.7579956054688,0.0,0.0,0.0,0.0,0.0,7]]"
        ),
    ],
)
def test_predict_tag(data):
    result = runner.invoke(app, ["predict-smoke", f"--data={data}"])
    assert result.exit_code == 0
