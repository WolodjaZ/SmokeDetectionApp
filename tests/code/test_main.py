import tempfile
from pathlib import Path

from typer.testing import CliRunner

from config import config
from src import main

# from src.main import app

runner = CliRunner()


def test_download_data(mocker) -> None:
    """Test download data."""
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        mocker.patch.object(main.config, "DATA_DIR", base_path)

        assert not Path(config.DATA_DIR / Path(config.DATA_RAW_NAME)).exists()
        # result = runner.invoke(app, ["download-data"]) # TODO cannot test download data with CLI
        # assert result.exit_code == 0
        main.download_data()
        assert Path(config.DATA_DIR / Path(config.DATA_RAW_NAME)).exists()
