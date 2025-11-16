# tests/test_main.py
from typer.testing import CliRunner
from bsort.main import app

runner = CliRunner()

def test_cli_help():
    """Tes apakah perintah --help berjalan."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "infer" in result.stdout