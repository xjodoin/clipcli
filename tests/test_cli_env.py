from pathlib import Path

from typer.testing import CliRunner

from clipcli.cli import app


def test_cli_loads_env_local_before_env(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAL_KEY", raising=False)
    (tmp_path / ".env").write_text("FAL_KEY=env\n")
    (tmp_path / ".env.local").write_text("FAL_KEY=local\n")

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "clipcli" in result.stdout
    import os

    assert os.environ["FAL_KEY"] == "local"
