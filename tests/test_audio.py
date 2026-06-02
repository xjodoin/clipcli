from pathlib import Path
from subprocess import CompletedProcess

import pytest

from clipcli import audio


def test_enhance_with_deepfilternet_uses_installed_executable(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    executable = tmp_path / "deepFilter"
    executable.write_text("#!/bin/sh\n")
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")

    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: executable)

    def fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / source.name).write_bytes(b"enhanced")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    output = audio.enhance_with_deepfilternet(
        source,
        tmp_path / "final.wav",
        work_dir=tmp_path / "work",
    )

    assert output.read_bytes() == b"enhanced"
    assert str(executable) == captured["command"][0]
    assert "--pf" in captured["command"]
    assert "--no-suffix" in captured["command"]


def test_enhance_with_deepfilternet_reports_missing_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: None)

    with pytest.raises(audio.AudioEnhancementError, match="DeepFilterNet is not installed"):
        audio.enhance_with_deepfilternet(
            tmp_path / "raw.wav",
            tmp_path / "final.wav",
            work_dir=tmp_path / "work",
        )
