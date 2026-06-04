from pathlib import Path

import pytest

from clipcli import audio


class _FakePopen:
    def __init__(self, command, *, returncode: int | None = 0, stderr_text: str = "", **kwargs):
        self.command = command
        self._returncode = returncode
        self.stderr = None
        self.killed = False

    def poll(self):
        return self._returncode

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout=None):
        return self._returncode


def test_enhance_with_deepfilternet_uses_installed_executable(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    executable = tmp_path / "deepFilter"
    executable.write_text("#!/bin/sh\n")
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")

    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: executable)

    def fake_popen(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / source.name).write_bytes(b"enhanced")
        return _FakePopen(command, returncode=0)

    monkeypatch.setattr(audio.subprocess, "Popen", fake_popen)

    output = audio.enhance_with_deepfilternet(
        source,
        tmp_path / "final.wav",
        work_dir=tmp_path / "work",
    )

    assert output.read_bytes() == b"enhanced"
    assert str(executable) == captured["command"][0]
    assert "--pf" in captured["command"]
    assert "--no-suffix" in captured["command"]


def test_enhance_with_deepfilternet_survives_shutdown_deadlock(monkeypatch, tmp_path: Path) -> None:
    """deep-filter-py can hang in interpreter shutdown after writing its output."""
    executable = tmp_path / "deepFilter"
    executable.write_text("#!/bin/sh\n")
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")

    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: executable)
    monkeypatch.setattr(audio.time, "sleep", lambda seconds: None)

    processes = []

    def fake_popen(command, **kwargs):
        out_dir = Path(command[command.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / source.name).write_bytes(b"enhanced")
        process = _FakePopen(command, returncode=None)  # never exits
        processes.append(process)
        return process

    monkeypatch.setattr(audio.subprocess, "Popen", fake_popen)

    output = audio.enhance_with_deepfilternet(
        source,
        tmp_path / "final.wav",
        work_dir=tmp_path / "work",
    )

    assert output.read_bytes() == b"enhanced"
    assert processes[0].killed  # the stuck process is cleaned up


def test_enhance_with_deepfilternet_reports_failure(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "deepFilter"
    executable.write_text("#!/bin/sh\n")
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")

    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: executable)
    monkeypatch.setattr(
        audio.subprocess, "Popen", lambda command, **kwargs: _FakePopen(command, returncode=1)
    )

    with pytest.raises(audio.AudioEnhancementError, match="DeepFilterNet failed"):
        audio.enhance_with_deepfilternet(
            source,
            tmp_path / "final.wav",
            work_dir=tmp_path / "work",
        )


def test_enhance_with_deepfilternet_reports_missing_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "_deepfilternet_executable", lambda: None)

    with pytest.raises(audio.AudioEnhancementError, match="DeepFilterNet is not installed"):
        audio.enhance_with_deepfilternet(
            tmp_path / "raw.wav",
            tmp_path / "final.wav",
            work_dir=tmp_path / "work",
        )
