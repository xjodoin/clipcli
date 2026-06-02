import pytest

from clipcli.transcribe import resolve_whisperx_device


def test_resolve_whisperx_device_rejects_mps() -> None:
    with pytest.raises(RuntimeError, match="does not support device='mps'"):
        resolve_whisperx_device("mps")


def test_resolve_whisperx_device_accepts_cpu() -> None:
    assert resolve_whisperx_device("cpu") == "cpu"
