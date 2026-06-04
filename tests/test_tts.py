from pathlib import Path

import pytest

from clipcli import ffmpeg, tts


def test_say_tts_builds_commands(monkeypatch, tmp_path: Path) -> None:
    commands = []

    def fake_run(command, timeout=None):
        commands.append(command)

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    monkeypatch.setattr(ffmpeg, "probe_duration", lambda path: 2.5)

    lines = tts.synthesize_voiceover_lines(
        ["Bonjour  le monde.", "Deuxième ligne."],
        tmp_path,
        provider="say",
        voice="Amélie",
    )

    assert len(lines) == 2
    assert lines[0].text == "Bonjour le monde."
    assert lines[0].duration == 2.5
    say_commands = [command for command in commands if command[0] == "say"]
    assert say_commands[0][:3] == ["say", "-o", str(tmp_path / "vo-01.aiff")]
    assert "-v" in say_commands[0] and "Amélie" in say_commands[0]
    convert_commands = [command for command in commands if command[0] == "ffmpeg"]
    assert len(convert_commands) == 2


def test_synthesize_rejects_empty_line(tmp_path: Path) -> None:
    with pytest.raises(tts.VoiceoverError):
        tts.synthesize_voiceover_lines(["   "], tmp_path, provider="say")


def test_write_pcm_wav_roundtrip(tmp_path: Path) -> None:
    import wave

    output = tmp_path / "tone.wav"
    tts._write_pcm_wav(b"\x00\x01" * 2400, output)
    with wave.open(str(output)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 2400


class _FakeChatterbox:
    sr = 24000

    def __init__(self, takes):
        self.takes = list(takes)
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append(kwargs)
        return self.takes.pop(0)


def _take(truncated: bool):
    import torch

    wav = torch.zeros(1, 24000)
    wav[:, :12000] = 0.3  # speech in the first half
    if truncated:
        wav[:, -2000:] = 0.3  # still speaking at the very edge
    return wav


def test_chatterbox_retries_truncated_takes(tmp_path: Path) -> None:
    model = _FakeChatterbox([_take(truncated=True), _take(truncated=False)])

    tts._chatterbox_tts(
        model,
        "Bonjour tout le monde.",
        tmp_path / "vo.wav",
        language="fr-CA",
        reference_audio=None,
        exaggeration=0.6,
    )

    assert len(model.calls) == 2  # retried once, stopped on the clean take
    assert (tmp_path / "vo.wav").exists()


def test_chatterbox_drops_cfg_for_cross_language_default_voice(tmp_path: Path) -> None:
    model = _FakeChatterbox([_take(truncated=False)])

    tts._chatterbox_tts(
        model,
        "Bonjour.",
        tmp_path / "vo.wav",
        language="fr-CA",
        reference_audio=None,
        exaggeration=0.5,
    )

    assert model.calls[0]["language_id"] == "fr"
    assert model.calls[0]["cfg_weight"] == 0.0


def test_chatterbox_keeps_cfg_with_reference_voice(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    model = _FakeChatterbox([_take(truncated=False)])

    tts._chatterbox_tts(
        model,
        "Bonjour.",
        tmp_path / "vo.wav",
        language="fr-CA",
        reference_audio=str(reference),
        exaggeration=0.5,
    )

    assert model.calls[0]["audio_prompt_path"] == str(reference)
    assert "cfg_weight" not in model.calls[0]
