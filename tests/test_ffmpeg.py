from pathlib import Path

from clipcli import ffmpeg
from clipcli.models import ClipPlan


def test_render_clip_builds_vertical_ffmpeg_command(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command
        captured["timeout"] = timeout

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    plan = ClipPlan(
        title="Test Clip",
        start=12.5,
        end=42.5,
        summary="summary",
        hook="hook",
        caption="caption",
    )

    captions = tmp_path / "caption file.ass"
    ffmpeg.render_clip(Path("input.mp4"), plan, tmp_path / "clip.mp4", captions=captions)

    command = captured["command"]
    assert command[:5] == ["ffmpeg", "-y", "-ss", "12.500", "-i"]
    assert "-vf" in command
    video_filter = command[command.index("-vf") + 1]
    assert "crop=1080:1920" in video_filter
    assert "subtitles=filename=" in video_filter
    assert command[-1].endswith("clip.mp4")


def test_render_clip_with_broll_builds_filter_complex(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command
        captured["timeout"] = timeout

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    plan = ClipPlan(
        title="Test Clip",
        start=10,
        end=40,
        summary="summary",
        hook="hook",
        caption="caption",
    )
    broll_path = tmp_path / "broll.mp4"
    broll_path.write_bytes(b"placeholder")
    captions = tmp_path / "captions.ass"

    ffmpeg.render_clip(
        Path("input.mp4"),
        plan,
        tmp_path / "clip.mp4",
        captions=captions,
        broll=[ffmpeg.BrollOverlay(path=broll_path, start=5, duration=3)],
    )

    command = captured["command"]
    assert "-filter_complex" in command
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "[0:v]trim=duration=30.000" in filter_complex
    assert "[1:v]trim=duration=3.000" in filter_complex
    assert "overlay=0:0:enable='between(t,5.000,8.000)'" in filter_complex
    assert "fade=t=in" in filter_complex
    assert "fade=t=out" in filter_complex
    assert "subtitles=filename=" in filter_complex
    assert "-map" in command
    assert "0:a?" in command
    assert command[-1].endswith("clip.mp4")


def test_vertical_left_mode_anchors_crop_left(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    plan = ClipPlan(title="x", start=0, end=20, summary="x", hook="x", caption="x")

    ffmpeg.render_clip(Path("input.mp4"), plan, tmp_path / "clip.mp4", mode="vertical_left")

    video_filter = captured["command"][captured["command"].index("-vf") + 1]
    assert "crop=1080:1920:0:" in video_filter


def test_crop_x_overrides_vertical_anchor(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    plan = ClipPlan(title="x", start=0, end=20, summary="x", hook="x", caption="x")

    ffmpeg.render_clip(Path("input.mp4"), plan, tmp_path / "clip.mp4", crop_x=0.15)

    video_filter = captured["command"][captured["command"].index("-vf") + 1]
    assert "crop=1080:1920:(iw-1080)*0.1500:" in video_filter


def test_vertical_auto_uses_detected_crop_anchor(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    class Anchor:
        x = 0.2

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    monkeypatch.setattr(ffmpeg, "detect_speaker_crop_x", lambda *args, **kwargs: Anchor())
    plan = ClipPlan(title="x", start=0, end=20, summary="x", hook="x", caption="x")

    ffmpeg.render_clip(Path("input.mp4"), plan, tmp_path / "clip.mp4", mode="vertical_auto")

    video_filter = captured["command"][captured["command"].index("-vf") + 1]
    assert "crop=1080:1920:(iw-1080)*0.2000:" in video_filter


def test_image_data_uri_encodes_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"abc")

    assert ffmpeg.image_data_uri(path) == "data:image/jpeg;base64,YWJj"


def test_extract_frame_can_crop_reference(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.extract_frame(Path("input.mp4"), tmp_path / "frame.jpg", 21.0, crop_x=0.0)

    command = captured["command"]
    assert "-vf" in command
    assert "crop=ih*9/16:ih:(iw-ih*9/16)*0.0000:0,scale=720:1280" in command


def test_extract_clip_audio_builds_48k_wav_command(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.extract_clip_audio(Path("input.mp4"), tmp_path / "clip.wav", start=12.5, duration=17.25)

    command = captured["command"]
    assert command[:5] == ["ffmpeg", "-y", "-ss", "12.500", "-i"]
    assert "-t" in command
    assert command[command.index("-t") + 1] == "17.250"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_extract_audio_accepts_custom_sample_rate(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.extract_audio(Path("input.mp4"), tmp_path / "audio.wav", sample_rate=48000)

    assert captured["command"][captured["command"].index("-ar") + 1] == "48000"


def test_master_speech_audio_builds_processing_chain(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.master_speech_audio(Path("input.wav"), tmp_path / "mastered.wav")

    command = captured["command"]
    audio_filter = command[command.index("-af") + 1]
    assert "highpass=f=70" in audio_filter
    assert "acompressor=" in audio_filter
    assert "loudnorm=I=-16" in audio_filter
    assert "alimiter=limit=0.95" in audio_filter


def test_replace_audio_copies_video_and_encodes_aac(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.replace_audio(Path("video.mp4"), Path("audio.wav"), tmp_path / "output.mp4")

    command = captured["command"]
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in command
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"


def test_mix_sound_bed_ducks_background_under_voice(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command
        captured["timeout"] = timeout

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.mix_sound_bed(
        Path("video.mp4"),
        Path("sound.mp3"),
        tmp_path / "output.mp4",
        duration=30,
        volume=0.055,
    )

    command = captured["command"]
    assert "-stream_loop" in command
    assert command[command.index("-stream_loop") + 1] == "-1"
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "sidechaincompress=" in filter_complex
    assert "volume=0.0550" in filter_complex
    assert "amix=inputs=2:duration=first" in filter_complex
    assert command[command.index("-c:v") + 1] == "copy"
