import json
import sys
import types
from pathlib import Path

import pytest

from clipcli import ffmpeg, gemma, promo
from clipcli.models import PromoPlan, PromoScene, Transcript, TranscriptSegment
from clipcli.promo import PromoOptions, _load_or_plan


def _transcript() -> Transcript:
    return Transcript(
        language="fr",
        segments=[
            TranscriptSegment(text="Bienvenue au lancement.", start=2.0, end=6.0),
            TranscriptSegment(text="Une plateforme de santé connectée.", start=30.0, end=36.0),
        ],
    )


def _promo_response() -> str:
    return json.dumps(
        {
            "title": "TechCare",
            "tagline": "La santé connectée pour tous",
            "language": "fr",
            "target_seconds": 30,
            "music_query": "uplifting corporate technology",
            "vo_style": "confident and warm",
            "scenes": [
                {"start": 5.0, "end": 9.0, "vo": "La santé se transforme.", "key_message": "SANTÉ CONNECTÉE"},
                {"start": 80.0, "end": 84.0, "vo": "Une plateforme pour tous.", "key_message": ""},
            ],
        }
    )


def _clips_response() -> str:
    return json.dumps(
        {
            "summary": "Lancement d'une plateforme santé.",
            "topics": ["santé"],
            "clips": [
                {
                    "title": "lancement",
                    "start": 10.0,
                    "end": 40.0,
                    "summary": "Annonce clé.",
                    "hook": "La santé se transforme",
                    "caption": "La santé se transforme",
                }
            ],
        }
    )


def _install_fake_mlx(monkeypatch, captured: dict, response_text: str) -> None:
    """Fake mlx_vlm: listening calls (audio attached) return notes; the final call returns the plan."""
    mlx_vlm = types.ModuleType("mlx_vlm")
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    utils = types.ModuleType("mlx_vlm.utils")
    captured["calls"] = []

    def load(name, **kwargs):
        captured["loaded"] = name
        return ("model-object", "processor-object")

    def load_config(name, **kwargs):
        return {"model_type": "gemma4"}

    def apply_chat_template(processor, config, prompt, num_images=0, num_audios=0, **kwargs):
        captured["prompt"] = prompt
        captured["num_images"] = num_images
        captured["num_audios"] = num_audios
        return f"<chat>{prompt}</chat>"

    class _Result:
        def __init__(self, text: str) -> None:
            self.text = text

    def generate(model, processor, prompt, image=None, audio=None, **kwargs):
        captured["image"] = image
        captured["audio"] = audio
        captured["generate_kwargs"] = kwargs
        captured["calls"].append({"prompt": prompt, "image": image, "audio": audio})
        if audio:
            return _Result(f"On annonce le projet Take Care. ({audio[0]})")
        return _Result(response_text)

    mlx_vlm.load = load
    mlx_vlm.generate = generate
    mlx_vlm.prompt_utils = prompt_utils
    mlx_vlm.utils = utils
    prompt_utils.apply_chat_template = apply_chat_template
    utils.load_config = load_config
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    monkeypatch.setattr(gemma, "_MODEL_CACHE", {})


def test_plan_promo_with_gemma_feeds_frames_and_audio(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    _install_fake_mlx(monkeypatch, captured, _promo_response())

    frames = [(tmp_path / f"frame-{i:02d}.jpg", at) for i, at in enumerate((11.4, 102.9, 290.0))]

    def fake_keyframes(source, output_dir, *, count, height=448, duration=None):
        captured["frame_count"] = count
        captured["frame_duration"] = duration
        return frames

    chunk_args: list[tuple[float, float]] = []

    def fake_extract_clip_audio(source, output, *, start, duration, sample_rate=48000):
        captured["sample_rate"] = sample_rate
        chunk_args.append((start, duration))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")
        return output

    monkeypatch.setattr(ffmpeg, "extract_keyframes", fake_keyframes)
    monkeypatch.setattr(ffmpeg, "extract_clip_audio", fake_extract_clip_audio)

    plan = gemma.plan_promo_with_gemma(
        _transcript(),
        video=tmp_path / "in.mp4",
        duration=30.0,
        scenes=6,
        video_duration=365.0,
        model="fake/gemma-promo",
        work_dir=tmp_path / "work",
        raw_path=tmp_path / "raw.json",
    )

    assert plan.title == "TechCare"
    assert len(plan.scenes) == 2
    assert captured["loaded"] == "fake/gemma-promo"
    # Gemma hears at most ~30s per clip and mlx-vlm takes one clip per prompt:
    # 365s becomes 13 listening calls plus the final planning call.
    listen_calls = [call for call in captured["calls"] if call["audio"]]
    assert len(listen_calls) == 13
    assert all(len(call["audio"]) == 1 for call in listen_calls)
    assert "00:00 to 00:30" in listen_calls[0]["prompt"]
    assert "06:00 to 06:05" in listen_calls[-1]["prompt"]
    assert chunk_args[0] == (0.0, 30.0)
    assert chunk_args[-1] == (360.0, pytest.approx(5.0))
    assert captured["sample_rate"] == 16000
    assert captured["frame_count"] == 24  # scenes * 4
    assert captured["frame_duration"] == 365.0
    # The final planning call sees frames plus the timestamped listening notes.
    final = captured["calls"][-1]
    assert final["audio"] is None
    assert final["image"] == [str(path) for path, _at in frames]
    assert captured["num_images"] == 3
    assert captured["num_audios"] == 0
    assert "Frame 1: 00:11 (11.4s)" in captured["prompt"]
    assert "Frame 3: 04:50 (290.0s)" in captured["prompt"]
    assert "[00:00-00:30] On annonce le projet Take Care." in captured["prompt"]
    assert "[06:00-06:05]" in captured["prompt"]
    assert "central announcement" in captured["prompt"]
    assert "commercial editor" in captured["prompt"]  # system guidance folded in
    assert "365 seconds long" in captured["prompt"]
    assert "Bienvenue au lancement." in captured["prompt"]  # WhisperX text grounding
    assert (tmp_path / "raw.json").read_text() == _promo_response()


def test_plan_promo_with_gemma_transcript_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    _install_fake_mlx(monkeypatch, captured, _promo_response())

    plan = gemma.plan_promo_with_gemma(
        _transcript(),
        video=None,
        model="fake/gemma-promo-text",
        work_dir=tmp_path,
    )

    assert plan.title == "TechCare"
    assert captured["num_images"] == 0
    assert captured["num_audios"] == 0
    assert captured["image"] is None
    assert captured["audio"] is None
    assert "still frames" not in captured["prompt"]
    assert "central announcement" in captured["prompt"]  # anchoring survives text-only


def test_plan_promo_with_gemma_requires_transcript_or_video() -> None:
    with pytest.raises(ValueError):
        gemma.plan_promo_with_gemma(None, video=None)


def test_plan_clips_with_gemma_grounds_on_source_audio(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    _install_fake_mlx(monkeypatch, captured, _clips_response())

    def fake_extract_clip_audio(source, output, *, start, duration, sample_rate=48000):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")
        return output

    monkeypatch.setattr(ffmpeg, "extract_clip_audio", fake_extract_clip_audio)
    monkeypatch.setattr(ffmpeg, "probe_duration", lambda path: 65.0)

    result = gemma.plan_clips_with_gemma(
        _transcript(),
        clips=3,
        min_seconds=15,
        max_seconds=60,
        model="fake/gemma-clips",
        source=tmp_path / "in.mp4",
        work_dir=tmp_path / "work",
    )

    assert len(result.clips) == 1
    assert result.clips[0].title == "lancement"
    listen_calls = [call for call in captured["calls"] if call["audio"]]
    assert len(listen_calls) == 3  # 30 + 30 + 5 seconds
    final = captured["calls"][-1]
    assert final["audio"] is None
    assert "[01:00-01:05] On annonce le projet Take Care." in captured["prompt"]
    assert "prefer your listening notes" in captured["prompt"]
    assert "Bienvenue au lancement." in captured["prompt"]


def test_plan_clips_with_gemma_works_without_source(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mlx(monkeypatch, captured, _clips_response())

    result = gemma.plan_clips_with_gemma(
        _transcript(),
        clips=3,
        min_seconds=15,
        max_seconds=60,
        model="fake/gemma-clips-text",
    )

    assert len(result.clips) == 1
    assert captured["num_audios"] == 0
    assert captured["audio"] is None


def test_missing_mlx_vlm_raises_install_hint(monkeypatch) -> None:
    monkeypatch.setattr(gemma, "_MODEL_CACHE", {})
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    with pytest.raises(RuntimeError, match=r"clipcli\[local\]"):
        gemma._load_model("fake/missing")


def test_promo_load_or_plan_routes_to_gemma_without_proxy(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    plan = PromoPlan(
        title="TechCare",
        scenes=[PromoScene(start=5.0, end=9.0, vo="La santé se transforme.")],
    )

    def fake_plan_promo(transcript, **kwargs):
        captured["transcript"] = transcript
        captured.update(kwargs)
        return plan

    def forbid_proxy(*args, **kwargs):
        raise AssertionError("gemma planning must not build a cloud upload proxy")

    whisper_transcript = _transcript()
    monkeypatch.setattr(promo, "plan_promo_with_gemma", fake_plan_promo)
    monkeypatch.setattr(promo, "_transcribe_for_gemma", lambda source, work_dir: whisper_transcript)
    monkeypatch.setattr(ffmpeg, "make_proxy", forbid_proxy)

    options = PromoOptions(
        source=tmp_path / "in.mp4",
        output_dir=tmp_path,
        planner="gemma",
        gemma_model="fake/gemma-routing",
    )

    result = _load_or_plan(options, tmp_path / "in.mp4", tmp_path / "work", video_duration=365.0)

    assert result is plan
    assert captured["video"] == tmp_path / "in.mp4"
    assert captured["model"] == "fake/gemma-routing"
    assert captured["video_duration"] == 365.0
    assert captured["work_dir"] == tmp_path / "work"
    # WhisperX text grounding is generated automatically when none is supplied.
    assert captured["transcript"] is whisper_transcript


def test_extract_keyframes_samples_evenly(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(ffmpeg, "run", lambda command, timeout=None: commands.append(command))

    frames = ffmpeg.extract_keyframes(
        tmp_path / "in.mp4",
        tmp_path / "frames",
        count=4,
        duration=100.0,
    )

    assert [at for _path, at in frames] == [12.5, 37.5, 62.5, 87.5]
    assert len(commands) == 4
    assert commands[0][commands[0].index("-ss") + 1] == "12.500"
    assert "scale=-2:448" in commands[0]
    assert frames[0][0].name == "frame-00.jpg"
