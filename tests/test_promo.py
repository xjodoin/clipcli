import json
from pathlib import Path

import pytest

from clipcli import ffmpeg, promo
from clipcli.captions import write_promo_overlays
from clipcli.gemini import parse_promo_json
from clipcli.models import PromoScene
from clipcli.promo import PromoOptions, _montage_timing, _synthesize_and_time
from clipcli.tts import VoiceoverLine


def _raw_plan() -> dict:
    return {
        "title": "TechCare",
        "tagline": "La santé connectée pour tous",
        "language": "fr",
        "target_seconds": 30,
        "music_query": "uplifting corporate technology",
        "vo_style": "confident and warm",
        "scenes": [
            {"start": 5.0, "end": 9.0, "vo": "La santé se transforme.", "key_message": "SANTÉ CONNECTÉE"},
            {"start": 80.0, "end": 84.0, "vo": "Une plateforme pour tous.", "key_message": ""},
            {"start": 300.0, "end": 305.0, "voiceover": "Dès aujourd'hui.", "text": "DISPONIBLE MAINTENANT"},
        ],
    }


def test_parse_promo_json_normalizes_and_drops_out_of_range_scenes() -> None:
    raw = _raw_plan()
    raw["scenes"].append({"start": 500.0, "end": 520.0, "vo": "hors limites"})
    raw["scenes"].append({"start": 360.0, "end": 380.0, "vo": "léger dépassement"})

    plan = parse_promo_json(f"```json\n{json.dumps(raw)}\n```", video_duration=365.0)

    assert plan.title == "TechCare"
    assert plan.language == "fr"
    assert len(plan.scenes) == 4  # the hallucinated 500s scene is dropped
    assert plan.scenes[2].vo == "Dès aujourd'hui."
    assert plan.scenes[2].key_message == "DISPONIBLE MAINTENANT"
    assert plan.scenes[3].start == 360.0
    assert plan.scenes[3].end == 365.0  # overrun trimmed to the video end
    assert "transforme" in plan.voiceover_script


def test_parse_promo_json_rejects_empty_scenes() -> None:
    with pytest.raises(ValueError):
        parse_promo_json(json.dumps({"title": "x", "scenes": []}))


def test_synthesize_and_time_aligns_scene_durations_to_voiceover(monkeypatch, tmp_path: Path) -> None:
    plan = parse_promo_json(json.dumps(_raw_plan()), video_duration=365.0)

    def fake_synthesize(lines, output_dir, **kwargs):
        return [
            VoiceoverLine(text=text, path=tmp_path / f"vo-{index}.wav", duration=3.0 + index)
            for index, text in enumerate(lines)
        ]

    monkeypatch.setattr(promo, "synthesize_voiceover_lines", fake_synthesize)
    options = PromoOptions(source=Path("in.mp4"), output_dir=tmp_path, duration=15.0)

    timed = _synthesize_and_time(options, plan, tmp_path, video_duration=365.0)

    assert timed[0].duration == pytest.approx(3.7)
    assert timed[1].duration == pytest.approx(4.7)
    assert timed[2].duration == pytest.approx(5.7)
    assert timed[0].montage_start == 0.0
    assert timed[1].montage_start == pytest.approx(3.7 - 0.35)
    total, end_card_start = _montage_timing(timed, options)
    assert end_card_start == pytest.approx(timed[2].montage_start + 5.7 - 0.35)
    assert total == pytest.approx(end_card_start + options.end_card_seconds)


def test_synthesize_and_time_nudges_overlapping_shots_forward(monkeypatch, tmp_path: Path) -> None:
    raw = _raw_plan()
    raw["scenes"] = [
        {"start": 53.0, "end": 55.4, "vo": "Première phrase courte.", "key_message": ""},
        {"start": 54.8, "end": 58.5, "vo": "Deuxième phrase courte.", "key_message": ""},
        {"start": 200.0, "end": 204.0, "vo": "Très loin dans la vidéo.", "key_message": ""},
    ]
    plan = parse_promo_json(json.dumps(raw), video_duration=365.0)

    def fake_synthesize(lines, output_dir, **kwargs):
        return [
            VoiceoverLine(text=text, path=tmp_path / f"vo-{index}.wav", duration=4.0)
            for index, text in enumerate(lines)
        ]

    monkeypatch.setattr(promo, "synthesize_voiceover_lines", fake_synthesize)
    options = PromoOptions(source=Path("in.mp4"), output_dir=tmp_path, duration=15.0)

    timed = _synthesize_and_time(options, plan, tmp_path, video_duration=365.0)

    assert timed[0].start == 53.0
    assert timed[1].start == pytest.approx(53.0 + 4.7)  # nudged past the first shot
    assert timed[2].start == 200.0  # large jumps are intentional and untouched


def test_synthesize_and_time_fills_toward_target_with_fast_narration(monkeypatch, tmp_path: Path) -> None:
    plan = parse_promo_json(json.dumps(_raw_plan()), video_duration=365.0)

    def fake_synthesize(lines, output_dir, **kwargs):
        return [
            VoiceoverLine(text=text, path=tmp_path / f"vo-{index}.wav", duration=2.0)
            for index, text in enumerate(lines)
        ]

    monkeypatch.setattr(promo, "synthesize_voiceover_lines", fake_synthesize)
    options = PromoOptions(source=Path("in.mp4"), output_dir=tmp_path, duration=30.0)

    timed = _synthesize_and_time(options, plan, tmp_path, video_duration=365.0)

    # Base scene length would be 2.7s; the deficit pushes each up by the 1.5s cap.
    assert all(item.duration == pytest.approx(2.7 + 1.5) for item in timed)


def test_concat_montage_builds_xfade_chain(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    monkeypatch.setattr(ffmpeg, "probe_duration", lambda path: 4.0)

    segments = [tmp_path / "01.mp4", tmp_path / "02.mp4", tmp_path / "03.mp4"]
    captions = tmp_path / "overlays.ass"
    ffmpeg.concat_montage(segments, tmp_path / "out.mp4", captions=captions)

    command = captured["command"]
    filter_complex = command[command.index("-filter_complex") + 1]
    assert filter_complex.count("xfade=transition=fade") == 2
    assert "offset=3.650" in filter_complex
    assert "offset=7.300" in filter_complex
    assert "subtitles=filename=" in filter_complex
    assert "-an" in command


def test_mix_voiceover_music_builds_ducked_mix(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, timeout=None):
        captured["command"] = command

    monkeypatch.setattr(ffmpeg, "run", fake_run)

    ffmpeg.mix_voiceover_music(
        tmp_path / "silent.mp4",
        [(tmp_path / "vo-01.wav", 0.45), (tmp_path / "vo-02.wav", 4.1)],
        tmp_path / "promo.mp4",
        music=tmp_path / "bed.mp3",
        duration=30.0,
        music_volume=0.3,
    )

    command = captured["command"]
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "adelay=450|450" in filter_complex
    assert "adelay=4100|4100" in filter_complex
    assert "sidechaincompress" in filter_complex
    assert "volume=0.3000" in filter_complex
    assert "silenceremove=start_periods=1" in filter_complex
    assert filter_complex.count("loudnorm") == 2  # voiceover and music bed
    assert "-stream_loop" in command


def test_mix_voiceover_music_requires_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ffmpeg, "run", lambda command, timeout=None: None)
    with pytest.raises(ffmpeg.FfmpegError):
        ffmpeg.mix_voiceover_music(
            tmp_path / "silent.mp4",
            [],
            tmp_path / "promo.mp4",
            music=None,
            duration=30.0,
        )


def test_write_promo_overlays_emits_key_messages_and_end_card(tmp_path: Path) -> None:
    output = write_promo_overlays(
        [(0.3, 3.5, "Santé connectée"), (4.0, 7.0, "")],
        tmp_path / "overlays.ass",
        width=1920,
        height=1080,
        title="TechCare",
        tagline="La santé connectée pour tous",
        end_start=25.0,
        end_end=28.0,
    )

    content = output.read_text()
    assert "SANTÉ CONNECTÉE" in content
    assert content.count("Dialogue:") == 3
    assert "TECHCARE" in content
    assert "La santé connectée pour tous" in content
    assert "PlayResX: 1920" in content


def test_promo_scene_validates_bounds() -> None:
    with pytest.raises(ValueError):
        PromoScene(start=10.0, end=10.0)


def test_resolve_vo_voice_extracts_source_reference(monkeypatch, tmp_path: Path) -> None:
    from clipcli import ffmpeg as ffmpeg_module
    from clipcli.promo import _resolve_vo_voice

    captured = {}

    def fake_extract(source, output, *, start, duration, sample_rate=48000):
        captured["start"] = start
        captured["duration"] = duration
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")
        return output

    monkeypatch.setattr(ffmpeg_module, "extract_clip_audio", fake_extract)
    options = PromoOptions(
        source=tmp_path / "in.mp4",
        output_dir=tmp_path,
        vo_voice="source:278.6-296",
    )

    voice = _resolve_vo_voice(options, tmp_path / "work")

    assert captured["start"] == 278.6
    assert captured["duration"] == pytest.approx(17.4)
    assert voice.endswith("reference.wav")


def test_resolve_vo_voice_passes_plain_values_through(tmp_path: Path) -> None:
    from clipcli.promo import _resolve_vo_voice

    options = PromoOptions(source=tmp_path / "in.mp4", output_dir=tmp_path, vo_voice="Charon")
    assert _resolve_vo_voice(options, tmp_path) == "Charon"
    options.vo_voice = None
    assert _resolve_vo_voice(options, tmp_path) is None
