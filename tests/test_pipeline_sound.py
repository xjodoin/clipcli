from pathlib import Path

from clipcli import pipeline
from clipcli.models import ClipPlan, ClipPlanResult, Transcript, TranscriptSegment
from clipcli.sound import SoundAsset, SoundCandidate
from clipcli.transcribe import save_transcript


def test_generate_can_mix_sound_bed_after_render(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    transcript_path = tmp_path / "transcript.json"
    save_transcript(
        Transcript(segments=[TranscriptSegment(text="hello", start=0, end=20)]),
        transcript_path,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        ClipPlanResult(
            summary="x",
            clips=[ClipPlan(title="Test Clip", start=0, end=20, summary="x", hook="x", caption="x")],
        ).model_dump_json()
    )

    captured = {}
    monkeypatch.setattr(pipeline.ffmpeg, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "write_ass_captions", lambda *args, **kwargs: tmp_path / "captions.ass")

    def fake_render_clip(source, clip, output, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered")
        return output

    def fake_mix_sound_bed(video, sound, output, **kwargs):
        captured["video"] = video
        captured["sound"] = sound
        captured["kwargs"] = kwargs
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mixed")
        return output

    sound_path = tmp_path / "sound.mp3"
    sound_path.write_bytes(b"sound")
    metadata_path = tmp_path / "sound.json"
    candidate = SoundCandidate(
        source="freesound",
        id="1",
        title="bed",
        author="artist",
        page_url="https://freesound.org/s/1/",
        preview_url="https://example.com/bed.mp3",
        license="Creative Commons 0",
        duration=20,
        tags=[],
    )
    monkeypatch.setattr(pipeline.ffmpeg, "render_clip", fake_render_clip)
    monkeypatch.setattr(pipeline.ffmpeg, "mix_sound_bed", fake_mix_sound_bed)
    monkeypatch.setattr(
        pipeline,
        "find_sound_bed",
        lambda *args, **kwargs: SoundAsset(sound_path, metadata_path, candidate),
    )

    result = pipeline.generate(
        pipeline.GenerateOptions(
            source=source,
            output_dir=tmp_path / "out",
            clips=1,
            transcript_path=transcript_path,
            plan_path=plan_path,
            sound_search=True,
        )
    )

    assert result.clips[0].read_bytes() == b"mixed"
    assert result.sounds == [metadata_path]
    assert captured["sound"] == sound_path
    assert captured["kwargs"]["volume"] == 0.055
