import json
from pathlib import Path

from clipcli import pipeline
from clipcli.models import BrollPrompt, ClipPlan, ClipPlanResult


def test_generate_broll_accepts_custom_reference_anchor(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class Client:
        def create_video(self, prompt: BrollPrompt):
            captured["image_url"] = prompt.image_url

            class Video:
                url = "https://example.com/video.mp4"

            return Video()

        def download(self, video, output: Path):
            output.parent.mkdir(parents=True)
            output.write_bytes(b"video")
            return output

    monkeypatch.setattr(pipeline, "SeedDanceClient", Client)
    monkeypatch.setattr(
        pipeline.ffmpeg,
        "extract_frame",
        lambda source, output, at, crop_x=None: captured.setdefault("crop_x", crop_x) or output,
    )
    monkeypatch.setattr(pipeline.ffmpeg, "image_data_uri", lambda path: "data:image/jpeg;base64,abc")

    plan = ClipPlanResult(
        summary="x",
        clips=[
            ClipPlan(
                title="x",
                start=0,
                end=20,
                summary="x",
                hook="x",
                caption="x",
                broll=[BrollPrompt(at=4, duration=4, prompt="x")],
            )
        ],
    )

    result = pipeline._generate_broll(
        plan,
        tmp_path / "broll",
        source=tmp_path / "source.mp4",
        work_dir=tmp_path / "work",
        limit=1,
        reference_mode="source-frame:0.12",
    )

    assert captured["crop_x"] == 0.12
    assert captured["image_url"] == "data:image/jpeg;base64,abc"
    assert result[1][0].path.exists()
