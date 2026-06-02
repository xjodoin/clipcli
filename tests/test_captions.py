from pathlib import Path

from clipcli.captions import write_ass_captions
from clipcli.models import ClipPlan, Transcript, TranscriptSegment, Word


def test_write_ass_captions_uses_word_timing(tmp_path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="this is a useful clip",
                start=10,
                end=12,
                words=[
                    Word(text="this", start=10.0, end=10.2),
                    Word(text="is", start=10.2, end=10.4),
                    Word(text="useful", start=10.4, end=10.8),
                    Word(text="clip", start=10.8, end=11.2),
                ],
            )
        ]
    )
    clip = ClipPlan(title="x", start=10, end=12, summary="x", hook="x", caption="x")

    path = write_ass_captions(transcript, clip, tmp_path / "captions.ass", max_words=2)
    content = path.read_text()

    assert "[Events]" in content
    assert "THIS IS" in content
    assert "USEFUL CLIP" in content


def test_viral_captions_include_hook_style(tmp_path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="connected health matters",
                start=0,
                end=3,
                words=[
                    Word(text="connected", start=0, end=0.4),
                    Word(text="health", start=0.4, end=0.8),
                    Word(text="matters", start=0.8, end=1.2),
                ],
            )
        ]
    )
    clip = ClipPlan(
        title="x",
        start=0,
        end=3,
        summary="x",
        hook="connected health changes care",
        caption="x",
    )

    path = write_ass_captions(transcript, clip, tmp_path / "viral.ass", style="viral")
    content = path.read_text()

    assert "Style: Hook" in content
    assert "POV: CONNECTED HEALTH CHANGES CARE" in content
