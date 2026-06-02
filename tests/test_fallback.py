from clipcli.fallback import fallback_plan_from_transcript
from clipcli.models import Transcript, TranscriptSegment


def test_fallback_plan_selects_contiguous_window() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(text="Intro.", start=0, end=5),
            TranscriptSegment(text="The important connected health claim.", start=5, end=18),
            TranscriptSegment(text="The payoff for citizens at home.", start=18, end=32),
        ]
    )

    plan = fallback_plan_from_transcript(
        transcript,
        clips=1,
        min_seconds=15,
        max_seconds=40,
        summary="summary",
        topics=["Digital Health"],
    )

    assert len(plan.clips) == 1
    assert 15 <= plan.clips[0].duration <= 40
    assert plan.clips[0].hashtags == ["#DigitalHealth"]
    assert "Fallback" in plan.clips[0].summary


def test_fallback_plan_prefers_topic_keywords_over_density() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(text="Generic dense formal words " * 20, start=0, end=20),
            TranscriptSegment(text="Connected health helps patients stay at home.", start=40, end=55),
            TranscriptSegment(text="Remote monitoring is proactive for citizens.", start=55, end=75),
        ]
    )

    plan = fallback_plan_from_transcript(
        transcript,
        clips=1,
        min_seconds=15,
        max_seconds=40,
        topics=["Health Tech"],
    )

    assert plan.clips[0].start == 40
