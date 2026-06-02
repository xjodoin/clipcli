from __future__ import annotations

from .models import ClipPlan, ClipPlanResult, Transcript, Virality


TOPIC_KEYWORDS = (
    "santé",
    "sante",
    "health",
    "patient",
    "patients",
    "domicile",
    "home",
    "connect",
    "connectée",
    "connected",
    "citoyen",
    "citizens",
    "remote",
    "monitoring",
    "proactive",
    "hôpital",
    "hopital",
)


def fallback_plan_from_transcript(
    transcript: Transcript,
    *,
    clips: int,
    min_seconds: int,
    max_seconds: int,
    summary: str = "",
    topics: list[str] | None = None,
) -> ClipPlanResult:
    windows = _candidate_windows(transcript, min_seconds=min_seconds, max_seconds=max_seconds)
    clip_plans: list[ClipPlan] = []
    for index, window in enumerate(windows[:clips], start=1):
        text = _window_text(transcript, window[0], window[1])
        hook = _first_sentence(text) or "A key moment from the source video"
        clip_plans.append(
            ClipPlan(
                title=f"fallback-clip-{index}",
                start=window[0],
                end=window[1],
                summary=(
                    "Fallback clip selected from a contiguous transcript window "
                    "because Gemini returned no clip candidates."
                ),
                hook=hook,
                caption=hook,
                hashtags=[_topic_hashtag(topic) for topic in (topics or [])[:3]],
                virality=Virality(
                    hook_score=12,
                    engagement_score=12,
                    value_score=16,
                    shareability_score=10,
                    total_score=50,
                    hook_type="statement",
                    reasoning="Deterministic fallback based on contiguous speech timing.",
                ),
            )
        )
    return ClipPlanResult(
        summary=summary or "Fallback plan generated from transcript timing.",
        topics=topics or [],
        clips=clip_plans,
    )


def _candidate_windows(
    transcript: Transcript,
    *,
    min_seconds: int,
    max_seconds: int,
) -> list[tuple[float, float]]:
    segments = [segment for segment in transcript.segments if segment.text.strip()]
    if not segments:
        return []

    windows: list[tuple[float, float, int]] = []
    for start_index, start_segment in enumerate(segments):
        text_chars = 0
        text_parts: list[str] = []
        for segment in segments[start_index:]:
            duration = segment.end - start_segment.start
            if duration > max_seconds:
                break
            text_chars += len(segment.text)
            text_parts.append(segment.text)
            if duration >= min_seconds:
                windows.append((start_segment.start, segment.end, _score_window(" ".join(text_parts), text_chars)))
                break
    windows.sort(key=lambda item: (item[2], item[1] - item[0]), reverse=True)
    return [(start, end) for start, end, _score in windows]


def _score_window(text: str, text_chars: int) -> int:
    normalized = text.lower()
    keyword_hits = sum(normalized.count(keyword) for keyword in TOPIC_KEYWORDS)
    return text_chars + keyword_hits * 500


def _window_text(transcript: Transcript, start: float, end: float) -> str:
    parts = [
        segment.text.strip()
        for segment in transcript.segments
        if segment.end > start and segment.start < end and segment.text.strip()
    ]
    return " ".join(parts)


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    for marker in (". ", "? ", "! "):
        if marker in normalized:
            return normalized.split(marker, 1)[0].strip() + marker.strip()
    return normalized[:120].strip()


def _topic_hashtag(topic: str) -> str:
    normalized = "".join(char for char in topic.title() if char.isalnum())
    return f"#{normalized}" if normalized else "#Clip"
