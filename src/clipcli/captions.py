from __future__ import annotations

from pathlib import Path

from .models import ClipPlan, RenderStyle, Transcript, TranscriptSegment, Word


def write_ass_captions(
    transcript: Transcript,
    clip: ClipPlan,
    output: Path,
    *,
    max_words: int = 7,
    max_chars: int = 48,
    style: RenderStyle = "viral",
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if style == "viral":
        max_words = min(max_words, 5)
        max_chars = min(max_chars, 32)
    events = list(_caption_events(transcript, clip, max_words=max_words, max_chars=max_chars))
    output.write_text(_ass_document(events, clip=clip, style=style))
    return output


def _caption_events(
    transcript: Transcript,
    clip: ClipPlan,
    *,
    max_words: int,
    max_chars: int,
):
    words = _clip_words(transcript, clip)
    if words:
        chunk: list[Word] = []
        for word in words:
            candidate = chunk + [word]
            text = _words_text(candidate)
            if chunk and (len(candidate) > max_words or len(text) > max_chars):
                yield _relative(chunk[0].start, clip.start), _relative(chunk[-1].end, clip.start), _words_text(chunk)
                chunk = [word]
            else:
                chunk = candidate
        if chunk:
            yield _relative(chunk[0].start, clip.start), _relative(chunk[-1].end, clip.start), _words_text(chunk)
        return

    for segment in transcript.segments:
        if segment.end <= clip.start or segment.start >= clip.end:
            continue
        start = _relative(max(segment.start, clip.start), clip.start)
        end = _relative(min(segment.end, clip.end), clip.start)
        for text in _chunk_text(segment.text, max_chars=max_chars):
            yield start, end, text


def _clip_words(transcript: Transcript, clip: ClipPlan) -> list[Word]:
    words: list[Word] = []
    for segment in transcript.segments:
        for word in segment.words:
            if word.end > clip.start and word.start < clip.end and word.text:
                words.append(word)
    return sorted(words, key=lambda item: item.start)


def _words_text(words: list[Word]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _relative(value: float, clip_start: float) -> float:
    return max(0.0, value - clip_start)


def _ass_document(
    events: list[tuple[float, float, str]],
    *,
    clip: ClipPlan,
    style: RenderStyle,
) -> str:
    if style == "viral":
        return _viral_ass_document(events, clip)
    return _clean_ass_document(events)


def _clean_ass_document(events: list[tuple[float, float, str]]) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Default,Arial,82,&H00FFFFFF,&H0000FFFF,&H00000000,&HAA000000,"
            "-1,0,0,0,100,100,0,0,1,5,0,2,70,70,220,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in events:
        if end <= start:
            end = start + 0.5
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_escape_ass(text.upper())}"
        )
    return "\n".join(lines) + "\n"


def _viral_ass_document(events: list[tuple[float, float, str]], clip: ClipPlan) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Default,Arial,88,&H00FFFFFF,&H0000FFFF,&H00000000,&HAA000000,"
            "-1,0,0,0,100,100,0,0,1,6,1,2,70,70,205,1"
        ),
        (
            "Style: Highlight,Arial,88,&H0000D7FF,&H0000FFFF,&H00000000,&HAA000000,"
            "-1,0,0,0,100,100,0,0,1,6,1,2,70,70,205,1"
        ),
        (
            "Style: Hook,Arial,58,&H0000D7FF,&H0000FFFF,&H00000000,&HAA000000,"
            "-1,0,0,0,100,100,0,0,1,5,1,8,64,64,92,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hook = _viral_hook_text(clip)
    if hook:
        lines.append(
            f"Dialogue: 1,0:00:00.00,{_ass_time(min(4.2, clip.duration))},Hook,,0,0,0,,"
            f"{{\\fad(120,220)}}{_escape_ass(hook)}"
        )
    for index, (start, end, text) in enumerate(events):
        if end <= start:
            end = start + 0.5
        style_name = "Highlight" if index % 3 == 0 else "Default"
        lines.append(
            f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},{style_name},,0,0,0,,"
            f"{{\\fad(45,45)}}{_escape_ass(text.upper())}"
        )
    return "\n".join(lines) + "\n"


def _viral_hook_text(clip: ClipPlan) -> str:
    text = (clip.hook or clip.title).strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > 54:
        text = text[:51].rstrip() + "..."
    return f"POV: {text.upper()}"


def write_promo_overlays(
    key_messages: list[tuple[float, float, str]],
    output: Path,
    *,
    width: int,
    height: int,
    title: str = "",
    tagline: str = "",
    end_start: float | None = None,
    end_end: float | None = None,
) -> Path:
    """Write the promo ASS overlay: per-scene key messages plus an end card."""
    output.parent.mkdir(parents=True, exist_ok=True)
    base = min(width, height)
    key_size = max(28, int(base * 0.058))
    title_size = max(40, int(base * 0.085))
    tagline_size = max(24, int(base * 0.042))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Key,Arial,{key_size},&H00FFFFFF,&H0000FFFF,&H00101010,&H66000000,"
            f"-1,0,0,0,100,100,1,0,1,4,2,2,{int(width * 0.08)},{int(width * 0.08)},{int(height * 0.085)},1"
        ),
        (
            f"Style: Title,Arial,{title_size},&H00FFFFFF,&H0000FFFF,&H00101010,&H66000000,"
            f"-1,0,0,0,100,100,2,0,1,0,0,5,{int(width * 0.06)},{int(width * 0.06)},0,1"
        ),
        (
            f"Style: Tagline,Arial,{tagline_size},&H00E6D7A8,&H0000FFFF,&H00101010,&H66000000,"
            f"0,0,0,0,100,100,1,0,1,0,0,5,{int(width * 0.06)},{int(width * 0.06)},0,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in key_messages:
        message = " ".join(text.split())
        if not message or end <= start:
            continue
        lines.append(
            f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Key,,0,0,0,,"
            f"{{\\fad(180,180)}}{_escape_ass(message.upper())}"
        )
    if end_start is not None and end_end is not None and end_end > end_start:
        if title.strip():
            offset = int(height * 0.05)
            lines.append(
                f"Dialogue: 1,{_ass_time(end_start)},{_ass_time(end_end)},Title,,0,0,0,,"
                f"{{\\fad(260,0)\\pos({width // 2},{height // 2 - offset})}}"
                f"{_escape_ass(title.strip().upper())}"
            )
        if tagline.strip():
            offset = int(height * 0.06)
            lines.append(
                f"Dialogue: 1,{_ass_time(min(end_end, end_start + 0.35))},{_ass_time(end_end)},Tagline,,0,0,0,,"
                f"{{\\fad(260,0)\\pos({width // 2},{height // 2 + offset})}}"
                f"{_escape_ass(tagline.strip())}"
            )
    output.write_text("\n".join(lines) + "\n")
    return output


def _ass_time(seconds: float) -> str:
    centiseconds = int(round(seconds * 100))
    total_seconds, cs = divmod(centiseconds, 100)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
