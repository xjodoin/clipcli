from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Word(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bounds(self) -> "TranscriptSegment":
        if self.end <= self.start:
            raise ValueError("segment end must be after start")
        return self


class Transcript(BaseModel):
    source: str | None = None
    language: str | None = None
    segments: list[TranscriptSegment]

    def as_prompt_lines(self, max_chars: int = 90_000) -> str:
        lines: list[str] = []
        used = 0
        for segment in self.segments:
            speaker = f" {segment.speaker}:" if segment.speaker else ""
            line = (
                f"[{format_seconds(segment.start)} - {format_seconds(segment.end)}]"
                f"{speaker} {segment.text.strip()}"
            )
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


class Virality(BaseModel):
    hook_score: int = Field(default=15, ge=0, le=25)
    engagement_score: int = Field(default=15, ge=0, le=25)
    value_score: int = Field(default=15, ge=0, le=25)
    shareability_score: int = Field(default=15, ge=0, le=25)
    total_score: int = Field(default=60, ge=0, le=100)
    hook_type: str = "none"
    reasoning: str = "Selected as a solid standalone clip."


class BrollPrompt(BaseModel):
    at: float = Field(ge=0)
    duration: float = Field(default=4.0, ge=1.0, le=15.0)
    prompt: str
    negative_prompt: str | None = None
    image_url: str | None = None


class ClipPlan(BaseModel):
    title: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    summary: str
    hook: str
    caption: str
    hashtags: list[str] = Field(default_factory=list)
    virality: Virality = Field(default_factory=Virality)
    broll: list[BrollPrompt] = Field(default_factory=list)

    @field_validator("hashtags", mode="before")
    @classmethod
    def normalize_hashtags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [part.strip() for part in value.replace(",", " ").split()]
        return [
            tag if str(tag).startswith("#") else f"#{tag}"
            for tag in value
            if str(tag).strip()
        ]

    @model_validator(mode="after")
    def validate_bounds(self) -> "ClipPlan":
        if self.end <= self.start:
            raise ValueError("clip end must be after start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipPlanResult(BaseModel):
    summary: str
    topics: list[str] = Field(default_factory=list)
    clips: list[ClipPlan]


class PromoAsset(BaseModel):
    """A visual shown instead of source footage: a real logo, a fetched/local image, or a generated one."""

    kind: Literal["generate", "logo", "url", "file"]
    value: str
    """Image prompt (generate), website domain (logo), direct image URL (url), or local path (file)."""
    fit: Literal["cover", "card"] | None = None
    """Override framing: cover fills the frame, card centers on a backdrop. Defaults by kind."""
    card_color: str | None = None
    """Card backdrop color (hex like 0xFFFFFF); defaults to white for file, brand-dark otherwise."""


class PromoScene(BaseModel):
    """One shot of a marketing montage: a source range, a voiceover line, and a key message."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    vo: str = ""
    key_message: str = ""
    duration_hint: float | None = Field(default=None, gt=0)
    asset: PromoAsset | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "PromoScene":
        if self.end <= self.start:
            raise ValueError("scene end must be after start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class PromoPlan(BaseModel):
    """A short marketing montage plan: shots, voiceover script, key messages, music."""

    title: str
    tagline: str = ""
    language: str = "en"
    target_seconds: float = Field(default=30.0, gt=0)
    music_query: str = "uplifting corporate technology music"
    vo_style: str = ""
    scenes: list[PromoScene]

    @property
    def voiceover_script(self) -> str:
        return " ".join(scene.vo.strip() for scene in self.scenes if scene.vo.strip())


class SeedDanceVideo(BaseModel):
    id: str | None = None
    status: str | None = None
    url: str | None = None
    local_path: Path | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class RenderMode(str):
    pass


OutputMode = Literal["vertical", "vertical_auto", "vertical_left", "vertical_right", "original"]
PlannerBackend = Literal["gemini", "gemma"]
RenderStyle = Literal["viral", "clean"]
SoundSource = Literal["freesound"]
SoundIntensity = Literal["low", "medium", "high"]


def format_seconds(seconds: float) -> str:
    whole = int(seconds)
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def parse_timestamp(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    parts = str(value).strip().split(":")
    if not parts:
        raise ValueError("empty timestamp")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    raise ValueError(f"invalid timestamp: {value}")
