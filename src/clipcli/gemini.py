from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import (
    BrollPrompt,
    ClipPlan,
    ClipPlanResult,
    PromoPlan,
    Transcript,
    Virality,
    parse_timestamp,
)


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


SYSTEM_PROMPT = """You are a state-of-the-art short-form video producer.

Return JSON only. Select source-grounded clip ranges from the transcript. Do not invent spoken content or non-contiguous ranges.

Each clip must be a self-contained short with a strong opening hook, clear payoff, and social caption metadata. Prefer concrete claims, emotional turns, useful lessons, surprising numbers, disagreement, before/after framing, and complete mini-stories.

Never return an empty clips list when the transcript contains usable speech. If no segment is ideal, choose the strongest available contiguous range and explain the limitation in the clip summary.

The JSON schema is:
{
  "summary": "string",
  "topics": ["string"],
  "clips": [
    {
      "title": "short filename-safe title",
      "start": 12.3,
      "end": 45.6,
      "summary": "why this works",
      "hook": "the first on-screen hook",
      "caption": "social caption",
      "hashtags": ["#tag"],
      "virality": {
        "hook_score": 0,
        "engagement_score": 0,
        "value_score": 0,
        "shareability_score": 0,
        "total_score": 0,
        "hook_type": "question|statement|statistic|story|contrast|none",
        "reasoning": "brief grounded rationale"
      },
      "broll": [
        {
          "at": 16.0,
          "duration": 4.0,
          "prompt": "SeedDance-ready cinematic vertical video prompt",
          "negative_prompt": "optional"
        }
      ]
    }
  ]
}
"""


PROMO_SYSTEM_PROMPT = """You are a senior commercial editor and copywriter for product launch videos.

Return JSON only. Plan a short, high-energy marketing montage cut from the provided source video. The montage replaces the original audio with a professional voiceover and a music bed, so pick shots for their VISUAL value: product close-ups, demos, screens, audience energy, smiles, handshakes, wide establishing shots. Avoid long static talking-head shots; if a speaker shot is essential keep it under 3 seconds.

Rules:
- Write the voiceover and key messages in the requested language, in a confident, warm, advertising tone. No filler, no hedging.
- The voiceover must be grounded in what the video actually says and shows. Do not invent product claims.
- One short voiceover line per scene (max 14 words) and one optional on-screen key message per scene (max 6 words, punchy, no final period).
- Scene source ranges must use original-video timestamps in seconds and must not overlap.
- Open with the strongest visual hook, close with the product/brand promise.
- music_query is always concise English keywords for a royalty-free music search.

The JSON schema is:
{
  "title": "short brand/product title for the end card",
  "tagline": "one-line promise for the end card",
  "language": "fr",
  "target_seconds": 30,
  "music_query": "uplifting corporate technology music",
  "vo_style": "short direction for the narrator, in English",
  "scenes": [
    {
      "start": 12.3,
      "end": 16.1,
      "vo": "one voiceover sentence",
      "key_message": "ON-SCREEN KEY MESSAGE",
      "asset": {"kind": "generate|logo|url|file", "value": "image prompt, organization website domain, direct image URL, or provided local file path"}
    }
  ]
}

The "asset" field is optional: when present the scene shows that visual instead of source footage. Use it sparingly — partner/branding cards ("logo" with the organization's website domain, or "file" with the exact path of a provided document image) or a diagram/visual the footage cannot show ("generate" with a detailed image prompt). Never use "generate" for a real organization's logo.
"""


def plan_clips_with_gemini(
    transcript: Transcript,
    *,
    clips: int,
    min_seconds: int,
    max_seconds: int,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
) -> ClipPlanResult:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed.") from exc

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini planning.")

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(transcript, clips=clips, min_seconds=min_seconds, max_seconds=max_seconds)
    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.4,
    }
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    except TypeError:
        pass

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = getattr(response, "text", "") or ""
    return parse_plan_json(text, min_seconds=min_seconds, max_seconds=max_seconds)


def plan_promo_with_gemini(
    transcript: Transcript | None,
    *,
    video: Path | None = None,
    duration: float = 30.0,
    scenes: int = 6,
    language: str | None = None,
    video_duration: float | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    document: Any | None = None,
    raw_path: Path | None = None,
) -> PromoPlan:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed.") from exc

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini planning.")
    if transcript is None and video is None:
        raise ValueError("Promo planning needs a transcript, a video, or both.")

    client = genai.Client(api_key=api_key)
    prompt = _build_promo_prompt(
        transcript,
        duration=duration,
        scenes=scenes,
        language=language,
        with_video=video is not None,
        video_duration=video_duration,
        document=document,
    )
    contents: list[Any] = []
    uploaded = None
    if video is not None:
        uploaded = _upload_video(client, video)
        contents.append(uploaded)
    contents.append(prompt)

    config_kwargs: dict[str, Any] = {
        "system_instruction": PROMO_SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.5,
    }
    try:
        # Shot selection needs real visual reasoning over the footage; plan with
        # a higher thinking budget than transcript-only clip planning.
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")
    except TypeError:
        pass

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
    text = getattr(response, "text", "") or ""
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text)
    return parse_promo_json(text, video_duration=video_duration)


def parse_promo_json(text: str, *, video_duration: float | None = None) -> PromoPlan:
    data = json.loads(_json_payload(text))
    scenes = []
    for item in data.get("scenes", []) or []:
        scene = dict(item)
        if "start" in scene:
            scene["start"] = parse_timestamp(scene["start"])
        if "end" in scene:
            scene["end"] = parse_timestamp(scene["end"])
        if video_duration is not None and not scene.get("asset"):
            # Drop hallucinated out-of-range shots; only trim ones that overrun
            # slightly. Asset scenes don't read the source, so they are exempt.
            if float(scene.get("start", 0.0)) >= video_duration - 1.0:
                continue
            scene["end"] = min(float(scene.get("end", scene["start"] + 3.0)), video_duration)
            if scene["end"] <= scene["start"]:
                continue
        scene.setdefault("vo", scene.get("voiceover", ""))
        scene.setdefault("key_message", scene.get("text", ""))
        scenes.append(scene)
    data["scenes"] = scenes
    try:
        plan = PromoPlan.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Gemini returned invalid promo JSON: {exc}") from exc
    if not plan.scenes:
        raise ValueError("Gemini returned a promo plan with no usable scenes.")
    return plan


def _build_promo_prompt(
    transcript: Transcript | None,
    *,
    duration: float,
    scenes: int,
    language: str | None,
    with_video: bool,
    video_duration: float | None = None,
    document: Any | None = None,
) -> str:
    target_language = language or (transcript.language if transcript else None) or "the source language"
    lines = [
        f"Plan a marketing montage of about {duration:.0f} seconds with {max(3, scenes - 1)} to {scenes + 1} scenes.",
        "",
        "Constraints:",
        f"- Write voiceover, key messages, title, and tagline in: {target_language}.",
        (
            "- Anchor the montage on the video's central announcement: name the specific "
            "product or project being launched and make every voiceover line tell its "
            "story (what it is, who it serves, what it promises) instead of generic praise."
        ),
    ]
    if video_duration is not None:
        lines.append(
            f"- The source video is {video_duration:.0f} seconds long. Every scene start and end "
            f"must be real timestamps between 0 and {video_duration:.0f} seconds."
        )
    lines += [
        f'- Set "language" to the BCP-47 code of that language and "target_seconds" to {duration:.0f}.',
        (
            f"- HARD LIMIT: the combined voiceover lines must total at most {int(duration * 1.7)} words "
            "so the narration fits the target duration. Short, punchy sentences."
        ),
    ]
    if with_video:
        lines.append(
            "- Watch the attached video and choose visually strong, non-overlapping shots; "
            "timestamps must match the attached video."
        )
    if document is not None:
        lines.extend(["", document.as_prompt_block()])
        if getattr(document, "images", None):
            lines.append(
                "Images extracted from the document are available as scene assets — "
                "reference them with asset kind \"file\" and the exact path:"
            )
            lines.extend(f"- {image}" for image in document.images)
    if transcript is not None:
        lines.extend(
            [
                "- The transcript below is machine-generated and may contain recognition errors; "
                "trust the video audio over the transcript when they disagree.",
                "",
                "Transcript:",
                transcript.as_prompt_lines(),
            ]
        )
    return "\n".join(lines)


def _upload_video(client: Any, video: Path, *, timeout: float = 600.0) -> Any:
    uploaded = client.files.upload(file=str(video))
    waited = 0.0
    interval = 5.0
    while str(getattr(uploaded, "state", "")).upper().endswith("PROCESSING"):
        if waited >= timeout:
            raise RuntimeError(f"Gemini file processing timed out for {video.name}")
        time.sleep(interval)
        waited += interval
        uploaded = client.files.get(name=uploaded.name)
    state = str(getattr(uploaded, "state", "")).upper()
    if not state.endswith("ACTIVE"):
        raise RuntimeError(f"Gemini file upload failed for {video.name}: {state or 'unknown state'}")
    return uploaded


def parse_plan_json(
    text: str,
    *,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
) -> ClipPlanResult:
    data = json.loads(_json_payload(text))
    if isinstance(data, list):
        data = {"summary": "", "topics": [], "clips": data}
    if "clips" not in data and "most_relevant_segments" in data:
        data["clips"] = data["most_relevant_segments"]
    if "topics" not in data and "key_topics" in data:
        data["topics"] = data["key_topics"]
    data["clips"] = [_normalize_clip(item) for item in data.get("clips", [])]
    try:
        result = ClipPlanResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Gemini returned invalid clip JSON: {exc}") from exc

    if min_seconds is not None or max_seconds is not None:
        filtered = []
        for clip in result.clips:
            if min_seconds is not None and clip.duration < min_seconds:
                continue
            if max_seconds is not None and clip.duration > max_seconds:
                continue
            filtered.append(clip)
        result.clips = filtered
    return result


def _build_prompt(
    transcript: Transcript,
    *,
    clips: int,
    min_seconds: int,
    max_seconds: int,
) -> str:
    return f"""Select the best {clips} clips from this transcript.

Constraints:
- Each clip must be {min_seconds}-{max_seconds} seconds.
- Times must refer to the original source video.
- Use only transcript evidence.
- Include 1-3 SeedDance b-roll prompts per clip only when visuals would materially improve the edit.
- Return the highest-value clips first.
- Return at least one clip when the transcript contains speech. Do not return an empty clips list just because the content is formal, technical, French, or not obviously viral.
- If the transcript is not in English, keep clip timing grounded in the original transcript and write metadata in concise English.

Transcript:
{transcript.as_prompt_lines()}
"""


def _json_payload(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    start = min([idx for idx in (text.find("{"), text.find("[")) if idx >= 0], default=-1)
    if start < 0:
        raise ValueError("No JSON object found in Gemini response")
    return text[start:].strip()


def _normalize_clip(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    for key in ("start", "start_time"):
        if key in item:
            normalized["start"] = parse_timestamp(item[key])
            break
    for key in ("end", "end_time"):
        if key in item:
            normalized["end"] = parse_timestamp(item[key])
            break
    virality = normalized.get("virality") or {}
    if "virality_reasoning" in virality and "reasoning" not in virality:
        virality["reasoning"] = virality["virality_reasoning"]
    normalized["virality"] = Virality.model_validate(virality).model_dump()
    normalized["broll"] = [
        BrollPrompt(
            at=parse_timestamp(entry.get("at", entry.get("timestamp", normalized["start"]))),
            duration=float(entry.get("duration", 4.0)),
            prompt=str(entry.get("prompt") or entry.get("search_term") or entry.get("visual") or ""),
            negative_prompt=entry.get("negative_prompt"),
        ).model_dump()
        for entry in normalized.get("broll", []) or normalized.get("broll_opportunities", []) or []
        if entry.get("prompt") or entry.get("search_term") or entry.get("visual")
    ]
    normalized.setdefault("title", f"clip-{int(normalized.get('start', 0))}")
    normalized.setdefault("summary", normalized.get("reasoning", "Selected clip candidate."))
    normalized.setdefault("hook", normalized.get("text", normalized["title"]))
    normalized.setdefault("caption", normalized["hook"])
    return normalized
