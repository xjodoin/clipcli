from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import ValidationError

from .models import BrollPrompt, ClipPlan, ClipPlanResult, Transcript, Virality, parse_timestamp


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
