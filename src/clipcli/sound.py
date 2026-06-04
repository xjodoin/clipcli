from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx

from .models import ClipPlan, SoundSource


class SoundSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SoundCandidate:
    source: str
    id: str
    title: str
    author: str
    page_url: str
    preview_url: str
    license: str
    duration: float
    tags: list[str]
    score: float = 0.0


@dataclass(frozen=True)
class SoundAsset:
    path: Path
    metadata_path: Path
    candidate: SoundCandidate


def find_sound_bed(
    clip: ClipPlan,
    output_dir: Path,
    *,
    query: str | None = None,
    source: SoundSource = "freesound",
    max_candidates: int = 12,
) -> SoundAsset:
    return _find_audio(
        query or sound_query_for_clip(clip),
        output_dir,
        duration=clip.duration,
        source=source,
        max_candidates=max_candidates,
    )


def find_music(
    query: str,
    output_dir: Path,
    *,
    duration: float,
    source: SoundSource = "freesound",
    max_candidates: int = 15,
) -> SoundAsset:
    """Find a music bed for a montage: prefer tracks at least as long as the edit.

    Full-text search treats every word as a constraint, so progressively relax the
    query (and finally the duration floor) instead of failing on a wordy query.
    """
    terms = query.lower()
    if "music" not in terms:
        query = f"{query} music"
    words = query.split()
    queries = [query]
    if len(words) > 3:
        queries.append(" ".join(words[:2] + ["music"]))
    queries.extend(["uplifting corporate music", "inspiring background music"])
    attempts = [(candidate, max(15.0, duration)) for candidate in queries]
    attempts.extend((candidate, 8.0) for candidate in queries)
    last_error: SoundSearchError | None = None
    for attempt_query, min_duration in attempts:
        try:
            return _find_audio(
                attempt_query,
                output_dir,
                duration=duration,
                source=source,
                max_candidates=max_candidates,
                min_duration=min_duration,
                max_duration=max(60.0, duration * 6),
                profile="music",
            )
        except SoundSearchError as exc:
            last_error = exc
    raise SoundSearchError(f"No safe music candidates found for query: {query}") from last_error


def _find_audio(
    search_query: str,
    output_dir: Path,
    *,
    duration: float,
    source: SoundSource = "freesound",
    max_candidates: int = 12,
    min_duration: float | None = None,
    max_duration: float | None = None,
    profile: str = "bed",
) -> SoundAsset:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source == "freesound":
        candidates = search_freesound(
            search_query,
            clip_duration=duration,
            max_candidates=max_candidates,
            min_duration=min_duration,
            max_duration=max_duration,
            profile=profile,
        )
    else:
        raise SoundSearchError(f"Unsupported sound source: {source}")

    if not candidates:
        raise SoundSearchError(f"No safe sound candidates found for query: {search_query}")

    selected = max(candidates, key=lambda candidate: candidate.score)
    extension = _extension_from_url(selected.preview_url)
    audio_path = output_dir / f"{_safe_stem(selected.source)}-{_safe_stem(selected.id)}{extension}"
    metadata_path = output_dir / f"{audio_path.stem}.json"
    download_sound_preview(selected, audio_path)
    metadata_path.write_text(
        json.dumps(
            {
                "query": search_query,
                "selected": _candidate_json(selected),
                "license_note": (
                    "Verify the source page before publishing commercial content. "
                    "The audio was selected from metadata that did not indicate "
                    "NonCommercial or Sampling-only licensing."
                ),
            },
            indent=2,
        )
    )
    return SoundAsset(path=audio_path, metadata_path=metadata_path, candidate=selected)


def sound_query_for_clip(clip: ClipPlan) -> str:
    text = " ".join([clip.hook, clip.caption, clip.summary]).lower()
    if any(word in text for word in ("health", "sant", "medical", "care", "clinic")):
        return "subtle corporate technology ambient music"
    if any(word in text for word in ("launch", "product", "startup", "innovation")):
        return "modern corporate technology music"
    return "subtle background ambient music"


def search_freesound(
    query: str,
    *,
    clip_duration: float,
    max_candidates: int = 12,
    api_key: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    profile: str = "bed",
) -> list[SoundCandidate]:
    token = api_key or os.environ.get("FREESOUND_API_KEY") or os.environ.get("FREESOUND_TOKEN")
    if not token:
        raise SoundSearchError(
            "FREESOUND_API_KEY is not set. Create a Freesound API token, add it to "
            ".env.local, then run again."
        )

    low = int(min_duration if min_duration is not None else 8)
    high = int(max_duration if max_duration is not None else max(12, int(clip_duration * 2)))
    filter_parts = [f"duration:[{low} TO {high}]"]
    if profile == "music":
        # The best corporate/upbeat tracks on Freesound are mostly NonCommercial;
        # filter server-side so commercial-safe music is what gets ranked.
        filter_parts.append('license:("Attribution" OR "Creative Commons 0")')
    params = {
        "query": query,
        "fields": "id,name,username,license,duration,tags,previews,url,avg_rating,num_downloads",
        "filter": " ".join(filter_parts),
        "sort": "rating_desc",
        "page_size": str(max(1, max_candidates)),
    }
    url = "https://freesound.org/apiv2/search/text/?" + urlencode(params)
    headers = {"Authorization": f"Token {token}"}
    try:
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SoundSearchError(f"Freesound search failed: {exc}") from exc

    data = response.json()
    candidates = [
        _candidate_from_freesound(item, clip_duration=clip_duration, profile=profile)
        for item in data.get("results", [])
    ]
    candidates = [candidate for candidate in candidates if _is_commercially_plausible(candidate)]
    if profile == "music":
        candidates = [candidate for candidate in candidates if _is_promo_music(candidate)]
    return candidates


def download_sound_preview(candidate: SoundCandidate, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", candidate.preview_url, timeout=120) as response:
            response.raise_for_status()
            with output.open("wb") as file:
                for chunk in response.iter_bytes():
                    if chunk:
                        file.write(chunk)
    except httpx.HTTPError as exc:
        raise SoundSearchError(f"Sound preview download failed: {exc}") from exc
    return output


def _candidate_from_freesound(item: dict, *, clip_duration: float, profile: str = "bed") -> SoundCandidate:
    previews = item.get("previews") or {}
    preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
    if not preview_url:
        preview_url = previews.get("preview-hq-ogg") or previews.get("preview-lq-ogg") or ""
    duration = float(item.get("duration") or 0)
    license_name = str(item.get("license") or "")
    tags = [str(tag).lower() for tag in item.get("tags") or []]
    scorer = _score_music if profile == "music" else _score_freesound
    return SoundCandidate(
        source="freesound",
        id=str(item.get("id") or ""),
        title=str(item.get("name") or "untitled"),
        author=str(item.get("username") or "unknown"),
        page_url=str(item.get("url") or ""),
        preview_url=preview_url,
        license=license_name,
        duration=duration,
        tags=tags,
        score=scorer(item, duration=duration, clip_duration=clip_duration, license_name=license_name),
    )


def _score_freesound(
    item: dict,
    *,
    duration: float,
    clip_duration: float,
    license_name: str,
) -> float:
    rating = float(item.get("avg_rating") or 0)
    downloads = min(float(item.get("num_downloads") or 0), 10_000) / 10_000
    duration_fit = max(0.0, 1.0 - abs(duration - clip_duration) / max(clip_duration, 1.0))
    license_bonus = 0.5 if "zero" in license_name.lower() or "public domain" in license_name.lower() else 0.0
    terms = _candidate_terms(
        str(item.get("name") or ""),
        [str(tag).lower() for tag in item.get("tags") or []],
    )
    preferred = (
        "background",
        "music",
        "ambient",
        "ambience",
        "calm",
        "chill",
        "corporate",
        "technology",
        "hopeful",
        "uplifting",
        "piano",
        "guitar",
    )
    preferred_bonus = sum(0.12 for term in preferred if term in terms)
    return rating + downloads + duration_fit + license_bonus + min(preferred_bonus, 0.72)


def _score_music(
    item: dict,
    *,
    duration: float,
    clip_duration: float,
    license_name: str,
) -> float:
    """Score for a marketing montage bed: energetic and present, not wallpaper."""
    rating = float(item.get("avg_rating") or 0)
    downloads = min(float(item.get("num_downloads") or 0), 10_000) / 10_000
    duration_fit = 1.0 if duration >= clip_duration else duration / max(clip_duration, 1.0)
    license_bonus = 0.5 if "zero" in license_name.lower() or "public domain" in license_name.lower() else 0.0
    terms = _candidate_terms(
        str(item.get("name") or ""),
        [str(tag).lower() for tag in item.get("tags") or []],
    )
    energetic = (
        "upbeat",
        "uplifting",
        "energetic",
        "corporate",
        "inspiring",
        "inspirational",
        "motivational",
        "driving",
        "anthem",
        "pop",
        "rock",
        "electronic",
        "dance",
        "beat",
        "drums",
        "synth",
        "groove",
        "happy",
        "positive",
    )
    energetic_bonus = sum(0.2 for term in energetic if term in terms)
    passive = (
        "ambient",
        "ambience",
        "atmosphere",
        "atmospheric",
        "calm",
        "drone",
        "ethereal",
        "gentle",
        "meditation",
        "relax",
        "relaxing",
        "sleep",
        "soundscape",
        "tranquil",
    )
    passive_penalty = sum(0.35 for term in passive if term in terms)
    return rating + downloads + duration_fit + license_bonus + min(energetic_bonus, 1.2) - passive_penalty


def _is_promo_music(candidate: SoundCandidate) -> bool:
    """Reject non-music and pure-ambience results that survive the generic filter."""
    terms = _candidate_terms(candidate.title, candidate.tags)
    not_music = (
        "asmr",
        "birds",
        "drone",
        "forest",
        "lullaby",
        "meditation",
        "nature",
        "ocean",
        "river",
        "sleep",
        "soundscape",
        "wind",
    )
    return not any(term in terms for term in not_music)


def _is_commercially_plausible(candidate: SoundCandidate) -> bool:
    license_name = candidate.license.lower()
    if not candidate.preview_url or not candidate.page_url:
        return False
    blocked = ("noncommercial", "non-commercial", "by-nc", "/nc/", "sampling+", "sampling plus")
    if any(token in license_name for token in blocked):
        return False
    terms = _candidate_terms(candidate.title, candidate.tags)
    unsuitable = (
        "applause",
        "applaud",
        "clap",
        "chatter",
        "conference",
        "conversation",
        "creature",
        "creepy",
        "crowd",
        "dark",
        "demon",
        "drip",
        "field-recording",
        "groan",
        "horror",
        "noise",
        "oxygen",
        "rain",
        "siren",
        "street",
        "traffic",
        "walla",
        "water",
    )
    return not any(term in terms for term in unsuitable)


def _candidate_terms(title: str, tags: list[str]) -> str:
    return " ".join([title.lower(), *[tag.lower() for tag in tags]]).replace("_", " ")


def _candidate_json(candidate: SoundCandidate) -> dict:
    data = asdict(candidate)
    data["tags"] = list(candidate.tags)
    return data


def _extension_from_url(url: str) -> str:
    lower = url.split("?", 1)[0].lower()
    if lower.endswith(".ogg"):
        return ".ogg"
    if lower.endswith(".wav"):
        return ".wav"
    return ".mp3"


def _safe_stem(value: str) -> str:
    stem = "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
    return stem[:80] or "sound"
