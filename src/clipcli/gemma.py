"""Local multimodal planning with Gemma 4 on Apple Silicon (MLX).

Gemma 4's encoder-free architecture takes images and raw audio directly in the
LLM backbone, so promo/clip planning can watch keyframes and listen to the
soundtrack entirely on-device — no proxy upload, no API key.

Audio is consumed in two complementary ways: a WhisperX transcript (when
available) carries the full timeline as text, and Gemma itself listens to the
soundtrack in 30s clips — the processor truncates anything longer and mlx-vlm
accepts one clip per prompt — building a timestamped digest that the planning
pass reads alongside the frames.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from . import ffmpeg
from .gemini import (
    PROMO_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    _build_prompt,
    _build_promo_prompt,
    parse_plan_json,
    parse_promo_json,
)
from .models import ClipPlanResult, PromoPlan, Transcript, format_seconds

DEFAULT_GEMMA_MODEL = "mlx-community/gemma-4-12B-it-4bit"

PLANNING_AUDIO_SAMPLE_RATE = 16000

# Gemma 4's processor hears at most 750 audio tokens (40ms each) per attached
# clip; longer audio is silently truncated. Split the soundtrack into clips of
# at most this length so the model hears the whole video, not just the intro.
AUDIO_CHUNK_SECONDS = 30.0

LISTEN_SYSTEM_PROMPT = "You are a precise audio analyst for video production."

LISTEN_PROMPT = (
    "This audio clip covers {start} to {end} of a longer event video. "
    "Transcribe the key spoken content in its original language, and note product "
    "or project names, who seems to be speaking, announcements, applause, and the "
    "room's energy. Reply with 2-4 concise lines, no preamble."
)

_MODEL_CACHE: dict[str, tuple[Any, Any, dict[str, Any]]] = {}


def plan_clips_with_gemma(
    transcript: Transcript,
    *,
    clips: int,
    min_seconds: int,
    max_seconds: int,
    model: str = DEFAULT_GEMMA_MODEL,
    source: Path | None = None,
    work_dir: Path | None = None,
) -> ClipPlanResult:
    """Select clips locally; listen to the source audio so garbled transcripts still ground."""
    prompt = _build_prompt(transcript, clips=clips, min_seconds=min_seconds, max_seconds=max_seconds)
    if source is not None:
        chunks = _planning_audio_chunks(source, work_dir)
        digest = _describe_audio_chunks(model, chunks)
        prompt = (
            _digest_block(digest)
            + "\nThe transcript below is machine-generated; prefer your listening "
            "notes when they disagree.\n\n"
        ) + prompt
    text = _generate(
        model,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        images=[],
        audio=[],
        temperature=0.4,
    )
    return parse_plan_json(text, min_seconds=min_seconds, max_seconds=max_seconds)


def plan_promo_with_gemma(
    transcript: Transcript | None,
    *,
    video: Path | None = None,
    duration: float = 30.0,
    scenes: int = 6,
    language: str | None = None,
    video_duration: float | None = None,
    model: str = DEFAULT_GEMMA_MODEL,
    work_dir: Path | None = None,
    frame_count: int | None = None,
    document: Any | None = None,
    raw_path: Path | None = None,
) -> PromoPlan:
    """Plan a promo montage by watching keyframes and listening to the soundtrack on-device."""
    if transcript is None and video is None:
        raise ValueError("Promo planning needs a transcript, a video, or both.")

    frames: list[tuple[Path, float]] = []
    digest: list[str] = []
    if video is not None:
        media_dir = _media_dir(work_dir)
        count = frame_count or max(12, min(32, scenes * 4))
        frames = ffmpeg.extract_keyframes(
            video,
            media_dir / "frames",
            count=count,
            duration=video_duration,
        )
        chunks = _planning_audio_chunks(video, work_dir, duration=video_duration)
        digest = _describe_audio_chunks(model, chunks)

    prompt = _build_promo_prompt(
        transcript,
        duration=duration,
        scenes=scenes,
        language=language,
        with_video=False,
        video_duration=video_duration,
        document=document,
    )
    if frames:
        prompt = _media_preamble(frames, digest) + "\n\n" + prompt

    text = _generate(
        model,
        system=PROMO_SYSTEM_PROMPT,
        prompt=prompt,
        images=[path for path, _at in frames],
        audio=[],
        temperature=0.5,
    )
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text)
    return parse_promo_json(text, video_duration=video_duration)


def _media_preamble(frames: list[tuple[Path, float]], digest: list[str]) -> str:
    lines = [
        f"You are given {len(frames)} still frames sampled from the source video"
        + (" plus your own notes from listening to its full soundtrack." if digest else "."),
        "The frames are attached in order; their source-video timestamps are:",
    ]
    lines.extend(
        f"- Frame {index}: {format_seconds(at)} ({at:.1f}s)"
        for index, (_path, at) in enumerate(frames, start=1)
    )
    if digest:
        lines.append(_digest_block(digest))
        lines.append(
            "From these notes, identify the central announcement — what is being "
            "launched, by whom, and what it promises — and quote its vocabulary "
            "in the voiceover."
        )
    lines.append(
        "Anchor each scene's source range near frames with strong visuals; "
        "interpolate timestamps between frames when the best moment falls between samples."
    )
    return "\n".join(lines)


def _digest_block(digest: list[str]) -> str:
    lines = [
        "Your listening notes from the soundtrack, clip by clip "
        "(timestamps refer to the source video):"
    ]
    lines.extend(f"- {entry}" for entry in digest)
    return "\n".join(lines)


def _describe_audio_chunks(
    model: str,
    chunks: list[tuple[Path, float, float]],
) -> list[str]:
    """Listen to the soundtrack one Gemma-sized clip at a time and take notes."""
    digest: list[str] = []
    for path, start, end in chunks:
        text = _generate(
            model,
            system=LISTEN_SYSTEM_PROMPT,
            prompt=LISTEN_PROMPT.format(start=format_seconds(start), end=format_seconds(end)),
            images=[],
            audio=[path],
            temperature=0.1,
            max_tokens=220,
        )
        digest.append(f"[{format_seconds(start)}-{format_seconds(end)}] {' '.join(text.split())}")
    return digest


def _planning_audio_chunks(
    source: Path,
    work_dir: Path | None,
    *,
    duration: float | None = None,
) -> list[tuple[Path, float, float]]:
    """Split the soundtrack into Gemma-sized clips so none of it is truncated."""
    media_dir = _media_dir(work_dir)
    total = duration if duration is not None else ffmpeg.probe_duration(source)
    chunks: list[tuple[Path, float, float]] = []
    start = 0.0
    index = 0
    while start < total:
        length = min(AUDIO_CHUNK_SECONDS, total - start)
        if length < 1.0:
            break
        output = media_dir / f"audio-{index:02d}.wav"
        if not output.exists():
            ffmpeg.extract_clip_audio(
                source,
                output,
                start=start,
                duration=length,
                sample_rate=PLANNING_AUDIO_SAMPLE_RATE,
            )
        chunks.append((output, start, start + length))
        index += 1
        start += AUDIO_CHUNK_SECONDS
    return chunks


def _media_dir(work_dir: Path | None) -> Path:
    base = work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="clipcli-gemma-"))
    media = base / "gemma-media"
    media.mkdir(parents=True, exist_ok=True)
    return media


def _load_model(model: str) -> tuple[Any, Any, dict[str, Any]]:
    cached = _MODEL_CACHE.get(model)
    if cached is not None:
        return cached
    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install local planning with: pip install 'clipcli[local]'"
        ) from exc
    try:
        import mlx.core as mx

        # Long multimodal prompts on shared consumer machines: keep MLX's buffer
        # cache from growing unbounded so the OS doesn't thrash to swap.
        mx.set_cache_limit(2 * 1024**3)
    except Exception:
        pass
    loaded_model, processor = load(model)
    config = load_config(model)
    _MODEL_CACHE[model] = (loaded_model, processor, config)
    return _MODEL_CACHE[model]


def _generate(
    model: str,
    *,
    system: str,
    prompt: str,
    images: list[Path],
    audio: list[Path],
    temperature: float,
    max_tokens: int = 2048,
) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    loaded_model, processor, config = _load_model(model)
    # Gemma chat templates fold system guidance into the user turn.
    full_prompt = f"{system}\n\n{prompt}"
    formatted = apply_chat_template(
        processor,
        config,
        full_prompt,
        num_images=len(images),
        num_audios=len(audio),
    )
    result = generate(
        loaded_model,
        processor,
        formatted,
        image=[str(path) for path in images] or None,
        audio=[str(path) for path in audio] or None,
        max_tokens=max_tokens,
        temperature=temperature,
        # A whole soundtrack plus keyframes is a long prompt; small prefill
        # chunks keep Metal command buffers short so consumer Macs don't hit
        # GPU timeouts. (KV quantization is NYI for Gemma's rotating caches;
        # its sliding-window layers bound KV growth on their own.)
        prefill_step_size=512,
        verbose=False,
    )
    text = getattr(result, "text", None)
    return text if text is not None else str(result)
